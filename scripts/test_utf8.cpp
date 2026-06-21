/**
 *  @brief  UTF-8 counting/newline/whitespace equivalence, normalization, and C++ API semantics.
 *  @file   scripts/test_utf8.cpp
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
#include <cctype>  // `std::isxdigit`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <sstream>       // `std::ostringstream`
#include <string>        // Baseline
#include <string_view>   // Baseline
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "test_stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

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
 *  @brief Known-answer + C++ API coverage for UTF-8 counting, nth-character finding, codepoint
 *         iteration, boundary detection, streaming unpacking, and Unicode-aware splitting.
 *
 *  Exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through
 *  the C++ `sz::string_view` wrappers, so a regression that the serial-vs-SIMD agreement tests would
 *  miss - because both share a wrong constant - is still caught against an external ground truth. This
 *  is the isolated coverage for `sz_utf8_newlines`, `sz_utf8_whitespaces`, `sz_utf8_find_nth`
 *  (whose SIMD variants are otherwise untested), and `sz_utf8_unpack_chunk` (otherwise untested in C++).
 */
void test_utf8_unit() {
    std::printf("  - testing UTF-8 known-answer vectors...\n");

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
    assert(sz::string_view(mixed, mixed_length).utf8_count() == 3u); // C++ wrapper

    // `sz_utf8_find_nth`: codepoint starts land at byte offsets 0, 1, 3; the 4th (n=3) is one-past-the-end.
    assert(sz_utf8_find_nth(mixed, mixed_length, 0u) == mixed + 0);           // Dispatched: 'a' at byte 0
    assert(sz_utf8_find_nth(mixed, mixed_length, 1u) == mixed + 1);           // Dispatched: 'ß' at byte 1
    assert(sz_utf8_find_nth(mixed, mixed_length, 2u) == mixed + 3);           // Dispatched: '中' at byte 3
    assert(sz_utf8_find_nth(mixed, mixed_length, 3u) == SZ_NULL_CHAR);        // Dispatched: beyond end
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 0u) == mixed + 0);    // Manual: serial kernel
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 2u) == mixed + 3);    // Manual: serial kernel
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 3u) == SZ_NULL_CHAR); // Manual: serial kernel
#if SZ_USE_HASWELL
    assert(sz_utf8_find_nth_haswell(mixed, mixed_length, 2u) == mixed + 3);    // Manual: haswell kernel
    assert(sz_utf8_find_nth_haswell(mixed, mixed_length, 3u) == SZ_NULL_CHAR); // Manual: haswell kernel
#endif
#if SZ_USE_ICELAKE
    assert(sz_utf8_find_nth_icelake(mixed, mixed_length, 2u) == mixed + 3);    // Manual: icelake kernel
    assert(sz_utf8_find_nth_icelake(mixed, mixed_length, 3u) == SZ_NULL_CHAR); // Manual: icelake kernel
#endif
    assert(sz::string_view(mixed, mixed_length).utf8_find_nth(2u) == 3u); // C++ wrapper (byte offset)

    // `sz_utf8_unpack_chunk` is a streaming chunk decoder: per its documented contract each call unpacks
    // a prefix of the remaining bytes - not necessarily all of it when the chunk mixes codepoint widths -
    // and reports how many runes it produced and how far the cursor advanced. Driven as designed (in a
    // loop) it must decode "aß中" into exactly {0x61, 0xDF, 0x4E2D}; every call must advance the cursor so
    // the loop terminates.
    sz_rune_t const mixed_expected[] = {0x61u, 0xDFu, 0x4E2Du};
    auto check_unpack_streaming = [&](sz_cptr_t (*unpack)(sz_cptr_t, sz_size_t, sz_rune_t *, sz_size_t, sz_size_t *)) {
        sz_rune_t collected[64];
        sz_size_t collected_count = 0;
        sz_cptr_t position = mixed;
        sz_cptr_t const end = mixed + mixed_length;
        while (position < end) {
            sz_rune_t chunk_runes[64];
            sz_size_t chunk_count = 0;
            sz_cptr_t const next = unpack(position, (sz_size_t)(end - position), chunk_runes, 64u, &chunk_count);
            assert(next > position); // Must advance, otherwise the streaming loop would never terminate
            for (sz_size_t index = 0; index != chunk_count; ++index) collected[collected_count++] = chunk_runes[index];
            position = next;
        }
        assert(collected_count == 3u);
        for (sz_size_t index = 0; index != 3u; ++index) assert(collected[index] == mixed_expected[index]);
    };
    check_unpack_streaming(sz_utf8_unpack_chunk);        // Dispatched (automatic kernel)
    check_unpack_streaming(sz_utf8_unpack_chunk_serial); // Manual propagation to the serial kernel
#if SZ_USE_ICELAKE
    check_unpack_streaming(sz_utf8_unpack_chunk_icelake); // Manual: icelake kernel
#endif

    // `sz_utf8_norm` / `sz_utf8_norm_violation`: "café" with a precomposed é (U+00E9) is already NFC, but
    // breaks NFD (é decomposes into e + U+0301). NFC normalization is a no-op; NFD expands it to 5 bytes.
    char const cafe_nfc[] = "caf\xC3\xA9"; // U+00E9 (precomposed é), 5 bytes
    sz_size_t const cafe_length = (sz_size_t)(sizeof(cafe_nfc) - 1);
    assert(sz_utf8_norm_violation(cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           SZ_NULL_CHAR); // Dispatched: already NFC
    assert(sz_utf8_norm_violation(cafe_nfc, cafe_length, sz_normal_form_nfd_k) != SZ_NULL_CHAR); // Dispatched: not NFD
    assert(sz_utf8_norm_violation_serial(cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           SZ_NULL_CHAR); // Manual: serial
    assert(sz_utf8_norm_violation_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k) !=
           SZ_NULL_CHAR); // Manual: serial
    {
        char norm_buffer[64];
        sz_size_t const nfc_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfc_k, norm_buffer);
        assert(nfc_length == cafe_length && std::memcmp(norm_buffer, cafe_nfc, cafe_length) == 0); // NFC no-op
        sz_size_t const nfd_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfd_k, norm_buffer);
        assert(nfd_length == 6u); // "caf" + 'e' + U+0301 (2-byte combining acute)
        sz_size_t const nfd_length_serial = sz_utf8_norm_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k,
                                                                norm_buffer); // Manual: serial
        assert(nfd_length_serial == 6u);
    }

    // C++ API battery: character counting vs byte length through the `sz::string_view` wrappers.
    assert("hello"_sv.utf8_count() == 5);
    assert("hello"_sv.size() == 5);

    // ASCII text: bytes == characters
    assert("Hello World"_sv.utf8_count() == 11);
    assert(sz::string_view("").utf8_count() == 0);

    // Mixed ASCII and multi-byte characters
    assert(sz::string_view("Hello 世界").utf8_count() == 8); // "Hello " (6) + "世界" (2 chars)
    assert(sz::string_view("Hello 世界").size() == 12);      // "Hello " (6) + "世界" (6 bytes)

    // Emojis (4-byte UTF-8)
    assert(sz::string_view("Hello 😀").utf8_count() == 7); // "Hello " (6) + emoji (1 char)
    assert(sz::string_view("Hello 😀").size() == 10);      // "Hello " (6) + emoji (4 bytes)
    assert(sz::string_view("😀😁😂").utf8_count() == 3);
    assert(sz::string_view("😀😁😂").size() == 12);

    // Cyrillic (2-byte UTF-8)
    assert(sz::string_view("Привет").utf8_count() == 6);
    assert(sz::string_view("Привет").size() == 12);

    // Finding byte offset of nth character
    {
        sz::string_view text = "Hello";
        assert(text.utf8_find_nth(0) == 0);                     // First char at byte 0
        assert(text.utf8_find_nth(1) == 1);                     // Second char at byte 1
        assert(text.utf8_find_nth(4) == 4);                     // Fifth char at byte 4
        assert(text.utf8_find_nth(5) == sz::string_view::npos); // Beyond end
        assert(text.utf8_find_nth(100) == sz::string_view::npos);
    }

    {
        sz::string_view text = "Hello 世界";
        assert(text.utf8_find_nth(0) == 0); // 'H' at byte 0
        assert(text.utf8_find_nth(5) == 5); // ' ' at byte 5
        assert(text.utf8_find_nth(6) == 6); // '世' at byte 6
        assert(text.utf8_find_nth(7) == 9); // '界' at byte 9
        assert(text.utf8_find_nth(8) == sz::string_view::npos);
    }

    {
        sz::string_view text = "😀😁😂";
        assert(text.utf8_find_nth(0) == 0); // First emoji at byte 0
        assert(text.utf8_find_nth(1) == 4); // Second emoji at byte 4
        assert(text.utf8_find_nth(2) == 8); // Third emoji at byte 8
        assert(text.utf8_find_nth(3) == sz::string_view::npos);
    }

    // Iterate over UTF-32 codepoints
    {
        auto chars = [](char const *t) {
            return sz::string_view(t).utf8_runes().template to<std::vector<sz_rune_t>>();
        };

        // Basic ASCII and edge cases
        let_assert(auto c = chars("Hello"), c.size() == 5 && c[0] == 'H' && c[4] == 'o');
        let_assert(auto c = chars(""), c.size() == 0);
        let_assert(auto c = chars("A"), c.size() == 1 && c[0] == 'A');

        // CJK (3-byte UTF-8)
        let_assert(auto c = chars("世界"), c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x754C);
        let_assert(auto c = chars("你好"), c.size() == 2 && c[0] == 0x4F60 && c[1] == 0x597D);

        // Cyrillic (2-byte UTF-8)
        let_assert(auto c = chars("Привет"), c.size() == 6 && c[0] == 0x041F && c[5] == 0x0442);

        // Arabic/RTL (2-byte UTF-8)
        let_assert(auto c = chars("مرحبا"), c.size() == 5 && c[0] == 0x0645 && c[4] == 0x0627);

        // Hebrew/RTL (2-byte UTF-8)
        let_assert(auto c = chars("שלום"), c.size() == 4 && c[0] == 0x05E9 && c[3] == 0x05DD);

        // Thai (3-byte UTF-8)
        let_assert(auto c = chars("สวัสดี"), c.size() == 6 && c[0] == 0x0E2A);

        // Devanagari/Hindi (3-byte UTF-8)
        let_assert(auto c = chars("नमस्ते"), c.size() == 6 && c[0] == 0x0928);

        // Emoji: basic smileys (4-byte UTF-8)
        let_assert(auto c = chars("😀😁😂"), c.size() == 3 && c[0] == 0x1F600 && c[2] == 0x1F602);

        // Emoji: with variation selector
        let_assert(auto c = chars("❤️"), c.size() == 2 && c[0] == 0x2764 && c[1] == 0xFE0F);

        // Emoji: various categories
        let_assert(auto c = chars("🚀🎉🔥"), c.size() == 3 && c[0] == 0x1F680);

        // Maximum valid Unicode codepoint (U+10FFFF)
        let_assert(auto c = chars("\xF4\x8F\xBF\xBF"), c.size() == 1 && c[0] == 0x10FFFF);

        // Deseret alphabet (4-byte UTF-8, U+10400 range)
        let_assert(auto c = chars("𐐷"), c.size() == 1 && c[0] == 0x10437);

        // Mixed scripts
        let_assert(auto c = chars("Hello世界"), c.size() == 7 && c[4] == 'o' && c[5] == 0x4E16);
        let_assert(auto c = chars("a𐐷b"), c.size() == 3 && c[0] == 'a' && c[1] == 0x10437 && c[2] == 'b');

        // Zero-width characters
        let_assert(auto c = chars("a​b"), c.size() == 3 && c[0] == 'a' && c[1] == 0x200B && c[2] == 'b');
        let_assert(auto c = chars("﻿"), c.size() == 1 && c[0] == 0xFEFF); // BOM

        // Combining diacritics (é as e + combining acute)
        let_assert(auto c = chars("é"), c.size() == 2 && c[0] == 'e' && c[1] == 0x0301);

        // Precomposed vs decomposed normalization
        let_assert(auto c = chars("é"), c.size() == 1 && c[0] == 0x00E9); // Precomposed

        // Missing transitions: 1→2, 2→1, 2→3, 3→2, 2→4, 4→2, 3→4, 4→3
        let_assert(auto c = chars("aП"), c.size() == 2 && c[0] == 'a' && c[1] == 0x041F);       // 1→2
        let_assert(auto c = chars("Пa"), c.size() == 2 && c[0] == 0x041F && c[1] == 'a');       // 2→1
        let_assert(auto c = chars("П世"), c.size() == 2 && c[0] == 0x041F && c[1] == 0x4E16);   // 2→3
        let_assert(auto c = chars("世П"), c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x041F);   // 3→2
        let_assert(auto c = chars("П😀"), c.size() == 2 && c[0] == 0x041F && c[1] == 0x1F600);  // 2→4
        let_assert(auto c = chars("😀П"), c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x041F);  // 4→2
        let_assert(auto c = chars("世😀"), c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x1F600); // 3→4
        let_assert(auto c = chars("😀世"), c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x4E16); // 4→3

        // Extended transitions with same-length runs
        let_assert(auto c = chars("ПРС"), c.size() == 3 && c[0] == 0x041F && c[2] == 0x0421);    // 2→2→2
        let_assert(auto c = chars("世界人"), c.size() == 3 && c[0] == 0x4E16 && c[2] == 0x4EBA); // 3→3→3

        // Asymmetric alternating patterns (2:3, 3:2) - stress homogeneity assumption
        let_assert(auto c = chars("xxПППxxППП"), c.size() == 10);           // 2 ASCII, 3 Cyrillic
        let_assert(auto c = chars("xxxППxxxПП"), c.size() == 10);           // 3 ASCII, 2 Cyrillic
        let_assert(auto c = chars("xx世世世xx世世世"), c.size() == 10);     // 2 ASCII, 3 CJK
        let_assert(auto c = chars("ПП世世世ПП世世世"), c.size() == 10);     // 2 Cyrillic, 3 CJK
        let_assert(auto c = chars("世世😀😀😀世世😀😀😀"), c.size() == 10); // 2 CJK, 3 Emoji
        let_assert(auto c = chars("xxx😀😀xxx😀😀"), c.size() == 10);       // 3 ASCII, 2 Emoji

        // Pathological mixed patterns
        let_assert(auto c = chars("xxПППП世世世世😀😀😀😀😀"), c.size() == 15); // 2-4-4-5
        let_assert(auto c = chars("xxПППxx😀😀😀😀世世世ПП"), c.size() == 16);  // 2-3-2-4-3-2

        // Extended asymmetric: 30x "xxППП" = 150 chars, 210 bytes (crosses multiple 64-byte chunks)
        scope_assert(std::string asym_long, for (int i = 0; i < 30; ++i) asym_long += "xxППП",
                     sz::string_view(asym_long).utf8_count() == 150);
    }

    // Test 64-byte chunk boundaries and batch limits
    {
        // Critical 63, 64, 65 byte boundaries
        let_assert(std::string s63(63, 'x'), sz::string_view(s63).utf8_runes().size() == 63);
        let_assert(std::string s64(64, 'x'), sz::string_view(s64).utf8_runes().size() == 64);
        let_assert(std::string s65(65, 'x'), sz::string_view(s65).utf8_runes().size() == 65);

        // ASCII batch limit: 16 characters max per Ice Lake iteration
        let_assert(std::string s17(17, 'x'), sz::string_view(s17).utf8_runes().size() == 17);
        let_assert(std::string s20(20, 'x'), sz::string_view(s20).utf8_runes().size() == 20);

        // 2-byte batch limit: 32 characters (64 bytes) max per iteration
        scope_assert(std::string cyr32, for (int i = 0; i < 32; ++i) cyr32 += "П",
                     sz::string_view(cyr32).utf8_count() == 32);
        scope_assert(std::string cyr33, for (int i = 0; i < 33; ++i) cyr33 += "П",
                     sz::string_view(cyr33).utf8_count() == 33);

        // 3-byte batch limit: 16 characters (48 bytes) max per iteration
        scope_assert(std::string cjk16, for (int i = 0; i < 16; ++i) cjk16 += "世",
                     sz::string_view(cjk16).utf8_count() == 16);
        scope_assert(std::string cjk17, for (int i = 0; i < 17; ++i) cjk17 += "世",
                     sz::string_view(cjk17).utf8_count() == 17);

        // 4-byte batch limit: 16 characters (64 bytes) max per iteration
        scope_assert(std::string emoji16, for (int i = 0; i < 16; ++i) emoji16 += "😀",
                     sz::string_view(emoji16).utf8_count() == 16);
        scope_assert(std::string emoji17, for (int i = 0; i < 17; ++i) emoji17 += "😀",
                     sz::string_view(emoji17).utf8_count() == 17);

        // Asymmetric at chunk boundary: 60 ASCII + "ПП世" = 63 chars, 67 bytes
        scope_assert(std::string boundary_asym(60, 'x'), boundary_asym += "ПП世",
                     sz::string_view(boundary_asym).utf8_count() == 63);

        // Test sequences exceeding batch limits
        // 100 consecutive 2-byte (exceeds 32-char limit 3x)
        scope_assert(std::string cyr100, for (int i = 0; i < 100; ++i) cyr100 += "П",
                     sz::string_view(cyr100).utf8_runes().size() == 100);

        // 50 consecutive 3-byte (exceeds 16-char limit 3x)
        scope_assert(std::string cjk50, for (int i = 0; i < 50; ++i) cjk50 += "世",
                     sz::string_view(cjk50).utf8_runes().size() == 50);

        // 50 consecutive 4-byte (exceeds 16-char limit 3x)
        scope_assert(std::string emoji50, for (int i = 0; i < 50; ++i) emoji50 += "😀",
                     sz::string_view(emoji50).utf8_runes().size() == 50);

        // Asymmetric overflow: 20x (2 ASCII + 3 Cyrillic) = 100 chars, 140 bytes
        scope_assert(std::string overflow_asym, for (int i = 0; i < 20; ++i) overflow_asym += "aaПРС",
                     sz::string_view(overflow_asym).utf8_count() == 100);

        // Test transitions at chunk boundaries
        // 63 bytes ASCII + 2-byte char (transition at 64-byte boundary)
        scope_assert(std::string boundary_test(63, 'x'), boundary_test += "П",
                     sz::string_view(boundary_test).utf8_runes().size() == 64);

        // Asymmetric spanning boundary: 60 ASCII + 24 Cyrillic = 84 chars, 108 bytes
        scope_assert(
            std::string span_asym,
            {
                for (int i = 0; i < 30; ++i) span_asym += "aa";
                for (int i = 0; i < 8; ++i) span_asym += "ПРС";
            },
            sz::string_view(span_asym).utf8_count() == 84);

        // Transition exactly at 64-byte boundary
        scope_assert(std::string exact_boundary(64, 'x'), exact_boundary += "П世😀",
                     sz::string_view(exact_boundary).utf8_count() == 67);

        // A buffer well past the L2 cache (>2 MB) of mixed-width codepoints: every repeat contributes one ASCII
        // 'x', one 2-byte 'П', one 3-byte '世', and one 4-byte '😀' - 4 codepoints in 10 bytes. The dispatched
        // and C++ counts must equal the serial reference and the exact known codepoint total over the long run.
        {
            std::string l2_mixed;
            std::size_t const l2_repeats = 220000; // 220000 * 10 bytes = 2.2 MB > a typical 256 KB-1 MB L2
            char const l2_unit[] = "x\xD0\x9F\xE4\xB8\xAD\xF0\x9F\x98\x80"; // 'x' + 'П' + '世' + '😀', 10 bytes
            l2_mixed.reserve(l2_repeats * (sizeof(l2_unit) - 1));
            for (std::size_t repeat = 0; repeat != l2_repeats; ++repeat) l2_mixed += l2_unit;
            sz_size_t const expected_codepoints = (sz_size_t)(l2_repeats * 4);
            sz_size_t const count_serial = sz_utf8_count_serial(l2_mixed.data(), l2_mixed.size());
            assert(count_serial == expected_codepoints);
            assert(sz_utf8_count(l2_mixed.data(), l2_mixed.size()) == count_serial); // Dispatched matches serial
            assert(sz::string_view(l2_mixed).utf8_count() == count_serial);          // C++ wrapper matches serial
        }
    }

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
        sz::string str = "Hello 世界";
        assert(str.utf8_count() == 8);
        assert(str.utf8_find_nth(6) == 6);
        let_assert(auto c = str.utf8_runes().template to<std::vector<sz_rune_t>>(), c.size() == 8 && c[6] == 0x4E16);

        sz::string multiline = "a\nb\nc";
        let_assert(auto l = multiline.utf8_lines().template to<std::vector<std::string>>(),
                   l.size() == 3 && l[1] == "b");

        sz::string words_str = "foo bar baz";
        let_assert(auto w = words_str.utf8_tokens().template to<std::vector<std::string>>(),
                   w.size() == 3 && w[2] == "baz");
    }

    // Test Unicode case folding
    {
        auto uncased_fold = [](sz::string s) -> sz::string {
            assert(s.try_utf8_uncased_fold());
            return s;
        };

        // ASCII uppercase to lowercase
        assert(uncased_fold("HELLO WORLD") == "hello world");
        assert(uncased_fold("ABC") == "abc");
        assert(uncased_fold("abc") == "abc"); // already lowercase
        assert(uncased_fold("123") == "123"); // no change for digits
        assert(uncased_fold("") == "");       // empty string

        // German Eszett - one-to-many expansion
        assert(uncased_fold("\xC3\x9F") == "ss");    // U+00DF -> ss
        assert(uncased_fold("STRAẞE") == "strasse"); // Capital U+1E9E -> ss

        // Cyrillic uppercase to lowercase
        assert(uncased_fold("ПРИВЕТ") == "привет");

        // Greek uppercase to lowercase
        assert(uncased_fold("ΑΒΓΔ") == "αβγδ");

        // Latin Extended characters
        assert(uncased_fold("ÀÁÂ") == "àáâ");

        // Armenian
        assert(uncased_fold("Ա") == "ա"); // U+0531 -> U+0561

        // Mixed case preservation for non-alphabetic
        assert(uncased_fold("Hello 123 World!") == "hello 123 world!");

        // Unicode characters without case folding should pass through unchanged
        assert(uncased_fold("日本語") == "日本語"); // Japanese (no case)
        assert(uncased_fold("中文") == "中文");     // Chinese (no case)
    }
}

/**
 *  @brief Test ligature/expansion matching semantics for UTF-8 uncased search.
 *
 *  SEMANTIC: fold(needle) should be a substring of fold(haystack).
 *  Offsets are reported in the ORIGINAL (pre-folded) haystack.
 */
void test_utf8_ligature_unit() {
    using str = sz::string_view;

    // Ligature in haystack, ASCII needle

    // "fi" in "ﬃ": fold("ﬃ")="ffi", "fi" is suffix of "ffi" → MATCH
    let_assert(auto m = str("\xEF\xAC\x83").utf8_uncased_find("fi"), m.offset == 0);

    // "ff" in "ﬃ": fold("ﬃ")="ffi", "ff" is prefix of "ffi" → MATCH
    let_assert(auto m = str("\xEF\xAC\x83").utf8_uncased_find("ff"), m.offset == 0);

    // "ffi" in "ﬃ": exact match → MATCH
    let_assert(auto m = str("\xEF\xAC\x83").utf8_uncased_find("ffi"), m.offset == 0);

    // "if" in "ﬃf": fold("ﬃf")="ffif", "if" at folded position 2 → MATCH
    let_assert(auto m = str("\xEF\xAC\x83" "f").utf8_uncased_find("if"), m.offset != str::npos);

    // "lf" in "ﬂﬃ": fold("ﬂﬃ")="flffi", "lf" at folded position 1 → MATCH
    let_assert(auto m = str("\xEF\xAC\x82\xEF\xAC\x83").utf8_uncased_find("lf"), m.offset != str::npos);

    // "xfi" in "xﬃ": fold("xﬃ")="xffi", "xfi" is NOT a substring → NO MATCH
    let_assert(auto m = str("x\xEF\xAC\x83").utf8_uncased_find("xfi"), m.offset == str::npos);

    // Ligature in needle, ASCII haystack

    // "ﬂ" in "ffll": fold("ﬂ")="fl", "fl" at position 1 → MATCH
    let_assert(auto m = str("ffll").utf8_uncased_find("\xEF\xAC\x82"), m.offset == 1);

    // "ﬁ" in "fil": fold("ﬁ")="fi", "fi" at position 0 → MATCH
    let_assert(auto m = str("fil").utf8_uncased_find("\xEF\xAC\x81"), m.offset == 0);

    // "ﬁ" in "ﬀi": fold("ﬁ")="fi", fold("ﬀi")="ffi", "fi" at folded position 1 → MATCH
    let_assert(auto m = str("\xEF\xAC\x80i").utf8_uncased_find("\xEF\xAC\x81"), m.offset != str::npos);

    // Eszett (ß) cases

    // "ss" in "ß": fold("ß")="ss" → exact MATCH
    let_assert(auto m = str("\xC3\x9F").utf8_uncased_find("ss"), m.offset == 0);

    // "s" in "ß": fold("ß")="ss", "s" is substring → MATCH
    let_assert(auto m = str("\xC3\x9F").utf8_uncased_find("s"), m.offset == 0);

    // "ß" in "ss": fold("ß")="ss", exact match → MATCH
    let_assert(auto m = str("ss").utf8_uncased_find("\xC3\x9F"), m.offset == 0);
}

/** @brief Golden checks for UTF-8 (UAX #29) word-boundary detection. */
void test_utf8_words_unit() {

    // Test Unicode word boundary detection (TR29 Word_Break)
    {
        // ASCII letters are word chars
        assert(sz_rune_is_word_char('A') == sz_true_k);
        assert(sz_rune_is_word_char('Z') == sz_true_k);
        assert(sz_rune_is_word_char('a') == sz_true_k);
        assert(sz_rune_is_word_char('z') == sz_true_k);

        // ASCII digits are word chars
        assert(sz_rune_is_word_char('0') == sz_true_k);
        assert(sz_rune_is_word_char('9') == sz_true_k);

        // ASCII underscore and mid-word punctuation
        assert(sz_rune_is_word_char('_') == sz_true_k);
        assert(sz_rune_is_word_char('\'') == sz_true_k); // apostrophe (mid-word)

        // ASCII whitespace and punctuation are NOT word chars
        assert(sz_rune_is_word_char(' ') == sz_false_k);
        assert(sz_rune_is_word_char('\n') == sz_false_k);
        assert(sz_rune_is_word_char('\t') == sz_false_k);
        assert(sz_rune_is_word_char('!') == sz_false_k);
        assert(sz_rune_is_word_char('@') == sz_false_k);
        assert(sz_rune_is_word_char('-') == sz_false_k);
        assert(sz_rune_is_word_char('/') == sz_false_k);

        // Latin Extended characters are word chars
        assert(sz_rune_is_word_char(0x00C0) == sz_true_k); // À
        assert(sz_rune_is_word_char(0x00E9) == sz_true_k); // é
        assert(sz_rune_is_word_char(0x00DF) == sz_true_k); // ß
        assert(sz_rune_is_word_char(0x0100) == sz_true_k); // Latin Extended-A start
        assert(sz_rune_is_word_char(0x017F) == sz_true_k); // Latin Extended-A end

        // Greek letters are word chars
        assert(sz_rune_is_word_char(0x0391) == sz_true_k); // Α (Alpha)
        assert(sz_rune_is_word_char(0x03B1) == sz_true_k); // α (alpha)
        assert(sz_rune_is_word_char(0x03C9) == sz_true_k); // ω (omega)

        // Cyrillic letters are word chars
        assert(sz_rune_is_word_char(0x0410) == sz_true_k); // А
        assert(sz_rune_is_word_char(0x0430) == sz_true_k); // а
        assert(sz_rune_is_word_char(0x044F) == sz_true_k); // я

        // Hebrew letters are word chars
        assert(sz_rune_is_word_char(0x05D0) == sz_true_k); // א (Alef)
        assert(sz_rune_is_word_char(0x05EA) == sz_true_k); // ת (Tav)

        // Arabic letters are word chars
        assert(sz_rune_is_word_char(0x0627) == sz_true_k); // ا (Alef)
        assert(sz_rune_is_word_char(0x0628) == sz_true_k); // ب (Ba)

        // CJK ideographs are boundaries (NOT word chars for TR29)
        assert(sz_rune_is_word_char(0x4E00) == sz_false_k); // 一
        assert(sz_rune_is_word_char(0x4E2D) == sz_false_k); // 中
        assert(sz_rune_is_word_char(0x6587) == sz_false_k); // 文
        assert(sz_rune_is_word_char(0x9FFF) == sz_false_k); // CJK last

        // Hangul syllables ARE word chars
        assert(sz_rune_is_word_char(0xAC00) == sz_true_k); // 가 (first)
        assert(sz_rune_is_word_char(0xD7A3) == sz_true_k); // last

        // Spaces and punctuation are boundaries
        assert(sz_rune_is_word_char(0x2000) == sz_false_k); // En quad
        assert(sz_rune_is_word_char(0x2014) == sz_false_k); // Em dash
        assert(sz_rune_is_word_char(0x3000) == sz_false_k); // Ideographic space

        // Emoji are boundaries
        assert(sz_rune_is_word_char(0x1F600) == sz_false_k); // 😀
        assert(sz_rune_is_word_char(0x1F4A9) == sz_false_k); // 💩

        // Edge cases
        assert(sz_rune_is_word_char(0x0000) == sz_false_k); // NUL
        assert(sz_rune_is_word_char(0x007F) == sz_false_k); // DEL
        assert(sz_rune_is_word_char(0xFFFF) == sz_false_k); // BMP max
    }

    // Segment `text` into UAX-29 words through one of the plural kernels.
    using word_boundaries_t = sz_size_t (*)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *);
    auto uax29_segments = [](word_boundaries_t boundaries, sz::string_view text) {
        std::vector<sz_size_t> starts(text.size() + 1), lengths(text.size() + 1);
        sz_size_t consumed = 0;
        sz_size_t count = boundaries(text.data(), text.size(), starts.data(), lengths.data(), text.size() + 1,
                                     &consumed);
        std::vector<std::string> words;
        for (sz_size_t i = 0; i != count; ++i) words.emplace_back(text.data() + starts[i], lengths[i]);
        return words;
    };
    auto words = [&](sz::string_view text) { return uax29_segments(sz_utf8_words, text); };

    // Forward segmentation against hand-checked UAX-29 expectations.
    let_assert(auto w = words(""), w.empty());
    let_assert(auto w = words("a"), w.size() == 1 && w[0] == "a");
    let_assert(auto w = words("Hello, world!"),
               w.size() == 5 && w[0] == "Hello" && w[1] == "," && w[2] == " " && w[3] == "world" && w[4] == "!");
    let_assert(auto w = words("don't"), w.size() == 1 && w[0] == "don't"); // WB6/WB7: apostrophe between letters
    let_assert(auto w = words("3,14"), w.size() == 1 && w[0] == "3,14");   // WB11/WB12: comma between digits
    let_assert(auto w = words("3,"), w.size() == 2 && w[0] == "3" && w[1] == ","); // digit then non-digit → break
    let_assert(auto w = words("can't_stop"), w.size() == 1);               // underscore is ExtendNumLet (WB13a/b)
    let_assert(auto w = words("a\r\nb"), w.size() == 3 && w[1] == "\r\n"); // WB3: CR × LF stay together
    let_assert(auto w = words("你好"), w.size() == 2 && w[0] == "你" && w[1] == "好"); // CJK: each is its own word

    // For each input the active backend, the serial reference, and the forward C++ range must agree.
    auto check_consistency = [&](sz::string_view text) {
        auto forward = uax29_segments(sz_utf8_words, text);
        assert(uax29_segments(sz_utf8_words_serial, text) == forward && "backend ≠ serial reference");
        assert(text.utf8_words().template to<std::vector<std::string>>() == forward && "utf8_words ≠ kernel");
    };
    for (sz::string_view text : {"", "a", "Hello, world!", "don't", "l'avion", "3,14", "1,2,3", "3,", "can't_stop",
                                 "a\r\nb", "Größe привет мир 你好 42", "the quick brown fox, really!"})
        check_consistency(text);

    // Multi-window seam regressions: WB15/16 Regional_Indicator parity and WB6/7/11/12 Mid-bridge carry once
    // miscounted across the 64-byte window boundary. The single-window inputs above can never exercise a seam, so
    // these byte vectors (each > 64 bytes) are the regression guard — serial and icelake must still agree exactly.
    auto from_hex = [](char const *h) {
        std::string out;
        auto nibble = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
        for (; h[0] && h[1]; h += 2) out.push_back(static_cast<char>(nibble(h[0]) * 16 + nibble(h[1])));
        return out;
    };
    for (
        char const *seam_hex :
        {// `RI RI RI` after a newline, straddling the seam (parity miscount dropped the boundary at byte 54).
         "E382AB2D0AF09F87A662C2ADF09F87A6CC800A5FF09F87A6F09F87A6C2AD0ACC88F09F87BAF09F8FBBCC88C2ADE281A0CC88F09F" "87"
                                                                                                                    "A6"
                                                                                                                    "F0"
                                                                                                                    "9F"
                                                                                                                    "87"
                                                                                                                    "A6"
                                                                                                                    "E4"
                                                                                                                    "B8"
                                                                                                                    "AD"
                                                                                                                    "E2"
                                                                                                                    "81"
                                                                                                                    "A0"
                                                                                                                    "5"
                                                                                                                    "F",
         // 4-RI run with an inner Word_Joiner: the parity seed must thread across the seam (spurious break at 57).
         "5F27F09F87BAE4B8ADCC81F09F87A6F09F87A65FCC885FD7902DE29382CC8062F09F87A6E281A0F09F87BACC80CC88F09F8FBBF0" "9F"
                                                                                                                    "87"
                                                                                                                    "BA"
                                                                                                                    "CC"
                                                                                                                    "81"
                                                                                                                    "F0"
                                                                                                                    "9F"
                                                                                                                    "87"
                                                                                                                    "BA"
                                                                                                                    "0A"
                                                                                                                    "C2"
                                                                                                                    "AD"
                                                                                                                    "F0"
                                                                                                                    "9F"
                                                                                                                    "87"
                                                                                                                    "BA"
                                                                                                                    "CC"
                                                                                                                    "88"
                                                                                                                    "E3"
                                                                                                                    "82"
                                                                                                                    "AB"
                                                                                                                    "5F"
                                                                                                                    "27"
                                                                                                                    "2E"
                                                                                                                    "CC"
                                                                                                                    "88"
                                                                                                                    "0A"
                                                                                                                    "E4"
                                                                                                                    "B8"
                                                                                                                    "A"
                                                                                                                    "D",
         // WB12 `5 ' 5` bridge whose left digit is two codepoints back across a long ignorable run (spurious @60).
         "5FE281A05F352CF09F988027F09F98805FCC88CC81E4B8ADE281A05F62CC8061E4B8ADE382AB35E281A0CC88CC88F09F8FBBCC88" "E2"
                                                                                                                    "81"
                                                                                                                    "A0"
                                                                                                                    "CC"
                                                                                                                    "80"
                                                                                                                    "CC"
                                                                                                                    "81"
                                                                                                                    "27"
                                                                                                                    "35"
                                                                                                                    "CC"
                                                                                                                    "80"
                                                                                                                    "E4"
                                                                                                                    "B8"
                                                                                                                    "AD"
                                                                                                                    "CC"
                                                                                                                    "81"
                                                                                                                    "E2"
                                                                                                                    "80"
                                                                                                                    "8D"
                                                                                                                    "C3"
                                                                                                                    "A9"
                                                                                                                    "E3"
                                                                                                                    "80"
                                                                                                                    "80"
                                                                                                                    "35"
                                                                                                                    "2"
                                                                                                                    "7"})
        check_consistency(from_hex(seam_hex));

    // A long mixed string exercises the wide SIMD windows and their tails.
    scope_assert(
        std::string big,
        [&] {
            for (int i = 0; i != 200; ++i) big += "hello world, foo_bar 42 don't ";
        }(),
        uax29_segments(sz_utf8_words, big) == uax29_segments(sz_utf8_words_serial, big));
}

#pragma region Segmentation Helpers

/** @brief Upper bound on the inputs the unit/safety segmentation layers feed (<=70 bytes, so <=70 segments). */
static constexpr sz_size_t utf8_unit_capacity_k = 70;

/** @brief Drive any boundary finder one-shot over @p text and return the emitted segments as byte strings. */
static std::vector<std::string> utf8_segments_(sz_utf8_segmenter_t finder, sz::string_view text) {
    std::vector<sz_size_t> starts(text.size() + 1), lengths(text.size() + 1);
    sz_size_t consumed = 0;
    sz_size_t const count = finder(text.data(), text.size(), starts.data(), lengths.data(), text.size() + 1, &consumed);
    std::vector<std::string> segments;
    for (sz_size_t index = 0; index != count; ++index)
        segments.push_back(std::string(text.data() + starts[index], lengths[index]));
    return segments;
}

/**
 *  @brief A small SMP/astral fixture set: the random differential corpora are pure BMP, so emoji flags
 *         (Regional-Indicator pairs), ZWJ sequences, and lone astral codepoints would otherwise be covered
 *         by golden vectors only. Reused by every family's safety + differential drivers.
 */
static std::vector<std::string> utf8_astral_fixtures_() {
    return {
        "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8",                                 // 🇺🇸 RI(U) RI(S) flag pair
        "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", // 🇺🇸🇫🇷 two flags
        "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D"          // 👩‍👩‍👧 family ZWJ sequence
        "\xF0\x9F\x91\xA7",
        "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD",     // 👍🏽 emoji + skin-tone modifier
        "\xF0\x9F\x98\x80\xF0\x9F\x98\x81",     // 😀😁 two astral emoji
        "a\xF0\x9F\x98\x80\x62",                // ASCII around an astral codepoint
        "\xF0\x90\x80\x80\xF0\x90\x80\x81",     // U+10000 U+10001 plain astral pair
        "\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8", // RI ASCII RI - RI parity must reset
    };
}

/** @brief Which UAX algorithm a motif corpus targets, selecting the family-specific corner-case alphabet. */
enum class utf8_segment_family_t {
    grapheme_k,
    sentence_k,
    line_k,
};

/**
 *  @brief UAX-29 grapheme-cluster (GB) corner motifs: emoji-ZWJ chains, VS16, skin-tone modifiers,
 *         regional-indicator runs of varying parity, Indic virama clusters, Hangul jamo combinations,
 *         and bare CR/LF shapes. All non-ASCII bytes are `\xHH` escapes.
 */
static std::vector<std::string> utf8_grapheme_motifs_() {
    return {
        "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D" // 👩‍👩‍👧 three-link ZWJ chain
        "\xF0\x9F\x91\xA7",
        "\xF0\x9F\x91\xA9\xE2\x80\x8D",     // 👩‍ dangling trailing ZWJ
        "\xE2\x9C\x8B\xEF\xB8\x8F",         // ✋️ emoji + VS16 (U+FE0F)
        "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD", // 👍🏽 emoji + skin-tone modifier (U+1F3FD)
        "\xF0\x9F\x87\xBA",                 // 🇺 single regional indicator (run length 1)
        "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", // 🇺🇸 regional-indicator run length 2
        "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"  // 🇺🇸🇫 regional-indicator run length 3 (odd parity)
        "\xF0\x9F\x87\xAB",
        "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8" // 🇺🇸🇫🇷🇩 regional-indicator run length 5 (odd parity)
        "\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7\xF0\x9F\x87\xA9",
        "\xE0\xA6\x95\xE0\xA7\x8D\xE0\xA6\x95", // Bengali ক + virama + ক (U+09CD)
        "\xE0\xB4\x95\xE0\xB5\x8D\xE0\xB4\x95", // Malayalam ക + virama + ക (U+0D4D)
        "\xE0\xAE\x95\xE0\xAF\x8D\xE0\xAE\x95", // Tamil க + virama + க (U+0BCD)
        "\xEA\xB0\x80",                         // 가 Hangul LV syllable
        "\xEC\x95\x8A",                         // 않 Hangul LVT syllable
        "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", // Hangul L+V+T jamo combination
        "\r",                                   // CR alone
        "\n",                                   // LF alone
        "\r\r\n",                               // CR CR LF (parity of CR runs)
    };
}

/**
 *  @brief UAX-29 sentence-break (SB) corner motifs: ATerm vs STerm, terminator + Close + Space + case,
 *         paragraph separators after a terminator, SContinue, abbreviation-like and numeric shapes.
 */
static std::vector<std::string> utf8_sentence_motifs_() {
    return {
        "End. Next",               // ATerm + space + uppercase (SB break)
        "End! Next",               // STerm + space + uppercase (SB break)
        "End? Next",               // STerm '?' + space + uppercase
        "end. next",               // SB8: '. ' before lowercase does not break
        "(End.) Next",             // terminator + Close ')' + space + uppercase
        "\"End.\" Next",           // terminator + Close '"' + space + uppercase
        "End.\xE2\x80\xA8" "Next", // ParaSep U+2028 after terminator
        "End.\xE2\x80\xA9" "Next", // ParaSep U+2029 after terminator
        "End.\xC2\x85" "Next",     // ParaSep U+0085 NEL after terminator
        "Item, more",              // SContinue ',' continuation
        "Item: more",              // SContinue ':' continuation
        "Mr. Smith",               // abbreviation-like
        "Dr. House",               // abbreviation-like
        "U.S.A. ok",               // dotted acronym
        "3.14 value",              // numeric ATerm (no break)
    };
}

/**
 *  @brief UAX-14 line-break (LB) corner motifs: mandatory breaks, ZWJ, and OP/CL/HY/GL/BA/EX/QU/IS/NU/PR
 *         adjacency plus numeric grouping. NBSP (U+00A0) is the glue (GL) representative.
 */
static std::vector<std::string> utf8_line_motifs_() {
    return {
        "a\nb",              // LF mandatory break
        "a\rb",              // CR mandatory break
        "a\r\nb",            // CRLF mandatory break (one segment)
        "a\xE2\x80\xA8" "b", // U+2028 LINE SEPARATOR mandatory
        "a\xE2\x80\xA9" "b", // U+2029 PARAGRAPH SEPARATOR mandatory
        "a\xC2\x85" "b",     // U+0085 NEL mandatory
        "a\xE2\x80\x8D" "b", // ZWJ (U+200D) adjacency
        "(word",             // OP open-punctuation before
        "word)",             // CL close-punctuation after
        "co-op",             // HY hyphen
        "a\xC2\xA0" "b",     // GL glue NBSP (U+00A0)
        "a b",               // BA break-after space
        "word!",             // EX exclamation
        "\"word\"",          // QU quotation
        "a:b",               // IS infix separator
        "12345",             // NU numbers
        "$50",               // PR prefix
        "3,000",             // numeric grouping comma
    };
}

/** @brief Return the family-specific motif corpus selected by @p family. */
static std::vector<std::string> utf8_family_motifs_(utf8_segment_family_t family) {
    switch (family) {
    case utf8_segment_family_t::grapheme_k: return utf8_grapheme_motifs_();
    case utf8_segment_family_t::sentence_k: return utf8_sentence_motifs_();
    case utf8_segment_family_t::line_k: return utf8_line_motifs_();
    }
    return {};
}

#pragma endregion // Segmentation Helpers

#pragma region Segmentation Unit

/** @brief One hand-checked segmentation golden vector: the source text and its expected segment list. */
struct utf8_unit_case_t {
    sz::string_view text;
    std::vector<std::string> expected;
};

/**
 *  @brief Drive one segmentation backend (dispatched / serial / ISA) over hand-checked golden vectors,
 *         asserting the forward segmentation matches.
 *         Mirrors `check_utf8_unit_` (caller invokes once per backend).
 */
static void check_utf8_segment_unit_(char const *family, sz_utf8_segmenter_t forward,
                                     std::vector<utf8_unit_case_t> const &cases) {
    for (utf8_unit_case_t const &one : cases) {
        std::vector<std::string> const got = utf8_segments_(forward, one.text);
        assert(got == one.expected && family && "segment forward != golden");
    }
}

/** @brief Hand-checked UAX-29 grapheme-cluster golden vectors. */
static std::vector<utf8_unit_case_t> utf8_grapheme_unit_cases_() {
    return {
        {"", {}},
        {"a", {"a"}},
        {"abc", {"a", "b", "c"}},
        {"a\r\nb", {"a", "\r\n", "b"}}, // GB3: CR × LF stay one cluster
        {"e\xCC\x81", {"e\xCC\x81"}},   // e + U+0301 combining acute → one cluster (GB9)
        {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", {"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"}}, // RI pair → one cluster (GB12/13)
        {"\xF0\x9F\x87\xBA\x61\xF0\x9F\x87\xB8",
         {"\xF0\x9F\x87\xBA", "a", "\xF0\x9F\x87\xB8"}}, // RI ASCII RI: parity reset
        {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9", // 👩‍👩 ZWJ joins → one cluster (GB11)
         {"\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA9"}},
        {"\xEA\xB0\x80", {"\xEA\xB0\x80"}}, // 가 Hangul LV syllable → one cluster
        {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8",
         {"\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"}}, // Hangul L+V+T jamo → one cluster (GB6/7/8)
        {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7",
         {"\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7"}}, // Indic क + virama + ष → one cluster (GB9c InCB)
    };
}

/** @brief Hand-checked UAX-29 sentence-break golden vectors. */
static std::vector<utf8_unit_case_t> utf8_sentence_unit_cases_() {
    return {
        {"", {}},
        {"Hello.", {"Hello."}},
        {"Hello. World.", {"Hello. ", "World."}},     // terminator + space ends the sentence
        {"Mr. Smith went.", {"Mr. ", "Smith went."}}, // UAX-29 has no abbreviation list: '. ' before uppercase breaks
        {"etc. and so on", {"etc. and so on"}},       // SB8: '. ' before a lowercase word does NOT break
        {"One.\nTwo.", {"One.\n", "Two."}},           // ParaSep after terminator
        {"A! B? C.", {"A! ", "B? ", "C."}},           // multiple terminators
    };
}

/** @brief Hand-checked UAX-14 line-break golden vectors (all break opportunities: mandatory hard breaks + soft wraps). */
static std::vector<utf8_unit_case_t> utf8_line_unit_cases_() {
    return {
        {"", {}},
        {"hello world", {"hello ", "world"}},          // soft wrap after the space
        {"a\nb", {"a\n", "b"}},                        // LF hard break
        {"a\r\nb", {"a\r\n", "b"}},                    // CRLF hard break, one segment
        {"a\xE2\x80\xA8" "b", {"a\xE2\x80\xA8", "b"}}, // U+2028 LINE SEPARATOR
        {"a\xE2\x80\xA9" "b", {"a\xE2\x80\xA9", "b"}}, // U+2029 PARAGRAPH SEPARATOR
    };
}

/** @brief Known-answer grapheme-cluster vectors driven through dispatched, serial, and each ISA backend. */
void test_utf8_grapheme_unit() {
    std::printf("  - testing UTF-8 grapheme-cluster known-answer vectors...\n");
    std::vector<utf8_unit_case_t> const cases = utf8_grapheme_unit_cases_();
    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes, cases); // Dispatched
    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes_serial, cases);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("grapheme", sz_utf8_graphemes_icelake, cases);
#endif
}

/** @brief Known-answer sentence-break vectors driven through dispatched, serial, and each ISA backend. */
void test_utf8_sentence_unit() {
    std::printf("  - testing UTF-8 sentence-break known-answer vectors...\n");
    std::vector<utf8_unit_case_t> const cases = utf8_sentence_unit_cases_();
    check_utf8_segment_unit_("sentence", sz_utf8_sentences, cases); // Dispatched
    check_utf8_segment_unit_("sentence", sz_utf8_sentences_serial, cases);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("sentence", sz_utf8_sentences_icelake, cases);
#endif
}

/** @brief Known-answer line-break vectors through dispatched, serial, and each ISA. */
void test_utf8_line_unit() {
    std::printf("  - testing UTF-8 line-break known-answer vectors...\n");
    std::vector<utf8_unit_case_t> const cases = utf8_line_unit_cases_();
    check_utf8_segment_unit_("line", sz_utf8_linewraps, cases); // Dispatched
    check_utf8_segment_unit_("line", sz_utf8_linewraps_serial, cases);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("line", sz_utf8_linewraps_icelake, cases);
#endif
}

#pragma endregion // Segmentation Unit

#pragma region Segmentation Safety

/**
 *  @brief Feed malformed / invalid UTF-8 through any boundary finder, asserting it survives, every emitted
 *         segment is in-bounds, and `bytes_consumed <= length`. Mirrors the structure of `check_utf8_safety_`.
 */
static void check_utf8_segment_safety_(char const *family, sz_utf8_segmenter_t forward,
                                       std::size_t random_input_count = scale_iterations(10000)) {
    sz_size_t offsets[utf8_unit_capacity_k + 1], lengths[utf8_unit_capacity_k + 1];

    auto probe = [&](sz_utf8_segmenter_t finder, char const *input, std::size_t input_length) {
        sz_size_t bytes_consumed = 0;
        sz_size_t const found = finder(input, (sz_size_t)input_length, offsets, lengths,
                                       (sz_size_t)(utf8_unit_capacity_k + 1), &bytes_consumed);
        assert(bytes_consumed <= input_length && "segment finder consumed past the input");
        for (sz_size_t index = 0; index != found; ++index) {
            if (offsets[index] + lengths[index] <= input_length) continue;
            std::fprintf(stderr, "%s emitted out-of-bounds segment (offset=%zu len=%zu, input=%zu)\n", family,
                         (std::size_t)offsets[index], (std::size_t)lengths[index], input_length);
            print_utf8_test_bytes_("input", input, input_length);
            assert(false && "segment finder emitted a span outside the input");
        }
    };
    auto check = [&](char const *input, std::size_t input_length) { probe(forward, input, input_length); };

    char input[utf8_unit_capacity_k];

    // Named adversarial shapes.
    check("\x80", 1);              // Lone continuation byte
    check("\xC0\x80", 2);          // Overlong encoding of NUL
    check("\xED\xA0\x80", 3);      // Surrogate-encoded codepoint (U+D800)
    check("hello\xF0\x9F\x98", 8); // Truncated 4-byte sequence at the very end

    // The SMP/astral fixtures the random corpora miss (RI parity, ZWJ, astral).
    for (std::string const &fixture : utf8_astral_fixtures_()) check(fixture.data(), fixture.size());

    // All 256 single bytes.
    for (std::size_t byte = 0; byte != 256; ++byte) { input[0] = (char)byte, check(input, 1); }

    // All 65,536 byte pairs.
    for (std::size_t first_byte = 0; first_byte != 256; ++first_byte)
        for (std::size_t second_byte = 0; second_byte != 256; ++second_byte) {
            input[0] = (char)first_byte, input[1] = (char)second_byte;
            check(input, 2);
        }

    // Random garbage at every sub-cache-line alignment.
    auto &rng = global_random_generator();
    std::uniform_int_distribution<std::size_t> length_distribution(1, utf8_unit_capacity_k);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (std::size_t iteration = 0; iteration != random_input_count; ++iteration) {
        std::size_t const input_length = length_distribution(rng);
        for (std::size_t index = 0; index != input_length; ++index) input[index] = (char)byte_distribution(rng);
        for_each_cacheline_offset_(input_length, [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, input, input_length);
            check(buffer, input_length);
        });
    }
}

/** @brief Malformed-input safety of the grapheme finders through serial, dispatched, and each ISA backend. */
void test_utf8_grapheme_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 grapheme kernels...\n");
    check_utf8_segment_safety_("grapheme (serial)", sz_utf8_graphemes_serial);
    check_utf8_segment_safety_("grapheme (dispatched)", sz_utf8_graphemes);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("grapheme (icelake)", sz_utf8_graphemes_icelake);
#endif
    std::printf("    grapheme safety passed!\n");
}

/** @brief Malformed-input safety of the sentence finder through serial, dispatched, and each ISA backend. */
void test_utf8_sentence_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 sentence kernels...\n");
    check_utf8_segment_safety_("sentence (serial)", sz_utf8_sentences_serial);
    check_utf8_segment_safety_("sentence (dispatched)", sz_utf8_sentences);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("sentence (icelake)", sz_utf8_sentences_icelake);
#endif
    std::printf("    sentence safety passed!\n");
}

/** @brief Malformed-input safety of the line finder through serial, dispatched, and each ISA backend. */
void test_utf8_line_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 line kernels...\n");
    check_utf8_segment_safety_("line (serial)", sz_utf8_linewraps_serial);
    check_utf8_segment_safety_("line (dispatched)", sz_utf8_linewraps);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("line (icelake)", sz_utf8_linewraps_icelake);
#endif
    std::printf("    line safety passed!\n");
}

#pragma endregion // Segmentation Safety

/** @brief Smoke-test the C++ binding wrappers for UTF-8 normalization. */
void test_norm_unit() {
    std::printf("  - testing C++ UTF-8 normalization bindings...\n");

    // Round-trip: NFC -> NFD -> NFC should recover the original NFC string.
    // "café" in NFC: c a f é(U+00E9 = NFC composed)
    char const cafe_nfc[] = "caf\xC3\xA9"; // U+00E9 as two-byte UTF-8
    sz::string nfc_str {cafe_nfc};
    assert(nfc_str.try_utf8_normalize(sz_normal_form_nfd_k));
    assert(nfc_str.try_utf8_normalize(sz_normal_form_nfc_k));
    assert(nfc_str == cafe_nfc);

    // is_normalized: NFC string is normalized under NFC, not NFD (é decomposes in NFD).
    sz::string_view nfc_view {cafe_nfc};
    assert(nfc_view.is_normalized(sz_normal_form_nfc_k));
    assert(!nfc_view.is_normalized(sz_normal_form_nfd_k));

    // is_normalized on the owning type mirrors the view behaviour.
    sz::string nfc_own {cafe_nfc};
    assert(nfc_own.is_normalized(sz_normal_form_nfc_k));
    assert(!nfc_own.is_normalized(sz_normal_form_nfd_k));

    // utf8_norm_violation returns non-null for a non-normalized string.
    assert(nfc_view.utf8_norm_violation(sz_normal_form_nfd_k) != SZ_NULL_CHAR);
    assert(nfc_own.utf8_norm_violation(sz_normal_form_nfd_k) != SZ_NULL_CHAR);

    // NFKD of "ﬁ" (U+FB01 LATIN SMALL LIGATURE FI) decomposes to "fi".
    char const ligature_fi[] = "\xEF\xAC\x81"; // U+FB01
    sz::string fi_str {ligature_fi};
    assert(fi_str.try_utf8_normalize(sz_normal_form_nfkd_k));
    assert(fi_str.contains('f'));
    assert(fi_str.contains('i'));

    std::printf("    C++ normalization bindings passed!\n");
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

/** @brief Wraps the UTF-8 normalization and normalization-violation kernels of one backend by their pointers. */
template <sz_utf8_norm_t norm_, sz_utf8_norm_violation_t violation_>
struct norm_from_sz_ {
    sz_size_t form(sz_cptr_t text, sz_size_t length, sz_normal_form_t normal_form, sz_ptr_t output) const noexcept {
        return norm_(text, length, normal_form, output);
    }
    sz_cptr_t violation(sz_cptr_t text, sz_size_t length, sz_normal_form_t normal_form) const noexcept {
        return violation_(text, length, normal_form);
    }
};

/**
 *  @brief Tests UTF-8 functions across different SIMD backends against the serial implementation.
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
void test_utf8_equivalence(reference_ reference, candidate_ candidate, //
                           std::size_t min_text_length = 4000, std::size_t min_iterations = scale_iterations(10000)) {

    // Enumerate every delimiter via repeated streaming calls (resuming from `bytes_consumed`), so a small
    // `capacity` exercises the SIMD vector loop, its resume logic, and the serial tail handoff.
    auto enumerate = [](auto matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
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
    auto reconstruct_segments = [](auto matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
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

/**
 *  @brief Differential fuzz for a UTF-8 normalizer backend: assert it is byte-exact vs the serial
 *         reference for all four forms (norm output + violation offset). The all-codepoint shuffle
 *         stresses canonical ordering, every decomposition/composition path, and SIMD-block straddles.
 *
 *  @param reference   Serial reference backend bundle (normalizer + normalization-violation finder).
 *  @param candidate   ISA-specific backend bundle under test (normalizer + normalization-violation finder).
 *  @param iterations  Number of all-codepoint shuffles to fuzz x 4 normal forms.
 */
template <typename reference_, typename candidate_>
void test_norm_equivalence(reference_ reference, candidate_ candidate, std::size_t iterations = scale_iterations(25)) {
    std::printf("  - testing normalization fuzz (%zu iterations x 4 forms)...\n", iterations);

    std::vector<sz_rune_t> all_runes;
    all_runes.reserve(0x10FFFF);
    for (sz_rune_t cp = 0; cp <= 0x10FFFF; ++cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF) continue; // skip surrogates
        all_runes.push_back(cp);
    }
    std::vector<char> input_buffer(all_runes.size() * 4);
    std::vector<char> output_reference(input_buffer.size() * 4 + 64); // decomposition can expand
    std::vector<char> output_candidate(input_buffer.size() * 4 + 64);
    auto &rng = global_random_generator();
    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};

    for (std::size_t it = 0; it <= iterations; ++it) {
        if (it > 0) std::shuffle(all_runes.begin(), all_runes.end(), rng);
        char *write_cursor = input_buffer.data();
        for (sz_rune_t cp : all_runes) write_cursor += sz_rune_export(cp, (sz_u8_t *)write_cursor);
        sz_size_t input_length = (sz_size_t)(write_cursor - input_buffer.data());

        for (sz_normal_form_t normal_form : norm_forms) {
            sz_size_t len_reference = reference.form(input_buffer.data(), input_length, normal_form,
                                                     output_reference.data());
            sz_size_t len_candidate = candidate.form(input_buffer.data(), input_length, normal_form,
                                                     output_candidate.data());
            if (len_reference != len_candidate ||
                std::memcmp(output_reference.data(), output_candidate.data(), len_reference) != 0) {
                std::fprintf(stderr, "norm mismatch (form=%d, iter=%zu): reference_len=%zu candidate_len=%zu\n",
                             (int)normal_form, it, (size_t)len_reference, (size_t)len_candidate);
                assert(false);
            }
            sz_cptr_t viol_reference = reference.violation(input_buffer.data(), input_length, normal_form);
            sz_cptr_t viol_candidate = candidate.violation(input_buffer.data(), input_length, normal_form);
            if (viol_reference != viol_candidate) {
                std::fprintf(stderr, "norm violation mismatch (form=%d, iter=%zu)\n", (int)normal_form, it);
                assert(false);
            }
        }
    }
    std::printf("    normalization fuzzing passed!\n");
}

#pragma region Segmentation Equivalence

/** @brief Whether a generated corpus must stay well-formed UTF-8 or may have malformed classes mixed in. */
enum class utf8_corpus_flavor_t {
    valid_k,
    malformed_k,
};

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_export` (silently skips invalid runes). */
static void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_export(codepoint, bytes);
    if (length == sz_utf8_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
}

/**
 *  @brief Append one randomly chosen malformed-UTF-8 class to @p text, deliberately injecting bytes the
 *         validator rejects: overlong encodings, surrogates, lone continuations, invalid leads, truncated
 *         tails, out-of-range leads, and noncharacters. Drawn from @p rng so `SZ_TESTS_SEED` reproduces.
 */
static void append_malformed_class_(std::string &text, std::mt19937 &rng) {
    static char const *malformed[] = {
        "\xC0\x80",         // overlong 2-byte encoding of NUL
        "\xC1\xBF",         // overlong 2-byte encoding of U+007F
        "\xE0\x80\x80",     // overlong 3-byte encoding of NUL
        "\xE0\x9F\xBF",     // overlong 3-byte encoding of U+07FF
        "\xF0\x80\x80\x80", // overlong 4-byte encoding of NUL
        "\xF0\x8F\xBF\xBF", // overlong 4-byte encoding of U+FFFF
        "\xED\xA0\x80",     // surrogate U+D800 in 3-byte form
        "\xED\xBF\xBF",     // surrogate U+DFFF in 3-byte form
        "\x80",             // lone continuation (low)
        "\xBF",             // lone continuation (high)
        "\xFE",             // invalid lead 0xFE
        "\xFF",             // invalid lead 0xFF
        "\xC2",             // truncated 2-byte tail (mid-string)
        "\xE2\x80",         // truncated 3-byte tail (mid-string)
        "\xF0\x9F\x98",     // truncated 4-byte tail (mid-string)
        "\xF5\x80\x80\x80", // out-of-range lead >U+10FFFF (0xF5)
        "\xEF\xB7\x90",     // noncharacter U+FDD0
        "\xEF\xB7\xAF",     // noncharacter U+FDEF
        "\xEF\xBF\xBE",     // plane-ender noncharacter U+FFFE
        "\xEF\xBF\xBF",     // plane-ender noncharacter U+FFFF
    };
    std::size_t const count = sizeof(malformed) / sizeof(malformed[0]);
    std::uniform_int_distribution<std::size_t> pick(0, count - 1);
    text.append(malformed[pick(rng)]);
}

/**
 *  @brief Apply structural mutation passes to @p text: NUL injection (10%), random byte-swap (10%), a
 *         truncate-last-codepoint pass, and a stray-continuation insertion. All randomness flows through
 *         @p rng so `SZ_TESTS_SEED` reproduces.
 */
static void apply_mutation_passes_(std::string &text, std::mt19937 &rng) {
    if (text.empty()) return;
    std::uniform_int_distribution<std::size_t> byte_index(0, text.size() - 1);
    std::size_t const to_corrupt = text.size() / 10;
    for (std::size_t index = 0; index != to_corrupt; ++index) text[byte_index(rng)] = '\0';
    for (std::size_t index = 0; index != to_corrupt; ++index) std::swap(text[byte_index(rng)], text[byte_index(rng)]);

    // Truncate the last codepoint: drop a 1-3 byte trailing run so a multi-byte sequence loses its tail.
    std::uniform_int_distribution<std::size_t> truncate_distribution(1, 3);
    std::size_t const truncate_by = std::min(truncate_distribution(rng), text.size());
    text.resize(text.size() - truncate_by);

    // Insert a stray continuation byte at a random position, breaking codepoint alignment.
    if (!text.empty()) {
        std::uniform_int_distribution<std::size_t> insert_at(0, text.size());
        std::uniform_int_distribution<int> continuation(0x80, 0xBF);
        text.insert(text.begin() + insert_at(rng), (char)continuation(rng));
    }
}

/**
 *  @brief Build a random UTF-8 corpus from a weighted corner-case alphabet (all four byte lengths, byte-length
 *         boundary codepoints, BOM / ZWJ / VS16, and the snippet/astral motifs) plus per-family UAX motifs,
 *         until at least @p min_length bytes. The @p malformed_k flavor mixes malformed classes into the
 *         otherwise-valid text; @p valid_k stays well-formed (modulo any later mutation passes the caller adds).
 */
static std::string utf8_random_segmentation_corpus_(std::size_t min_length, utf8_corpus_flavor_t flavor,
                                                    utf8_segment_family_t family) {
    static sz_rune_t const boundary_codepoints[] = {
        0x007F, 0x0080, 0x07FF, 0x0800,  0xFFFF, 0x10000, 0x10FFFF, // byte-length boundaries
        0xFEFF,                                                     // BOM U+FEFF
        0x200D,                                                     // ZWJ U+200D
        0xFE0F,                                                     // VS16 U+FE0F
        0x0041, 0x00DF, 0x4E2D, 0x1F600,                            // one codepoint per byte length
    };
    static char const *snippets[] = {
        "a",
        "Hello, world! ",
        "don't ",
        "3,14 ",
        "\xE4\xBD\xA0\xE5\xA5\xBD", // 你好
        "\r\n",
        "one. two. ",
        "A! B? ",
        "e\xCC\x81",
        "\xEA\xB0\x80",                         // combining + Hangul
        "\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB7", // Indic conjunct
        "line\nbreak ",
        "soft wrap here ",
        "\xE2\x80\xA8",
        "\xE2\x80\xA9", // LINE/PARA separators
    };
    std::vector<std::string> const astral = utf8_astral_fixtures_();
    std::vector<std::string> const motifs = utf8_family_motifs_(family);
    auto &rng = global_random_generator();
    std::size_t const boundary_count = sizeof(boundary_codepoints) / sizeof(boundary_codepoints[0]);
    std::size_t const snippet_count = sizeof(snippets) / sizeof(snippets[0]);

    // Weighted choice: snippets are the bulk, boundary codepoints / astral / family motifs sprinkled in,
    // and (for the malformed flavor) a malformed class injected with a small probability.
    std::uniform_int_distribution<int> weight(0, 99);
    std::uniform_int_distribution<std::size_t> boundary_pick(0, boundary_count - 1);
    std::uniform_int_distribution<std::size_t> snippet_pick(0, snippet_count - 1);
    std::uniform_int_distribution<std::size_t> astral_pick(0, astral.size() - 1);
    std::uniform_int_distribution<std::size_t> motif_pick(0, motifs.empty() ? 0 : motifs.size() - 1);

    std::string text;
    while (text.size() < min_length) {
        int const roll = weight(rng);
        if (flavor == utf8_corpus_flavor_t::malformed_k && roll < 10) append_malformed_class_(text, rng);
        else if (roll < 30) append_codepoint_(text, boundary_codepoints[boundary_pick(rng)]);
        else if (roll < 45) text.append(astral[astral_pick(rng)]);
        else if (roll < 65 && !motifs.empty()) text.append(motifs[motif_pick(rng)]);
        else text.append(snippets[snippet_pick(rng)]);
    }
    return text;
}

/**
 *  @brief Metamorphic invariant: the emitted segments must exactly tile `[0, length)` with no gaps or
 *         overlaps. For @p valid_k corpora every non-empty segment additionally starts on a codepoint
 *         boundary (the lead byte is not a continuation byte); malformed input may legitimately split
 *         mid-byte, so that alignment check is gated on the flavor. Tiling holds for every flavor.
 */
static void assert_segment_invariants_(std::vector<sz_size_t> const &offsets, std::vector<sz_size_t> const &lengths,
                                       sz_cptr_t data, sz_size_t length, utf8_corpus_flavor_t flavor) {
    sz_size_t running_cursor = 0;
    for (std::size_t index = 0; index != offsets.size(); ++index) {
        assert(offsets[index] == running_cursor && "segments do not tile the input contiguously");
        if (flavor == utf8_corpus_flavor_t::valid_k && lengths[index] != 0)
            assert((((sz_u8_t)data[offsets[index]]) & 0xC0u) != 0x80u && "segment starts mid-codepoint");
        running_cursor += lengths[index];
    }
    assert(running_cursor == length && "segments do not cover the whole input");
}

/**
 *  @brief Differential fuzz of any ISA finder against the serial reference, enumerating segments via repeated
 *         streaming calls across the 6 capacities {len+64,65,63,16,3,1}, over random UTF-8 (valid and
 *         malformed) plus an alignment sweep. Asserts tiling/alignment invariants and capacity-independence
 *         (every capacity yields identical output).
 */
static void test_utf8_segment_equivalence_(char const *family, sz_utf8_segmenter_t reference,
                                           sz_utf8_segmenter_t candidate, utf8_segment_family_t corpus_family,
                                           std::size_t iterations = scale_iterations(2000)) {
    std::printf("  - testing %s serial-vs-ISA differential...\n", family);

    // Hoisted batch scratch reused across every streaming call (resized once per `compare` to the max capacity).
    std::vector<sz_size_t> batch_offsets, batch_lengths;

    // Enumerate every segment via repeated streaming calls (resuming from `bytes_consumed`), so a small
    // `capacity` exercises the window loop, its resume logic, and the serial tail handoff.
    auto enumerate = [&](sz_utf8_segmenter_t matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
                         std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
        offsets.clear(), lengths.clear();
        sz_size_t base = 0;
        while (base < length) {
            sz_size_t consumed = 0;
            sz_size_t const got = matcher(data + base, length - base, batch_offsets.data(), batch_lengths.data(),
                                          capacity, &consumed);
            for (sz_size_t index = 0; index != got; ++index)
                offsets.push_back(base + batch_offsets[index]), lengths.push_back(batch_lengths[index]);
            if (got == 0 && consumed == 0) break;
            base += consumed;
        }
    };

    std::vector<sz_size_t> reference_offsets, reference_lengths, candidate_offsets, candidate_lengths;
    std::vector<sz_size_t> first_offsets, first_lengths;
    auto compare = [&](sz_utf8_segmenter_t left, sz_utf8_segmenter_t right, char const *what, sz_cptr_t data,
                       sz_size_t length, utf8_corpus_flavor_t flavor, bool check_invariants) {
        sz_size_t const capacities[] = {length + 64, 65, 63, 16, 3, 1};
        sz_size_t max_capacity = 1;
        for (sz_size_t capacity : capacities) max_capacity = std::max(max_capacity, capacity);
        if (batch_offsets.size() < max_capacity) batch_offsets.resize(max_capacity), batch_lengths.resize(max_capacity);
        bool first = true;
        for (sz_size_t capacity : capacities) {
            if (capacity == 0) continue;
            enumerate(left, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(right, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths && what);
            if (check_invariants)
                assert_segment_invariants_(candidate_offsets, candidate_lengths, data, length, flavor);
            // Resume-identity / capacity-independence: every capacity reproduces the capacity[0] enumeration.
            if (first) first_offsets = candidate_offsets, first_lengths = candidate_lengths, first = false;
            else
                assert(candidate_offsets == first_offsets && candidate_lengths == first_lengths &&
                       "capacity-dependent segmentation");
        }
    };
    auto check = [&](sz_cptr_t data, sz_size_t length, utf8_corpus_flavor_t flavor) {
        compare(reference, candidate, "segment offsets/lengths mismatch", data, length, flavor, true);
    };

    auto &rng = global_random_generator();
    for (std::size_t iteration = 0; iteration != iterations; ++iteration) {
        // Valid corpus: serial-vs-ISA agreement plus tiling/alignment invariants on well-formed input.
        std::string const valid = utf8_random_segmentation_corpus_(400, utf8_corpus_flavor_t::valid_k, corpus_family);
        check(valid.data(), valid.size(), utf8_corpus_flavor_t::valid_k);

        // Mutated valid corpus: NUL/swap/truncate/stray-continuation passes break codepoints, yet both
        // backends must still agree (alignment invariant relaxed via the malformed flavor).
        std::string mutated = valid;
        apply_mutation_passes_(mutated, rng);
        check(mutated.data(), mutated.size(), utf8_corpus_flavor_t::malformed_k);

        // Malformed corpus: malformed classes mixed directly into otherwise-valid text.
        std::string const malformed = utf8_random_segmentation_corpus_(400, utf8_corpus_flavor_t::malformed_k,
                                                                       corpus_family);
        check(malformed.data(), malformed.size(), utf8_corpus_flavor_t::malformed_k);
    }

    // Alignment sweep: drive a fixed corpus at every sub-cache-line offset so the load alignment is swept.
    std::string const probe = utf8_random_segmentation_corpus_(256, utf8_corpus_flavor_t::valid_k, corpus_family);
    for_each_cacheline_offset_(probe.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
        std::memcpy(buffer, probe.data(), probe.size());
        check(buffer, probe.size(), utf8_corpus_flavor_t::valid_k);
    });
}

#pragma endregion // Segmentation Equivalence

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Feeds malformed / invalid UTF-8 through one backend's counting, boundary-finding, normalization,
 *         and streaming-unpack kernels, asserting no crash, in-bounds output, and intact canary bytes.
 *
 *  Counting and the boundary finders are bounds-safe on arbitrary bytes, so they face the full malformed
 *  battery: the only requirements are that they survive and that no `(offset, length)` pair lands outside
 *  the input. `sz_utf8_norm` and `sz_utf8_unpack_chunk` instead document a valid-UTF-8 precondition (the
 *  decoder "performs no validity checks") and assert internally on garbage, so they are exercised on the
 *  `sz_utf8_valid` subset of the same adversarial shapes - the normalizer's output must stay within its
 *  documented 18x bound, the unpacker must report no more runes than the destination holds and never let
 *  its cursor escape the input. Every byte outside each documented output window keeps its `0xA5` canary;
 *  guards bracket the destination buffers, like `test_uncased_safety`.
 *
 *  @param count            Codepoint counter under test.
 *  @param newlines         Newline boundary finder under test.
 *  @param whitespaces      Whitespace boundary finder under test.
 *  @param norm             Single-pass normalizer under test (or NULL when this backend has no normalizer).
 *  @param unpack           Streaming chunk decoder under test (or NULL when this backend has no decoder).
 *  @param random_inputs    Number of random garbage buffers to fuzz on top of the exhaustive byte sweeps.
 */
static void check_utf8_safety_(sz_utf8_count_t count, sz_utf8_segmenter_t newlines, sz_utf8_segmenter_t whitespaces,
                               sz_utf8_norm_t norm, sz_utf8_unpack_chunk_t unpack,
                               std::size_t random_inputs = scale_iterations(10000)) {

    std::size_t const max_input_length = 70;
    // The normalizer's documented worst case is 18x the input for a single-codepoint compatibility
    // decomposition (see `utf8_norm.h`); a truncated trailing sequence may mis-decode one extra rune.
    std::size_t const norm_bound = max_input_length * 18 + 18;
    std::vector<sz_rune_t> rune_destination;

    auto check = [&](char const *input, std::size_t input_length) {
        // Counting and the case-invariant boundary finders just have to survive and stay in bounds.
        [[maybe_unused]] sz_size_t const counted = count(input, (sz_size_t)input_length);

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
        check_boundaries(newlines, "newline finder");
        check_boundaries(whitespaces, "whitespace finder");

        // Streaming unpack: like `sz_utf8_norm`, the chunk decoder "performs no validity checks" (see its
        // docs) and assumes valid UTF-8, so it is only exercised on inputs that pass `sz_utf8_valid`. Each
        // call must report no more runes than the destination holds and a cursor that never runs past the
        // input. A truncated trailing sequence legitimately stalls (returns the same cursor) - the "need more
        // bytes" signal - so we stop rather than spin.
        if (unpack && sz_utf8_valid(input, (sz_size_t)input_length) == sz_true_k) {
            std::size_t const rune_capacity = max_input_length + 4;
            rune_destination.assign(rune_capacity, (sz_rune_t)0);
            sz_cptr_t cursor = input;
            sz_cptr_t const end = input + input_length;
            while (cursor < end) {
                sz_size_t produced = 0;
                sz_cptr_t const next = unpack(cursor, (sz_size_t)(end - cursor), rune_destination.data(),
                                              (sz_size_t)rune_capacity, &produced);
                assert(produced <= rune_capacity && "Unpack reported more runes than the destination holds");
                assert(next >= cursor && next <= end && "Unpack cursor escaped the input");
                if (next == cursor) break; // Stalled on a truncated trailing sequence - the caller would refill
                cursor = next;
            }
        }

        // Normalization (when present): unlike counting and the boundary finders, `sz_utf8_norm` documents a
        // valid-UTF-8 precondition and asserts internally on malformed bytes, so it is only driven on inputs
        // that pass `sz_utf8_valid`. The output must still stay within the bound; `with_guarded_buffer_`
        // brackets the destination with canaries and asserts they survive, like `test_uncased_safety`.
        if (norm && sz_utf8_valid(input, (sz_size_t)input_length) == sz_true_k) {
            with_guarded_buffer_(norm_bound, [&](sz_ptr_t output, std::size_t) {
                sz_size_t const normalized = norm(input, (sz_size_t)input_length, sz_normal_form_nfkd_k, output);
                if (normalized > norm_bound) {
                    std::fprintf(stderr, "Norm of invalid input returned %zu bytes for %zu input bytes (bound %zu)\n",
                                 (std::size_t)normalized, input_length, norm_bound);
                    print_utf8_test_bytes_("input", input, input_length);
                    assert(false && "Normalizer output must stay within the documented bound");
                }
            });
        }
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
void test_utf8_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 kernels...\n");

    // Serial baseline and the dispatched (automatic kernel resolution) entry points face the same contract.
    check_utf8_safety_(sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial, sz_utf8_norm_serial,
                       sz_utf8_unpack_chunk_serial);
    check_utf8_safety_(sz_utf8_count, sz_utf8_newlines, sz_utf8_whitespaces, sz_utf8_norm, sz_utf8_unpack_chunk);

#if SZ_USE_HASWELL
    check_utf8_safety_(sz_utf8_count_haswell, sz_utf8_newlines_haswell, sz_utf8_whitespaces_haswell,
                       sz_utf8_norm_haswell, SZ_NULL);
#endif
#if SZ_USE_SKYLAKE
    check_utf8_safety_(sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial, sz_utf8_norm_skylake,
                       SZ_NULL);
#endif
#if SZ_USE_ICELAKE
    check_utf8_safety_(sz_utf8_count_icelake, sz_utf8_newlines_icelake, sz_utf8_whitespaces_icelake,
                       sz_utf8_norm_icelake, sz_utf8_unpack_chunk_icelake);
#endif
#if SZ_USE_NEON
    check_utf8_safety_(sz_utf8_count_neon, sz_utf8_newlines_neon, sz_utf8_whitespaces_neon, sz_utf8_norm_neon, SZ_NULL);
#endif
#if SZ_USE_SVE
    check_utf8_safety_(sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial, sz_utf8_norm_sve,
                       SZ_NULL);
#endif
#if SZ_USE_SVE2
    check_utf8_safety_(sz_utf8_count_sve2, sz_utf8_newlines_sve2, sz_utf8_whitespaces_sve2, sz_utf8_norm_sve2, SZ_NULL);
#endif
#if SZ_USE_V128
    check_utf8_safety_(sz_utf8_count_v128, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128, sz_utf8_norm_v128, SZ_NULL);
#endif
#if SZ_USE_V128RELAXED
    // Relaxed-V128 only specializes counting; boundaries and normalization fall back to the V128 kernels.
    check_utf8_safety_(sz_utf8_count_v128relaxed, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128,
                       sz_utf8_norm_v128relaxed, SZ_NULL);
#endif
#if SZ_USE_RVV
    check_utf8_safety_(sz_utf8_count_rvv, sz_utf8_newlines_rvv, sz_utf8_whitespaces_rvv, sz_utf8_norm_rvv, SZ_NULL);
#endif
#if SZ_USE_LASX
    check_utf8_safety_(sz_utf8_count_lasx, sz_utf8_newlines_lasx, sz_utf8_whitespaces_lasx, sz_utf8_norm_lasx, SZ_NULL);
#endif
#if SZ_USE_POWERVSX
    check_utf8_safety_(sz_utf8_count_powervsx, sz_utf8_newlines_powervsx, sz_utf8_whitespaces_powervsx,
                       sz_utf8_norm_powervsx, SZ_NULL);
#endif

    std::printf("    malformed-input safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Run the UTF-8 count/newline/whitespace differential against every compiled SIMD backend. */
void test_utf8_all() {
    using reference_t = utf8_from_sz_<sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial>;
#if SZ_USE_HASWELL
    test_utf8_equivalence(
        reference_t {}, utf8_from_sz_<sz_utf8_count_haswell, sz_utf8_newlines_haswell, sz_utf8_whitespaces_haswell> {});
#endif
#if SZ_USE_ICELAKE
    test_utf8_equivalence(
        reference_t {}, utf8_from_sz_<sz_utf8_count_icelake, sz_utf8_newlines_icelake, sz_utf8_whitespaces_icelake> {});
#endif
#if SZ_USE_NEON
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_neon, sz_utf8_newlines_neon, sz_utf8_whitespaces_neon> {});
#endif
#if SZ_USE_SVE2
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_sve2, sz_utf8_newlines_sve2, sz_utf8_whitespaces_sve2> {});
#endif
#if SZ_USE_V128
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_v128, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128> {});
#endif
#if SZ_USE_V128RELAXED
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_v128relaxed, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128> {});
#endif
#if SZ_USE_RVV
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_rvv, sz_utf8_newlines_rvv, sz_utf8_whitespaces_rvv> {});
#endif
#if SZ_USE_LASX
    test_utf8_equivalence(reference_t {},
                          utf8_from_sz_<sz_utf8_count_lasx, sz_utf8_newlines_lasx, sz_utf8_whitespaces_lasx> {});
#endif
#if SZ_USE_POWERVSX
    test_utf8_equivalence(
        reference_t {},
        utf8_from_sz_<sz_utf8_count_powervsx, sz_utf8_newlines_powervsx, sz_utf8_whitespaces_powervsx> {});
#endif
}

/** @brief Run the normalization differential fuzz against every compiled SIMD backend. */
void test_norm_all() {
    using reference_t = norm_from_sz_<sz_utf8_norm_serial, sz_utf8_norm_violation_serial>;
#if SZ_USE_HASWELL
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_haswell, sz_utf8_norm_violation_haswell> {});
#endif
#if SZ_USE_SKYLAKE
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_skylake, sz_utf8_norm_violation_skylake> {});
#endif
#if SZ_USE_ICELAKE
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_icelake, sz_utf8_norm_violation_icelake> {});
#endif
#if SZ_USE_NEON
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_neon, sz_utf8_norm_violation_neon> {});
#endif
#if SZ_USE_SVE
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_sve, sz_utf8_norm_violation_sve> {});
#endif
#if SZ_USE_SVE2
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_sve2, sz_utf8_norm_violation_sve2> {});
#endif
#if SZ_USE_RVV
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_rvv, sz_utf8_norm_violation_rvv> {});
#endif
#if SZ_USE_V128
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_v128, sz_utf8_norm_violation_v128> {});
#endif
#if SZ_USE_V128RELAXED
    test_norm_equivalence(reference_t {},
                          norm_from_sz_<sz_utf8_norm_v128relaxed, sz_utf8_norm_violation_v128relaxed> {});
#endif
#if SZ_USE_LASX
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_lasx, sz_utf8_norm_violation_lasx> {});
#endif
#if SZ_USE_POWERVSX
    test_norm_equivalence(reference_t {}, norm_from_sz_<sz_utf8_norm_powervsx, sz_utf8_norm_violation_powervsx> {});
#endif
}

/** @brief Run the grapheme-cluster differential against every compiled SIMD backend. */
void test_utf8_grapheme_all() {
#if SZ_USE_ICELAKE
    test_utf8_segment_equivalence_("grapheme", sz_utf8_graphemes_serial, sz_utf8_graphemes_icelake,
                                   utf8_segment_family_t::grapheme_k);
#endif
}

/** @brief Run the sentence-break differential against every compiled SIMD backend. */
void test_utf8_sentence_all() {
#if SZ_USE_ICELAKE
    test_utf8_segment_equivalence_("sentence", sz_utf8_sentences_serial, sz_utf8_sentences_icelake,
                                   utf8_segment_family_t::sentence_k);
#endif
}

/** @brief Run the line-break differential against every compiled SIMD backend. */
void test_utf8_line_all() {
#if SZ_USE_ICELAKE
    test_utf8_segment_equivalence_("line", sz_utf8_linewraps_serial, sz_utf8_linewraps_icelake,
                                   utf8_segment_family_t::line_k);
#endif
}

#pragma endregion // Drivers
