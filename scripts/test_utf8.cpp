/**
 *  @brief  UTF-8 counting/newline/whitespace equivalence, normalization, and C++ API semantics.
 *  @file   scripts/test_utf8.cpp
 *  @author Ash Vardanian
 *  @date   2026-06-16
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

#include <cassert>       // C-style assertions
#include <algorithm>     // `std::transform`
#include <cstdio>        // `std::printf`
#include <cstring>       // `std::memcpy`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <sstream>       // `std::ostringstream`
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <set>           // `std::set`
#include <vector>        // `std::vector`

#include <string>      // Baseline
#include <string_view> // Baseline

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


#pragma region Unit

/**
 *  @brief Known-answer + C++ API coverage for UTF-8 counting, nth-character finding, codepoint
 *         iteration, boundary detection, streaming unpacking, and Unicode-aware splitting.
 *
 *  Exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through
 *  the C++ `sz::string_view` wrappers, so a regression that the serial-vs-SIMD agreement tests would
 *  miss - because both share a wrong constant - is still caught against an external ground truth. This
 *  is the isolated coverage for `sz_utf8_find_newlines`, `sz_utf8_find_whitespaces`, `sz_utf8_find_nth`
 *  (whose SIMD variants are otherwise untested), and `sz_utf8_unpack_chunk` (otherwise untested in C++).
 */
void test_utf8_unit() {
    std::printf("  - testing UTF-8 known-answer vectors...\n");

    // The mixed-script anchor: "aß中" is `a` (1 byte) + `ß` U+00DF (2 bytes) + `中` U+4E2D (3 bytes),
    // so 6 bytes encode exactly 3 codepoints {0x61, 0xDF, 0x4E2D}.
    char const mixed[] = "a\xC3\x9F\xE4\xB8\xAD"; // "aß中"
    sz_size_t const mixed_length = (sz_size_t)(sizeof(mixed) - 1);
    assert(mixed_length == 6u);

    // `sz_utf8_count`: 6 bytes, 3 codepoints.
    assert(sz_utf8_count(mixed, mixed_length) == 3u);        // Dispatched (automatic kernel)
    assert(sz_utf8_count_serial(mixed, mixed_length) == 3u); // Manual propagation to the serial kernel
#if SZ_USE_HASWELL
    assert(sz_utf8_count_haswell(mixed, mixed_length) == 3u); // Manual: haswell kernel
#endif
#if SZ_USE_ICELAKE
    assert(sz_utf8_count_icelake(mixed, mixed_length) == 3u); // Manual: icelake kernel
#endif
    assert(sz::string_view(mixed, mixed_length).utf8_count() == 3u); // C++ wrapper

    // `sz_utf8_find_nth`: codepoint starts land at byte offsets 0, 1, 3; the 4th (n=3) is one-past-the-end.
    assert(sz_utf8_find_nth(mixed, mixed_length, 0u) == mixed + 0);            // Dispatched: 'a' at byte 0
    assert(sz_utf8_find_nth(mixed, mixed_length, 1u) == mixed + 1);            // Dispatched: 'ß' at byte 1
    assert(sz_utf8_find_nth(mixed, mixed_length, 2u) == mixed + 3);            // Dispatched: '中' at byte 3
    assert(sz_utf8_find_nth(mixed, mixed_length, 3u) == SZ_NULL_CHAR);         // Dispatched: beyond end
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 0u) == mixed + 0);     // Manual: serial kernel
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 2u) == mixed + 3);     // Manual: serial kernel
    assert(sz_utf8_find_nth_serial(mixed, mixed_length, 3u) == SZ_NULL_CHAR);  // Manual: serial kernel
#if SZ_USE_HASWELL
    assert(sz_utf8_find_nth_haswell(mixed, mixed_length, 2u) == mixed + 3);    // Manual: haswell kernel
    assert(sz_utf8_find_nth_haswell(mixed, mixed_length, 3u) == SZ_NULL_CHAR); // Manual: haswell kernel
#endif
#if SZ_USE_ICELAKE
    assert(sz_utf8_find_nth_icelake(mixed, mixed_length, 2u) == mixed + 3);    // Manual: icelake kernel
    assert(sz_utf8_find_nth_icelake(mixed, mixed_length, 3u) == SZ_NULL_CHAR); // Manual: icelake kernel
#endif
    assert(sz::string_view(mixed, mixed_length).utf8_find_nth(2u) == 3u); // C++ wrapper (byte offset)

    // Runs one boundary kernel and asserts the emitted offsets/lengths match the expected lists.
    auto check_boundaries = [](sz_utf8_find_boundaries_t matcher, sz_cptr_t text, sz_size_t length,
                               std::vector<sz_size_t> const &expected_offsets,
                               std::vector<sz_size_t> const &expected_lengths) {
        sz_size_t found_offsets[16], found_lengths[16], consumed = 0;
        sz_size_t const got = matcher(text, length, found_offsets, found_lengths, 16u, &consumed);
        assert(got == expected_offsets.size());
        for (sz_size_t i = 0; i != got; ++i) {
            assert(found_offsets[i] == expected_offsets[i]);
            assert(found_lengths[i] == expected_lengths[i]);
        }
    };

    // `sz_utf8_find_newlines`: in "a\nb\r\nc" the `\n` is a length-1 newline at byte 1, and the `\r\n` is a
    // single length-2 newline at byte 3 (CRLF merges into one match).
    char const newline_text[] = "a\nb\r\nc";
    sz_size_t const newline_length = (sz_size_t)(sizeof(newline_text) - 1);
    std::vector<sz_size_t> const newline_offsets = {1u, 3u}, newline_lengths = {1u, 2u};
    check_boundaries(sz_utf8_find_newlines, newline_text, newline_length, newline_offsets, newline_lengths); // Dispatched
    check_boundaries(sz_utf8_find_newlines_serial, newline_text, newline_length, newline_offsets, newline_lengths); // serial
#if SZ_USE_HASWELL
    check_boundaries(sz_utf8_find_newlines_haswell, newline_text, newline_length, newline_offsets, newline_lengths); // haswell
#endif
#if SZ_USE_ICELAKE
    check_boundaries(sz_utf8_find_newlines_icelake, newline_text, newline_length, newline_offsets, newline_lengths); // icelake
#endif

    // `sz_utf8_find_whitespaces`: in "a b\tc" the space is a length-1 match at byte 1, the tab at byte 3
    // (there is no CRLF merging in the whitespace set - each codepoint is its own match).
    char const whitespace_text[] = "a b\tc";
    sz_size_t const whitespace_length = (sz_size_t)(sizeof(whitespace_text) - 1);
    std::vector<sz_size_t> const whitespace_offsets = {1u, 3u}, whitespace_lengths = {1u, 1u};
    check_boundaries(sz_utf8_find_whitespaces, whitespace_text, whitespace_length, whitespace_offsets,
                     whitespace_lengths); // Dispatched
    check_boundaries(sz_utf8_find_whitespaces_serial, whitespace_text, whitespace_length, whitespace_offsets,
                     whitespace_lengths); // serial
#if SZ_USE_HASWELL
    check_boundaries(sz_utf8_find_whitespaces_haswell, whitespace_text, whitespace_length, whitespace_offsets,
                     whitespace_lengths); // haswell
#endif
#if SZ_USE_ICELAKE
    check_boundaries(sz_utf8_find_whitespaces_icelake, whitespace_text, whitespace_length, whitespace_offsets,
                     whitespace_lengths); // icelake
#endif

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
    assert(sz_utf8_norm_violation(cafe_nfc, cafe_length, sz_normal_form_nfc_k) == SZ_NULL_CHAR);     // Dispatched: already NFC
    assert(sz_utf8_norm_violation(cafe_nfc, cafe_length, sz_normal_form_nfd_k) != SZ_NULL_CHAR);     // Dispatched: not NFD
    assert(sz_utf8_norm_violation_serial(cafe_nfc, cafe_length, sz_normal_form_nfc_k) == SZ_NULL_CHAR); // Manual: serial
    assert(sz_utf8_norm_violation_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k) != SZ_NULL_CHAR); // Manual: serial
    {
        char norm_buffer[64];
        sz_size_t const nfc_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfc_k, norm_buffer);
        assert(nfc_length == cafe_length && std::memcmp(norm_buffer, cafe_nfc, cafe_length) == 0); // NFC no-op
        sz_size_t const nfd_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfd_k, norm_buffer);
        assert(nfd_length == 6u); // "caf" + 'e' + U+0301 (2-byte combining acute)
        sz_size_t const nfd_length_serial =
            sz_utf8_norm_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k, norm_buffer); // Manual: serial
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
            return sz::string_view(t).utf8_chars().template to<std::vector<sz_rune_t>>();
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
        let_assert(std::string s63(63, 'x'), sz::string_view(s63).utf8_chars().size() == 63);
        let_assert(std::string s64(64, 'x'), sz::string_view(s64).utf8_chars().size() == 64);
        let_assert(std::string s65(65, 'x'), sz::string_view(s65).utf8_chars().size() == 65);

        // ASCII batch limit: 16 characters max per Ice Lake iteration
        let_assert(std::string s17(17, 'x'), sz::string_view(s17).utf8_chars().size() == 17);
        let_assert(std::string s20(20, 'x'), sz::string_view(s20).utf8_chars().size() == 20);

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
                     sz::string_view(cyr100).utf8_chars().size() == 100);

        // 50 consecutive 3-byte (exceeds 16-char limit 3x)
        scope_assert(std::string cjk50, for (int i = 0; i < 50; ++i) cjk50 += "世",
                     sz::string_view(cjk50).utf8_chars().size() == 50);

        // 50 consecutive 4-byte (exceeds 16-char limit 3x)
        scope_assert(std::string emoji50, for (int i = 0; i < 50; ++i) emoji50 += "😀",
                     sz::string_view(emoji50).utf8_chars().size() == 50);

        // Asymmetric overflow: 20x (2 ASCII + 3 Cyrillic) = 100 chars, 140 bytes
        scope_assert(std::string overflow_asym, for (int i = 0; i < 20; ++i) overflow_asym += "aaПРС",
                     sz::string_view(overflow_asym).utf8_count() == 100);

        // Test transitions at chunk boundaries
        // 63 bytes ASCII + 2-byte char (transition at 64-byte boundary)
        scope_assert(std::string boundary_test(63, 'x'), boundary_test += "П",
                     sz::string_view(boundary_test).utf8_chars().size() == 64);

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
    }

    // Split by Unicode newlines
    {
        auto lines = [](sz::string_view t) { return t.utf8_split_lines().template to<std::vector<std::string>>(); };

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
        let_assert(auto l = lines("a b"), l.size() >= 1);
        let_assert(auto l = lines("a b"), l.size() >= 1);

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
        auto words = [](sz::string_view t) { return t.utf8_split().template to<std::vector<std::string>>(); };

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
        let_assert(auto w = words("ab"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0085 NEL (Next Line)
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+00A0 NBSP (No-Break Space)

        // Triple-byte whitespace (17 chars) - various space widths
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+1680 OGHAM SPACE MARK
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2000 EN QUAD
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2001 EM QUAD
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2002 EN SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2003 EM SPACE
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2004 THREE-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2005 FOUR-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2006 SIX-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2007 FIGURE SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2008 PUNCTUATION SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2009 THIN SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+200A HAIR SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2028 LINE SEPARATOR
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2029 PARAGRAPH SEPARATOR
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+202F NARROW NO-BREAK SPACE
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+205F MEDIUM MATHEMATICAL SPACE
        let_assert(auto w = words("a　b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+3000 IDEOGRAPHIC SPACE

        // Mixed byte-length whitespace patterns
        let_assert(auto w = words("a   b"), w.size() == 4);             // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("a\t　b"), w.size() == 4);            // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("Hello 世界 Привет"), w.size() == 3); // Unicode content + spaces

        // Edge cases
        let_assert(auto w = words(""), w.size() == 1 && w[0] == "");
        let_assert(auto w = words("   "), w.size() == 4);                // "" "" "" ""
        let_assert(auto w = words("\t\n\r\v\f"), w.size() == 6);         // All single-byte whitespace
        let_assert(auto w = words(" "), w.size() == 3);       // All double-byte whitespace
        let_assert(auto w = words("  　"), w.size() == 4); // All triple-byte whitespace
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
        let_assert(auto w = words("a \t\n\r\vb"), w.size() == 6);                // 5 whitespace chars between a and b
        let_assert(auto w = words("a   　b"), w.size() == 5); // 1+2+3+3 byte: 4 delims -> 5 segs

        // Long sequences to test chunk boundaries - N delimiters yield N+1 segments
        scope_assert(std::string long_ws, for (int i = 0; i < 100; ++i) long_ws += " ",
                     sz::string_view(long_ws).utf8_split().template to<std::vector<std::string>>().size() ==
                         101); // 100 spaces = 101 empty segments

        scope_assert(
            std::string long_mixed,
            {
                for (int i = 0; i < 50; ++i) long_mixed += "word ";
                long_mixed.pop_back();
            }, // Remove trailing space
            sz::string_view(long_mixed).utf8_split().template to<std::vector<std::string>>().size() == 50); // 50 words
    }

    // Test with `sz::string` - not just `sz::string_view`
    {
        sz::string str = "Hello 世界";
        assert(str.utf8_count() == 8);
        assert(str.utf8_find_nth(6) == 6);
        let_assert(auto c = str.utf8_chars().template to<std::vector<sz_rune_t>>(), c.size() == 8 && c[6] == 0x4E16);

        sz::string multiline = "a\nb\nc";
        let_assert(auto l = multiline.utf8_split_lines().template to<std::vector<std::string>>(),
                   l.size() == 3 && l[1] == "b");

        sz::string words_str = "foo bar baz";
        let_assert(auto w = words_str.utf8_split().template to<std::vector<std::string>>(),
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

/** @brief Golden and reverse-equals-forward checks for UTF-8 (UAX #29) word-boundary detection. */
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

    // Segment `text` into UAX-29 words through one of the plural kernels (forward or reverse variant).
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
    auto words = [&](sz::string_view text) { return uax29_segments(sz_utf8_word_find_boundaries, text); };

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

    // For each input the active backend, the serial reference, and the forward & reverse C++ ranges must agree.
    auto check_consistency = [&](sz::string_view text) {
        auto forward = uax29_segments(sz_utf8_word_find_boundaries, text);
        std::vector<std::string> reversed(forward.rbegin(), forward.rend());
        assert(uax29_segments(sz_utf8_word_find_boundaries_serial, text) == forward && "backend ≠ serial reference");
        assert(uax29_segments(sz_utf8_word_rfind_boundaries, text) == reversed &&
               "reverse pass ≠ reversed forward list");
        assert(text.utf8_split_words().template to<std::vector<std::string>>() == forward &&
               "utf8_split_words ≠ kernel");
        assert(text.utf8_rsplit_words().template to<std::vector<std::string>>() == reversed &&
               "utf8_rsplit_words ≠ kernel");
    };
    for (sz::string_view text : {"", "a", "Hello, world!", "don't", "l'avion", "3,14", "1,2,3", "3,", "can't_stop",
                                 "a\r\nb", "Größe привет мир 你好 42", "the quick brown fox, really!"})
        check_consistency(text);

    // A long mixed string exercises the wide SIMD windows and their tails.
    scope_assert(
        std::string big,
        [&] {
            for (int i = 0; i != 200; ++i) big += "hello world, foo_bar 42 don't ";
        }(),
        uax29_segments(sz_utf8_word_find_boundaries, big) == uax29_segments(sz_utf8_word_find_boundaries_serial, big));
}

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

/** @brief Wraps a UTF-8 codepoint-counting backend by its kernel pointer. */
template <sz_utf8_count_t count_>
struct utf8_count_from_sz_ {
    sz_size_t operator()(sz_cptr_t text, sz_size_t length) const noexcept { return count_(text, length); }
};

/** @brief Wraps a UTF-8 boundary-finding backend (newlines or whitespaces) by its kernel pointer. */
template <sz_utf8_find_boundaries_t find_>
struct utf8_find_boundaries_from_sz_ {
    sz_size_t operator()(sz_cptr_t text, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths, sz_size_t capacity,
                         sz_size_t *consumed) const noexcept {
        return find_(text, length, offsets, lengths, capacity, consumed);
    }
};

/** @brief Wraps a UTF-8 normalization backend by its kernel pointer. */
template <sz_utf8_norm_t norm_>
struct utf8_norm_from_sz_ {
    sz_size_t operator()(sz_cptr_t text, sz_size_t length, sz_normal_form_t form, sz_ptr_t output) const noexcept {
        return norm_(text, length, form, output);
    }
};

/** @brief Wraps a UTF-8 normalization-violation backend by its kernel pointer. */
template <sz_utf8_norm_violation_t violation_>
struct utf8_norm_violation_from_sz_ {
    sz_cptr_t operator()(sz_cptr_t text, sz_size_t length, sz_normal_form_t form) const noexcept {
        return violation_(text, length, form);
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
 *  @param count_reference         Serial reference codepoint counter (wrapper instance).
 *  @param count_candidate         ISA-specific codepoint counter under test (wrapper instance).
 *  @param newlines_reference      Serial reference newline finder (wrapper instance).
 *  @param newlines_candidate      ISA-specific newline finder under test (wrapper instance).
 *  @param whitespaces_reference   Serial reference whitespace finder (wrapper instance).
 *  @param whitespaces_candidate   ISA-specific whitespace finder under test (wrapper instance).
 *  @param min_text_length         Minimum byte length of each generated string.
 *  @param min_iterations          Number of random strings to generate and check.
 */
template <typename count_reference_, typename count_candidate_, typename newlines_reference_,
          typename newlines_candidate_, typename whitespaces_reference_, typename whitespaces_candidate_>
void test_utf8_equivalence(                                              //
    count_reference_ count_reference, count_candidate_ count_candidate,  //
    newlines_reference_ newlines_reference, newlines_candidate_ newlines_candidate,
    whitespaces_reference_ whitespaces_reference, whitespaces_candidate_ whitespaces_candidate,
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
            sz_size_t delimiters =
                matcher(data + suffix, region, batch_offsets.data(), batch_lengths.data(), capacity, &consumed);
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

    auto check = [&](std::string const &text) {
        sz_cptr_t data = text.data();
        sz_size_t length = text.size();

        // Test `sz_utf8_count` equivalence
        sz_size_t count_result_reference = count_reference(data, length);
        sz_size_t count_result_candidate = count_candidate(data, length);
        assert(count_result_reference == count_result_candidate);

        // Sweep capacities: one huge (one-shot), the awkward 65/63 straddling the 64-byte AVX-512 window, the
        // binding default 16, and tiny 3/1 - stressing the per-window / mid-window capacity cut at every boundary.
        sz_size_t const capacities[] = {length + 64, 65, 63, 16, 3, 1};
        std::vector<sz_size_t> reference_offsets, reference_lengths, candidate_offsets, candidate_lengths;
        for (sz_size_t capacity : capacities) {
            if (capacity == 0) continue;
            enumerate(newlines_reference, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(newlines_candidate, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && "Mismatch in newline offsets");
            assert(reference_lengths == candidate_lengths && "Mismatch in newline lengths");

            enumerate(whitespaces_reference, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(whitespaces_candidate, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && "Mismatch in whitespace offsets");
            assert(reference_lengths == candidate_lengths && "Mismatch in whitespace lengths");

            // Segment-level (iterator) equivalence: catches a `bytes_consumed` overshoot at a window-aligned fill.
            reconstruct_segments(newlines_reference, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments(newlines_candidate, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "Mismatch in newline segments");
            reconstruct_segments(whitespaces_reference, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments(whitespaces_candidate, data, length, capacity, candidate_offsets, candidate_lengths);
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
    std::size_t const spaecial_chars_count = sizeof(special_chars) / sizeof(special_chars[0]);
    std::size_t const total_strings_to_sample = utf8_content_count + spaecial_chars_count;
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
        check(text);

        // Now, let's replace 10% of bytes in the sequence with a NUL character, thus breaking many valid codepoints
        std::size_t num_bytes_to_corrupt = text.size() / 10;
        std::uniform_int_distribution<std::size_t> byte_index_dist(0, text.size() - 1);
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index = byte_index_dist(rng);
            text[byte_index] = '\0';
        }
        check(text);

        // Swap 10% of bytes at random positions, creating malformed UTF-8 sequences
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index_1 = byte_index_dist(rng);
            std::size_t byte_index_2 = byte_index_dist(rng);
            std::swap(text[byte_index_1], text[byte_index_2]);
        }
        check(text);
    }
}

/**
 *  @brief Differential fuzz for a UTF-8 normalizer backend: assert it is byte-exact vs the serial
 *         reference for all four forms (norm output + violation offset). The all-codepoint shuffle
 *         stresses canonical ordering, every decomposition/composition path, and SIMD-block straddles.
 *
 *  @param norm_reference        Serial reference normalizer (wrapper instance).
 *  @param norm_candidate        ISA-specific normalizer under test (wrapper instance).
 *  @param violation_reference   Serial reference normalization-violation finder (wrapper instance).
 *  @param violation_candidate   ISA-specific normalization-violation finder under test (wrapper instance).
 *  @param iterations            Number of all-codepoint shuffles to fuzz x 4 normal forms.
 */
template <typename norm_reference_, typename norm_candidate_, typename violation_reference_,
          typename violation_candidate_>
void test_norm_equivalence(norm_reference_ norm_reference, norm_candidate_ norm_candidate,
                           violation_reference_ violation_reference, violation_candidate_ violation_candidate,
                           std::size_t iterations = scale_iterations(25)) {
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

        for (sz_normal_form_t form : norm_forms) {
            sz_size_t len_reference = norm_reference(input_buffer.data(), input_length, form, output_reference.data());
            sz_size_t len_candidate = norm_candidate(input_buffer.data(), input_length, form, output_candidate.data());
            if (len_reference != len_candidate ||
                std::memcmp(output_reference.data(), output_candidate.data(), len_reference) != 0) {
                std::fprintf(stderr, "norm mismatch (form=%d, iter=%zu): reference_len=%zu candidate_len=%zu\n",
                             (int)form, it, (size_t)len_reference, (size_t)len_candidate);
                assert(false);
            }
            sz_cptr_t viol_reference = violation_reference(input_buffer.data(), input_length, form);
            sz_cptr_t viol_candidate = violation_candidate(input_buffer.data(), input_length, form);
            if (viol_reference != viol_candidate) {
                std::fprintf(stderr, "norm violation mismatch (form=%d, iter=%zu)\n", (int)form, it);
                assert(false);
            }
        }
    }
    std::printf("    normalization fuzzing passed!\n");
}

#pragma endregion // Equivalence

#pragma region Drivers

/** @brief Run the UTF-8 count/newline/whitespace differential against every compiled SIMD backend. */
void test_utf8_all() {
#if SZ_USE_HASWELL
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_haswell> {},                               //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_haswell> {},             //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_haswell> {});
#endif
#if SZ_USE_ICELAKE
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_icelake> {},                               //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_icelake> {},             //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_icelake> {});
#endif
#if SZ_USE_NEON
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_neon> {},                                  //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_neon> {},                //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_neon> {});
#endif
#if SZ_USE_SVE2
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_sve2> {},                                  //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_sve2> {},                //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_sve2> {});
#endif
#if SZ_USE_V128
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_v128> {},                                  //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_v128> {},                //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_v128> {});
#endif
#if SZ_USE_V128RELAXED
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_v128relaxed> {},                           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_v128relaxed> {},         //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_v128relaxed> {});
#endif
#if SZ_USE_RVV
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_rvv> {},                                   //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_rvv> {},                 //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_rvv> {});
#endif
#if SZ_USE_LASX
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_lasx> {},                                  //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_lasx> {},                //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_lasx> {});
#endif
#if SZ_USE_POWERVSX
    test_utf8_equivalence(                                                           //
        utf8_count_from_sz_<sz_utf8_count_serial> {},                                //
        utf8_count_from_sz_<sz_utf8_count_powervsx> {},                              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_serial> {},              //
        utf8_find_boundaries_from_sz_<sz_utf8_find_newlines_powervsx> {},            //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_serial> {},           //
        utf8_find_boundaries_from_sz_<sz_utf8_find_whitespaces_powervsx> {});
#endif
}

/** @brief Run the normalization differential fuzz against every compiled SIMD backend. */
void test_norm_all() {
#if SZ_USE_HASWELL
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_haswell> {},                        //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_haswell> {});
#endif
#if SZ_USE_SKYLAKE
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_skylake> {},                        //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_skylake> {});
#endif
#if SZ_USE_ICELAKE
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_icelake> {},                        //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_icelake> {});
#endif
#if SZ_USE_NEON
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_neon> {},                           //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_neon> {});
#endif
#if SZ_USE_SVE
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_sve> {},                            //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_sve> {});
#endif
#if SZ_USE_SVE2
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_sve2> {},                           //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_sve2> {});
#endif
#if SZ_USE_RVV
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_rvv> {},                            //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_rvv> {});
#endif
#if SZ_USE_V128
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_v128> {},                           //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_v128> {});
#endif
#if SZ_USE_V128RELAXED
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_v128relaxed> {},                    //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_v128relaxed> {});
#endif
#if SZ_USE_LASX
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_lasx> {},                           //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_lasx> {});
#endif
#if SZ_USE_POWERVSX
    test_norm_equivalence(                                                  //
        utf8_norm_from_sz_<sz_utf8_norm_serial> {},                         //
        utf8_norm_from_sz_<sz_utf8_norm_powervsx> {},                       //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_serial> {},     //
        utf8_norm_violation_from_sz_<sz_utf8_norm_violation_powervsx> {});
#endif
}

#pragma endregion // Drivers
