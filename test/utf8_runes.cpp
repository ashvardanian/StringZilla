/**
 *  @brief  UTF-8 codepoint counting, nth-character finding, and streaming rune-unpacking tests.
 *  @file scripts/test_utf8_runes.cpp
 *  @author Ash Vardanian
 *  @date June 20, 2026
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

#include <algorithm> // `std::transform`
#include <iterator>  // `std::distance`
#include <random>    // `std::random_device`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`

#pragma region Helpers

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_encode` (silently skips invalid runes). */
static void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_encode(codepoint, bytes);
    if (length == sz_rune_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
}

/**
 *  @brief Builds a random, well-formed UTF-8 string whose codepoints span all four byte-widths and
 *         every 1->2->3->4 transition, so the count/find-nth/unpack kernels hit their mixed-width paths.
 *
 *  @param target_codepoints  How many codepoints to emit (the exact count, recorded by the caller).
 *  @param rng                Random source; drawing through it keeps `SZ_TESTS_SEED` reproducible.
 *  @return The encoded UTF-8 bytes.
 */
static std::string random_valid_utf8_(std::size_t target_codepoints, std::mt19937 &rng) {
    // Disjoint ranges, one per byte-width, chosen to avoid surrogates and noncharacters.
    static struct {
        sz_rune_t low, high;
    } const ranges[] = {
        {0x000020u, 0x00007Eu}, // 1-byte ASCII printable
        {0x0000A1u, 0x0007FFu}, // 2-byte
        {0x000800u, 0x00CFFFu}, // 3-byte (stays below the U+D800 surrogate block)
        {0x010000u, 0x0EFFFFu}, // 4-byte (avoids the U+FFFE/U+FFFF plane enders below 0x10000)
    };
    std::uniform_int_distribution<int> width_pick(0, 3);
    std::string text;
    text.reserve(target_codepoints * 4);
    for (std::size_t index = 0; index != target_codepoints; ++index) {
        int const width = width_pick(rng);
        std::uniform_int_distribution<sz_rune_t> codepoint(ranges[width].low, ranges[width].high);
        std::size_t const before = text.size();
        // Retry until one valid codepoint is actually appended, so the emitted count is exact.
        while (text.size() == before) append_codepoint_(text, codepoint(rng));
    }
    return text;
}

/**
 *  @brief Streams `unpack` over the entire @p text, collecting every decoded rune.
 *
 *  Mirrors the documented streaming contract: each call decodes a prefix of the remaining bytes and
 *  reports how many runes it produced and how far the cursor advanced; on valid UTF-8 every call must
 *  advance, so the loop terminates.
 *
 *  @param unpack    Streaming chunk decoder under test.
 *  @param text      Valid UTF-8 input bytes.
 *  @param length    Byte length of @p text.
 *  @param[out] out  Receives the decoded codepoints in order.
 */
static void collect_unpacked_runes_(sz_utf8_decode_t unpack, sz_cptr_t text, sz_size_t length,
                                    std::vector<sz_rune_t> &out) {
    out.clear();
    sz_size_t const chunk_capacity = 64u;
    sz_cptr_t position = text;
    sz_cptr_t const end = text + length;
    while (position < end) {
        sz_rune_t chunk_runes[64];
        sz_size_t chunk_count = 0;
        sz_cptr_t const next = unpack(position, (sz_size_t)(end - position), chunk_runes, chunk_capacity, &chunk_count);
        assert(next > position); // Must advance, otherwise the streaming loop would never terminate
        // Fill-or-drain contract: a call that did not reach the end must have filled the capacity, so an iterator
        // issues one decode per buffer regardless of script width (no early stop after a single register-worth).
        assert((next == end || chunk_count == chunk_capacity) && "Unpack stopped before filling capacity");
        for (sz_size_t index = 0; index != chunk_count; ++index) out.push_back(chunk_runes[index]);
        position = next;
    }
}

/**
 *  @brief Runs one UTF-8 codepoint backend (count + nth-finder + chunk-unpacker) over the known-answer
 *         anchor and asserts the produced count, byte offsets, and decoded runes match the expectations.
 *
 *  Mirrors `check_sha256_unit_` in `test_hash.cpp`: the caller drives it once per backend (dispatched,
 *  serial, and each natively-compiled kernel), so a wrong constant shared by the serial-vs-SIMD agreement
 *  tests is still caught against an external ground truth.
 *
 *  @param count           Codepoint counter under test.
 *  @param find_nth        Nth-codepoint byte-offset finder under test.
 *  @param unpack          Streaming chunk decoder under test (or NULL when this backend has no decoder).
 *  @param text            Anchor text whose codepoints are counted / located / decoded.
 *  @param length          Byte length of @p text.
 *  @param expected_count  Expected codepoint count of @p text.
 *  @param expected_runes  Expected decoded codepoints, in order.
 */
static void check_utf8_runes_unit_(                                          //
    sz_utf8_count_t count, sz_utf8_seek_t find_nth, sz_utf8_decode_t unpack, //
    sz_cptr_t text, sz_size_t length, sz_size_t expected_count,              //
    std::vector<sz_rune_t> const &expected_runes) {

    // Codepoint count is the external ground truth - not derived from a sibling kernel.
    assert(count(text, length) == expected_count);

    // `sz_utf8_seek`: codepoint starts land at the byte offset of each rune; the one-past-the-end
    // index returns the NUL sentinel.
    sz_size_t byte_offset = 0;
    for (sz_size_t rune_index = 0; rune_index != expected_count; ++rune_index) {
        assert(find_nth(text, length, rune_index) == text + byte_offset);
        sz_u8_t bytes[4];
        byte_offset += (sz_size_t)sz_rune_encode(expected_runes[rune_index], bytes);
    }
    assert(find_nth(text, length, expected_count) == SZ_NULL_CHAR); // Beyond the last codepoint

    // `sz_utf8_decode`: streaming the decoder must reproduce exactly the expected runes.
    if (unpack) {
        std::vector<sz_rune_t> decoded;
        collect_unpacked_runes_(unpack, text, length, decoded);
        assert(decoded.size() == expected_count);
        for (sz_size_t index = 0; index != expected_count; ++index) assert(decoded[index] == expected_runes[index]);
    }
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer unit tests for the UTF-8 codepoints family on simple, hand-verifiable inputs.
 *
 *  Exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through
 *  the C++ `sz::string_view` wrappers, so a regression that the serial-vs-SIMD agreement tests would
 *  miss - because both share a wrong constant - is still caught against an external ground truth. This
 *  is the isolated coverage for `sz_utf8_seek` and `sz_utf8_decode`, whose SIMD variants are
 *  otherwise only fuzzed against serial.
 */
void test_utf8_runes_unit() {
    std::printf("  - testing UTF-8 codepoints known-answer vectors...\n");

    // The mixed-script anchor: "a\xC3\x9F\xE4\xB8\xAD" is `a` (1 byte) + U+00DF (2 bytes) + U+4E2D (3 bytes),
    // so 6 bytes encode exactly 3 codepoints {0x61, 0xDF, 0x4E2D}.
    char const mixed[] = "a\xC3\x9F\xE4\xB8\xAD";
    sz_size_t const mixed_length = (sz_size_t)(sizeof(mixed) - 1);
    assert(mixed_length == 6u);
    std::vector<sz_rune_t> const mixed_runes = {0x61u, 0xDFu, 0x4E2Du};

    // Drive the count + find-nth + unpack known-answer through the dispatched, serial, and native kernels.
    check_utf8_runes_unit_(sz_utf8_count, sz_utf8_seek, sz_utf8_decode, // Dispatched
                           mixed, mixed_length, 3u, mixed_runes);
    check_utf8_runes_unit_(sz_utf8_count_serial, sz_utf8_seek_serial, sz_utf8_decode_serial, // serial
                           mixed, mixed_length, 3u, mixed_runes);
#if SZ_USE_HASWELL
    check_utf8_runes_unit_(sz_utf8_count_haswell, sz_utf8_seek_haswell,
                           sz_utf8_decode_haswell, // haswell
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_ICELAKE
    check_utf8_runes_unit_(sz_utf8_count_icelake, sz_utf8_seek_icelake,
                           sz_utf8_decode_icelake, // icelake
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_NEON
    check_utf8_runes_unit_(sz_utf8_count_neon, sz_utf8_seek_neon,
                           sz_utf8_decode_neon, // neon
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_SVE2
    check_utf8_runes_unit_(sz_utf8_count_sve2, sz_utf8_seek_sve2,
                           sz_utf8_decode_sve2, // sve2
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_V128
    check_utf8_runes_unit_(sz_utf8_count_v128, sz_utf8_seek_v128,
                           sz_utf8_decode_v128, // v128
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_RVV
    check_utf8_runes_unit_(sz_utf8_count_rvv, sz_utf8_seek_rvv,
                           sz_utf8_decode_rvv, // rvv
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_POWERVSX
    check_utf8_runes_unit_(sz_utf8_count_powervsx, sz_utf8_seek_powervsx,
                           sz_utf8_decode_powervsx, // powervsx
                           mixed, mixed_length, 3u, mixed_runes);
#endif
#if SZ_USE_LASX
    check_utf8_runes_unit_(sz_utf8_count_lasx, sz_utf8_seek_lasx,
                           sz_utf8_decode_lasx, // lasx
                           mixed, mixed_length, 3u, mixed_runes);
#endif

    // C++ API: character counting vs byte length through the `sz::string_view` wrappers.
    assert(sz::string_view(mixed, mixed_length).utf8_count() == 3u);
    assert("hello"_sv.utf8_count() == 5);
    assert("hello"_sv.size() == 5);
    assert("Hello World"_sv.utf8_count() == 11);
    assert(sz::string_view("").utf8_count() == 0);
    assert(sz::string_view("Hello \xE4\xB8\x96\xE7\x95\x8C").utf8_count() == 8); // "Hello " (6) + 2 CJK chars
    assert(sz::string_view("Hello \xE4\xB8\x96\xE7\x95\x8C").size() == 12);      // "Hello " (6) + 6 bytes
    assert(sz::string_view("Hello \xF0\x9F\x98\x80").utf8_count() == 7);         // "Hello " (6) + 1 emoji
    assert(sz::string_view("Hello \xF0\x9F\x98\x80").size() == 10);              // "Hello " (6) + 4 bytes
    assert(sz::string_view("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82").utf8_count() == 3);
    assert(sz::string_view("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82").size() == 12);
    assert(sz::string_view("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82").utf8_count() == 6);
    assert(sz::string_view("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82").size() == 12);

    // C++ API: byte offset of the nth character (and the npos beyond-end sentinel).
    {
        sz::string_view text = "Hello";
        assert(text.utf8_seek(0) == 0);
        assert(text.utf8_seek(1) == 1);
        assert(text.utf8_seek(4) == 4);
        assert(text.utf8_seek(5) == sz::string_view::npos);
        assert(text.utf8_seek(100) == sz::string_view::npos);
    }
    {
        sz::string_view text = "Hello \xE4\xB8\x96\xE7\x95\x8C";
        assert(text.utf8_seek(0) == 0); // 'H' at byte 0
        assert(text.utf8_seek(5) == 5); // ' ' at byte 5
        assert(text.utf8_seek(6) == 6); // '世' at byte 6
        assert(text.utf8_seek(7) == 9); // '界' at byte 9
        assert(text.utf8_seek(8) == sz::string_view::npos);
    }
    {
        sz::string_view text = "\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82";
        assert(text.utf8_seek(0) == 0); // First emoji at byte 0
        assert(text.utf8_seek(1) == 4); // Second emoji at byte 4
        assert(text.utf8_seek(2) == 8); // Third emoji at byte 8
        assert(text.utf8_seek(3) == sz::string_view::npos);
    }

    // C++ API: codepoint (rune) iteration materialized as a vector - never a range-for over the view range,
    // whose sentinel comparison is a C++17 extension that errors at C++11.
    {
        auto runes_of = [](char const *t) {
            return sz::string_view(t).utf8_runes().template to<std::vector<sz_rune_t>>();
        };

        // Basic ASCII and edge cases
        let_assert(auto c = runes_of("Hello"), c.size() == 5 && c[0] == 'H' && c[4] == 'o');
        let_assert(auto c = runes_of(""), c.size() == 0);
        let_assert(auto c = runes_of("A"), c.size() == 1 && c[0] == 'A');

        // CJK (3-byte UTF-8)
        let_assert(auto c = runes_of("\xE4\xB8\x96\xE7\x95\x8C"), c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x754C);
        let_assert(auto c = runes_of("\xE4\xBD\xA0\xE5\xA5\xBD"), c.size() == 2 && c[0] == 0x4F60 && c[1] == 0x597D);

        // Cyrillic (2-byte UTF-8)
        let_assert(auto c = runes_of("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82"),
                   c.size() == 6 && c[0] == 0x041F && c[5] == 0x0442);

        // Arabic/RTL (2-byte UTF-8)
        let_assert(auto c = runes_of("\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7"),
                   c.size() == 5 && c[0] == 0x0645 && c[4] == 0x0627);

        // Hebrew/RTL (2-byte UTF-8)
        let_assert(auto c = runes_of("\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D"),
                   c.size() == 4 && c[0] == 0x05E9 && c[3] == 0x05DD);

        // Thai (3-byte UTF-8)
        let_assert(auto c = runes_of("\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5"),
                   c.size() == 6 && c[0] == 0x0E2A);

        // Devanagari/Hindi (3-byte UTF-8)
        let_assert(auto c = runes_of("\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87"),
                   c.size() == 6 && c[0] == 0x0928);

        // Emoji: basic smileys (4-byte UTF-8)
        let_assert(auto c = runes_of("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82"),
                   c.size() == 3 && c[0] == 0x1F600 && c[2] == 0x1F602);

        // Emoji: with variation selector
        let_assert(auto c = runes_of("\xE2\x9D\xA4\xEF\xB8\x8F"), c.size() == 2 && c[0] == 0x2764 && c[1] == 0xFE0F);

        // Emoji: various categories
        let_assert(auto c = runes_of("\xF0\x9F\x9A\x80\xF0\x9F\x8E\x89\xF0\x9F\x94\xA5"),
                   c.size() == 3 && c[0] == 0x1F680);

        // Maximum valid Unicode codepoint (U+10FFFF)
        let_assert(auto c = runes_of("\xF4\x8F\xBF\xBF"), c.size() == 1 && c[0] == 0x10FFFF);

        // Deseret alphabet (4-byte UTF-8, U+10400 range)
        let_assert(auto c = runes_of("\xF0\x90\x90\xB7"), c.size() == 1 && c[0] == 0x10437);

        // Mixed scripts
        let_assert(auto c = runes_of("Hello\xE4\xB8\x96\xE7\x95\x8C"), c.size() == 7 && c[4] == 'o' && c[5] == 0x4E16);
        let_assert(auto c = runes_of("a\xF0\x90\x90\xB7" "b"),
                   c.size() == 3 && c[0] == 'a' && c[1] == 0x10437 && c[2] == 'b');

        // Zero-width characters
        let_assert(auto c = runes_of("a\xE2\x80\x8B" "b"),
                   c.size() == 3 && c[0] == 'a' && c[1] == 0x200B && c[2] == 'b');
        let_assert(auto c = runes_of("\xEF\xBB\xBF"), c.size() == 1 && c[0] == 0xFEFF); // BOM

        // Combining diacritics (e + combining acute) vs precomposed. Written as \xHH escapes so the decomposed
        // sequence cannot be NFC-composed away by an editor/normalizer.
        let_assert(auto c = runes_of("e\xCC\x81"), c.size() == 2 && c[0] == 'e' && c[1] == 0x0301); // e + U+0301
        let_assert(auto c = runes_of("\xC3\xA9"), c.size() == 1 && c[0] == 0x00E9); // precomposed U+00E9

        // Missing transitions: 1->2, 2->1, 2->3, 3->2, 2->4, 4->2, 3->4, 4->3
        let_assert(auto c = runes_of("a\xD0\x9F"), c.size() == 2 && c[0] == 'a' && c[1] == 0x041F);    // 1->2
        let_assert(auto c = runes_of("\xD0\x9F" "a"), c.size() == 2 && c[0] == 0x041F && c[1] == 'a'); // 2->1
        let_assert(auto c = runes_of("\xD0\x9F\xE4\xB8\x96"),
                   c.size() == 2 && c[0] == 0x041F && c[1] == 0x4E16); // 2->3
        let_assert(auto c = runes_of("\xE4\xB8\x96\xD0\x9F"),
                   c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x041F); // 3->2
        let_assert(auto c = runes_of("\xD0\x9F\xF0\x9F\x98\x80"),
                   c.size() == 2 && c[0] == 0x041F && c[1] == 0x1F600); // 2->4
        let_assert(auto c = runes_of("\xF0\x9F\x98\x80\xD0\x9F"),
                   c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x041F); // 4->2
        let_assert(auto c = runes_of("\xE4\xB8\x96\xF0\x9F\x98\x80"),
                   c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x1F600); // 3->4
        let_assert(auto c = runes_of("\xF0\x9F\x98\x80\xE4\xB8\x96"),
                   c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x4E16); // 4->3

        // Extended transitions with same-length runs
        let_assert(auto c = runes_of("\xD0\x9F\xD0\xA0\xD0\xA1"),
                   c.size() == 3 && c[0] == 0x041F && c[2] == 0x0421); // 2->2->2
        let_assert(auto c = runes_of("\xE4\xB8\x96\xE7\x95\x8C\xE4\xBA\xBA"),
                   c.size() == 3 && c[0] == 0x4E16 && c[2] == 0x4EBA); // 3->3->3

        // Asymmetric alternating patterns - stress the homogeneity assumption
        let_assert(auto c = runes_of("xx\xD0\x9F\xD0\x9F\xD0\x9Fxx\xD0\x9F\xD0\x9F\xD0\x9F"),
                   c.size() == 10);                                                              // 2 ASCII, 3 Cyrillic
        let_assert(auto c = runes_of("xxx\xD0\x9F\xD0\x9Fxxx\xD0\x9F\xD0\x9F"), c.size() == 10); // 3 ASCII, 2 Cyrillic
        let_assert(auto c = runes_of("xx\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96xx\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96"),
                   c.size() == 10); // 2 ASCII, 3 CJK
        let_assert(
            auto c = runes_of(
                "\xD0\x9F\xD0\x9F\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xD0\x9F\xD0\x9F\xE4\xB8\x96" "\xE4\xB8\x96" "\xE4" "\xB8" "\x96"),
            c.size() == 10); // 2 Cyrillic, 3 CJK
        let_assert(
            auto c = runes_of(
                "\xE4\xB8\x96\xE4\xB8\x96\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xE4\xB8" "\x96\xE4\xB8" "\x96" "\xF0" "\x9F" "\x98" "\x80" "\xF0\x9F\x98\x80" "\xF0\x9F\x98\x80"),
            c.size() == 10); // 2 CJK, 3 Emoji
        let_assert(auto c = runes_of("xxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80xxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80"),
                   c.size() == 10); // 3 ASCII, 2 Emoji

        // Pathological mixed patterns
        let_assert(
            auto c = runes_of(
                "xx\xD0\x9F\xD0\x9F\xD0\x9F\xD0\x9F\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8" "\x96\xF0" "\x9F\x98" "\x80\xF0" "\x9F\x98" "\x80\xF0" "\x9F\x98" "\x80\xF0" "\x9F\x98" "\x80\xF0" "\x9F" "\x98\x80"),
            c.size() == 15); // 2-4-4-5
        let_assert(
            auto c = runes_of(
                "xx\xD0\x9F\xD0\x9F\xD0\x9Fxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0" "\x9F\x98\x80" "\xE4" "\xB8" "\x96" "\xE4" "\xB8" "\x96\xE4\xB8\x96" "\xD0\x9F\xD0\x9F"),
            c.size() == 16); // 2-3-2-4-3-2

        // Extended asymmetric: 30x "xxППП" = 150 chars, 210 bytes (crosses multiple 64-byte chunks)
        scope_assert(std::string asym_long, for (int i = 0; i < 30; ++i) asym_long += "xx\xD0\x9F\xD0\x9F\xD0\x9F",
                     sz::string_view(asym_long).utf8_count() == 150);
    }

    // 64-byte chunk boundaries and batch limits, materialized via the vector wrapper.
    {
        // Critical 63, 64, 65 byte boundaries
        let_assert(std::string s63(63, 'x'), sz::string_view(s63).utf8_runes().size() == 63);
        let_assert(std::string s64(64, 'x'), sz::string_view(s64).utf8_runes().size() == 64);
        let_assert(std::string s65(65, 'x'), sz::string_view(s65).utf8_runes().size() == 65);

        // ASCII batch limit: 16 characters max per Ice Lake iteration
        let_assert(std::string s17(17, 'x'), sz::string_view(s17).utf8_runes().size() == 17);
        let_assert(std::string s20(20, 'x'), sz::string_view(s20).utf8_runes().size() == 20);

        // 2-byte batch limit: 32 characters (64 bytes) max per iteration
        scope_assert(std::string cyr32, for (int i = 0; i < 32; ++i) cyr32 += "\xD0\x9F",
                     sz::string_view(cyr32).utf8_count() == 32);
        scope_assert(std::string cyr33, for (int i = 0; i < 33; ++i) cyr33 += "\xD0\x9F",
                     sz::string_view(cyr33).utf8_count() == 33);

        // 3-byte batch limit: 16 characters (48 bytes) max per iteration
        scope_assert(std::string cjk16, for (int i = 0; i < 16; ++i) cjk16 += "\xE4\xB8\x96",
                     sz::string_view(cjk16).utf8_count() == 16);
        scope_assert(std::string cjk17, for (int i = 0; i < 17; ++i) cjk17 += "\xE4\xB8\x96",
                     sz::string_view(cjk17).utf8_count() == 17);

        // 4-byte batch limit: 16 characters (64 bytes) max per iteration
        scope_assert(std::string emoji16, for (int i = 0; i < 16; ++i) emoji16 += "\xF0\x9F\x98\x80",
                     sz::string_view(emoji16).utf8_count() == 16);
        scope_assert(std::string emoji17, for (int i = 0; i < 17; ++i) emoji17 += "\xF0\x9F\x98\x80",
                     sz::string_view(emoji17).utf8_count() == 17);

        // Asymmetric at chunk boundary: 60 ASCII + "ПП世" = 63 chars, 67 bytes
        scope_assert(std::string boundary_asym(60, 'x'), boundary_asym += "\xD0\x9F\xD0\x9F\xE4\xB8\x96",
                     sz::string_view(boundary_asym).utf8_count() == 63);

        // Sequences exceeding batch limits
        scope_assert(std::string cyr100, for (int i = 0; i < 100; ++i) cyr100 += "\xD0\x9F",
                     sz::string_view(cyr100).utf8_runes().size() == 100);
        scope_assert(std::string cjk50, for (int i = 0; i < 50; ++i) cjk50 += "\xE4\xB8\x96",
                     sz::string_view(cjk50).utf8_runes().size() == 50);
        scope_assert(std::string emoji50, for (int i = 0; i < 50; ++i) emoji50 += "\xF0\x9F\x98\x80",
                     sz::string_view(emoji50).utf8_runes().size() == 50);

        // Asymmetric overflow: 20x (2 ASCII + 3 Cyrillic) = 100 chars, 140 bytes
        scope_assert(std::string overflow_asym,
                     for (int i = 0; i < 20; ++i) overflow_asym += "aa\xD0\x9F\xD0\xA0\xD0\xA1",
                     sz::string_view(overflow_asym).utf8_count() == 100);

        // Transitions at chunk boundaries
        scope_assert(std::string boundary_test(63, 'x'), boundary_test += "\xD0\x9F",
                     sz::string_view(boundary_test).utf8_runes().size() == 64);
        scope_assert(
            std::string span_asym,
            {
                for (int i = 0; i < 30; ++i) span_asym += "aa";
                for (int i = 0; i < 8; ++i) span_asym += "\xD0\x9F\xD0\xA0\xD0\xA1";
            },
            sz::string_view(span_asym).utf8_count() == 84);
        scope_assert(std::string exact_boundary(64, 'x'), exact_boundary += "\xD0\x9F\xE4\xB8\x96\xF0\x9F\x98\x80",
                     sz::string_view(exact_boundary).utf8_count() == 67);
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief Cross-checks the serial UTF-8 codepoint kernels against a candidate SIMD backend on random,
 *         well-formed inputs: the chunk-unpacked runes, the nth-codepoint byte offsets, and the count.
 *
 *  The known-answer anchors live in `test_utf8_runes_unit`; this is the serial-vs-ISA differential,
 *  the only coverage that exercises the SIMD `sz_utf8_decode` and `sz_utf8_seek` variants.
 *
 *  @param count_serial      Reference (serial) codepoint counter.
 *  @param count_candidate   Candidate codepoint counter to validate against the reference.
 *  @param find_nth_serial   Reference (serial) nth-codepoint byte-offset finder.
 *  @param find_nth_cand     Candidate nth-codepoint finder to validate against the reference.
 *  @param unpack_serial     Reference (serial) streaming chunk decoder.
 *  @param unpack_candidate  Candidate streaming chunk decoder (or NULL when the backend has none).
 *  @param inputs            Number of random inputs to fuzz, scaled by the global multiplier.
 */
static inline void test_utf8_runes_equivalence(                        //
    sz_utf8_count_t count_serial, sz_utf8_count_t count_candidate,     //
    sz_utf8_seek_t find_nth_serial, sz_utf8_seek_t find_nth_candidate, //
    sz_utf8_decode_t unpack_serial, sz_utf8_decode_t unpack_candidate, //
    sz_size_t inputs) {

    auto &rng = global_random_generator();
    std::vector<sz_rune_t> runes_serial, runes_candidate;

    auto check = [&](std::string const &text) {
        sz_cptr_t const data = text.data();
        sz_size_t const length = (sz_size_t)text.size();

        // Count must agree between serial and candidate.
        sz_size_t const count_reference = count_serial(data, length);
        assert(count_candidate(data, length) == count_reference);

        // `sz_utf8_seek`: sweep every codepoint index plus a couple past the end.
        for (sz_size_t n = 0; n <= count_reference + 2u; ++n)
            assert(find_nth_candidate(data, length, n) == find_nth_serial(data, length, n));

        // `sz_utf8_decode`: streaming both backends must decode identical runes.
        if (unpack_candidate) {
            collect_unpacked_runes_(unpack_serial, data, length, runes_serial);
            collect_unpacked_runes_(unpack_candidate, data, length, runes_candidate);
            assert(runes_serial.size() == count_reference);
            assert(runes_candidate == runes_serial);
        }
    };

    // Structured length ladder around the SIMD window boundaries, every codepoint count exercised.
    sz_size_t const ladder[] = {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 100u, 200u};
    for (sz_size_t codepoints : ladder) check(random_valid_utf8_(codepoints, rng));

    // Fuzzed inputs of random codepoint counts, each placed at every sub-cache-line offset so serial-vs-ISA
    // agreement is checked across all alignments the SIMD kernels may hit.
    std::uniform_int_distribution<std::size_t> codepoint_distribution(0, 96);
    for (sz_size_t iteration = 0; iteration != inputs; ++iteration) {
        std::string const text = random_valid_utf8_(codepoint_distribution(rng), rng);
        for_each_cacheline_offset_(text.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, text.data(), text.size());
            check(std::string(buffer, text.size()));
        });
    }
}

/**
 *  @brief Folds in the moved large-buffer count agreement: a few hundred KB of mixed-width codepoints
 *         where the dispatched and C++ counts must equal the serial reference and the exact known total.
 */
static void test_utf8_runes_large_count() {
    // Every repeat contributes one ASCII 'x', one 2-byte, one 3-byte, and one 4-byte codepoint - 4
    // codepoints in 10 bytes - so the total is exactly `repeats * 4`.
    char const unit[] = "x\xD0\x9F\xE4\xB8\xAD\xF0\x9F\x98\x80"; // 'x' + U+041F + U+4E2D + U+1F600, 10 bytes
    std::size_t const repeats = scale_iterations(30000);         // ~300 KB at the default multiplier
    std::string mixed;
    mixed.reserve(repeats * (sizeof(unit) - 1));
    for (std::size_t repeat = 0; repeat != repeats; ++repeat) mixed.append(unit, sizeof(unit) - 1);

    sz_size_t const expected_codepoints = (sz_size_t)(repeats * 4);
    sz_size_t const count_serial = sz_utf8_count_serial(mixed.data(), mixed.size());
    assert(count_serial == expected_codepoints);
    assert(sz_utf8_count(mixed.data(), mixed.size()) == count_serial); // Dispatched matches serial
    assert(sz::string_view(mixed).utf8_count() == count_serial);       // C++ wrapper matches serial
}

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Feeds malformed / invalid UTF-8 through one backend's counting and streaming-unpack kernels,
 *         asserting no crash, in-bounds output, and a cursor that never escapes the input.
 *
 *  Counting is bounds-safe on arbitrary bytes, so it faces the full malformed battery: it must merely
 *  survive. `sz_utf8_decode` documents a valid-UTF-8 precondition (the decoder "performs no
 *  validity checks") and asserts internally on garbage, so it is only exercised on the `sz_utf8_find_malformed`
 *  subset; each call must report no more runes than the destination holds and never let its cursor run
 *  past the input.
 *
 *  @param count          Codepoint counter under test.
 *  @param unpack         Streaming chunk decoder under test (or NULL when this backend has no decoder).
 *  @param random_inputs  Number of random garbage buffers to fuzz on top of the exhaustive byte sweeps.
 */
static void check_utf8_runes_safety_(sz_utf8_count_t count, sz_utf8_decode_t unpack,
                                     std::size_t random_inputs = scale_iterations(10000)) {

    std::size_t const max_input_length = 70;
    std::vector<sz_rune_t> rune_destination;

    auto check = [&](char const *input, std::size_t input_length) {
        // Counting just has to survive and return some value on arbitrary bytes.
        sz_size_t const counted = count(input, (sz_size_t)input_length);
        sz_unused_(counted);

        // Streaming unpack is total under the unified contract: it runs on arbitrary bytes, substituting U+FFFD
        // for ill-formed input. Each call must report no more runes than the destination holds, a cursor that
        // never runs past the input, and only valid Unicode scalar values (no surrogate, none beyond U+10FFFF) -
        // the precondition every binding relies on to convert runes without re-validation. A well-formed but
        // truncated trailing sequence legitimately stalls (returns the same cursor) - so we stop rather than spin.
        if (unpack) {
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
                for (sz_size_t rune_index = 0; rune_index != produced; ++rune_index) {
                    sz_rune_t const rune = rune_destination[rune_index];
                    assert(rune <= 0x10FFFFu && !(rune >= 0xD800u && rune <= 0xDFFFu) &&
                           "Unpack emitted a non-scalar value (surrogate or out of range)");
                }
                if (next == cursor) break; // Stalled on a truncated trailing sequence - the caller would refill
                cursor = next;
            }
        }
    };

    char input[max_input_length];

    // The named adversarial shapes, exercised directly.
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

/**
 *  @brief One UTF-8 codepoint backend: its three kernels plus whether its streaming decoder is hardened against
 *         malformed input. Built once here and iterated by every driver, so the unit/safety and equivalence
 *         passes can never drift apart in which ISAs they cover. The always-present `dispatched` entry keeps the
 *         table non-empty (and the helpers live) even on a baseline target with no SIMD tier compiled in.
 */
struct utf8_runes_backend_t {
    char const *name;
    sz_utf8_count_t count;
    sz_utf8_seek_t seek;
    sz_utf8_decode_t decode; // Streaming decoder, or `SZ_NULL` when the backend has none (e.g. `v128relaxed`).
};

static utf8_runes_backend_t const utf8_runes_backends[] = {
    {"dispatched", sz_utf8_count, sz_utf8_seek, sz_utf8_decode},
#if SZ_USE_HASWELL
    {"haswell", sz_utf8_count_haswell, sz_utf8_seek_haswell, sz_utf8_decode_haswell},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_utf8_count_icelake, sz_utf8_seek_icelake, sz_utf8_decode_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_utf8_count_neon, sz_utf8_seek_neon, sz_utf8_decode_neon},
#endif
#if SZ_USE_SVE2
    {"sve2", sz_utf8_count_sve2, sz_utf8_seek_sve2, sz_utf8_decode_sve2},
#endif
#if SZ_USE_V128
    {"v128", sz_utf8_count_v128, sz_utf8_seek_v128, sz_utf8_decode_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_utf8_count_v128relaxed, sz_utf8_seek_v128relaxed, SZ_NULL},
#endif
#if SZ_USE_RVV
    {"rvv", sz_utf8_count_rvv, sz_utf8_seek_rvv, sz_utf8_decode_rvv},
#endif
#if SZ_USE_LASX
    {"lasx", sz_utf8_count_lasx, sz_utf8_seek_lasx, sz_utf8_decode_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_utf8_count_powervsx, sz_utf8_seek_powervsx, sz_utf8_decode_powervsx},
#endif
};

/** @brief Drive the malformed-input safety probe through serial, dispatched, and every native backend. */
void test_utf8_runes_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 codepoint kernels...\n");

    // Serial is the reference contract; the dispatched and native backends below face the same probe.
    check_utf8_runes_safety_(sz_utf8_count_serial, sz_utf8_decode_serial);
    for (utf8_runes_backend_t const &backend : utf8_runes_backends)
        check_utf8_runes_safety_(backend.count, backend.decode);

    std::printf("    malformed-input safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD UTF-8 codepoint differential (unpack + find-nth + count) across every
 *         backend compiled on this target, plus the moved large-buffer count agreement.
 */
void test_utf8_runes_all() {
    sz_size_t const inputs = (sz_size_t)scale_iterations(200);

    // Serial is the reference; the dispatched entry and every native backend are differenced against it. A
    // `SZ_NULL` decoder (e.g. `v128relaxed`) simply skips the streaming-decode leg inside the helper.
    for (utf8_runes_backend_t const &backend : utf8_runes_backends)
        test_utf8_runes_equivalence(sz_utf8_count_serial, backend.count, //
                                    sz_utf8_seek_serial, backend.seek,   //
                                    sz_utf8_decode_serial, backend.decode, inputs);

    // Fold in the moved large-buffer count agreement (serial == dispatched == C++ wrapper == known total).
    test_utf8_runes_large_count();
}

#pragma endregion // Drivers
