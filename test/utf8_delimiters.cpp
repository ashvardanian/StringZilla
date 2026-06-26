/**
 *  @brief  UTF-8 delimiter (punctuation/symbol/separator/whitespace) scanning tests.
 *  @file scripts/test_utf8_delimiters.cpp
 *  @author Ash Vardanian
 *  @date June 26, 2026
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

/** @brief Function-pointer type of a per-ISA `sz_find_delimiters_utf8_<isa>` backend. */
typedef sz_cptr_t (*sz_find_delimiter_utf8_t)(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_encode` (silently skips invalid runes). */
static void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_encode(codepoint, bytes);
    if (length == sz_rune_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
}

/**
 *  @brief Builds a random, well-formed UTF-8 string whose codepoints span all four byte-widths and
 *         every 1->2->3->4 transition, mixing in both delimiter and non-delimiter codepoints so the
 *         scan kernels hit their mixed-width paths and resolve a match at every alignment.
 *
 *  @param target_codepoints  How many codepoints to emit.
 *  @param rng                Random source; drawing through it keeps `SZ_TESTS_SEED` reproducible.
 *  @return The encoded UTF-8 bytes.
 */
static std::string random_valid_utf8_(std::size_t target_codepoints, std::mt19937 &rng) {
    // Disjoint ranges, one per byte-width, chosen to avoid surrogates and noncharacters. The spans
    // straddle delimiter blocks (ASCII punctuation/space, Latin-1 symbols, general punctuation, etc.)
    // and letters alike, so a scan finds delimiters interspersed with ordinary text.
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
        // Retry until one valid codepoint is actually appended, so the emitted count is exact.
        while (text.size() == before) append_codepoint_(text, codepoint(rng));
    }
    return text;
}

/**
 *  @brief Runs one UTF-8 delimiter-scan backend over the known-answer anchor and asserts the produced
 *         match pointer and the matched delimiter byte-length agree with the expected values.
 *
 *  Mirrors `check_sha256_unit_` in `test_hash.cpp`: the caller drives it once per backend (dispatched,
 *  serial, and each natively-compiled kernel), so a wrong constant shared by the serial-vs-SIMD agreement
 *  tests is still caught against an external ground truth.
 *
 *  @param find_delimiter     Delimiter-scan kernel under test.
 *  @param text               Anchor text to scan.
 *  @param length             Byte length of @p text.
 *  @param expected_offset    Expected byte offset of the first delimiter, or `length` if none.
 *  @param expected_length    Expected byte length of the matched delimiter codepoint (0 if none).
 */
static void check_utf8_delimiters_unit_(sz_find_delimiter_utf8_t find_delimiter, //
                                        sz_cptr_t text, sz_size_t length,        //
                                        sz_size_t expected_offset, sz_size_t expected_length) {

    sz_size_t matched_length = 12345u; // Poison value: the kernel must overwrite it.
    sz_cptr_t const match = find_delimiter(text, length, &matched_length);
    if (expected_length == 0u) {
        assert(match == SZ_NULL_CHAR); // No delimiter present
        assert(matched_length == 0u);
    }
    else {
        assert(match == text + expected_offset);
        assert(matched_length == expected_length);
    }
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer unit tests for the UTF-8 delimiter scanner on simple, hand-verifiable inputs.
 *
 *  Exercises the scanner through the dispatched C API (automatic kernel resolution) and through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), so a regression
 *  that the serial-vs-SIMD agreement tests would miss - because both share a wrong constant - is still
 *  caught against an external ground truth.
 */
void test_utf8_delimiters_unit() {
    std::printf("  - testing UTF-8 delimiter known-answer vectors...\n");

    // "abc def" - the first delimiter is the ASCII space at byte 3, a length-1 codepoint.
    char const ascii_space[] = "abc def";
    // "hello" - all letters, no delimiter at all.
    char const no_delimiter[] = "hello";
    // "ab," - the first delimiter is the ASCII comma at byte 2, a length-1 codepoint.
    char const ascii_comma[] = "ab,";
    // "ab\xC2\xA0" - U+00A0 NO-BREAK SPACE (Zs) is a 2-byte delimiter starting at byte 2.
    char const nbsp[] = "ab\xC2\xA0";
    // "ab\xE2\x80\x94" - U+2014 EM DASH (Pd) is a 3-byte delimiter starting at byte 2.
    char const em_dash[] = "ab\xE2\x80\x94";
    // "ab\xF0\x9F\x98\x80" - U+1F600 GRINNING FACE (So) is a 4-byte delimiter starting at byte 2.
    char const emoji[] = "ab\xF0\x9F\x98\x80";
    // "a\xC3\x9F\xE4\xB8\xAD" - `a` + U+00DF + U+4E2D, all letters: no delimiter, 6 bytes.
    char const mixed_letters[] = "a\xC3\x9F\xE4\xB8\xAD";

    struct {
        sz_find_delimiter_utf8_t find;
        char const *name;
    } const backends[] = {
        {sz_find_delimiter_utf8, "dispatched"},       {sz_find_delimiters_utf8_serial, "serial"},
#if SZ_USE_HASWELL
        {sz_find_delimiters_utf8_haswell, "haswell"},
#endif
#if SZ_USE_ICELAKE
        {sz_find_delimiters_utf8_icelake, "icelake"},
#endif
#if SZ_USE_NEON
        {sz_find_delimiters_utf8_neon, "neon"},
#endif
    };

    for (auto const &backend : backends) {
        sz_unused_(backend.name);
        check_utf8_delimiters_unit_(backend.find, ascii_space, (sz_size_t)(sizeof(ascii_space) - 1), 3u, 1u);
        check_utf8_delimiters_unit_(backend.find, no_delimiter, (sz_size_t)(sizeof(no_delimiter) - 1), 0u, 0u);
        check_utf8_delimiters_unit_(backend.find, ascii_comma, (sz_size_t)(sizeof(ascii_comma) - 1), 2u, 1u);
        check_utf8_delimiters_unit_(backend.find, nbsp, (sz_size_t)(sizeof(nbsp) - 1), 2u, 2u);
        check_utf8_delimiters_unit_(backend.find, em_dash, (sz_size_t)(sizeof(em_dash) - 1), 2u, 3u);
        check_utf8_delimiters_unit_(backend.find, emoji, (sz_size_t)(sizeof(emoji) - 1), 2u, 4u);
        check_utf8_delimiters_unit_(backend.find, mixed_letters, (sz_size_t)(sizeof(mixed_letters) - 1), 0u, 0u);
        check_utf8_delimiters_unit_(backend.find, "", 0u, 0u, 0u); // Empty input: nothing to scan
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief Cross-checks the serial UTF-8 delimiter scanner against a candidate SIMD backend on random,
 *         well-formed inputs: both the match offset and the matched delimiter byte-length must agree.
 *
 *  The known-answer anchors live in `test_utf8_delimiters_unit`; this is the serial-vs-ISA differential.
 *
 *  @param find_serial     Reference (serial) delimiter scanner.
 *  @param find_candidate  Candidate delimiter scanner to validate against the reference.
 *  @param inputs          Number of random inputs to fuzz, scaled by the global multiplier.
 */
static void test_utf8_delimiters_equivalence(sz_find_delimiter_utf8_t find_serial,    //
                                             sz_find_delimiter_utf8_t find_candidate, //
                                             sz_size_t inputs) {

    auto &rng = global_random_generator();

    auto check = [&](std::string const &text) {
        sz_cptr_t const data = text.data();
        sz_size_t const length = (sz_size_t)text.size();

        sz_size_t serial_length = 12345u, candidate_length = 67890u;
        sz_cptr_t const serial_match = find_serial(data, length, &serial_length);
        sz_cptr_t const candidate_match = find_candidate(data, length, &candidate_length);

        // The match pointer and the matched delimiter byte-length must agree between serial and candidate.
        assert(candidate_match == serial_match && "Mismatch in delimiter offset");
        if (serial_match) assert(candidate_length == serial_length && "Mismatch in delimiter length");
        else assert(candidate_length == 0u && serial_length == 0u);
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

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Feeds malformed / invalid UTF-8 through one backend's delimiter scanner, asserting no crash,
 *         in-bounds output, and a match pointer that never escapes the input.
 *
 *  The scan is bounds-safe on arbitrary bytes: a truncated trailing sequence is never over-read, and any
 *  byte that does not begin a well-formed codepoint is skipped and never reported. Each call must return
 *  either NULL or a pointer inside `[input, input + length)`, with a matched length that stays within the
 *  remaining bytes.
 *
 *  @param find_delimiter  Delimiter scanner under test.
 *  @param random_inputs   Number of random garbage buffers to fuzz on top of the exhaustive byte sweeps.
 */
static void check_utf8_delimiters_safety_(sz_find_delimiter_utf8_t find_delimiter,
                                          std::size_t random_inputs = scale_iterations(10000)) {

    std::size_t const max_input_length = 70;

    auto check = [&](char const *input, std::size_t input_length) {
        sz_size_t matched_length = 12345u;
        sz_cptr_t const match = find_delimiter(input, (sz_size_t)input_length, &matched_length);
        if (!match) {
            assert(matched_length == 0u && "No-match scan must report a zero matched length");
            return;
        }
        // The match must land inside the input, and its codepoint must fit within the remaining bytes.
        assert(match >= input && match < input + input_length && "Delimiter scan escaped the input");
        assert(matched_length >= 1u && matched_length <= 4u && "Delimiter matched an impossible byte length");
        assert((std::size_t)(match - input) + matched_length <= input_length &&
               "Delimiter match span outside the input");
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

/** @brief Drive the malformed-input safety probe through serial, dispatched, and every native backend. */
void test_utf8_delimiters_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 delimiter kernels...\n");

    // Serial baseline and the dispatched (automatic kernel resolution) entry points face the same contract.
    check_utf8_delimiters_safety_(sz_find_delimiters_utf8_serial);
    check_utf8_delimiters_safety_(sz_find_delimiter_utf8);

#if SZ_USE_HASWELL
    check_utf8_delimiters_safety_(sz_find_delimiters_utf8_haswell);
#endif
#if SZ_USE_ICELAKE
    check_utf8_delimiters_safety_(sz_find_delimiters_utf8_icelake);
#endif
#if SZ_USE_NEON
    check_utf8_delimiters_safety_(sz_find_delimiters_utf8_neon);
#endif

    std::printf("    malformed-input safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD UTF-8 delimiter-scan differential across every backend compiled on
 *         this target.
 */
void test_utf8_delimiters_all() {
    sz_size_t const inputs = (sz_size_t)scale_iterations(200);
    sz_unused_(inputs);

#if SZ_USE_HASWELL
    test_utf8_delimiters_equivalence(sz_find_delimiters_utf8_serial, sz_find_delimiters_utf8_haswell, inputs);
#endif
#if SZ_USE_ICELAKE
    test_utf8_delimiters_equivalence(sz_find_delimiters_utf8_serial, sz_find_delimiters_utf8_icelake, inputs);
#endif
#if SZ_USE_NEON
    test_utf8_delimiters_equivalence(sz_find_delimiters_utf8_serial, sz_find_delimiters_utf8_neon, inputs);
#endif
}

#pragma endregion // Drivers
