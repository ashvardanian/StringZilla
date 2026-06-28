/**
 *  @brief  Uncased UTF-8 case-folding equivalence/fuzzing and uncased substring search tests.
 *  @file   scripts/test_uncased.cpp
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

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`
using sz::literals::operator""_bs; // for `sz::byteset`

#if SZ_IS_CPP17_
using namespace std::literals; // for ""sv
#endif

#pragma region Helpers

/** @brief Runs one uncased-find backend over a known case and asserts the match offset and length. */
static void check_uncased_find_unit_(                                               //
    sz_utf8_uncased_search_t find, char const *haystack, sz_size_t haystack_length, //
    char const *needle, sz_size_t needle_length,                                    //
    sz_size_t expected_offset, sz_size_t expected_length) {
    sz_utf8_uncased_needle_metadata_t metadata = {};
    sz_size_t matched_length = 0;
    sz_cptr_t match = find(haystack, haystack_length, needle, needle_length, &metadata, &matched_length);
    assert(match != SZ_NULL_CHAR);
    assert((sz_size_t)(match - haystack) == expected_offset);
    assert(matched_length == expected_length);
}

/** @brief Runs one uncased-fold backend over `input` and asserts the folded bytes equal `expected`. */
static void check_uncased_fold_unit_( //
    sz_utf8_uncased_fold_t fold, char const *input, sz_size_t input_length, char const *expected) {
    char produced[64];
    sz_size_t produced_length = fold(input, input_length, produced);
    sz_size_t const expected_length = (sz_size_t)std::strlen(expected);
    assert(produced_length == expected_length);
    assert(std::memcmp(produced, expected, expected_length) == 0);
}

/** @brief Prints one labeled hex dump line to `stderr`; used by the adversarial UTF-8 case tests below. */
static void print_uncased_test_bytes_(char const *label, char const *bytes, std::size_t length) {
    std::fprintf(stderr, "  %s (%zu bytes): ", label, length);
    for (std::size_t i = 0; i < length; ++i) std::fprintf(stderr, "%02X ", (unsigned char)bytes[i]);
    std::fprintf(stderr, "\n");
}

/**
 *  @brief Independent ground-truth uncased search: `fold(needle)` as a contiguous run of `fold(haystack)`.
 *
 *  Folds the haystack codepoint by codepoint into fixed-size arrays, recording each folded rune's source
 *  byte span, then folds the needle and slides it over the folded stream. The earliest run wins; the
 *  reported offset and length snap to the source codepoint boundaries, so a match that begins or ends
 *  mid-expansion ("sss" inside "ssss" from "ßß") is reported in original haystack bytes. Depends on no
 *  production kernel, so it catches bugs the base-vs-SIMD differential cannot: those where every backend
 *  agrees yet all of them are wrong. Mirrors the Python `_reference_case_insensitive_find`.
 */
static sz_cptr_t reference_uncased_find_(char const *haystack, std::size_t haystack_length, //
                                         char const *needle, std::size_t needle_length, std::size_t *match_length) {

    // The adversarial harnesses below cap haystacks at 128 bytes and needles at 32; folding can triple
    // the rune count, so these fixed windows stay comfortably ahead of the longest input.
    sz_rune_t needle_folded[128];
    std::size_t needle_folded_count = 0;
    for (char const *cursor = needle, *end = needle + needle_length; cursor < end;) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(cursor, end, &rune);
        if (rune_length == sz_rune_invalid_k) { // Malformed byte is its own 1-byte maximal subpart, copied unfolded
            assert(needle_folded_count < 128 && "reference needle buffer overflow");
            needle_folded[needle_folded_count++] = (sz_u8_t)*cursor;
            ++cursor;
            continue;
        }
        sz_rune_t folded[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t index = 0; index < folded_count; ++index) {
            assert(needle_folded_count < 128 && "reference needle buffer overflow");
            needle_folded[needle_folded_count++] = folded[index];
        }
        cursor += rune_length;
    }
    if (needle_folded_count == 0) {
        *match_length = 0;
        return haystack;
    }

    sz_rune_t haystack_folded[512];
    sz_cptr_t source_begin[512], source_end[512];
    std::size_t haystack_folded_count = 0;
    for (char const *cursor = haystack, *end = haystack + haystack_length; cursor < end;) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(cursor, end, &rune);
        if (rune_length == sz_rune_invalid_k) { // Malformed byte is its own 1-byte maximal subpart, copied unfolded
            assert(haystack_folded_count < 512 && "reference haystack buffer overflow");
            haystack_folded[haystack_folded_count] = (sz_u8_t)*cursor;
            source_begin[haystack_folded_count] = cursor;
            source_end[haystack_folded_count] = cursor + 1;
            ++haystack_folded_count;
            ++cursor;
            continue;
        }
        sz_rune_t folded[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t index = 0; index < folded_count; ++index) {
            assert(haystack_folded_count < 512 && "reference haystack buffer overflow");
            haystack_folded[haystack_folded_count] = folded[index];
            source_begin[haystack_folded_count] = cursor;
            source_end[haystack_folded_count] = cursor + rune_length;
            ++haystack_folded_count;
        }
        cursor += rune_length;
    }

    for (std::size_t start = 0; start + needle_folded_count <= haystack_folded_count; ++start) {
        bool equal = true;
        for (std::size_t index = 0; index < needle_folded_count; ++index)
            if (haystack_folded[start + index] != needle_folded[index]) {
                equal = false;
                break;
            }
        if (!equal) continue;
        *match_length = (std::size_t)(source_end[start + needle_folded_count - 1] - source_begin[start]);
        return source_begin[start];
    }
    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Runs one uncased find query through two backends and the reference, demanding all three agree.
 *
 *  Both the match pointer offset and the matched length must agree across the base backend, the SIMD
 *  backend, and the independent fold-subset reference, including the not-found case. Because the SIMD
 *  kernels delegate short needles to the serial path, a base-vs-SIMD-only check is blind to a bug both
 *  share; the reference leg closes that gap. On mismatch prints the needle and haystack as hex.
 */
static void check_uncased_find_three_way_(                                  //
    sz_utf8_uncased_search_t find_base, sz_utf8_uncased_search_t find_simd, //
    char const *haystack, std::size_t haystack_length,                      //
    char const *needle, std::size_t needle_length, char const *test_name) {

    sz_utf8_uncased_needle_metadata_t base_metadata = {}, simd_metadata = {};
    sz_size_t base_matched = 0, simd_matched = 0;
    sz_cptr_t base_result = find_base(haystack, haystack_length, needle, needle_length, &base_metadata, &base_matched);
    sz_cptr_t simd_result = find_simd(haystack, haystack_length, needle, needle_length, &simd_metadata, &simd_matched);

    std::size_t reference_matched = 0;
    sz_cptr_t reference_result = reference_uncased_find_(haystack, haystack_length, needle, needle_length,
                                                         &reference_matched);

    bool const base_matches_simd = base_result == simd_result &&
                                   (base_result == SZ_NULL_CHAR || base_matched == simd_matched);
    bool const base_matches_reference = base_result == reference_result &&
                                        (base_result == SZ_NULL_CHAR || base_matched == reference_matched);
    bool const simd_matches_reference = simd_result == reference_result &&
                                        (simd_result == SZ_NULL_CHAR || simd_matched == reference_matched);
    if (base_matches_simd && base_matches_reference && simd_matches_reference) return;

    long const base_offset = base_result ? (long)(base_result - haystack) : -1L;
    long const simd_offset = simd_result ? (long)(simd_result - haystack) : -1L;
    long const reference_offset = reference_result ? (long)(reference_result - haystack) : -1L;
    std::fprintf(
        stderr, "%s FAIL: base offset=%ld len=%zu | simd offset=%ld len=%zu kernel=%u | reference offset=%ld len=%zu\n",
        test_name, base_offset, (std::size_t)base_matched, simd_offset, (std::size_t)simd_matched,
        simd_metadata.kernel_id, reference_offset, (std::size_t)reference_matched);
    print_uncased_test_bytes_("needle  ", needle, needle_length);
    print_uncased_test_bytes_("haystack", haystack, haystack_length);
    assert(base_matches_reference && "Uncased find base backend disagrees with the reference");
    assert(simd_matches_reference && "Uncased find SIMD backend disagrees with the reference");
    assert(base_matches_simd && "Uncased find backends disagree with each other");
}

/**
 *  @brief Fuzz tests uncased UTF-8 substring search with controlled haystack sizes.
 *
 *  Uses two verification modes:
 *  - Exhaustive (max_needles_per_haystack == 0): Tests ALL N*(N+1)/2 substrings of each folded haystack
 *  - Sampled (max_needles_per_haystack > 0): Tests up to that many random substrings per haystack
 *
 *  Algorithm:
 *  1. Generate random haystack of ~haystack_length runes from character pool
 *  2. Case-fold the haystack
 *  3. Extract needles from folded haystack (guarantees needle exists uncasedly)
 *  4. Search needle in ORIGINAL (unfolded) haystack with both serial and SIMD
 *  5. Both must return identical positions
 *
 *  @param haystack_length Target number of bytes in each haystack
 *  @param max_needles_per_haystack  0 = exhaustive, >0 = sample this many per haystack
 *  @param total_queries Total needle searches to perform across all haystacks
 */
static void test_uncased_find_fuzz(sz_utf8_uncased_search_t find_serial, sz_utf8_uncased_search_t find_simd,
                                   sz_utf8_uncased_fold_t uncased_fold, sz_utf8_seek_t utf8_seek,
                                   sz_utf8_count_t utf8_count, std::size_t haystack_length,
                                   std::size_t max_needles_per_haystack, std::size_t total_queries) {

    char const *mode = max_needles_per_haystack == 0 ? "exhaustive" : "sampled";
    std::printf("    - fuzz testing (%s, haystack_len=%zu, queries=%zu)...\n", mode, haystack_length, total_queries);

    auto &rng = global_random_generator();

    // Character pool with normal + weird Unicode characters from safety profiles
    char const *char_pool[] = {
        // Normal ASCII (individual characters for mixing)
        // clang-format off
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "0", "1", "2", "3", " ", ".", ",", "!", "?",
        // clang-format on
        // ASCII words for realistic text patterns
        "Hello",
        "World",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",

        // Folded representations of multi-byte danger chars (ASCII equivalents)
        // These are what the complex characters fold TO - essential for testing both directions
        "ss",  // ß (U+00DF, C3 9F) and ẞ (U+1E9E, E1 BA 9E) fold to this
        "fi",  // ﬁ (U+FB01, EF AC 81) folds to this
        "fl",  // ﬂ (U+FB02, EF AC 82) folds to this
        "ff",  // ﬀ (U+FB00, EF AC 80) folds to this
        "ffi", // ﬃ (U+FB03, EF AC 83) folds to this
        "ffl", // ﬄ (U+FB04, EF AC 84) folds to this
        "st",  // ﬅ (U+FB05), ﬆ (U+FB06) fold to this
        "k",   // K (U+212A, E2 84 AA) Kelvin folds to this
        "s",   // ſ (U+017F, C5 BF) Long S folds to this

        // Latin-1/Extended (Western European)
        "\xC3\x9F", // 'ß' (U+00DF, C3 9F) - Latin Small Letter Sharp S (folds to ss)
        "\xC3\xB6", // 'ö' (U+00F6, C3 B6) - Latin Small Letter O with Diaeresis
        "\xC3\x96", // 'Ö' (U+00D6, C3 96) - Latin Capital Letter O with Diaeresis
        "\xC3\xBC", // 'ü' (U+00FC, C3 BC) - Latin Small Letter U with Diaeresis
        "\xC3\x9C", // 'Ü' (U+00DC, C3 9C) - Latin Capital Letter U with Diaeresis
        "\xC3\xA4", // 'ä' (U+00E4, C3 A4) - Latin Small Letter A with Diaeresis
        "\xC3\x84", // 'Ä' (U+00C4, C3 84) - Latin Capital Letter A with Diaeresis
        "\xC3\xA9", // 'é' (U+00E9, C3 A9) - Latin Small Letter E with Acute
        "\xC3\x89", // 'É' (U+00C9, C3 89) - Latin Capital Letter E with Acute
        "\xC3\xA0", // 'à' (U+00E0, C3 A0) - Latin Small Letter A with Grave
        "\xC3\x80", // 'À' (U+00C0, C3 80) - Latin Capital Letter A with Grave
        "\xC3\xB1", // 'ñ' (U+00F1, C3 B1) - Latin Small Letter N with Tilde
        "\xC3\x91", // 'Ñ' (U+00D1, C3 91) - Latin Capital Letter N with Tilde
        "\xC2\xAA", // 'ª' (U+00AA, C2 AA) - Feminine Ordinal Indicator (caseless)
        "\xC2\xBA", // 'º' (U+00BA, C2 BA) - Masculine Ordinal Indicator (caseless)
        "\xC2\xB5", // 'µ' (U+00B5, C2 B5) - Micro Sign (folds to Greek mu)
        "\xC3\x85", // 'Å' (U+00C5, C3 85) - Latin Capital Letter A with Ring Above
        "\xC3\xA5", // 'å' (U+00E5, C3 A5) - Latin Small Letter A with Ring Above
        "\xC5\xBF", // 'ſ' (U+017F, C5 BF) - Latin Small Letter Long S (folds to regular s)

        // Kelvin sign and Angstrom sign
        "\xE2\x84\xAA", // 'K' (U+212A, E2 84 AA) - Kelvin Sign (folds to ASCII k)
        "\xE2\x84\xAB", // 'Å' (U+212B, E2 84 AB) - Angstrom Sign (folds to Latin-1 a-ring)

        // Turkish
        "\xC4\xB0", // 'İ' (U+0130, C4 B0) - Latin Capital Letter I with Dot Above
        "\xC4\xB1", // 'ı' (U+0131, C4 B1) - Latin Small Letter Dotless I

        // Cyrillic (Russian, Ukrainian)
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", // "привет" (Cyrillic privet) - Russian Hello
        "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0", // "Москва" (Cyrillic Moskva) - Moscow
        "\xD0\xB0",                                         // 'а' (U+0430, D0 B0) - Cyrillic Small Letter A
        "\xD0\x90",                                         // 'А' (U+0410, D0 90) - Cyrillic Capital Letter A
        "\xD0\xB1",                                         // 'б' (U+0431, D0 B1) - Cyrillic Small Letter Be
        "\xD0\x91",                                         // 'Б' (U+0411, D0 91) - Cyrillic Capital Letter Be
        "\xD0\xB2",                                         // 'в' (U+0432, D0 B2) - Cyrillic Small Letter Ve
        "\xD0\x92",                                         // 'В' (U+0412, D0 92) - Cyrillic Capital Letter Ve

        // Greek (including final sigma)
        "\xCE\xB1",                         // 'α' (U+03B1, CE B1) - Greek Small Letter Alpha
        "\xCE\x91",                         // 'Α' (U+0391, CE 91) - Greek Capital Letter Alpha
        "\xCE\xB2",                         // 'β' (U+03B2, CE B2) - Greek Small Letter Beta
        "\xCE\x92",                         // 'Β' (U+0392, CE 92) - Greek Capital Letter Beta
        "\xCF\x83",                         // 'σ' (U+03C3, CF 83) - Greek Small Letter Sigma
        "\xCE\xA3",                         // 'Σ' (U+03A3, CE A3) - Greek Capital Letter Sigma
        "\xCF\x82",                         // 'ς' (U+03C2, CF 82) - Greek Small Letter Final Sigma
        "\xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC", // "κόσμ" (Greek kosm) - World

        // Greek symbol forms (fold to normal counterparts - danger zone chars)
        "\xCF\x90", // 'ϐ' (U+03D0, CF 90) - Greek Beta Symbol -> β
        "\xCF\x91", // 'ϑ' (U+03D1, CF 91) - Greek Theta Symbol -> θ
        "\xCF\x95", // 'ϕ' (U+03D5, CF 95) - Greek Phi Symbol -> φ
        "\xCF\x96", // 'ϖ' (U+03D6, CF 96) - Greek Pi Symbol -> π
        "\xCF\xB0", // 'ϰ' (U+03F0, CF B0) - Greek Kappa Symbol -> κ
        "\xCF\xB1", // 'ϱ' (U+03F1, CF B1) - Greek Rho Symbol -> ρ
        "\xCF\xB5", // 'ϵ' (U+03F5, CF B5) - Greek Lunate Epsilon Symbol -> ε

        // Greek with dialytika + tonos (expand to base + combining marks)
        "\xCE\x90", // 'ΐ' (U+0390, CE 90) - Greek Small Letter Iota with Dialytika and Tonos
        "\xCE\xB0", // 'ΰ' (U+03B0, CE B0) - Greek Small Letter Upsilon with Dialytika and Tonos

        // Armenian
        "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE", // "բարև" (Armenian barev) - Hello
        "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E", // "ԲԱՐԵՒ" (Armenian BAREV) - HELLO
        "\xD5\xA5",                                 // 'ե' (U+0565, D5 A5) - Armenian Small Letter Ech
        "\xD6\x87",                                 // 'և' (U+0587, D6 87) - Armenian Small Ligature Ech Yiwn

        // Vietnamese/Latin Extended Additional
        "\xE1\xBB\x87", // 'ệ' (U+1EC7, E1 BB 87) - Latin Small Letter E with Circumflex and Dot Below
        "\xE1\xBB\x86", // 'Ệ' (U+1EC6, E1 BB 86) - Latin Capital Letter E with Circumflex and Dot Below
        "\xE1\xBA\xA1", // 'ạ' (U+1EA1, E1 BA A1) - Latin Small Letter A with Dot Below
        "\xE1\xBA\xA0", // 'Ạ' (U+1EA0, E1 BA A0) - Latin Capital Letter A with Dot Below
        "\xC4\x90",     // 'Đ' (U+0110, C4 90) - Latin Capital Letter D with Stroke
        "\xC4\x91",     // 'đ' (U+0111, C4 91) - Latin Small Letter D with Stroke
        "\xC6\xA0",     // 'Ơ' (U+01A0, C6 A0) - Latin Capital Letter O with Horn
        "\xC6\xA1",     // 'ơ' (U+01A1, C6 A1) - Latin Small Letter O with Horn

        // Latin ligatures (one-to-many folding)
        "\xEF\xAC\x80", // 'ﬀ' (U+FB00, EF AC 80) - Latin Small Ligature ff
        "\xEF\xAC\x81", // 'ﬁ' (U+FB01, EF AC 81) - Latin Small Ligature fi
        "\xEF\xAC\x82", // 'ﬂ' (U+FB02, EF AC 82) - Latin Small Ligature fl
        "\xEF\xAC\x83", // 'ﬃ' (U+FB03, EF AC 83) - Latin Small Ligature ffi
        "\xEF\xAC\x84", // 'ﬄ' (U+FB04, EF AC 84) - Latin Small Ligature ffl
        "\xEF\xAC\x85", // 'ﬅ' (U+FB05, EF AC 85) - Latin Small Ligature Long S T
        "\xEF\xAC\x86", // 'ﬆ' (U+FB06, EF AC 86) - Latin Small Ligature St

        // Armenian ligatures (one-to-many folding)
        "\xEF\xAC\x93", // 'ﬓ' (U+FB13, EF AC 93) - Armenian Small Ligature Men Now
        "\xEF\xAC\x94", // 'ﬔ' (U+FB14, EF AC 94) - Armenian Small Ligature Men Ech
        "\xEF\xAC\x95", // 'ﬕ' (U+FB15, EF AC 95) - Armenian Small Ligature Men Ini
        "\xEF\xAC\x96", // 'ﬖ' (U+FB16, EF AC 96) - Armenian Small Ligature Vew Now
        "\xEF\xAC\x97", // 'ﬗ' (U+FB17, EF AC 97) - Armenian Small Ligature Men Xeh

        // One-to-many expansions (U+1E96-1E9A)
        "\xE1\xBA\x96", // 'ẖ' (U+1E96, E1 BA 96) - Latin Small Letter H with Line Below
        "\xE1\xBA\x97", // 'ẗ' (U+1E97, E1 BA 97) - Latin Small Letter T with Diaeresis
        "\xE1\xBA\x98", // 'ẘ' (U+1E98, E1 BA 98) - Latin Small Letter W with Ring Above
        "\xE1\xBA\x99", // 'ẙ' (U+1E99, E1 BA 99) - Latin Small Letter Y with Ring Above
        "\xE1\xBA\x9A", // 'ẚ' (U+1E9A, E1 BA 9A) - Latin Small Letter A with Right Half Ring

        // Capital Eszett (U+1E9E) - folds to ss
        "\xE1\xBA\x9E", // 'ẞ' (U+1E9E, E1 BA 9E) - Latin Capital Letter Sharp S

        // Lowercase Sharp S (U+00DF) - folds to ss (critical danger char!)
        "\xC3\x9F", // 'ß' (U+00DF, C3 9F) - Latin Small Letter Sharp S -> ss

        // Long S with dot above (U+1E9B) - folds to 'ṡ'
        "\xE1\xBA\x9B", // 'ẛ' (U+1E9B, E1 BA 9B) - Latin Small Letter Long S with Dot Above

        // Ohm sign - folds to Greek omega (danger char!)
        "\xE2\x84\xA6", // 'Ω' (U+2126, E2 84 A6) - Ohm Sign -> ω (U+03C9)

        // Afrikaans n-apostrophe (U+0149) -> 'n
        "\xC5\x89", // 'ŉ' (U+0149, C5 89) - Latin Small Letter N Preceded by Apostrophe

        // J-caron (U+01F0) -> j + combining caron
        "\xC7\xB0", // 'ǰ' (U+01F0, C7 B0) - Latin Small Letter J with Caron

        // Modifier letter apostrophe (U+02BC) - context for n
        "\xCA\xBC", // 'ʼ' (U+02BC, CA BC) - Modifier Letter Apostrophe

        // Modifier letter right half ring (U+02BE) - context for a
        "\xCA\xBE", // 'ʾ' (U+02BE, CA BE) - Modifier Letter Right Half Ring

        // Georgian (exercises the Georgian SIMD fold kernel across its three case blocks)
        "\xE1\x82\xA0", // 'Ⴀ' (U+10A0, E1 82 A0) - Georgian Capital Letter An (Asomtavruli, folds to Mkhedruli)
        "\xE1\x82\xB0", // 'Ⴐ' (U+10B0, E1 82 B0) - Georgian Capital Letter Ras (Asomtavruli)
        "\xE1\x83\x90", // 'ა' (U+10D0, E1 83 90) - Georgian Letter An (Mkhedruli, caseless lowercase)
        "\xE1\x83\x91", // 'ბ' (U+10D1, E1 83 91) - Georgian Letter Ban (Mkhedruli)
        "\xE1\xB2\x90", // 'Ა' (U+1C90, E1 B2 90) - Georgian Mtavruli Capital Letter An (folds to Mkhedruli)
        "\xE1\xB2\xBF", // 'Ჿ' (U+1CBF, E1 B2 BF) - Georgian Mtavruli Capital Letter Caucasian Cudzin

        // Caseless scripts (for coverage)
        "\xE4\xB8\xAD\xE6\x96\x87", // "中文" (CJK zhongwen) - Caseless
        "\xE3\x81\x82\xE3\x81\x84", // "あい" (Hiragana ai) - Caseless
        "\xF0\x9F\x98\x80",         // '😀' (U+1F600, F0 9F 98 80) - Grinning Face
    };
    std::size_t const pool_size = sizeof(char_pool) / sizeof(char_pool[0]);
    std::uniform_int_distribution<std::size_t> pool_dist(0, pool_size - 1);

    std::size_t queries_remaining = total_queries;
    std::size_t haystacks_tested = 0;
    std::size_t total_passed = 0;

    // Pre-allocate buffers - reusable between iterations
    std::string haystack;
    std::vector<char> haystack_folded;
    haystack.reserve(haystack_length);

    while (queries_remaining > 0) {
        // 1. Generate random haystack of ~haystack_length bytes
        haystack.clear();
        while (haystack.size() < haystack_length) haystack += char_pool[pool_dist(rng)];

        // 2. Case-fold the haystack - expands up to 3x
        haystack_folded.resize(haystack.size() * 3);
        haystack_folded.resize(uncased_fold(haystack.data(), haystack.size(), haystack_folded.data()));
        if (haystack_folded.empty()) continue;

        // 3. Count runes in folded haystack
        sz_size_t runes_in_folded_haystack = utf8_count(haystack_folded.data(), haystack_folded.size());
        if (runes_in_folded_haystack < 2) continue;

        // 4. Calculate needles for this haystack
        std::size_t const needles_in_this_haystack =
            max_needles_per_haystack == 0 ? ((runes_in_folded_haystack * (runes_in_folded_haystack + 1)) / 2)
                                          : std::min(max_needles_per_haystack, queries_remaining);

        // Helper to extract needle from folded haystack and test both implementations
        auto test_needle = [&](sz_size_t start, sz_size_t rune_count) -> bool {
            sz_cptr_t needle_start = (start == 0) ? haystack_folded.data()
                                                  : utf8_seek(haystack_folded.data(), haystack_folded.size(), start);
            sz_cptr_t needle_end = ((start + rune_count) == runes_in_folded_haystack)
                                       ? haystack_folded.data() + haystack_folded.size()
                                       : utf8_seek(haystack_folded.data(), haystack_folded.size(), start + rune_count);
            if (!needle_start || !needle_end || needle_end <= needle_start) return false;

            sz_size_t needle_bytes = needle_end - needle_start;
            sz_size_t serial_matched = 0, simd_matched = 0;
            sz_utf8_uncased_needle_metadata_t serial_meta = {}, simd_meta = {};

            sz_cptr_t serial_result = find_serial(haystack.data(), haystack.size(), needle_start, needle_bytes,
                                                  &serial_meta, &serial_matched);
            sz_cptr_t simd_result = find_simd(haystack.data(), haystack.size(), needle_start, needle_bytes, &simd_meta,
                                              &simd_matched);

            if (serial_result != simd_result || serial_matched != simd_matched) {
                std::fprintf(stderr, "FUZZ FAIL haystack=%zu start=%zu rune_count=%zu\n", haystacks_tested, start,
                             rune_count);
                std::fprintf(stderr, "  Haystack len=%zu, needle len=%zu\n", haystack.size(), needle_bytes);

                std::fprintf(stderr, "  Needle bytes: ");
                for (sz_size_t j = 0; j < needle_bytes && j < 50; ++j)
                    std::fprintf(stderr, "%02X ", (unsigned char)needle_start[j]);
                std::fprintf(stderr, "\n");

                sz_size_t serial_off = serial_result ? (sz_size_t)(serial_result - haystack.data()) : SZ_SIZE_MAX;
                sz_size_t simd_off = simd_result ? (sz_size_t)(simd_result - haystack.data()) : SZ_SIZE_MAX;
                std::fprintf(stderr, "  Serial: offset=%zu, len=%zu\n",
                             serial_off == SZ_SIZE_MAX ? (sz_size_t)-1 : serial_off, serial_matched);
                std::fprintf(stderr, "  SIMD:   offset=%zu, len=%zu\n",
                             simd_off == SZ_SIZE_MAX ? (sz_size_t)-1 : simd_off, simd_matched);
                std::fprintf(stderr, "  SIMD metadata: kernel=%u offset_in_unfolded=%zu, length_in_unfolded=%zu\n",
                             simd_meta.kernel_id, simd_meta.offset_in_unfolded, simd_meta.length_in_unfolded);
                // Print haystack bytes around the match
                std::fprintf(stderr, "  Haystack bytes (offset %zu-20 to offset %zu+20):\n    ",
                             serial_off == SZ_SIZE_MAX ? 0 : serial_off, serial_off == SZ_SIZE_MAX ? 0 : serial_off);
                sz_size_t print_start = (serial_off != SZ_SIZE_MAX && serial_off > 20) ? serial_off - 20 : 0;
                sz_size_t print_end = (serial_off != SZ_SIZE_MAX)
                                          ? std::min(serial_off + needle_bytes + 20, haystack.size())
                                          : std::min((sz_size_t)50, haystack.size());
                for (sz_size_t j = print_start; j < print_end; ++j)
                    std::fprintf(stderr, "%02X ", (unsigned char)haystack[j]);
                std::fprintf(stderr, "\n");
                assert(serial_result == simd_result && "Fuzz offset mismatch");
                assert(serial_matched == simd_matched && "Fuzz length mismatch");
            }
            return true;
        };

        // 5. Generate and test needles
        if (max_needles_per_haystack == 0) {
            // Exhaustive mode: every possible (start, length) pair from folded haystack
            for (sz_size_t start = 0; start < runes_in_folded_haystack && queries_remaining > 0; ++start) {
                for (sz_size_t rune_count = 1; rune_count <= runes_in_folded_haystack - start && queries_remaining > 0;
                     ++rune_count) {
                    if (test_needle(start, rune_count)) {
                        ++total_passed;
                        --queries_remaining;
                    }
                }
            }
        }
        else {
            // Sampled mode: random (start, length) pairs
            std::uniform_int_distribution<sz_size_t> start_dist(0, runes_in_folded_haystack - 1);
            for (std::size_t i = 0; i < needles_in_this_haystack && queries_remaining > 0; ++i) {
                sz_size_t start = start_dist(rng);
                std::uniform_int_distribution<sz_size_t> rune_count_dist(1, runes_in_folded_haystack - start);
                if (test_needle(start, rune_count_dist(rng))) {
                    ++total_passed;
                    --queries_remaining;
                }
            }
        }

        ++haystacks_tested;
    }

    std::printf("    passed %zu fuzz tests across %zu haystacks\n", total_passed, haystacks_tested);
}

/**
 *  @brief Differential adversarial test for uncased search over @b all fold preimages.
 *
 *  Random fuzzing rarely places a rare preimage (like 'ϴ' U+03F4 → 'θ') right where a SIMD
 *  danger-detection "alarm" crosses a chunk boundary - structured enumeration does. For every
 *  codepoint whose `sz_unicode_fold_codepoint_` output differs from identity, the folded output
 *  becomes the needle (bare, "x"-prefixed, "x"-suffixed) and the UTF-8 @b preimage hides in
 *  'y'-padded haystacks at offsets straddling the 64-byte SIMD chunk boundary. Haystacks are
 *  built both with and without the mirroring "x" context, so the not-found path is exercised
 *  with the same adversarial shapes.
 */
void test_uncased_find_preimages_fuzz(sz_utf8_uncased_search_t find_base, sz_utf8_uncased_search_t find_simd) {

    std::printf("  - testing uncased find against all fold preimages...\n");

    std::size_t const offsets[] = {0, 14, 15, 16, 17, 30, 31, 32, 33, 61, 62, 63, 64, 65};
    std::size_t const offsets_count = sizeof(offsets) / sizeof(offsets[0]);
    std::size_t const padding_length = 16;
    std::size_t preimages_tested = 0, cases_tested = 0;

    char needle[16];   // Longest folded form is 9 bytes, plus one ASCII context byte
    char haystack[96]; // Largest offset 65, plus context, plus a 4-byte preimage, plus padding

    for (sz_rune_t preimage = 1; preimage <= 0x10FFFF; ++preimage) {
        if (preimage >= 0xD800 && preimage <= 0xDFFF) continue; // Surrogates aren't valid UTF-8
        sz_rune_t folded_runes[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(preimage, folded_runes);
        if (folded_count == 1 && folded_runes[0] == preimage) continue; // Identity folds aren't adversarial

        sz_u8_t preimage_utf8[4];
        std::size_t const preimage_length = (std::size_t)sz_rune_encode(preimage, preimage_utf8);
        sz_u8_t folded_utf8[12];
        std::size_t folded_length = 0;
        for (sz_size_t i = 0; i < folded_count; ++i)
            folded_length += (std::size_t)sz_rune_encode(folded_runes[i], folded_utf8 + folded_length);
        ++preimages_tested;

        // Needle variants: 0 = bare folded form, 1 = "x"-prefixed, 2 = "x"-suffixed
        for (int variant = 0; variant < 3; ++variant) {
            std::size_t needle_length = 0;
            if (variant == 1) needle[needle_length++] = 'x';
            std::memcpy(needle + needle_length, folded_utf8, folded_length);
            needle_length += folded_length;
            if (variant == 2) needle[needle_length++] = 'x';

            // With the mirroring "x" context the needle must match through the fold expansion;
            // without it both backends must agree on the not-found result. The bare variant has
            // no context to mirror, so it runs once.
            for (int with_context = variant == 0 ? 1 : 0; with_context < 2; ++with_context) {
                for (std::size_t offset_index = 0; offset_index < offsets_count; ++offset_index) {
                    std::size_t haystack_length = 0;
                    while (haystack_length < offsets[offset_index]) haystack[haystack_length++] = 'y';
                    if (variant == 1 && with_context) haystack[haystack_length++] = 'x';
                    std::memcpy(haystack + haystack_length, preimage_utf8, preimage_length);
                    haystack_length += preimage_length;
                    if (variant == 2 && with_context) haystack[haystack_length++] = 'x';
                    for (std::size_t i = 0; i < padding_length; ++i) haystack[haystack_length++] = 'y';

                    check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                  needle_length, "preimage find");
                    ++cases_tested;
                }
            }
        }
    }
    std::printf("    passed %zu cases across %zu fold preimages\n", cases_tested, preimages_tested);
}

/**
 *  @brief Differential test for matches sitting at the very tail of the haystack, where the
 *         haystack span is wider or narrower than the folded needle window.
 *
 *  Expanding preimages ('ᾳ' U+1FB3 → "αι", 'ß' → "ss", the ﬁ/ﬀ/ﬃ ligatures, 'ŉ' U+0149) and
 *  shrinking ones ('K' Kelvin U+212A → "k", 'Å' Angstrom U+212B → "å") break the byte-for-byte
 *  relation between haystack and folded needle - exactly where SIMD tail danger-windows get cut
 *  short. The set is derived generatively: every preimage whose folded UTF-8 byte length differs
 *  from its own. Each lands within the last `needle_window` bytes (windows 4..16) of haystacks
 *  whose filler also sweeps the 64-byte SIMD chunk boundary.
 */
void test_uncased_find_tails_fuzz(sz_utf8_uncased_search_t find_base, sz_utf8_uncased_search_t find_simd) {

    std::printf("  - testing uncased find with expanding preimages at haystack tails...\n");

    struct expanding_preimage_t {
        sz_u8_t preimage_utf8[4];
        std::size_t preimage_length;
        sz_u8_t folded_utf8[12];
        std::size_t folded_length;
    };
    std::vector<expanding_preimage_t> expanding_preimages;

    for (sz_rune_t preimage = 1; preimage <= 0x10FFFF; ++preimage) {
        if (preimage >= 0xD800 && preimage <= 0xDFFF) continue; // Surrogates aren't valid UTF-8
        sz_rune_t folded_runes[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(preimage, folded_runes);
        if (folded_count == 1 && folded_runes[0] == preimage) continue; // Identity folds can't expand

        expanding_preimage_t entry;
        entry.preimage_length = (std::size_t)sz_rune_encode(preimage, entry.preimage_utf8);
        entry.folded_length = 0;
        for (sz_size_t i = 0; i < folded_count; ++i)
            entry.folded_length += (std::size_t)sz_rune_encode(folded_runes[i],
                                                               entry.folded_utf8 + entry.folded_length);
        if (entry.folded_length == entry.preimage_length) continue; // Same width → not a tail-expansion shape
        expanding_preimages.push_back(entry);
    }

    std::size_t const filler_lengths[] = {0, 1, 2, 14, 15, 16, 17, 30, 31, 32, 33, 59, 60, 61, 62, 63, 64, 65};
    std::size_t const filler_lengths_count = sizeof(filler_lengths) / sizeof(filler_lengths[0]);
    std::size_t const needle_window_max = 16; // Tail paddings 0..15 keep the preimage within the last 4..16 bytes
    std::size_t cases_tested = 0;

    char needle[32];    // Two folded forms of up to 12 bytes, or one with an ASCII context byte
    char haystack[128]; // Largest filler 65, plus context, plus two 4-byte preimages, plus tail padding

    for (std::size_t entry_index = 0; entry_index < expanding_preimages.size(); ++entry_index) {
        expanding_preimage_t const &entry = expanding_preimages[entry_index];

        // Needle variants: 0 = folded form, 1 = "x"-suffixed, 2 = "x"-prefixed, 3 = folded form twice
        for (int variant = 0; variant < 4; ++variant) {
            std::size_t needle_length = 0;
            if (variant == 2) needle[needle_length++] = 'x';
            std::memcpy(needle + needle_length, entry.folded_utf8, entry.folded_length);
            needle_length += entry.folded_length;
            if (variant == 1) needle[needle_length++] = 'x';
            if (variant == 3) {
                std::memcpy(needle + needle_length, entry.folded_utf8, entry.folded_length);
                needle_length += entry.folded_length;
            }

            for (std::size_t filler_index = 0; filler_index < filler_lengths_count; ++filler_index) {
                for (std::size_t tail_padding = 0; tail_padding < needle_window_max; ++tail_padding) {
                    std::size_t haystack_length = 0;
                    while (haystack_length < filler_lengths[filler_index]) haystack[haystack_length++] = 'o';
                    if (variant == 2) haystack[haystack_length++] = 'x';
                    std::memcpy(haystack + haystack_length, entry.preimage_utf8, entry.preimage_length);
                    haystack_length += entry.preimage_length;
                    if (variant == 3) {
                        std::memcpy(haystack + haystack_length, entry.preimage_utf8, entry.preimage_length);
                        haystack_length += entry.preimage_length;
                    }
                    if (variant == 1) haystack[haystack_length++] = 'x';
                    for (std::size_t i = 0; i < tail_padding; ++i) haystack[haystack_length++] = 'y';

                    check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                  needle_length, "expanding tail find");
                    ++cases_tested;
                }
            }
        }
    }
    std::printf("    passed %zu cases across %zu expanding preimages\n", cases_tested, expanding_preimages.size());
}

/**
 *  @brief Differential + ground-truth test for matches whose folded runes cross the boundary between
 *         two adjacent multi-rune-folding codepoints.
 *
 *  The byte-history serial helpers are most fragile when a needle's folded runes begin mid-way through
 *  one expanding codepoint and end mid-way through the next - e.g. needle "sss" sits inside haystack
 *  "ßß" -> "ssss", beginning in the first 'ß' and finishing in the second. The preimage / expanding-tail
 *  fuzzers never reproduce this: they pad expansions with ASCII filler, so a folded run never straddles
 *  two expansions. Here we place two multi-rune-folding codepoints back to back, enumerate every proper
 *  sub-run of the combined folded stream that genuinely crosses the join, and use that sub-run (as
 *  folded bytes) as the needle - swept across the 64-byte SIMD chunk boundary - validated against the
 *  fold-subset reference.
 */
void test_uncased_find_crossing_fuzz(sz_utf8_uncased_search_t find_base, sz_utf8_uncased_search_t find_simd) {

    std::printf("  - testing uncased find across adjacent expansion boundaries...\n");

    // Codepoints whose fold emits more than one rune, so a needle can slice through the middle of their
    // expansion. The set is small (ligatures, sharp-s, decomposed accents), so a fixed cap avoids any
    // dynamic allocation on this cold path.
    enum { expanders_capacity = 1024 };
    sz_rune_t expanders[expanders_capacity];
    std::size_t expanders_count = 0;
    for (sz_rune_t codepoint = 1; codepoint <= 0x10FFFF; ++codepoint) {
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) continue; // Surrogates are not valid UTF-8
        sz_rune_t folded[3];
        if (sz_unicode_fold_codepoint_(codepoint, folded) < 2) continue;
        assert(expanders_count < expanders_capacity && "expander buffer overflow");
        expanders[expanders_count++] = codepoint;
    }

    std::size_t const filler_lengths[] = {0, 1, 14, 15, 16, 17, 30, 31, 32, 33, 60, 61, 62, 63, 64, 65};
    std::size_t const filler_lengths_count = sizeof(filler_lengths) / sizeof(filler_lengths[0]);
    std::size_t cases_tested = 0;

    char haystack[128];
    char needle[32];

    for (std::size_t first_index = 0; first_index < expanders_count; ++first_index) {
        for (std::size_t second_index = 0; second_index < expanders_count; ++second_index) {
            // Fold the adjacent pair into a flat rune stream, tagging which source codepoint (0 or 1)
            // produced each folded rune so we keep only the needles that cross the join.
            sz_rune_t folded_runes[6];
            int rune_owner[6];
            std::size_t folded_total = 0;
            sz_u8_t pair_utf8[8];
            std::size_t pair_length = 0;
            for (int which = 0; which < 2; ++which) {
                sz_rune_t const codepoint = expanders[which == 0 ? first_index : second_index];
                pair_length += (std::size_t)sz_rune_encode(codepoint, pair_utf8 + pair_length);
                sz_rune_t emitted[3];
                sz_size_t emitted_count = sz_unicode_fold_codepoint_(codepoint, emitted);
                for (sz_size_t index = 0; index < emitted_count; ++index) {
                    folded_runes[folded_total] = emitted[index];
                    rune_owner[folded_total] = which;
                    ++folded_total;
                }
            }

            // Every proper sub-run [start, start + length) that begins in the first codepoint and ends
            // in the second is a needle whose folded runes straddle the expansion boundary.
            for (std::size_t start = 0; start < folded_total; ++start) {
                for (std::size_t length = 1; start + length <= folded_total; ++length) {
                    std::size_t const last = start + length - 1;
                    if (rune_owner[start] != 0 || rune_owner[last] != 1) continue;

                    std::size_t needle_length = 0;
                    for (std::size_t index = 0; index < length; ++index)
                        needle_length += (std::size_t)sz_rune_encode(folded_runes[start + index],
                                                                     (sz_u8_t *)needle + needle_length);

                    for (std::size_t filler_index = 0; filler_index < filler_lengths_count; ++filler_index) {
                        std::size_t haystack_length = 0;
                        while (haystack_length < filler_lengths[filler_index]) haystack[haystack_length++] = 'o';
                        std::memcpy(haystack + haystack_length, pair_utf8, pair_length);
                        haystack_length += pair_length;
                        for (int tail = 0; tail < 4; ++tail) haystack[haystack_length++] = 'y';

                        check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                      needle_length, "crossing expansion find");
                        ++cases_tested;
                    }
                }
            }
        }
    }
    std::printf("    passed %zu cases across %zu adjacent expander pairs\n", cases_tested,
                expanders_count * expanders_count);
}

/**
 *  @brief Reference-validated coverage for the rolling-hash (Rabin-Karp) path when a match crosses
 *         expansion boundaries.
 *
 *  The short-needle helpers cover needles folding to 1-3 runes; this drives the 4+-rune path with
 *  matches that begin and end mid-expansion. A haystack of repeated expanding codepoints (e.g. "ßßßß"
 *  folds to "ssssssss") is searched for every folded sub-run of 4 or more runes - which bypasses the
 *  short helpers - swept across the 64-byte chunk boundary, each result checked against the independent
 *  fold-subset reference via the three-way `check_uncased_find_three_way_`.
 */
static void test_uncased_find_long_crossing_fuzz(sz_utf8_uncased_search_t find_base,
                                                 sz_utf8_uncased_search_t find_simd) {
    std::printf("  - testing uncased find across long (Rabin-Karp) expansion runs...\n");

    struct expander_t {
        char const *utf8;
        char const *folded;
    };
    expander_t const expanders[] = {
        {"\xC3\x9F", "ss"},      // ß
        {"\xEF\xAC\x80", "ff"},  // ﬀ
        {"\xEF\xAC\x83", "ffi"}, // ﬃ
    };
    std::size_t const fillers[] = {0, 30, 62, 63, 64, 65};
    std::size_t cases_tested = 0;

    char haystack[256];
    char needle[64];

    for (expander_t const &expander : expanders) {
        std::size_t const utf8_length = std::strlen(expander.utf8);
        std::size_t const folded_length = std::strlen(expander.folded);
        for (std::size_t copies = 3; copies <= 6; ++copies) {
            char folded_run[64];
            std::size_t folded_total = 0;
            for (std::size_t copy = 0; copy != copies; ++copy)
                for (std::size_t byte = 0; byte != folded_length; ++byte)
                    folded_run[folded_total++] = expander.folded[byte];

            // Every folded sub-run of 4+ runes bypasses the short helpers and lands in the rolling hash;
            // inside a haystack of `copies` expanders it necessarily crosses expansion boundaries.
            for (std::size_t needle_length = 4; needle_length <= folded_total; ++needle_length) {
                for (std::size_t start = 0; start + needle_length <= folded_total; ++start) {
                    std::memcpy(needle, folded_run + start, needle_length);
                    for (std::size_t filler_index = 0; filler_index != sizeof(fillers) / sizeof(fillers[0]);
                         ++filler_index) {
                        std::size_t haystack_length = 0;
                        while (haystack_length < fillers[filler_index]) haystack[haystack_length++] = 'o';
                        for (std::size_t copy = 0; copy != copies; ++copy) {
                            std::memcpy(haystack + haystack_length, expander.utf8, utf8_length);
                            haystack_length += utf8_length;
                        }
                        for (int tail = 0; tail != 4; ++tail) haystack[haystack_length++] = 'y';
                        check_uncased_find_three_way_(find_base, find_simd, haystack, haystack_length, needle,
                                                      needle_length, "long crossing expansion find");
                        ++cases_tested;
                    }
                }
            }
        }
    }
    std::printf("    passed %zu long Rabin-Karp crossing cases\n", cases_tested);
}

/**
 *  @brief The full differential + ground-truth battery for one backend's uncased find.
 *
 *  Runs the fuzzers and the structured adversarial enumerators (fold preimages, expanding tails, and
 *  cross-expansion needles) of `find_simd` against the serial baseline and the independent reference.
 *  Called once per backend so coverage stays uniform and a new backend cannot silently skip a test.
 */
static void run_uncased_find_battery_(sz_utf8_uncased_search_t find_simd) {
    sz_utf8_uncased_search_t const find_serial = sz_utf8_uncased_search_serial;
    std::size_t const queries = scale_iterations(100000);
    test_uncased_find_fuzz(find_serial, find_simd, sz_utf8_uncased_fold_serial, sz_utf8_seek_serial,
                           sz_utf8_count_serial, 16, 0, queries);
    test_uncased_find_fuzz(find_serial, find_simd, sz_utf8_uncased_fold_serial, sz_utf8_seek_serial,
                           sz_utf8_count_serial, 32, 0, queries);
    test_uncased_find_fuzz(find_serial, find_simd, sz_utf8_uncased_fold_serial, sz_utf8_seek_serial,
                           sz_utf8_count_serial, 100, 100, queries);
    test_uncased_find_fuzz(find_serial, find_simd, sz_utf8_uncased_fold_serial, sz_utf8_seek_serial,
                           sz_utf8_count_serial, 200, 100, queries);
    test_uncased_find_preimages_fuzz(find_serial, find_simd);
    test_uncased_find_tails_fuzz(find_serial, find_simd);
    test_uncased_find_crossing_fuzz(find_serial, find_simd);
    test_uncased_find_long_crossing_fuzz(find_serial, find_simd);

    // A long ASCII needle (well past the 32-rune ring buffer and the 3-rune short helpers) drives the
    // pure Rabin-Karp path inside a large random haystack, both where the needle was spliced in (so a
    // match exists) and where it was not (so the not-found path is exercised at scale).
    std::printf("  - testing uncased find with a long Rabin-Karp ASCII needle in a large haystack...\n");
    auto &random_generator = global_random_generator();
    std::uniform_int_distribution<int> letter_distribution(0, 25);
    // The independent reference oracle caps its folded haystack at 512 runes, so keep the haystack under
    // that while the 48-rune needle (well past the 32-rune ring buffer) still drives the Rabin-Karp path.
    std::string needle, haystack;
    needle.reserve(48);
    haystack.reserve(400);
    for (std::size_t length = 0; length != 48; ++length)
        needle.push_back((char)('A' + letter_distribution(random_generator)));
    haystack.clear();
    for (std::size_t length = 0; length != 400; ++length)
        haystack.push_back((char)('a' + letter_distribution(random_generator)));

    // The lowercase haystack cannot contain the uppercase needle by chance, so this is the not-found case.
    check_uncased_find_three_way_(find_serial, find_simd, haystack.data(), haystack.size(), needle.data(),
                                  needle.size(), "long ascii needle, not found");

    // Splice the needle near the middle of the haystack so the match exists.
    std::string spliced_haystack = haystack;
    std::size_t const splice_offset = spliced_haystack.size() / 2;
    spliced_haystack.replace(splice_offset, needle.size(), needle);
    check_uncased_find_three_way_(find_serial, find_simd, spliced_haystack.data(), spliced_haystack.size(),
                                  needle.data(), needle.size(), "long ascii needle, with match");
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer + C++ API coverage for the uncased UTF-8 family on simple, hand-verifiable inputs.
 *
 *  First exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through the
 *  C++ wrappers, so a regression that the serial-vs-SIMD agreement tests would miss - because both share
 *  a wrong constant - is still caught against an external, hand-derived ground truth. It then sweeps a
 *  broad battery of cross-script C++ wrapper cases: ordering, finding, ligatures, expansions, and
 *  SIMD-boundary regressions discovered by earlier fuzzing.
 */
void test_uncased_unit() {

    using str = sz::string_view;

    // Known-answer vectors through the dispatched C API, the backend kernels, and the C++ wrappers.
    {
        // `sz_utf8_uncased_search`: "world" matches case-insensitively in "Hello World" at byte offset 6, length 5.
        char const *greeting = "Hello World";
        sz_size_t const greeting_length = (sz_size_t)std::strlen(greeting);
        check_uncased_find_unit_(sz_utf8_uncased_search, greeting, greeting_length, // Dispatched (automatic kernel)
                                 "world", 5, 6, 5);
        check_uncased_find_unit_(sz_utf8_uncased_search_serial, greeting, greeting_length, // Manual: serial kernel
                                 "world", 5, 6, 5);
#if SZ_USE_HASWELL
        check_uncased_find_unit_(sz_utf8_uncased_search_haswell, greeting, greeting_length, // Manual: haswell kernel
                                 "world", 5, 6, 5);
#endif
#if SZ_USE_ICELAKE
        check_uncased_find_unit_(sz_utf8_uncased_search_icelake, greeting, greeting_length, // Manual: icelake kernel
                                 "world", 5, 6, 5);
#endif

        // `sz_utf8_uncased_search`: 'ß' (U+00DF, C3 9F) folds to "ss", so needle "SS" matches the whole 2-byte 'ß'.
        char const *sharp_s = "\xC3\x9F" "fox"; // 'ß' followed by "fox" → folds to "ssfox"
        sz_size_t const sharp_s_length = (sz_size_t)std::strlen(sharp_s);
        check_uncased_find_unit_(sz_utf8_uncased_search, sharp_s, sharp_s_length, // Dispatched (automatic kernel)
                                 "SS", 2, 0, 2);
        check_uncased_find_unit_(sz_utf8_uncased_search_serial, sharp_s, sharp_s_length, // Manual: serial kernel
                                 "SS", 2, 0, 2);
#if SZ_USE_HASWELL
        check_uncased_find_unit_(sz_utf8_uncased_search_haswell, sharp_s, sharp_s_length, // Manual: haswell kernel
                                 "SS", 2, 0, 2);
#endif
#if SZ_USE_ICELAKE
        check_uncased_find_unit_(sz_utf8_uncased_search_icelake, sharp_s, sharp_s_length, // Manual: icelake kernel
                                 "SS", 2, 0, 2);
#endif

        // C++ wrapper on `sz::string_view`: same two cases through `utf8_uncased_search`.
        { let_assert(auto match = str(greeting).utf8_uncased_search("world"), match.offset == 6 && match.length == 5); }
        { let_assert(auto match = str(sharp_s).utf8_uncased_search("SS"), match.offset == 0 && match.length == 2); }

        // `sz_utf8_uncased_fold`: "HeLLo" → "hello", and 'ß' (U+00DF) → "ss".
        check_uncased_fold_unit_(sz_utf8_uncased_fold, "HeLLo", 5, "hello");        // Dispatched (automatic kernel)
        check_uncased_fold_unit_(sz_utf8_uncased_fold_serial, "HeLLo", 5, "hello"); // Manual: serial kernel
        check_uncased_fold_unit_(sz_utf8_uncased_fold, "\xC3\x9F", 2, "ss");        // Dispatched (automatic kernel)
        check_uncased_fold_unit_(sz_utf8_uncased_fold_serial, "\xC3\x9F", 2, "ss"); // Manual: serial kernel
#if SZ_USE_ICELAKE
        check_uncased_fold_unit_(sz_utf8_uncased_fold_icelake, "HeLLo", 5, "hello"); // Manual: icelake kernel
        check_uncased_fold_unit_(sz_utf8_uncased_fold_icelake, "\xC3\x9F", 2, "ss"); // Manual: icelake kernel
#endif

        // C++ wrapper: in-place fold on a mutable `sz::string`.
        {
            sz::string folded("HeLLo");
            assert(folded.try_utf8_uncased_fold());
            assert(folded == "hello");
        }
        {
            sz::string folded("\xC3\x9F");
            assert(folded.try_utf8_uncased_fold());
            assert(folded == "ss");
        }

        // `sz_utf8_uncased_order`: "Hello" and "HELLO" compare equal ignoring case.
        assert(sz_utf8_uncased_order("Hello", 5, "HELLO", 5) == sz_equal_k);        // Dispatched (automatic kernel)
        assert(sz_utf8_uncased_order_serial("Hello", 5, "HELLO", 5) == sz_equal_k); // Manual: serial kernel
        assert(str("Hello").utf8_uncased_order("HELLO") == sz_equal_k);             // C++ wrapper

        // `sz_utf8_find_cased`: NULL for a fully-caseless string, else the FIRST cased codepoint.
        // "价格 123" is caseless (CJK + digits + space), so no rune participates in case → NULL.
        char const *caseless = "\xE4\xBB\xB7\xE6\xA0\xBC 123"; // "价格 123"
        sz_size_t const caseless_length = (sz_size_t)std::strlen(caseless);
        assert(sz_utf8_find_cased(caseless, caseless_length) == SZ_NULL_CHAR);        // Dispatched (automatic kernel)
        assert(sz_utf8_find_cased_serial(caseless, caseless_length) == SZ_NULL_CHAR); // Manual: serial kernel
#if SZ_USE_ICELAKE
        assert(sz_utf8_find_cased_icelake(caseless, caseless_length) == SZ_NULL_CHAR); // Manual: icelake kernel
#endif

        // "123Abc" has its first cased codepoint 'A' at byte offset 3.
        char const *mixed = "123Abc";
        sz_size_t const mixed_length = (sz_size_t)std::strlen(mixed);
        assert(sz_utf8_find_cased(mixed, mixed_length) == mixed + 3);        // Dispatched (automatic kernel)
        assert(sz_utf8_find_cased_serial(mixed, mixed_length) == mixed + 3); // Manual: serial kernel
#if SZ_USE_ICELAKE
        assert(sz_utf8_find_cased_icelake(mixed, mixed_length) == mixed + 3); // Manual: icelake kernel
#endif
    }

    // Equal strings (ASCII)
    assert(str("hello").utf8_uncased_order("HELLO") == sz_equal_k);
    assert(str("abc").utf8_uncased_order("ABC") == sz_equal_k);
    assert(str("HeLLo WoRLd").utf8_uncased_order("hello world") == sz_equal_k);

    // ASCII Extensions
    let_assert(auto m = str("prefixhello").utf8_uncased_search("HELLO"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("hello_suffix").utf8_uncased_search("HELLO"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("mid_hello_mid").utf8_uncased_search("HELLO"), m.offset == 4 && m.length == 5);

    // Less than
    assert(str("abc").utf8_uncased_order("abd") == sz_less_k);
    assert(str("ab").utf8_uncased_order("abc") == sz_less_k);
    assert(str("ABC").utf8_uncased_order("abd") == sz_less_k);

    // Greater than
    assert(str("abd").utf8_uncased_order("abc") == sz_greater_k);
    assert(str("abcd").utf8_uncased_order("abc") == sz_greater_k);
    assert(str("ABD").utf8_uncased_order("abc") == sz_greater_k);

    // Latin-1 Supplement & Latin Extended-A
    // German Umlauts
    assert(str("schöner").utf8_uncased_order("SCHÖNER") == sz_equal_k);
    let_assert(auto m = str("Das ist ein schöner Tag").utf8_uncased_search("SCHÖNER"),
               m.offset == 12 && m.length == 8); // 'ö' (U+00F6, C3 B6) is 2 bytes

    // French Accents
    assert(str("café").utf8_uncased_order("CAFÉ") == sz_equal_k);
    assert(str("naïve").utf8_uncased_order("NAÏVE") == sz_equal_k);
    assert(str("À la carte").utf8_uncased_order("à la CARTE") == sz_equal_k);

    // Spanish/Portuguese
    assert(str("niño").utf8_uncased_order("NIÑO") == sz_equal_k);

    // Polish / Central European (Latin Extended-A)
    // "ĄĆĘŁŃÓŚŹŻ" -> "ąćęłńóśźż"
    // "Zaółć gęślą jaźń" (classic Polish pangram fragment)
    assert(str("Zaółć gęślą jaźń").utf8_uncased_order("ZAÓŁĆ GĘŚLĄ JAŹŃ") == sz_equal_k);

    // Czech characters: ř (U+0159, C5 99), ž (U+017E, C5 BE), č (U+010D, C4 8D), ě (U+011B, C4 9B)
    assert(str("řžčě").utf8_uncased_order("ŘŽČĚ") == sz_equal_k);
    let_assert(auto m = str("Příklad").utf8_uncased_search("PŘÍKLAD"), m.offset == 0 && m.length == 9);
    let_assert(auto m = str("žena").utf8_uncased_search("ŽENA"), m.offset == 0 && m.length == 5);

    // Polish ł (U+0142, C5 82) in city name
    assert(str("Łódź").utf8_uncased_order("ŁÓDŹ") == sz_equal_k);
    let_assert(auto m = str("miasto Łódź").utf8_uncased_search("łódź"), m.offset == 7 && m.length == 7);

    // Hungarian: ő (U+0151, C5 91), ű (U+0171, C5 B1)
    assert(str("őű").utf8_uncased_order("ŐŰ") == sz_equal_k);
    let_assert(auto m = str("Erdő").utf8_uncased_search("ERDŐ"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("Győr").utf8_uncased_search("GYŐR"), m.offset == 0 && m.length == 5);

    // Central European at SIMD boundary (64 bytes)
    {
        std::string prefix(62, 'a');
        let_assert(auto m = str(prefix + "ž").utf8_uncased_search("Ž"), m.offset == 62 && m.length == 2);
        let_assert(auto m = str(prefix + "řž").utf8_uncased_search("ŘŽ"), m.offset == 62 && m.length == 4);
    }

    // German (Eszett 'ß')
    // 'ß' (U+00DF, C3 9F) -> "ss"
    // "straße" -> "strasse"
    // "STRASSE" -> "strasse"
    assert(str("straße").utf8_uncased_order("STRASSE") == sz_equal_k);
    assert(str("STRASSE").utf8_uncased_order("straße") == sz_equal_k);

    // Uppercase 'ẞ' (U+1E9E, E1 BA 9E) -> "ss" or "ß" depending on fold
    // StringZilla generally folds to lowercase first. 'ẞ' -> 'ss'.
    // Haystack uses 'ß' (2 bytes), Needle "SS".
    let_assert(auto m = str("straße").utf8_uncased_search("SS"),
               m.offset == 4 && m.length == 2); // Matches 'ß' (2 bytes)

    // Eszett Context Extensions
    let_assert(auto m = str("Eine straße").utf8_uncased_search("SS"),
               m.offset == 9 && m.length == 2); // "Eine " is 5 chars -> 5 bytes + "stra" (4) = 9
    let_assert(auto m = str("straßebahn").utf8_uncased_search("SS"), m.offset == 4 && m.length == 2);
    let_assert(auto m = str("Eine straßebahn").utf8_uncased_search("SS"), m.offset == 9 && m.length == 2);

    // Same case-folding, but different relation
    let_assert(auto m = str("HelloäeßHelloL").utf8_uncased_search("helloäesshellol"), m.offset == 0 && m.length == 16);
    let_assert(auto m = str("helloäesshellol").utf8_uncased_search("HelloäeßHelloL"), m.offset == 0 && m.length == 16);

    // Same case-folding, but different relation and needle length due to uppercase tripple-byte 'ẞ' (U+1E9E, E1 BA 9E)
    let_assert(auto m = str("HelloäeẞHelloL").utf8_uncased_search("helloäesshellol"), m.offset == 0 && m.length == 17);
    let_assert(auto m = str("helloäesshellol").utf8_uncased_search("HelloäeẞHelloL"), m.offset == 0 && m.length == 16);

    // Haystack "STRASSE", Needle "straße"
    let_assert(auto m = str("STRASSE").utf8_uncased_search("straße"),
               m.offset == 0 && m.length == 7); // Matches "STRASSE" (7 bytes)

    // "Maße" -> "MASSE"
    let_assert(auto m = str("Maße").utf8_uncased_search("MASSE"),
               m.offset == 0 && m.length == 5); // Matches "Maße" (5 bytes)

    // Haystack: "Fuss" (4 bytes) "u", "s", "s"
    // Needle: "Fuß" (4 bytes) "u", "ß"
    // They are equal in order, and searching "Fuß" in "Fuss" works.
    let_assert(auto m = str("Fuss").utf8_uncased_search("Fuß"),
               m.offset == 0 && m.length == 4); // Matches "Fuss"

    // Mid-expansion matching: needle starts with 's' (uses serial fallback)
    // Haystack: "ßfox" (5 bytes) → folds to "ssfox"
    // Needle "sfox" matches at position 1 in folded, but we report offset=0 (start of ß)
    // Length is 5 because we consume the entire ß character (can't point to half of it)
    let_assert(auto m = str("\xC3\x9F" "fox").utf8_uncased_search("sfox"), m.offset == 0 && m.length == 5);

    // Needle ends with 's' - suffix case (uses serial fallback)
    // Haystack: "foxß" → folds to "foxss"
    // Needle "foxs" matches through first 's' of expansion
    let_assert(auto m = str("fox\xC3\x9F").utf8_uncased_search("foxs"), m.offset == 0 && m.length == 5);

    // Cross-boundary case: "ßS" folds to "sss" (uses serial fallback)
    // Haystack: "ßStra" (6 bytes) → folds to "ssstra"
    // Needle "sstra" starts with 's', would match at position 1 (mid-ß) without the rule
    // Length is 6 because we consume the entire haystack (ß expands, consuming whole character)
    let_assert(auto m = str("\xC3\x9F" "Stra").utf8_uncased_search("sstra"), m.offset == 0 && m.length == 6);

    // Needle with 's' NOT at boundary - should use fast SIMD path
    let_assert(auto m = str("te\xC3\x9F" "t").utf8_uncased_search("tesst"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("ma\xC3\x9F" "e").utf8_uncased_search("masse"), m.offset == 0 && m.length == 5);

    // Needle with 'ss' at boundary - also uses serial (can't match across ß boundary)
    let_assert(auto m = str("fo\xC3\x9F").utf8_uncased_search("foss"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("\xC3\x9F" "fo").utf8_uncased_search("ssfo"), m.offset == 0 && m.length == 4);

    // Math Symbols
    // Multiplication × (U+00D7, C3 97) and Division ÷ (U+00F7, C3 B7)
    // Often confusable with 'x' and '+'/'=', but strictly they are distinct.
    // They should equal themselves but not each other.
    assert(str("×").utf8_uncased_order("×") == sz_equal_k); // × == ×
    assert(str("÷").utf8_uncased_order("÷") == sz_equal_k); // ÷ == ÷
    assert(str("×").utf8_uncased_order("÷") != sz_equal_k); // × ≠ ÷
    assert(str("a×b").utf8_uncased_order("A×B") == sz_equal_k);

    // Math Context Extensions
    let_assert(auto m = str("2×3=6").utf8_uncased_search("×"), m.offset == 1 && m.length == 2);
    let_assert(auto m = str("6÷2=3").utf8_uncased_search("÷"), m.offset == 1 && m.length == 2);

    // Empty strings
    assert(str("").utf8_uncased_order("") == sz_equal_k);
    assert(str("a").utf8_uncased_order("") == sz_greater_k);
    assert(str("").utf8_uncased_order("a") == sz_less_k);

    // Greek
    // Basic casing: "αβγδ" vs "ΑΒΓΔ"
    assert(str("αβγδ").utf8_uncased_order("ΑΒΓΔ") == sz_equal_k);
    let_assert(auto m = str("αβγδ").utf8_uncased_search("ΑΒΓΔ"),
               m.offset == 0 && m.length == 8); // 4 * 2 bytes = 8 bytes

    // Greek Context Extensions
    // "prefix " is 7 bytes.
    let_assert(auto m = str("prefix αβγδ").utf8_uncased_search("ΑΒΓΔ"), m.offset == 7 && m.length == 8);
    // " suffix" is 7 bytes. "αβγδ" is 8 bytes.
    let_assert(auto m = str("αβγδ suffix").utf8_uncased_search("ΑΒΓΔ"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("prefix αβγδ suffix").utf8_uncased_search("ΑΒΓΔ"), m.offset == 7 && m.length == 8);

    // Sigma: 'Σ' (U+03A3, CE A3) matches both 'σ' (U+03C3, CF 83, medial) and 'ς' (U+03C2, CF 82, final)
    // Haystack: "ΟΔΥΣΣΕΥΣ" (Odysseus uppercase)
    // Needle: "οδυσσευς" (lowercase with final sigma)
    // Lengths match byte-for-byte in this case.
    let_assert(auto m = str("ΟΔΥΣΣΕΥΣ").utf8_uncased_search("οδυσσευς"),
               m.offset == 0 && m.length == 16); // 8 chars * 2 bytes

    // Micro Sign 'µ' (U+00B5) vs Greek Mu 'μ' (U+03BC) vs 'Μ' (U+039C)
    // These should all fold to the same canonical representation.
    let_assert(auto m = str("µ").utf8_uncased_search("μ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("μ").utf8_uncased_search("µ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("µ").utf8_uncased_search("Μ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("Μ").utf8_uncased_search("µ"), m.offset == 0 && m.length == 2);
    // Context: Head/Tail/Middle
    let_assert(auto m = str("123µ456").utf8_uncased_search("123μ456"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("LongPrefix Μ Suffix").utf8_uncased_search("Prefix µ Suf"),
               m.offset == 4 && m.length == 13);

    // Greek Lunate Epsilon 'ϵ' (U+03F5) -> 'ε' (U+03B5)
    let_assert(auto m = str("ϵ").utf8_uncased_search("ε"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("start ϵ end").utf8_uncased_search("start ε end"), m.offset == 0 && m.length == 12);
    let_assert(auto m = str("...ϵ...").utf8_uncased_search(".ε."), m.offset == 2 && m.length == 4);
    // Greek Kappa Symbol 'ϰ' (U+03F0) -> 'κ' (U+03BA)
    let_assert(auto m = str("ϰ").utf8_uncased_search("κ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("text ϰ").utf8_uncased_search("text κ"), m.offset == 0 && m.length == 7); // 5 + 2
    let_assert(auto m = str("ϰ text").utf8_uncased_search("κ text"), m.offset == 0 && m.length == 7);

    // Greek Symbols & Anomalies
    // 'ϐ' (CF 90) -> 'β' (CE B2)
    let_assert(auto m = str("ϐ").utf8_uncased_search("β"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("alpha ϐ").utf8_uncased_search("alpha β"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("ϐ beta").utf8_uncased_search("β beta"), m.offset == 0 && m.length == 7);
    // 'ϑ' (CF 91) -> 'θ' (CE B8)
    let_assert(auto m = str("ϑ").utf8_uncased_search("θ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("1ϑ2").utf8_uncased_search("1θ2"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("prefix ϑ suffix").utf8_uncased_search("fix θ suf"), m.offset == 3 && m.length == 10);
    // 'ϖ' (CF 96) -> 'π' (CF 80)
    let_assert(auto m = str("ϖ").utf8_uncased_search("π"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("AϖB").utf8_uncased_search("AπB"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("Long string with ϖ in it").utf8_uncased_search("th π in"),
               m.offset == 14 && m.length == 8);

    // Greek Context Extensions (Symbols)
    let_assert(auto m = str("alpha ϖ omega").utf8_uncased_search("π"), m.offset == 6 && m.length == 2);

    // Dialytika with Tonos 'ΐ' (CE 90) -> Identity check mostly
    assert(str("ΐ").utf8_uncased_order("ΐ") == sz_equal_k);

    // Greek in Mixed Scripts (boundary checks)
    let_assert(auto m = str("ABCαβγ").utf8_uncased_search("abcΑΒΓ"),
               m.offset == 0 && m.length == 9); // 3 + 3*2 bytes

    // Cyrillic
    // Basic: "привет" vs "ПРИВЕТ"
    assert(str("привет").utf8_uncased_order("ПРИВЕТ") == sz_equal_k);
    let_assert(auto m = str("привет мир").utf8_uncased_search("ПРИВЕТ"),
               m.offset == 0 && m.length == 12); // 6 chars * 2 bytes

    // Cyrillic Context Extensions
    // "Check " is 6 bytes.
    let_assert(auto m = str("Check привет").utf8_uncased_search("ПРИВЕТ"), m.offset == 6 && m.length == 12);
    let_assert(auto m = str("привет check").utf8_uncased_search("ПРИВЕТ"), m.offset == 0 && m.length == 12);

    // Palochka 'Ӏ' (U+04C0, D3 80) -> 'ӏ' (U+04CF, D3 8F)
    // Used in Caucasian languages. Case agnostic.
    let_assert(auto m = str("Ӏ").utf8_uncased_search("ӏ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("ӏ").utf8_uncased_search("Ӏ"), m.offset == 0 && m.length == 2);

    // Ukrainian Ґ (U+0490) -> ґ (U+0491)
    let_assert(auto m = str("Ґ").utf8_uncased_search("ґ"), m.offset == 0 && m.length == 2);

    // Mixed Cyrillic
    let_assert(auto m = str("Москва is beautiful").utf8_uncased_search("МОСКВА"),
               m.offset == 0 && m.length == 12); // 6 chars * 2

    // Turkish
    // Dotted 'İ' (U+0130, C4 B0) -> 'i' (ASCII) + combining dot (U+0307, CC 87)
    // "İstanbul" (starts with İ) vs "i̇stanbul" (starts with i + dot)
    // StringZilla finds canonical equivalence. 'İ' (2 bytes) matches 'i̇' (3 bytes).
    let_assert(auto m = str("İstanbul").utf8_uncased_search("i̇stanbul"), // "i" + dot
               m.offset == 0 && m.length == 9);                          // Haystack length is 2 (İ) + 7 (stanbul) = 9
    // Needle starts with the combining dot (mid-expansion of 'İ'), so the match still anchors to 'İ'.
    let_assert(auto m = str("İstanbul").utf8_uncased_search("\xCC\x87" "stanbul"), m.offset == 0 && m.length == 9);

    // Turkish Context Extensions
    // "Welcome to " is 11 bytes.
    let_assert(auto m = str("Welcome to İstanbul").utf8_uncased_search("i̇stanbul"), m.offset == 11 && m.length == 9);
    let_assert(auto m = str("Welcome to İstanbul").utf8_uncased_search("\xCC\x87" "stanbul"),
               m.offset == 11 && m.length == 9);
    // "İstanbul city"
    let_assert(auto m = str("İstanbul city").utf8_uncased_search("i̇stanbul"), m.offset == 0 && m.length == 9);

    // Undotted 'ı' (U+0131)
    // Typically 'I' (ASCII) folds to 'i' (ASCII).
    // 'ı' folds to... itself? Or 'I' if we are in Turkish mode?
    // Default fold often treats 'ı' as distinct from 'i'.
    // 'I' -> 'i'. 'ı' -> 'ı'. So 'I' != 'ı'.
    let_assert(auto m = str("I").utf8_uncased_search("ı"), m.offset == str::npos);

    // Turkish Ğ (U+011E) -> ğ (U+011F) and Ş (U+015E) -> ş (U+015F)
    let_assert(auto m = str("ĞŞ").utf8_uncased_search("ğş"), m.offset == 0 && m.length == 4);

    // Armenian
    // Ligature: 'և' (U+0587, D6 87) -> 'ե' (U+0565, D5 A5) + 'ւ' (U+0582, D6 82)
    // Haystack: "և" (2 bytes). Needle: "եւ" (2 + 2 = 4 bytes).
    // Match should return haystack slice (2 bytes).
    let_assert(auto m = str("և").utf8_uncased_search("եւ"), m.offset == 0 && m.length == 2);

    // Armenian Context Extensions
    let_assert(auto m = str("abcև").utf8_uncased_search("եւ"), m.offset == 3 && m.length == 2);
    let_assert(auto m = str("ևabc").utf8_uncased_search("եւ"), m.offset == 0 && m.length == 2);
    // Reverse: Haystack "եւ" (4 bytes). Needle "և" (2 bytes).
    // Match should return haystack slice (4 bytes).
    let_assert(auto m = str("եւ").utf8_uncased_search("և"), m.offset == 0 && m.length == 4);

    // Armenian Context Extensions Reverse
    let_assert(auto m = str("abcեւ").utf8_uncased_search("և"), m.offset == 3 && m.length == 4);

    // Ligature: 'ﬓ' (U+FB13 Men-Now) -> 'մ' (U+0574) + 'ն' (U+0576)
    // Haystack 3 bytes (EF AC 93). Needle 4 bytes (D5 B4 D5 B6).
    let_assert(auto m = str("ﬓ").utf8_uncased_search("մն"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("abcﬓdef").utf8_uncased_search("մն"), m.offset == 3 && m.length == 3);
    let_assert(auto m = str("ﬓ start").utf8_uncased_search("մն start"), m.offset == 0 && m.length == 9);

    // Ligature: 'ﬔ' (U+FB14 Men-Ech) -> 'մ' (U+0574) + 'ե' (U+0565)
    let_assert(auto m = str("ﬔ").utf8_uncased_search("մե"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Some ﬔ text").utf8_uncased_search("մե"), m.offset == 5 && m.length == 3);
    let_assert(auto m = str("End ﬔ").utf8_uncased_search("End մե"), m.offset == 0 && m.length == 7);

    // Ligature: 'ﬕ' (U+FB15 Men-Ini) -> 'մ' (U+0574) + 'ի' (U+056B)
    let_assert(auto m = str("ﬕ").utf8_uncased_search("մի"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("123 ﬕ 456").utf8_uncased_search("123 մի 456"), m.offset == 0 && m.length == 11);
    let_assert(auto m = str("prefixﬕ").utf8_uncased_search("մի"), m.offset == 6 && m.length == 3);

    // Ligature: 'ﬖ' (U+FB16 Vew-Now) -> 'վ' (U+057E) + 'ն' (U+0576)
    let_assert(auto m = str("ﬖ").utf8_uncased_search("վն"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Test ﬖ Case").utf8_uncased_search("Test վն Case"), m.offset == 0 && m.length == 13);
    let_assert(auto m = str("ﬖ").utf8_uncased_search("վն"),
               m.offset == 0 && m.length == 3); // Redundant but safe

    // Ligature: 'ﬗ' (U+FB17 Men-Xeh) -> 'մ' (U+0574) + 'խ' (U+056D)
    let_assert(auto m = str("ﬗ").utf8_uncased_search("մխ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Mid ﬗ dle").utf8_uncased_search("մխ"), m.offset == 4 && m.length == 3);
    let_assert(auto m = str("Start ﬗ").utf8_uncased_search("Start մխ"), m.offset == 0 && m.length == 9);

    // Vietnamese / Latin Extended Additional
    // 'Ạ' (U+1EA0, E1 BA A0) -> 'ạ' (U+1EA1, E1 BA A1)
    let_assert(auto m = str("Ạ").utf8_uncased_search("ạ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Word Ạ End").utf8_uncased_search("Word ạ End"), m.offset == 0 && m.length == 12);
    let_assert(auto m = str("PrefixẠ").utf8_uncased_search("ạ"), m.offset == 6 && m.length == 3);

    // 'Ấ' (U+1EA4, E1 BA A4) -> 'ấ' (U+1EA5, E1 BA A5)
    let_assert(auto m = str("Ấ").utf8_uncased_search("ấ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Ấ Start").utf8_uncased_search("ấ Start"), m.offset == 0 && m.length == 9);
    let_assert(auto m = str("Mid Ấ dle").utf8_uncased_search("Mid ấ dle"), m.offset == 0 && m.length == 11);

    // Horn letters: Ơ (U+01A0, C6 A0) -> ơ (U+01A1, C6 A1), Ư (U+01AF, C6 AF) -> ư (U+01B0, C6 B0)
    let_assert(auto m = str("ƠƯ").utf8_uncased_search("ơư"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("Big ƠƯ Horns").utf8_uncased_search("Big ơư Horns"), m.offset == 0 && m.length == 14);
    let_assert(auto m = str("Prefix ƠƯ").utf8_uncased_search("ơư"), m.offset == 7 && m.length == 4);

    // Latin Extended Additional: Ḁ (U+1E80, E1 BA 80) -> ḁ (U+1E81, E1 BA 81)
    let_assert(auto m = str("Ḁ").utf8_uncased_search("ḁ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Code Ḁ").utf8_uncased_search("Code ḁ"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("StartḀ").utf8_uncased_search("Startḁ"), m.offset == 0 && m.length == 8);

    // Vietnamese Context Extensions
    let_assert(auto m = str("xin chào Ḁ").utf8_uncased_search("ḁ"), m.offset == 10 && m.length == 3);

    // Special Symbols (Latin)
    // Kelvin Sign U+212A (E2 84 AA) folds to 'k' (1 byte); the match spans the 3-byte source rune.
    let_assert(auto m = str("273 \xE2\x84\xAA").utf8_uncased_search("273 k"), m.offset == 0 && m.length == 7);

    // Reverse: haystack "273 k" (5 bytes), needle "273 " + Kelvin U+212A.
    let_assert(auto m = str("273 k").utf8_uncased_search("273 \xE2\x84\xAA"), m.offset == 0 && m.length == 5);

    // Angstrom Sign U+212B (E2 84 AB) folds to 'a' with ring U+00E5 (C3 A5).
    let_assert(auto m = str("\xE2\x84\xAB").utf8_uncased_search("\xC3\xA5"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("\xE2\x84\xAB").utf8_uncased_search("\xC3\x85"), m.offset == 0 && m.length == 3);

    // Context Extensions (Special Symbols)
    let_assert(auto m = str("Temp: 273 \xE2\x84\xAA").utf8_uncased_search("k"), m.offset == 10 && m.length == 3);
    let_assert(auto m = str("Unit: \xE2\x84\xAB").utf8_uncased_search("\xC3\xA5"), m.offset == 6 && m.length == 3);

    // Long S 'ſ' (U+017F) -> 's'
    // "Messer" vs "Meſſer"
    // Haystack "Meſſer": M(1) e(1) ſ(2) ſ(2) e(1) r(1) = 8 bytes.
    // Needle "MESSER": 6 bytes.
    let_assert(auto m = str("Meſſer").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("Ein Meſſer").utf8_uncased_search("MESSER"), m.offset == 4 && m.length == 8);
    let_assert(auto m = str("Meſſer block").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);

    // Ligature 'ﬅ' (U+FB05 "st") -> "st"
    // Haystack "ﬅ" (3 bytes). Needle "st" (2 bytes).
    let_assert(auto m = str("ﬅ").utf8_uncased_search("st"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Test ﬅ").utf8_uncased_search("Test st"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("ﬅart").utf8_uncased_search("start"), m.offset == 0 && m.length == 6);

    // Ligature 'ﬆ' (U+FB06, EF AC 86) -> "st"
    let_assert(auto m = str("ﬆ").utf8_uncased_search("st"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("My ﬆyle").utf8_uncased_search("My style"), m.offset == 0 && m.length == 9);
    let_assert(auto m = str("Faﬆ").utf8_uncased_search("Fast"), m.offset == 0 && m.length == 5);

    // Extended Ligature Contexts
    // "Messer" vs "Meſſer" ('ſ' is U+017F, C5 BF)
    let_assert(auto m = str("Das Meſſer schneidet").utf8_uncased_search("MESSER"),
               m.offset == 4 && m.length == 8); // "Das " (4) + "Meſſer" (8) = 12, start at 4
    let_assert(auto m = str("Meſſer").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("Großes Meſſer").utf8_uncased_search("MESSER"),
               m.offset == 8 && m.length == 8); // "Großes " (4+2+1+1+1 = 9 bytes? No. 'ß' is 2 bytes.
                                                // G(1)r(1)o(1)ß(2)e(1)s(1) (1) = 8 bytes. So offset 8.

    // 'ﬅ' (U+FB05, EF AC 85)
    let_assert(auto m = str("Ligature ﬅ check").utf8_uncased_search("st"), m.offset == 9 && m.length == 3);
    let_assert(auto m = str("end with ﬅ").utf8_uncased_search("st"), m.offset == 9 && m.length == 3);

    // More complex ligatures
    let_assert(auto m = str("ﬃJaCä").utf8_uncased_search("fija"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("ﬃJaCä").utf8_uncased_search("ﬁja"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("alﬃJaCä").utf8_uncased_search("fija"), m.offset == 2 && m.length == 5);
    let_assert(auto m = str("alﬃJaCä").utf8_uncased_search("ﬁja"), m.offset == 2 && m.length == 5);

    // Mid-expansion matches inside a single ligature: we still report the source rune span.
    // 'ﬃ' (EF AC 83) folds to "ffi", so "fi" occurs starting at index 1.
    let_assert(auto m = str("ﬃ").utf8_uncased_search("fi"), m.offset == 0 && m.length == 3);
    // 'ﬄ' (EF AC 84) folds to "ffl", so "fl" occurs starting at index 1.
    let_assert(auto m = str("ﬄ").utf8_uncased_search("fl"), m.offset == 0 && m.length == 3);

    // Combining diacritical marks: ǰ (U+01F0) folds to 'j' + combining caron (U+030C)
    // Needle starts with combining caron - can match mid-expansion of ǰ
    let_assert(auto m = str("ǰ0").utf8_uncased_search("\xCC\x8C" "0"), // caron + '0'
               m.offset == 0 && m.length == 3);                        // Match entire ǰ0 (2 byte ǰ + 1 byte 0)
    let_assert(auto m = str("abcǰ0def").utf8_uncased_search("\xCC\x8C" "0"),
               m.offset == 3 && m.length == 3); // "abc" = 3 bytes

    // Mid-expansion matches with ß (U+00DF) → "ss"
    // Needle "sfoxeepmº" should match in "ßfoxeEPMº" (folded: "ssfoxeepmº") at position 1 of folded text
    // Return position is byte 0 where ß starts (first contributing character)
    let_assert(auto m = str("ßfoxeEPMº").utf8_uncased_search("sfoxeepmº"),
               m.offset == 0 && m.length == 11); // Entire haystack

    // 'ﬆ' (U+FB06, EF AC 86)
    let_assert(auto m = str("Big ﬆ").utf8_uncased_search("st"), m.offset == 4 && m.length == 3);

    // Georgian
    // Mtavruli (Upper) -> Mkhedruli (Lower)
    // 'Ა' (U+1C90, E1 B2 90) -> 'ა' (U+10D0, E1 83 90)
    // Both are 3 bytes in UTF-8.
    // Georgian Context
    let_assert(auto m = str("Text Ა").utf8_uncased_search("ა"), m.offset == 5 && m.length == 3);

    // Cherokee
    // Cherokee Supplement (Lower, U+AB70, EA AD B0, 'ꭰ') -> Cherokee (Upper, U+13A0, E1 8E A0, 'Ꭰ')
    // Both 3 bytes.
    let_assert(auto m = str("ꭰ").utf8_uncased_search("Ꭰ"), m.offset == 0 && m.length == 3);

    // Cherokee Context
    let_assert(auto m = str("Syllable ꭰ").utf8_uncased_search("Ꭰ"), m.offset == 9 && m.length == 3);

    // Coptic (Extended)
    // Coptic Ⲡ (U+2C80, E2 B2 80) -> ⲡ (U+2C81, E2 B2 81)
    let_assert(auto m = str("Ⲡ").utf8_uncased_search("ⲡ"), m.offset == 0 && m.length == 3);

    // Glagolitic
    // Ⰰ (U+2C00, E2 B0 80) -> ⰰ (U+2C30, E2 B0 B0)
    let_assert(auto m = str("Ⰰ").utf8_uncased_search("ⰰ"), m.offset == 0 && m.length == 3);

    // Glagolitic Context
    let_assert(auto m = str("Letter Ⰰ").utf8_uncased_search("ⰰ"), m.offset == 7 && m.length == 3);

    // Caseless Scripts (CJK, Arabic, Hebrew, Emoji)
    // These generally don't fold, so they must match exactly or effectively be uncased by identity.

    // Arabic "Salam"
    assert(str("السلام").utf8_uncased_order("السلام") == sz_equal_k);

    // Hebrew "Shalom"
    assert(str("שלום").utf8_uncased_order("שלום") == sz_equal_k);

    // Numbers & Punctuation
    let_assert(auto m = str("12345!@#$%").utf8_uncased_search("345"), m.offset == 2 && m.length == 3);

    // Negative Tests
    // Not found in Cyrillic
    let_assert(auto m = str("Привет").utf8_uncased_search("xyz"), m.offset == str::npos);
    // Not found Cyrillic in ASCII
    let_assert(auto m = str("Hello World").utf8_uncased_search("При"), m.offset == str::npos);

    // CJK "Chinese"
    let_assert(auto m = str("中文测试").utf8_uncased_search("中文"), m.offset == 0 && m.length == 6);

    // Emoji
    let_assert(auto m = str("😀😁😂").utf8_uncased_search("😁"), m.offset == 4 && m.length == 4);

    // Emoji Context
    let_assert(auto m = str("smile 😀😁😂").utf8_uncased_search("😁"), m.offset == 10 && m.length == 4);

    // Regressions & Complex Cases
    // "Fuzz Regression": Needle "nԱԲՐԵշ" (Mixed case Armenian + ASCII)
    let_assert(auto m = str("nԱԲՐԵշ").utf8_uncased_search("nաբրեշ"), m.offset == 0 && m.length == 11);

    // Complex SIMD Regression Trigger
    // Needle includes: ǰ (Latin B), ẞ (Sharp S), Turkish ı, Emoji
    std::string complex_haystack =
        "\x66\x6F\x78\x74\xD0\xB2\x58\x77\x58\x20\x67\x31\x5A\xEF\xAC\x82\x46\x21\xC3\xA0\x31\x21\xC6\xA0\xEF\xAC" //
        "\x85\x57\x6F\x72\x6C\x64\xC4\x91\xE4\xB8\xAD\xE6\x96\x87\x43\xCF\x83\xE3\x81\x82\xE3\x81\x84\xD4\xB2\xD4" //
        "\xB1\xD5\x90\xD4\xB5\xD5\x8E\xC4\xB1\x6E\x32\xE4\xB8\xAD\xE6\x96\x87\x42\x30\x6E\xC3\x9F\x55\xCE\xBA\xCF" //
        "\x8C\xCF\x83\xCE\xBC\x30\x62\x72\x6F\x77\x6E\xCF\x83\x67\x66\x6F\x78\x21\xC2\xB5\x4D\xE4\xB8\xAD\xE6\x96" //
        "\x87\xC7\xB0\xE1\xBB\x86\xC4\xB0\x6A\x75\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80" //
        "\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3\xB1\x45\x7A\xC3\xBC\x49\x74\x68\x65\x61\xC5\xBF\xC3\x80\xC3\x85" //
        "\xD0\x91\xC5\xBF\x4C\x20\xC4\xB0\xCE\x91\x2C\x67\xE1\xBA\x96\xC3\xA0\x77\xC3\x91\x4D\x52\xE1\xBA\xA1\x4A" //
        "\xC6\xA0\xEF\xAC\x85\xE1\xBA\x9E\xF0\x9F\x98\x80\xEF\xAC\x80\xD0\xB1\xCF\x82\x65\x4B\x7A\xC3\xB1\x65\xC3" //
        "\x9C\x64\xC3\xB1\x55\xD0\xB0\xC3\xA4\x67\x41\x7A\xE1\xBB\x87\x5A\x4A\x71\x76\xC3\x89\xC6\xA0\x45\xCE\x91" //
        "\x66\x67\x6F\x41\xC3\x85\x4F\x6B\x58\xC3\xB1\x52\xE1\xBA\x98\xE1\xBA\xA1\x63\x47\xC2\xAA\xD4\xB2\xD4\xB1" //
        "\xD5\x90\xD4\xB5\xD5\x8E\xC3\x89\x77\x31\x46\xCF\x82\x76\xCE\xA3\x56\x56\xCA\xBE\xE1\xBA\x96\xD0\x91\x6F" //
        "\xCE\x92\x6A\x75\x6D\x70\x73\x33\xE1\xBA\xA1\x6A\x75\x6D\x70\x73\xE1\xBA\x98\xC3\x9F\xC3\x9C\xC6\xA1\x59" //
        "\xEF\xAC\x86\x59\x56\x2E\x33\xC3\xA9\x7A\x4C\x4C";

    std::string complex_needle =
        "\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3" //
        "\xB1\x45\x7A\xC3\xBC\x49\x74\x68\x65";

    let_assert(auto m = str(complex_haystack).utf8_uncased_search(complex_needle), m.length != 0);

    // Cross-Script Mixed Needles (Regression tests for kernel selection issues)

    // Capital Eszett (U+1E9E, E1 BA 9E) - folds to "ss"
    // Single Capital Eszett
    let_assert(auto m = str("\xE1\xBA\x9E").utf8_uncased_search("ss"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("ss").utf8_uncased_search("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Capital Eszett vs lowercase ß (C3 9F)
    let_assert(auto m = str("\xE1\xBA\x9E").utf8_uncased_search("\xC3\x9F"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("\xC3\x9F").utf8_uncased_search("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Double Capital Eszett
    let_assert(auto m = str("\xE1\xBA\x9E\xE1\xBA\x9E").utf8_uncased_search("ssss"), m.offset == 0 && m.length == 6);

    // Capital Eszett at boundaries
    let_assert(auto m = str("prefix\xE1\xBA\x9E" "suffix").utf8_uncased_search("xss"),
               m.offset == 5 && m.length == 4); // 'x'(1) + ẞ(3) = 4

    // Capital Eszett + Vietnamese (Western + Vietnamese kernels)
    // ẞ (E1 BA 9E) + ệ (E1 BB 87) - the exact failing pattern from fuzz tests
    let_assert(auto m = str("test\xE1\xBA\x9E\xE1\xBB\x87" "end").utf8_uncased_search("ss\xE1\xBB\x86"),
               m.offset == 4 && m.length == 6); // ẞ(3) + ệ(3) searched as ss + Ệ

    // Micro Sign + Greek (Western + Greek kernels)
    // µ (C2 B5) surrounded by Greek α (CE B1) and β (CE B2)
    let_assert(auto m = str("\xCE\xB1\xC2\xB5\xCE\xB2").utf8_uncased_search("\xCE\xB1\xCE\xBC\xCE\xB2"),
               m.offset == 0 && m.length == 6); // αµβ vs αμβ

    // Long S (C5 BF) + non-ASCII context
    let_assert(auto m = str("me\xC5\xBF\xC5\xBF" "age").utf8_uncased_search("MESSAGE"),
               m.offset == 0 && m.length == 9); // meſſage (9 bytes)

    // One-to-Many Expansions (U+1E96-1E9A range)
    // h with line below (U+1E96, E1 BA 96) -> h + combining line below (CC B1)
    let_assert(auto m = str("\xE1\xBA\x96").utf8_uncased_search("h\xCC\xB1"), m.offset == 0 && m.length == 3);

    // t with diaeresis (U+1E97, E1 BA 97) -> t + combining diaeresis (CC 88)
    let_assert(auto m = str("\xE1\xBA\x97").utf8_uncased_search("t\xCC\x88"), m.offset == 0 && m.length == 3);

    // w with ring above (U+1E98, E1 BA 98) -> w + combining ring above (CC 8A)
    let_assert(auto m = str("\xE1\xBA\x98").utf8_uncased_search("w\xCC\x8A"), m.offset == 0 && m.length == 3);

    // y with ring above (U+1E99, E1 BA 99) -> y + combining ring above (CC 8A)
    let_assert(auto m = str("\xE1\xBA\x99").utf8_uncased_search("y\xCC\x8A"), m.offset == 0 && m.length == 3);

    // Kelvin Sign (E2 84 AA) in mixed context
    let_assert(auto m = str("273 \xE2\x84\xAA test").utf8_uncased_search("273 k"),
               m.offset == 0 && m.length == 7); // K is 3 bytes

    // Angstrom Sign (E2 84 AB) with accented chars
    let_assert(auto m = str("10 \xE2\x84\xAB unit").utf8_uncased_search("10 \xC3\xA5"),
               m.offset == 0 && m.length == 6); // Å (3) vs å (2)

    // 64-byte Boundary Stress Tests

    // Capital Eszett at position 63 (just at SIMD boundary)
    {
        std::string prefix(63, 'x');
        let_assert(auto m = str((prefix + "\xE1\xBA\x9E" "end").c_str()).utf8_uncased_search("xss"),
                   m.offset == 62 && m.length == 4); // last 'x' + ẞ(3)
    }

    // Vietnamese char at position 62
    {
        std::string prefix(62, 'a');
        let_assert(auto m = str((prefix + "\xE1\xBB\x87" "b").c_str()).utf8_uncased_search("\xE1\xBB\x86" "B"),
                   m.offset == 62 && m.length == 4); // ệ(3) + b(1)
    }

    // Micro Sign at position 64 (just past SIMD boundary)
    {
        std::string prefix(64, 'z');
        let_assert(auto m = str((prefix + "\xC2\xB5" "test").c_str()).utf8_uncased_search("\xCE\xBC"),
                   m.offset == 64 && m.length == 2); // µ matches μ
    }

    // 'ﬄ' at position 63 (just at SIMD boundary), matching from inside its fold.
    {
        std::string prefix(63, 'x');
        let_assert(auto m = str((prefix + "\xEF\xAC\x84" "end").c_str()).utf8_uncased_search("fl"),
                   m.offset == 63 && m.length == 3); // consume whole ligature
    }

    // ASCII + ligature spanning the SIMD boundary: 'P' at 62 and 'ﬄ' at 63.
    {
        std::string prefix(62, 'x');
        let_assert(auto m = str((prefix + "P\xEF\xAC\x84" "end").c_str()).utf8_uncased_search("pf"),
                   m.offset == 62 && m.length == 4); // "P"(1) + "ﬄ"(3)
    }

    // Basic ASCII search
    let_assert(auto m = str("Hello World").utf8_uncased_search("WORLD"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("Hello World").utf8_uncased_search("world"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("HELLO").utf8_uncased_search("hello"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("Hello").utf8_uncased_search("xyz"), m.offset == str::npos);
    let_assert(auto m = str("Hello").utf8_uncased_search(""), m.offset == 0 && m.length == 0);

    // Fuzz-Discovered Regressions (Serial vs SIMD mismatches)
    // These patterns were discovered by the find fuzzers and expose
    // disagreements between serial and SIMD implementations.

    // Pattern 0: Ligature tail-match in mixed-case context (historical verify crash).
    // Haystack: C3 96 45 47 76 C3 91 2C 50 EF AC 84 ... EF AC 82 70
    // Needle:   67 76 C3 B1 2C 70 66
    {
        let_assert(
            auto m =
                str("\xC3\x96" "EGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p").utf8_uncased_search("gv\xC3\xB1,pf"),
            m.offset == 3 && m.length == 9);
        let_assert(
            auto m = str("\xC3\x96" "EGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p").utf8_uncased_search("pf"),
            m.offset == 8 && m.length == 4);
    }

    // Pattern 1: "st" + Latin-1 char (st ligature expansion issue?)
    // Needle: 73 74 C2 BA = "st" + º (masculine ordinal indicator)
    // Needle: 73 74 C3 B1 = "st" + ñ
    // Needle: 73 74 C3 A5 = "st" + å
    // Needle: 73 74 C3 A9 = "st" + é
    // Needle: 73 74 D5 A2 = "st" + Armenian բ
    // Needle: 73 74 CE B1 = "st" + Greek α
    // These trigger kernel=2 (Central Europe) with safe_window issues
    {
        // "st" followed by º - should this match "st" ligature + º?
        let_assert(auto m = str("test\xEF\xAC\x85\xC2\xBA" "end").utf8_uncased_search("st\xC2\xBA"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + º (2)

        // "st" followed by ñ
        let_assert(auto m = str("test\xEF\xAC\x85\xC3\xB1" "end").utf8_uncased_search("st\xC3\xB1"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + ñ (2)

        // "st" followed by Greek α
        let_assert(auto m = str("prefix\xEF\xAC\x85\xCE\xB1" "suffix").utf8_uncased_search("st\xCE\xB1"),
                   m.offset == 6 && m.length == 5); // st ligature (3) + α (2)
    }

    // Pattern 2: "ss" + Latin-1/Greek (Eszett expansion)
    // Needle: 73 73 CE B1 = "ss" + Greek α
    // Needle: 73 73 C3 A5 = "ss" + å
    {
        // "ss" followed by Greek α - should match ß + α
        let_assert(auto m = str("test\xC3\x9F\xCE\xB1" "end").utf8_uncased_search("ss\xCE\xB1"),
                   m.offset == 4 && m.length == 4); // ß (2) + α (2)

        // "ss" followed by å
        let_assert(auto m = str("prefix\xC3\x9F\xC3\xA5" "suffix").utf8_uncased_search("ss\xC3\xA5"),
                   m.offset == 6 && m.length == 4); // ß (2) + å (2)
    }

    // Pattern 3: ASCII + combining diacritical + other char
    // Needle: 68 CC B1 D5 A5 = "h" + combining macron below + Armenian ե
    // Needle: 77 CC 8A CE B2 = "w" + combining ring above + Greek β
    // Needle: 6A CC 8C D5 A2 = "j" + combining caron + Armenian բ
    // These test one-to-many expansions (U+1E96 range) mixed with other scripts
    {
        // h + combining macron below should match ẖ (U+1E96)
        let_assert(auto m = str("\xE1\xBA\x96\xD5\xA5").utf8_uncased_search("h\xCC\xB1\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẖ (3) + ե (2)
        // Needle starts with the combining mark (mid-expansion of ẖ).
        let_assert(auto m = str("\xE1\xBA\x96\xD5\xA5").utf8_uncased_search("\xCC\xB1\xD5\xA5"),
                   m.offset == 0 && m.length == 5);

        // w + combining ring above should match ẘ (U+1E98)
        let_assert(auto m = str("\xE1\xBA\x98\xCE\xB2").utf8_uncased_search("w\xCC\x8A\xCE\xB2"),
                   m.offset == 0 && m.length == 5); // ẘ (3) + β (2)
        // Needle starts with the combining mark (mid-expansion of ẘ).
        let_assert(auto m = str("\xE1\xBA\x98\xCE\xB2").utf8_uncased_search("\xCC\x8A\xCE\xB2"),
                   m.offset == 0 && m.length == 5);

        // j + combining caron should match ǰ (U+01F0)
        let_assert(auto m = str("\xC7\xB0\xD5\xA2").utf8_uncased_search("j\xCC\x8C\xD5\xA2"),
                   m.offset == 0 && m.length == 4); // ǰ (2) + բ (2)
        // Needle starts with the combining mark (mid-expansion of ǰ).
        let_assert(auto m = str("\xC7\xB0\xD5\xA2").utf8_uncased_search("\xCC\x8C\xD5\xA2"),
                   m.offset == 0 && m.length == 4);
    }

    // Pattern 4: Modifier letters + other chars
    // Needle: CA BC 6E CE BC = modifier apostrophe + "n" + Greek μ
    // Needle: 61 CA BE D5 A5 = "a" + modifier right half ring + Armenian ե
    // These test n-apostrophe (U+0149) and a-right-half-ring (U+1E9A) expansions
    {
        // 'n (U+0149) expands to modifier apostrophe + n
        let_assert(auto m = str("\xC5\x89\xCE\xBC").utf8_uncased_search("\xCA\xBC" "n\xCE\xBC"),
                   m.offset == 0 && m.length == 4); // ʼn (2) + μ (2)
        // Needle starts at the second rune of the expansion ("n..."), so it still anchors to 'ŉ'.
        let_assert(auto m = str("\xC5\x89\xCE\xBC").utf8_uncased_search("n\xCE\xBC"), m.offset == 0 && m.length == 4);

        // a + modifier right half ring should match ẚ (U+1E9A)
        let_assert(auto m = str("\xE1\xBA\x9A\xD5\xA5").utf8_uncased_search("a\xCA\xBE\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẚ (3) + ե (2)
        // Needle starts at the second rune of the expansion ("ʾ..."), so it still anchors to 'ẚ'.
        let_assert(auto m = str("\xE1\xBA\x9A\xD5\xA5").utf8_uncased_search("\xCA\xBE\xD5\xA5"),
                   m.offset == 0 && m.length == 5);
    }

    // Pattern 5: Armenian + combining chars / ligatures
    // Needle: D5 A5 D6 82 CE B2 = Armenian ech+yiwn + Greek β
    {
        // Armenian ech+yiwn characters followed by Greek
        let_assert(auto m = str("\xD5\xA5\xD6\x82\xCE\xB2").utf8_uncased_search("\xD5\xA5\xD6\x82\xCE\xB2"),
                   m.offset == 0 && m.length == 6);
    }

    // Pattern 6: Long complex needles crossing multiple scripts
    // These stress test the kernel selection and danger zone handling
    {
        // Armenian barev + Latin ligatures + Vietnamese
        std::string haystack = "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE" // barev
                               "\xEF\xAC\x83"                             // ffi ligature
                               "\xE1\xBB\x87";                            // Vietnamese ệ
        std::string needle = "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE"   // barev
                             "ffi"                                        // expanded
                             "\xE1\xBB\x86";                              // Vietnamese Ệ
        let_assert(auto m = str(haystack).utf8_uncased_search(needle), m.offset == 0 && m.length == 16);
    }

    // Long needle tests at ring buffer boundary (32 folded runes)
    // The serial implementation uses a 32-rune ring buffer for fold comparisons
    {
        // Exactly 32 ASCII characters (32 folded runes)
        std::string hay32(32, 'a');
        let_assert(auto m = str(hay32 + "xyz").utf8_uncased_search(hay32), m.offset == 0 && m.length == 32);

        // 33 ASCII characters (crosses ring buffer boundary)
        std::string hay33(33, 'a');
        let_assert(auto m = str(hay33 + "xyz").utf8_uncased_search(hay33), m.offset == 0 && m.length == 33);

        // 16 eszett characters → 32 folded runes (ss×16), exactly at boundary
        std::string hay_16_ss(16, '\xC3');
        for (size_t i = 0; i < 16; ++i) hay_16_ss.insert(i * 2 + 1, 1, '\x9F'); // Build "ßßßßßßßßßßßßßßßß"
        std::string needle_32_s(32, 's');
        let_assert(auto m = str(hay_16_ss + "end").utf8_uncased_search(needle_32_s), m.offset == 0 && m.length == 32);

        // 64 ASCII characters (tests double boundary)
        std::string hay64(64, 'b');
        let_assert(auto m = str(hay64 + "xyz").utf8_uncased_search(hay64), m.offset == 0 && m.length == 64);
    }

    // Eszett at SIMD 64-byte chunk boundaries
    {
        // ß at position 62 (ends exactly at 64-byte boundary)
        std::string prefix62(62, 'a');
        let_assert(auto m = str(prefix62 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 62 && m.length == 2);

        // ß straddling 64-byte boundary (starts at 63)
        std::string prefix63(63, 'a');
        let_assert(auto m = str(prefix63 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 63 && m.length == 2);

        // ß exactly at 64-byte boundary
        std::string prefix64(64, 'a');
        let_assert(auto m = str(prefix64 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 64 && m.length == 2);

        // Word with ß crossing boundary: "straße" starting at position 60
        std::string prefix60(60, 'a');
        let_assert(auto m = str(prefix60 + "stra\xC3\x9F" "e" + "zzz").utf8_uncased_search("strasse"),
                   m.offset == 60 && m.length == 7);
    }

    // Cross-script boundary tests (different SIMD kernels)
    {
        // ASCII → Greek transition at SIMD boundary
        std::string ascii60(60, 'x');
        let_assert(auto m = str(ascii60 + "\xCE\xB1\xCE\xB2\xCE\xB3").utf8_uncased_search("\xCE\x91\xCE\x92\xCE\x93"),
                   m.offset == 60 && m.length == 6); // ΑΒΓ matching αβγ

        // Latin-1 → Cyrillic transition
        std::string latin58(58, '\xC3');                                      // Build Latin-1 prefix
        for (size_t i = 0; i < 58; ++i) latin58.insert(i * 2 + 1, 1, '\xA4'); // "äääää..."
        // This creates 116-byte prefix of ä characters
    }

    // Minimal Divergence Cases (Ice Lake vs Serial)
    // These were discovered by multi-seed fuzzing and represent minimal inputs
    // that previously caused Serial/SIMD disagreement.

    // Pattern 7: "sss" prefix matching "Sß" (seed 5678, Kernel 2)
    // Haystack: "brown Sßà jumps" - bytes at [6]: 53 C3 9F C3 A0 ("Sßà") = 5 bytes
    // Needle: "sssà" - bytes: 73 73 73 C3 A0 = 5 bytes
    // 'S' → 's', 'ß' → "ss", 'à' → 'à', so "Sßà" → "sssà" (should match!)
    {
        // Simple case: "Sßà" should match "sssà" - match is 5 bytes (53 C3 9F C3 A0)
        let_assert(auto m = str("brown S\xC3\x9F\xC3\xA0 jumps").utf8_uncased_search("sss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Lowercase: "sßà" should match "sssà" - match is 5 bytes
        let_assert(auto m = str("brown s\xC3\x9F\xC3\xA0 jumps").utf8_uncased_search("sss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Uppercase ß (U+1E9E) when it exists - "ẞà" should match "ssà"
        let_assert(auto m = str("brown \xE1\xBA\x9E\xC3\xA0 jumps").utf8_uncased_search("ss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Triple-s with space (seed 1234): "sß " should match "sss "
        // Match starts at byte 7 where 's' is (byte 6 is space before 's')
        let_assert(auto m = str("\xC7\xB0" "bee3 s\xC3\x9F ee\xC3\xA9 nc").utf8_uncased_search("sss ee\xC3\xA9"),
                   m.offset == 7 && m.length == 8);

        // "ss" needle vs "ß" haystack (basic case)
        let_assert(auto m = str("\xC3\x9F" "abc").utf8_uncased_search("ssabc"), m.offset == 0 && m.length == 5);

        // "sss" needle vs "sß" haystack
        let_assert(auto m = str("s\xC3\x9F" "abc").utf8_uncased_search("sssabc"), m.offset == 0 && m.length == 6);
    }

    // Pattern 8: Greek Mu UTF-8 boundary (seed 300, 1000, 1700, Kernel 5)
    // Needle: CE BC (Greek μ - U+03BC)
    // Bug: SIMD was incorrectly matching mid-byte BC as standalone
    // Fix: Ensure proper UTF-8 character boundary validation
    {
        // Simple Greek mu search
        let_assert(auto m = str("hello \xCE\xBC world").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 6 && m.length == 2);

        // Greek mu NOT at position where 0xBC appears as second byte of another char
        // Create haystack with Latin-1 char ending in 0xBC, then Greek mu
        // This ensures we only match at valid UTF-8 boundaries
        let_assert(auto m = str("test \xC2\xBC thing \xCE\xBC end").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 14 && m.length == 2); // Only at actual μ, not at ¼

        // Multiple Greek chars around mu
        let_assert(auto m = str("\xCE\xB1\xCE\xBC\xCE\xB2").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 2 && m.length == 2);
    }

    // Pattern 9: Cyrillic Moscow case folding (seed 9999, 44444, 55555, Kernel 4)
    // Haystack: "се Москва" (uppercase М - D0 9C)
    // Needle: "се москва" (lowercase м - D0 BC)
    // Should match uncasedly
    {
        // Simple Moscow: Москва vs москва
        let_assert(auto m = str("\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0")
                                .utf8_uncased_search("\xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                   m.offset == 0 && m.length == 12);

        // Moscow with Latin prefix
        let_assert(auto m = str("se \xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0")
                                .utf8_uncased_search("se \xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                   m.offset == 0 && m.length == 15);

        // All Cyrillic uppercase vs lowercase
        let_assert(auto m = str("\xD0\x90\xD0\x91\xD0\x92")                       // АБВ
                                .utf8_uncased_search("\xD0\xB0\xD0\xB1\xD0\xB2"), // абв
                   m.offset == 0 && m.length == 6);

        // Mixed: ПРИВЕТ vs привет
        let_assert(auto m = str("\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2")                       // ПРИВЕТ
                                .utf8_uncased_search("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82"), // привет
                   m.offset == 0 && m.length == 12);
    }

    // Pattern 10: Ligature fi expansion (seed 500, Kernel 2)
    // Haystack contains ﬁ (EF AC 81 - U+FB01)
    // Needle has "fi" (66 69)
    // ﬁ should case-fold to "fi"
    {
        // Simple: ﬁ vs fi
        let_assert(auto m = str("\xEF\xAC\x81" "nd").utf8_uncased_search("find"), m.offset == 0 && m.length == 5);

        // With uppercase: ﬁ vs FI
        let_assert(auto m = str("\xEF\xAC\x81" "nd").utf8_uncased_search("FInd"), m.offset == 0 && m.length == 5);

        // ff ligature: ﬀ (EF AC 80) vs ff
        let_assert(auto m = str("\xEF\xAC\x80" "oo").utf8_uncased_search("ffoo"), m.offset == 0 && m.length == 5);

        // ffi ligature: ﬃ (EF AC 83) vs ffi
        let_assert(auto m = str("\xEF\xAC\x83" "ce").utf8_uncased_search("ffice"), m.offset == 0 && m.length == 5);

        // fl ligature: ﬂ (EF AC 82) vs fl
        let_assert(auto m = str("\xEF\xAC\x82" "oor").utf8_uncased_search("floor"), m.offset == 0 && m.length == 6);

        // ffl ligature: ﬄ (EF AC 84) vs ffl
        let_assert(auto m = str("wa\xEF\xAC\x84" "e").utf8_uncased_search("waffle"), m.offset == 0 && m.length == 6);
    }

    // Pattern 11: Combining marks vs precomposed (seed 123, 42, Kernel 2)
    // j + combining caron (6A CC 8C) vs ǰ (C7 B0 - U+01F0)
    // These are canonically equivalent in Unicode
    // Note: StringZilla may or may not perform normalization - document behavior
    {
        // Precomposed ǰ vs decomposed j+caron
        // If normalization is performed, these should match
        // If not, they won't match (current behavior TBD)
        let_assert(auto m = str("\xC7\xB0" "ump").utf8_uncased_search("j\xCC\x8C" "ump"),
                   m.offset == 0 && m.length == 5);

        // é precomposed (C3 A9) vs e+acute (65 CC 81)
        let_assert(auto m = str("\xC3\xA9" "lan").utf8_uncased_search("e\xCC\x81" "lan"), m.offset == str::npos);
    }

    // Pattern 12: Mixed script verification (seeds 456, 789, 22222, Kernels 3, 5, 6)
    // These test that case folding works correctly when multiple scripts are mixed
    {
        // Greek κόσμ mixed with Latin
        let_assert(auto m = str("brown \xCE\xBA\xCF\x8C\xCF\x83 end").utf8_uncased_search("\xCE\xBA\xCF\x8C\xCF\x83"),
                   m.offset == 6 && m.length == 6);

        // Greek sigma case: Σ (CE A3) vs σ (CF 83) vs ς (CF 82 - final sigma)
        let_assert(auto m = str("\xCE\xA3\xCE\xB5").utf8_uncased_search("\xCF\x83\xCE\xB5"),
                   m.offset == 0 && m.length == 4);

        // Armenian + Latin mixed
        let_assert(auto m = str("test \xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE world")
                                .utf8_uncased_search("\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE"),
                   m.offset == 5 && m.length == 10);

        // Armenian ligature: և (D6 87 - U+0587) vs ե+ւ (D5 A5 D6 82)
        let_assert(auto m = str("\xD6\x87" "nd").utf8_uncased_search("\xD5\xA5\xD6\x82" "nd"),
                   m.offset == 0 && m.length == 4);
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps a UTF-8 case-folding backend by its kernel pointer. */
template <sz_utf8_uncased_fold_t fold_>
struct uncased_fold_from_sz_ {
    sz_size_t operator()(sz_cptr_t text, sz_size_t length, sz_ptr_t output) const noexcept {
        return fold_(text, length, output);
    }
};

/** @brief Wraps a UTF-8 uncased-find backend by its kernel pointer. */
template <sz_utf8_uncased_search_t find_>
struct uncased_find_from_sz_ {
    sz_cptr_t operator()(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
                         sz_utf8_uncased_needle_metadata_t *metadata, sz_size_t *matched_length) const noexcept {
        return find_(haystack, haystack_length, needle, needle_length, metadata, matched_length);
    }
};

/**
 *  @brief Tests equivalence of case folding implementations (reference vs candidate).
 *
 *  Folds a broad battery of fixed UTF-8 strings, then random concatenations of them, and finally the
 *  exhaustive sweep of every valid Unicode codepoint (in order and shuffled), comparing the reference
 *  and candidate backends byte-by-byte. The random-input coverage that the standalone fuzz routine
 *  used to provide is folded into this single differential, so the driver only needs this one call.
 *
 *  Generates random UTF-8 strings containing:
 *  - ASCII text (uppercase and lowercase)
 *  - Multi-byte UTF-8 characters from various scripts (Cyrillic, Greek, Latin Extended)
 *  - Special cases like German ß
 *
 *  @param reference          Reference case-folding backend wrapper.
 *  @param candidate          Candidate case-folding backend wrapper to validate against the reference.
 *  @param min_text_length    Lower bound on the length of each random concatenation.
 *  @param min_iterations     Number of random concatenations to fold and compare.
 */
template <typename reference_, typename candidate_>
void test_fold_equivalence(reference_ reference, candidate_ candidate, sz_size_t min_text_length,
                           sz_size_t min_iterations) {

    // Output buffers (3x input for worst-case expansion)
    std::vector<char> output_base(min_text_length * 3 + 256);
    std::vector<char> output_simd(min_text_length * 3 + 256);

    auto check = [&](std::string const &text) {
        // Ensure buffers are large enough
        if (output_base.size() < text.size() * 3 + 64) {
            output_base.resize(text.size() * 3 + 64);
            output_simd.resize(text.size() * 3 + 64);
        }

        sz_size_t len_base = reference(text.data(), text.size(), output_base.data());
        sz_size_t len_simd = candidate(text.data(), text.size(), output_simd.data());

        if (len_base != len_simd) {
            std::fprintf(stderr, "Case fold length mismatch: base=%zu, simd=%zu, input_len=%zu\n", //
                         len_base, len_simd, text.size());
            // Print first divergence
            for (std::size_t i = 0; i < std::min(len_base, len_simd); ++i) {
                if (output_base[i] != output_simd[i]) {
                    std::fprintf(stderr, "First byte diff at output[%zu]: base=0x%02X, simd=0x%02X\n", //
                                 i, (unsigned char)output_base[i], (unsigned char)output_simd[i]);
                    break;
                }
            }
            assert(len_base == len_simd && "Case fold length mismatch");
        }

        for (sz_size_t i = 0; i < len_base; ++i) {
            if (output_base[i] != output_simd[i]) {
                std::fprintf(stderr, "Case fold content mismatch at byte %zu: base=0x%02X, simd=0x%02X\n", //
                             i, (unsigned char)output_base[i], (unsigned char)output_simd[i]);
                // Show context around the mismatch
                std::size_t start = i > 10 ? i - 10 : 0;
                std::size_t end = std::min(i + 10, (std::size_t)len_base);
                std::fprintf(stderr, "Base output[%zu..%zu]: ", start, end);
                for (std::size_t j = start; j < end; ++j) std::fprintf(stderr, "%02X ", (unsigned char)output_base[j]);
                std::fprintf(stderr, "\nSIMD output[%zu..%zu]: ", start, end);
                for (std::size_t j = start; j < end; ++j) std::fprintf(stderr, "%02X ", (unsigned char)output_simd[j]);
                std::fprintf(stderr, "\n");
                assert(output_base[i] == output_simd[i] && "Case fold content mismatch");
            }
        }
    };

    // Test content - mix of scripts with case folding rules
    static char const *utf8_content[] = {
        // ASCII
        "",
        "a",
        "A",
        "hello",
        "HELLO",
        "Hello World",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghijklmnopqrstuvwxyz",
        "0123456789",
        // German Eszett (both ß and ẞ fold to "ss")
        "\xC3\x9F", // ß (U+00DF) → ss
        "straße",
        // Latin-1 uppercase (À-Þ range, 2-byte UTF-8 starting with C3)
        "\xC3\x80", // À (U+00C0)
        "\xC3\x89", // É (U+00C9)
        "\xC3\x96", // Ö (U+00D6)
        "\xC3\x9C", // Ü (U+00DC)
        "\xC3\x9E", // Þ (U+00DE)
        // Cyrillic (2-byte UTF-8 starting with D0-D1)
        "\xD0\x90",                                         // А (U+0410)
        "\xD0\x9F",                                         // П (U+041F)
        "\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2", // ПРИВЕТ
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", // привет
        // Cyrillic Special
        "\xD0\x81", // Ё
        "\xD1\x91", // ё
        "\xD0\x84", // Є
        "\xD1\x94", // є
        "\xD0\x87", // Ї
        "\xD1\x97", // ї
        // Greek (2-byte UTF-8 starting with CE-CF)
        "\xCE\x91",                                         // Α (U+0391)
        "\xCE\xA9",                                         // Ω (U+03A9)
        "\xCE\x95\xCE\xBB\xCE\xBB\xCE\xAC\xCE\xB4\xCE\xB1", // Ελλάδα
        // Armenian (2-byte UTF-8 starting with D4-D5)
        "\xD4\xB1", // Ա (U+0531)
        "\xD5\x80", // Հ (U+0540)
        // Mixed content
        "Hello \xD0\x9C\xD0\xB8\xD1\x80!",      // Hello Мир!
        "Caf\xC3\xA9 \xCE\xB1\xCE\xB2\xCE\xB3", // Café αβγ
        // Georgian uppercase (3-byte UTF-8: E1 82 A0-BF, E1 83 80-85/87/8D)
        // These fold to lowercase Mkhedruli (E2 B4 XX)
        "\xE1\x82\xA0",                         // Ⴀ (U+10A0) → ა (U+2D00)
        "\xE1\x82\xB0",                         // Ⴐ (U+10B0) → ⴐ (U+2D10)
        "\xE1\x83\x80",                         // Ⴠ (U+10C0) → ⴠ (U+2D20)
        "\xE1\x83\x85",                         // Ⴥ (U+10C5) → ⴥ (U+2D25)
        "\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2", // ႠႡႢ → ⴀⴁⴂ
        "\xE1\x83\x90\xE1\x83\x91\xE1\x83\x92", // ა ბ გ (lowercase, no change)
        // Georgian mixed with ASCII (tests fast-path interaction)
        ("Hello \xE1\x82\xA0\xE1\x82\xA1 World"),          // Hello ႠႡ World
        ("ABC\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2" "DEF"), // ABCႠႡႢdef
        // Emojis (no case folding, should pass through)
        "\xF0\x9F\x98\x80",             // 😀
        "Hello \xF0\x9F\x8C\x8D World", // Hello 🌍 World
    };

    auto &rng = global_random_generator();
    std::size_t const content_count = sizeof(utf8_content) / sizeof(utf8_content[0]);
    std::uniform_int_distribution<std::size_t> content_dist(0, content_count - 1);

    // First, test all the fixed strings
    for (std::size_t i = 0; i < content_count; ++i) { check(utf8_content[i]); }

    // Generate and test many random strings
    for (std::size_t iteration = 0; iteration < min_iterations; ++iteration) {
        std::string text;

        // Build up a random string of at least `min_text_length` bytes
        while (text.size() < min_text_length) {
            std::size_t content_index = content_dist(rng);
            text.append(utf8_content[content_index]);
        }
        check(text);
    }

    // Exhaustive sweep over every valid Unicode codepoint: first in order (0x0..0x10FFFF), then shuffled.
    // This is the random-input coverage formerly provided by a standalone fuzz routine, folded in here so
    // a single fold differential drives both the structured strings and the whole-codepoint enumeration.
    std::printf("  - testing case folding fuzz (ordered + shuffled codepoint sweep)...\n");
    std::vector<sz_rune_t> all_runes;
    all_runes.reserve(0x10FFFF);
    for (sz_rune_t codepoint = 0; codepoint <= 0x10FFFF; ++codepoint) {
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) continue; // Skip surrogates
        all_runes.push_back(codepoint);
    }

    std::vector<char> input_buffer(all_runes.size() * 4); // Max UTF-8 size is 4 bytes per rune
    std::size_t const sweep_iterations = 1 + scale_iterations(100);
    for (std::size_t iteration = 0; iteration < sweep_iterations; ++iteration) {
        if (iteration > 0) std::shuffle(all_runes.begin(), all_runes.end(), rng);

        char *write_cursor = input_buffer.data();
        for (sz_rune_t codepoint : all_runes) write_cursor += sz_rune_encode(codepoint, (sz_u8_t *)write_cursor);
        std::string const text(input_buffer.data(), (std::size_t)(write_cursor - input_buffer.data()));
        check(text);
    }
    std::printf("    exhaustive fuzzing passed!\n");
}

/**
 *  @brief Closure property of the case-invariant classifier over the Unicode fold table.
 *
 *  A rune may be treated as case-invariant only if no uncased match can start or hide
 *  inside it. That demands two closures over `sz_unicode_fold_codepoint_`: every preimage with a
 *  non-identity fold participates in case, and every rune @b emitted by such a fold can appear
 *  inside a folded expansion (like 'ʾ' U+02BE inside 'ẚ' → "aʾ"), so neither may be invariant.
 *  Fully generative: a Unicode table update re-derives the expected set automatically.
 */
void test_uncased_invariant_reference() {

    std::printf("  - testing case-invariant closure over the fold table...\n");
    std::size_t preimages_checked = 0, outputs_checked = 0;

    for (sz_rune_t rune = 0; rune <= 0x10FFFF; ++rune) {
        if (rune >= 0xD800 && rune <= 0xDFFF) continue; // Surrogates aren't valid UTF-8
        sz_rune_t folded_runes[3];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        if (folded_count == 1 && folded_runes[0] == rune) continue; // Identity folds impose no constraint

        if (sz_rune_is_uncased_(rune) != sz_false_k) {
            std::fprintf(stderr, "Fold preimage U+%04X is wrongly classified as case-invariant\n", rune);
            assert(false && "Fold preimages must not be case-invariant");
        }
        ++preimages_checked;

        for (sz_size_t i = 0; i < folded_count; ++i) {
            if (sz_rune_is_uncased_(folded_runes[i]) != sz_false_k) {
                std::fprintf(stderr,
                             "Fold output U+%04X (from preimage U+%04X) is wrongly classified as case-invariant\n",
                             folded_runes[i], rune);
                assert(false && "Fold outputs must not be case-invariant");
            }
            ++outputs_checked;
        }
    }
    std::printf("    passed %zu preimages and %zu fold-output runes\n", preimages_checked, outputs_checked);
}

#pragma endregion // Equivalence

#pragma region Safety

/** @brief Wraps the uncased fold/find/violation kernels of one backend by their pointers. */
template <sz_utf8_uncased_fold_t fold_, sz_utf8_uncased_search_t find_, sz_utf8_find_cased_t violation_>
struct uncased_from_sz_ {
    sz_size_t fold(sz_cptr_t text, sz_size_t length, sz_ptr_t output) const noexcept {
        return fold_(text, length, output);
    }
    sz_cptr_t find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
                   sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) const noexcept {
        return find_(haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }
    sz_cptr_t violation(sz_cptr_t text, sz_size_t length) const noexcept { return violation_(text, length); }
};

/**
 *  @brief Smoke test feeding invalid UTF-8 into the case kernels of one backend.
 *
 *  The valid-UTF-8 contract stands, so outputs may be arbitrary - the only requirements are:
 *  no crash, the fold output stays within the documented 3x expansion bound, and no writes land
 *  outside the destination buffer. One nuance: a truncated multi-byte sequence at the very end
 *  of the input can mis-decode as a single rune and emit up to 4 bytes where fewer source bytes
 *  remain, so the enforced bound is `3 * input_length + 4`. Canary bytes guard both sides of
 *  the destination, like the audit's standalone safety probe did. Covers all 256 single bytes,
 *  all 65,536 byte pairs, and random garbage buffers of 1..70 bytes through fold, find (with a
 *  short valid needle), and the case-invariant check.
 */
template <typename candidate_>
static void check_uncased_safety_(candidate_ candidate, std::size_t random_inputs = scale_iterations(10000)) {

    std::printf("  - testing invalid-input safety of case kernels (%zu random buffers)...\n", random_inputs);

    std::size_t const max_input_length = 70;
    char const *needle = "st"; // Short valid needle: the folds of 'ﬅ' and 'ﬆ' collapse onto it

    auto check = [&](char const *input, std::size_t input_length) {
        // The documented bound is 3x for valid UTF-8; invalid input may add one mis-decoded
        // trailing rune of up to 4 bytes on top of it
        std::size_t const output_bound = input_length * 3 + 4;
        // A canary-guarded fold output buffer catches any write past the documented bound; the helper
        // asserts the flanking guards survive the call.
        with_guarded_buffer_(output_bound, [&](sz_ptr_t output, std::size_t length) {
            sz_size_t folded_length = candidate.fold(input, input_length, output);
            if (folded_length > length) {
                std::fprintf(stderr, "Fold of invalid input returned %zu bytes for %zu input bytes (bound is 3x + 4)\n",
                             (std::size_t)folded_length, input_length);
                print_uncased_test_bytes_("input", input, input_length);
                assert(false && "Fold output must stay within 3x the input length plus one mis-decoded rune");
            }
        });
        // The classifier and the finder return arbitrary verdicts on garbage - they just must survive
        [[maybe_unused]] sz_cptr_t const violation = candidate.violation(input, input_length);
        sz_utf8_uncased_needle_metadata_t needle_metadata = {};
        sz_size_t matched_length = 0;
        [[maybe_unused]] sz_cptr_t const match = candidate.find(input, input_length, needle, 2, &needle_metadata,
                                                                &matched_length);
    };

    char input[max_input_length];

    // All 256 single bytes: truncated leads, stray continuations, 0xFE/0xFF
    for (std::size_t byte = 0; byte < 256; ++byte) {
        input[0] = (char)byte;
        check(input, 1);
    }

    // All 65,536 byte pairs: every lead × continuation interaction, including overlong shapes
    for (std::size_t first_byte = 0; first_byte < 256; ++first_byte)
        for (std::size_t second_byte = 0; second_byte < 256; ++second_byte) {
            input[0] = (char)first_byte;
            input[1] = (char)second_byte;
            check(input, 2);
        }

    // Random garbage buffers spanning whole SIMD chunks
    auto &rng = global_random_generator();
    std::uniform_int_distribution<std::size_t> length_distribution(1, max_input_length);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (std::size_t iteration = 0; iteration < random_inputs; ++iteration) {
        std::size_t input_length = length_distribution(rng);
        for (std::size_t index = 0; index < input_length; ++index) input[index] = (char)byte_distribution(rng);
        check(input, input_length);
    }

    std::printf("    passed %zu cases (256 singles + 65536 pairs + %zu random)\n", //
                256 + 65536 + random_inputs, random_inputs);
}

/**
 *  @brief Adversarial invalid-input safety driver across every backend compiled on this target.
 *
 *  Mirrors `test_memory_safety()` / `test_utf8_runes_safety()`: the registered no-arg driver owns the per-ISA
 *  ladder while the file-local `check_uncased_safety_` checker runs the actual probes. The serial backend
 *  faces the same invalid-input contract as the SIMD ones, then one `#if SZ_USE_*` block per ISA.
 */
void test_uncased_safety() {

    check_uncased_safety_(
        uncased_from_sz_<sz_utf8_uncased_fold_serial, sz_utf8_uncased_search_serial, sz_utf8_find_cased_serial> {});
#if SZ_USE_HASWELL
    check_uncased_safety_(
        uncased_from_sz_<sz_utf8_uncased_fold_haswell, sz_utf8_uncased_search_haswell, sz_utf8_find_cased_haswell> {});
#endif
#if SZ_USE_ICELAKE
    check_uncased_safety_(
        uncased_from_sz_<sz_utf8_uncased_fold_icelake, sz_utf8_uncased_search_icelake, sz_utf8_find_cased_icelake> {});
#endif
#if SZ_USE_NEON
    check_uncased_safety_(
        uncased_from_sz_<sz_utf8_uncased_fold_neon, sz_utf8_uncased_search_neon, sz_utf8_find_cased_neon> {});
#endif
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief Drives the serial-vs-SIMD uncased fold/find differentials and the structured adversarial find
 *         enumerators across every backend compiled on this target.
 *
 *  Mirrors the per-ISA ladder of the legacy `test_equivalence`: the backend-independent fold-table
 *  closure always runs, then each `#if SZ_USE_*` block exercises that ISA's fold equivalence
 *  (serial = reference, ISA = candidate) and its full find battery. The invalid-input safety probes
 *  live in their own registered driver, `test_uncased_safety`.
 */
void test_uncased_all() {

    using fold_serial_t = uncased_fold_from_sz_<sz_utf8_uncased_fold_serial>;
    fold_serial_t const fold_serial;

    // Backend-independent: the fold table and the case-invariant classifier must stay closed
    test_uncased_invariant_reference();

#if SZ_USE_HASWELL
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_haswell> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_haswell);
#endif
#if SZ_USE_ICELAKE
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_icelake> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_icelake);
#endif
#if SZ_USE_NEON
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_neon> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_neon);
#endif
#if SZ_USE_V128
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_v128> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_v128);
#endif
#if SZ_USE_RVV
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_rvv> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_rvv);
#endif
#if SZ_USE_LASX
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_lasx> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_lasx);
#endif
#if SZ_USE_POWERVSX
    test_fold_equivalence(fold_serial, uncased_fold_from_sz_<sz_utf8_uncased_fold_powervsx> {}, 4000,
                          scale_iterations(10000));
    run_uncased_find_battery_(sz_utf8_uncased_search_powervsx);
#endif
}

#pragma endregion // Drivers
