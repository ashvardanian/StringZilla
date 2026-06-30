/**
 *  @brief  UTF-8 normalization (NFC/NFD/NFKC/NFKD) known-answer, serial-vs-ISA equivalence, and safety.
 *  @file   scripts/test_utf8_norm.cpp
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

/** @brief Prints one labeled hex dump line to `stderr`; used by the malformed-input safety test below. */
static void print_utf8_test_bytes_(char const *label, char const *bytes, std::size_t length) {
    std::fprintf(stderr, "  %s (%zu bytes): ", label, length);
    for (std::size_t index = 0; index < length; ++index) std::fprintf(stderr, "%02X ", (unsigned char)bytes[index]);
    std::fprintf(stderr, "\n");
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer unit tests for UTF-8 normalization on simple, hand-verifiable inputs.
 *
 *  Exercises the normalizer and the normalization-violation finder through the dispatched C API (automatic
 *  kernel resolution) and through the natively-compiled backend kernels directly, plus the C++
 *  `sz::string`/`sz::string_view` wrappers (`try_utf8_normalize`, `is_normalized`, `utf8_find_denormalized`),
 *  so a regression that the serial-vs-SIMD agreement tests would miss - because both share a wrong constant -
 *  is still caught against an external ground truth.
 */
void test_utf8_norm_unit() {
    std::printf("  - testing UTF-8 normalization known-answer vectors...\n");

    // `sz_utf8_norm` / `sz_utf8_find_denormalized`: "café" with a precomposed é (U+00E9) is already NFC, but
    // breaks NFD (é decomposes into e + U+0301). NFC normalization is a no-op; NFD expands it to 5 bytes.
    char const cafe_nfc[] = "caf\xC3\xA9"; // U+00E9 (precomposed é), 5 bytes
    sz_size_t const cafe_length = (sz_size_t)(sizeof(cafe_nfc) - 1);
    assert(sz_utf8_find_denormalized(cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           SZ_NULL_CHAR); // Dispatched: already NFC
    assert(sz_utf8_find_denormalized(cafe_nfc, cafe_length, sz_normal_form_nfd_k) !=
           SZ_NULL_CHAR); // Dispatched: not NFD
    assert(sz_utf8_find_denormalized_serial(cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           SZ_NULL_CHAR); // Manual: serial
    assert(sz_utf8_find_denormalized_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k) !=
           SZ_NULL_CHAR); // Manual: serial
#if SZ_USE_ICELAKE
    assert(sz_utf8_find_denormalized_icelake(cafe_nfc, cafe_length, sz_normal_form_nfc_k) ==
           SZ_NULL_CHAR); // Manual: icelake
    assert(sz_utf8_find_denormalized_icelake(cafe_nfc, cafe_length, sz_normal_form_nfd_k) !=
           SZ_NULL_CHAR); // Manual: icelake
#endif
    {
        char norm_buffer[64];
        sz_size_t const nfc_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfc_k, norm_buffer);
        assert(nfc_length == cafe_length && std::memcmp(norm_buffer, cafe_nfc, cafe_length) == 0); // NFC no-op
        sz_size_t const nfd_length = sz_utf8_norm(cafe_nfc, cafe_length, sz_normal_form_nfd_k, norm_buffer);
        assert(nfd_length == 6u); // "caf" + 'e' + U+0301 (2-byte combining acute)
        sz_size_t const nfd_length_serial = sz_utf8_norm_serial(cafe_nfc, cafe_length, sz_normal_form_nfd_k,
                                                                norm_buffer); // Manual: serial
        assert(nfd_length_serial == 6u);
#if SZ_USE_ICELAKE
        sz_size_t const nfd_length_icelake = sz_utf8_norm_icelake(cafe_nfc, cafe_length, sz_normal_form_nfd_k,
                                                                  norm_buffer); // Manual: icelake
        assert(nfd_length_icelake == 6u);
#endif
    }

    // C++ binding round-trip: NFC -> NFD -> NFC should recover the original NFC string.
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

    // utf8_find_denormalized returns non-null for a non-normalized string.
    assert(nfc_view.utf8_find_denormalized(sz_normal_form_nfd_k) != SZ_NULL_CHAR);
    assert(nfc_own.utf8_find_denormalized(sz_normal_form_nfd_k) != SZ_NULL_CHAR);

    // NFKD of "ﬁ" (U+FB01 LATIN SMALL LIGATURE FI) decomposes to "fi".
    char const ligature_fi[] = "\xEF\xAC\x81"; // U+FB01
    sz::string fi_str {ligature_fi};
    assert(fi_str.try_utf8_normalize(sz_normal_form_nfkd_k));
    assert(fi_str.contains('f'));
    assert(fi_str.contains('i'));

    std::printf("    normalization known-answer vectors passed!\n");
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief One backend's UTF-8 normalization + violation kernels, stored by pointer so the differential driver can
 *         iterate a table. The members are named for the call sites (`reference.form(...)`,
 *         `reference.violation(...)`), so each function-pointer member is invoked directly — the harness is unchanged.
 */
struct utf8_norm_backend_t {
    char const *name;
    sz_utf8_norm_t form;
    sz_utf8_find_denormalized_t violation;
};

/**
 *  @brief Tests UTF-8 normalization across SIMD backends against the serial implementation.
 *
 *  Encodes every assigned Unicode codepoint, shuffles the order, and normalizes the full corpus under all
 *  four normal forms, so the comparison stresses canonical ordering, every decomposition/composition path,
 *  and SIMD-block straddles.
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
        for (sz_rune_t cp : all_runes) write_cursor += sz_rune_encode(cp, (sz_u8_t *)write_cursor);
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

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Drives the normalization-violation finder and the normalizer through the malformed-byte battery.
 *
 *  `sz_utf8_find_denormalized` is bounds-safe on arbitrary bytes, so it faces the full malformed battery: the
 *  only requirements are that it survives and that any returned violation pointer stays inside
 *  `[text, text + length]`. `sz_utf8_norm` instead documents a valid-UTF-8 precondition (the decoder
 *  "performs no validity checks") and asserts internally on garbage, so it is only exercised on the
 *  `sz_utf8_find_malformed` subset of the same adversarial shapes - the normalizer's output must stay within its
 *  documented 18x bound; `with_guarded_buffer_` brackets the destination with canaries and asserts they
 *  survive, like `test_uncased_safety`.
 *
 *  @param norm             Single-pass normalizer under test.
 *  @param violation        Normalization-violation finder under test.
 *  @param random_inputs    Number of random garbage buffers to fuzz on top of the exhaustive byte sweeps.
 */
static void check_utf8_norm_safety_(sz_utf8_norm_t norm, sz_utf8_find_denormalized_t violation,
                                    std::size_t random_inputs = scale_iterations(10000)) {

    std::size_t const max_input_length = 70;
    // The normalizer's documented worst case is 18x the input for a single-codepoint compatibility
    // decomposition (see `utf8_norm.h`); a truncated trailing sequence may mis-decode one extra rune.
    std::size_t const norm_bound = max_input_length * 18 + 18;

    static sz_normal_form_t const norm_forms[4] = {sz_normal_form_nfd_k, sz_normal_form_nfc_k, sz_normal_form_nfkd_k,
                                                   sz_normal_form_nfkc_k};

    auto check = [&](char const *input, std::size_t input_length) {
        // The violation finder is bounds-safe on arbitrary bytes, so it just has to survive and keep any
        // returned pointer inside the input. It runs against every malformed shape, not just valid UTF-8.
        for (sz_normal_form_t normal_form : norm_forms) {
            sz_cptr_t const found = violation(input, (sz_size_t)input_length, normal_form);
            if (found == SZ_NULL_CHAR) continue;
            if (found >= input && found <= input + input_length) continue;
            std::fprintf(stderr, "norm violation returned out-of-bounds pointer (form=%d, input=%zu)\n",
                         (int)normal_form, input_length);
            print_utf8_test_bytes_("input", input, input_length);
            assert(false && "Normalization-violation finder returned a pointer outside the input");
        }

        // Normalization: unlike the violation finder, `sz_utf8_norm` documents a valid-UTF-8 precondition and
        // asserts internally on malformed bytes, so it is only driven on inputs that pass `sz_utf8_find_malformed`. The
        // output must still stay within the bound; `with_guarded_buffer_` brackets the destination with
        // canaries and asserts they survive, like `test_uncased_safety`.
        if (sz_utf8_find_malformed(input, (sz_size_t)input_length) == SZ_NULL_CHAR) {
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

/** @brief Drive the malformed-input normalization safety probe through serial, dispatched, and every backend. */
void test_utf8_norm_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 normalization kernels...\n");

    // Serial baseline and the dispatched (automatic kernel resolution) entry points face the same contract.
    check_utf8_norm_safety_(sz_utf8_norm_serial, sz_utf8_find_denormalized_serial);
    check_utf8_norm_safety_(sz_utf8_norm, sz_utf8_find_denormalized);

#if SZ_USE_HASWELL
    check_utf8_norm_safety_(sz_utf8_norm_haswell, sz_utf8_find_denormalized_haswell);
#endif
#if SZ_USE_SKYLAKE
    check_utf8_norm_safety_(sz_utf8_norm_skylake, sz_utf8_find_denormalized_skylake);
#endif
#if SZ_USE_ICELAKE
    check_utf8_norm_safety_(sz_utf8_norm_icelake, sz_utf8_find_denormalized_icelake);
#endif
#if SZ_USE_NEON
    check_utf8_norm_safety_(sz_utf8_norm_neon, sz_utf8_find_denormalized_neon);
#endif
#if SZ_USE_SVE
    check_utf8_norm_safety_(sz_utf8_norm_sve, sz_utf8_find_denormalized_sve);
#endif
#if SZ_USE_SVE2
    check_utf8_norm_safety_(sz_utf8_norm_sve2, sz_utf8_find_denormalized_sve2);
#endif
#if SZ_USE_RVV
    check_utf8_norm_safety_(sz_utf8_norm_rvv, sz_utf8_find_denormalized_rvv);
#endif
#if SZ_USE_V128
    check_utf8_norm_safety_(sz_utf8_norm_v128, sz_utf8_find_denormalized_v128);
#endif
#if SZ_USE_V128RELAXED
    check_utf8_norm_safety_(sz_utf8_norm_v128relaxed, sz_utf8_find_denormalized_v128relaxed);
#endif
#if SZ_USE_LASX
    check_utf8_norm_safety_(sz_utf8_norm_lasx, sz_utf8_find_denormalized_lasx);
#endif
#if SZ_USE_POWERVSX
    check_utf8_norm_safety_(sz_utf8_norm_powervsx, sz_utf8_find_denormalized_powervsx);
#endif

    std::printf("    normalization safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/**
 *  @brief The UTF-8 normalization backends compiled on this target. The always-present `dispatched` entry keeps the
 *         table non-empty on a baseline build; the differential below runs serial vs each.
 */
static utf8_norm_backend_t const utf8_norm_backends[] = {
    {"dispatched", sz_utf8_norm, sz_utf8_find_denormalized},
#if SZ_USE_HASWELL
    {"haswell", sz_utf8_norm_haswell, sz_utf8_find_denormalized_haswell},
#endif
#if SZ_USE_SKYLAKE
    {"skylake", sz_utf8_norm_skylake, sz_utf8_find_denormalized_skylake},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_utf8_norm_icelake, sz_utf8_find_denormalized_icelake},
#endif
#if SZ_USE_NEON
    {"neon", sz_utf8_norm_neon, sz_utf8_find_denormalized_neon},
#endif
#if SZ_USE_SVE
    {"sve", sz_utf8_norm_sve, sz_utf8_find_denormalized_sve},
#endif
#if SZ_USE_SVE2
    {"sve2", sz_utf8_norm_sve2, sz_utf8_find_denormalized_sve2},
#endif
#if SZ_USE_RVV
    {"rvv", sz_utf8_norm_rvv, sz_utf8_find_denormalized_rvv},
#endif
#if SZ_USE_V128
    {"v128", sz_utf8_norm_v128, sz_utf8_find_denormalized_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_utf8_norm_v128relaxed, sz_utf8_find_denormalized_v128relaxed},
#endif
#if SZ_USE_LASX
    {"lasx", sz_utf8_norm_lasx, sz_utf8_find_denormalized_lasx},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_utf8_norm_powervsx, sz_utf8_find_denormalized_powervsx},
#endif
};

/** @brief Run the normalization differential fuzz against every compiled backend (dispatched first). */
void test_utf8_norm_all() {
    utf8_norm_backend_t const serial {"serial", sz_utf8_norm_serial, sz_utf8_find_denormalized_serial};
    for (utf8_norm_backend_t const &backend : utf8_norm_backends) test_norm_equivalence(serial, backend);
}

#pragma endregion // Drivers
