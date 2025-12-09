/**
 *  @brief   Extensive @b unit-testing suite for StringZilla, written in C++.
 *  @note    It mostly tests one target hardware platform at a time and should be compiled and run separately for each.
 *           To override the default hardware platform, overrides the @b `SZ_USE_*` flags at the top of this file.
 *
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringzilla.cpp
 *  @author  Ash Vardanian
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
 #define SZ_USE_ICE 0
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

/**
 *  Instantiate all the templates to make the symbols visible and also check
 *  for weird compilation errors on uncommon paths.
 */
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
template class std::basic_string_view<char>;
#endif
template class sz::basic_string_slice<char>;
template class std::basic_string<char>;
template class sz::basic_string<char>;
template class sz::basic_byteset<char>;

template class std::vector<sz::string>;
template class std::map<sz::string, int>;
template class std::unordered_map<sz::string, int>;

template class std::vector<sz::string_view>;
template class std::map<sz::string_view, int>;
template class std::unordered_map<sz::string_view, int>;

/**
 *  @brief  Several string processing operations rely on computing integer logarithms.
 *          Failures in such operations will result in wrong `resize` outcomes and heap corruption.
 */
void test_arithmetical_utilities() {

    assert(sz_u64_clz(0x0000000000000001ull) == 63);
    assert(sz_u64_clz(0x0000000000000002ull) == 62);
    assert(sz_u64_clz(0x0000000000000003ull) == 62);
    assert(sz_u64_clz(0x0000000000000004ull) == 61);
    assert(sz_u64_clz(0x0000000000000007ull) == 61);
    assert(sz_u64_clz(0x8000000000000001ull) == 0);
    assert(sz_u64_clz(0xffffffffffffffffull) == 0);
    assert(sz_u64_clz(0x4000000000000000ull) == 1);

    assert(sz_size_log2i_nonzero(1) == 0);
    assert(sz_size_log2i_nonzero(2) == 1);
    assert(sz_size_log2i_nonzero(3) == 1);

    assert(sz_size_log2i_nonzero(4) == 2);
    assert(sz_size_log2i_nonzero(5) == 2);
    assert(sz_size_log2i_nonzero(7) == 2);

    assert(sz_size_log2i_nonzero(8) == 3);
    assert(sz_size_log2i_nonzero(9) == 3);

    assert(sz_size_bit_ceil(0) == 0);
    assert(sz_size_bit_ceil(1) == 1);

    assert(sz_size_bit_ceil(2) == 2);
    assert(sz_size_bit_ceil(3) == 4);
    assert(sz_size_bit_ceil(4) == 4);

    assert(sz_size_bit_ceil(77) == 128);
    assert(sz_size_bit_ceil(127) == 128);
    assert(sz_size_bit_ceil(128) == 128);

    assert(sz_size_bit_ceil(1000000ull) == (1ull << 20));
    assert(sz_size_bit_ceil(2000000ull) == (1ull << 21));
    assert(sz_size_bit_ceil(4000000ull) == (1ull << 22));
    assert(sz_size_bit_ceil(8000000ull) == (1ull << 23));

    assert(sz_size_bit_ceil(16000000ull) == (1ull << 24));
    assert(sz_size_bit_ceil(32000000ull) == (1ull << 25));
    assert(sz_size_bit_ceil(64000000ull) == (1ull << 26));

    assert(sz_size_bit_ceil(128000000ull) == (1ull << 27));
    assert(sz_size_bit_ceil(256000000ull) == (1ull << 28));
    assert(sz_size_bit_ceil(512000000ull) == (1ull << 29));

    assert(sz_size_bit_ceil(1000000000ull) == (1ull << 30));
    assert(sz_size_bit_ceil(2000000000ull) == (1ull << 31));

#if SZ_IS_64BIT_
    assert(sz_size_bit_ceil(4000000000ull) == (1ull << 32));
    assert(sz_size_bit_ceil(8000000000ull) == (1ull << 33));
    assert(sz_size_bit_ceil(16000000000ull) == (1ull << 34));

    assert(sz_size_bit_ceil((1ull << 62)) == (1ull << 62));
    assert(sz_size_bit_ceil((1ull << 62) + 1) == (1ull << 63));
    assert(sz_size_bit_ceil((1ull << 63)) == (1ull << 63));
#endif
}

/** @brief Validates `sz_sequence_t` and related construction utilities. */
void test_sequence_struct() {
    // Make sure the sequence helper functions work as expected
    // for both trivial c-style arrays and more complicated STL containers.
    {
        sz_sequence_t sequence;
        sz_cptr_t strings[] = {"banana", "apple", "cherry"};
        sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
        assert(sequence.count == 3);
        assert("banana"_sv == sequence.get_start(sequence.handle, 0));
        assert("apple"_sv == sequence.get_start(sequence.handle, 1));
        assert("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
    // Do the same for STL:
    {
        using strings_vector_t = std::vector<std::string>;
        strings_vector_t strings = {"banana", "apple", "cherry"};
        sz_sequence_t sequence;
        sequence.handle = &strings;
        sequence.count = strings.size();
        sequence.get_start =
            reinterpret_cast<sz_sequence_member_start_t>(+[](void *handle, sz_size_t index) noexcept -> sz_cptr_t {
                auto const &strings = *static_cast<strings_vector_t *>(handle);
                return strings[index].c_str();
            });
        sequence.get_length =
            reinterpret_cast<sz_sequence_member_length_t>(+[](void *handle, sz_size_t index) noexcept -> sz_size_t {
                auto const &strings = *static_cast<strings_vector_t *>(handle);
                return strings[index].size();
            });

        assert(sequence.count == 3);
        assert("banana"_sv == sequence.get_start(sequence.handle, 0));
        assert("apple"_sv == sequence.get_start(sequence.handle, 1));
        assert("cherry"_sv == sequence.get_start(sequence.handle, 2));
    }
}

/** @brief Validates `sz_memory_allocator_t` and related construction utilities. */
void test_memory_allocator_struct() {
    // Our behavior for `malloc(0)` is to return a NULL pointer,
    // while the standard is implementation-defined.
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        assert(alloc.allocate(0, alloc.handle) == nullptr);
    }

    // Non-NULL allocation
    {
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_default(&alloc);
        void *byte = alloc.allocate(1, alloc.handle);
        assert(byte != nullptr);
        alloc.free(byte, 1, alloc.handle);
    }

    // Use a fixed buffer
    {
        char buffer[1024];
        sz_memory_allocator_t alloc;
        sz_memory_allocator_init_fixed(&alloc, buffer, sizeof(buffer));
        void *byte = alloc.allocate(1, alloc.handle);
        assert(byte != nullptr);
        alloc.free(byte, 1, alloc.handle);
    }
}

/** @brief Validates `sz_byteset_t` and related construction utilities. */
void test_byteset_struct() {
    sz_byteset_t s;
    sz_byteset_init(&s);
    assert(sz_byteset_contains(&s, 'a') == sz_false_k);
    sz_byteset_add(&s, 'a');
    assert(sz_byteset_contains(&s, 'a') == sz_true_k);
    sz_byteset_add(&s, 'z');
    assert(sz_byteset_contains(&s, 'z') == sz_true_k);
    sz_byteset_invert(&s);
    assert(sz_byteset_contains(&s, 'a') == sz_false_k);
    assert(sz_byteset_contains(&s, 'z') == sz_false_k);
    assert(sz_byteset_contains(&s, 'b') == sz_true_k);
    sz_byteset_init_ascii(&s);
    assert(sz_byteset_contains(&s, 'A') == sz_true_k);
}

/**
 *  @brief  Hashes a string and compares the output between a serial and hardware-specific SIMD backend.
 *
 *  The test covers increasingly long and complex strings, starting with "abcabc..." repetitions and
 *  progressing towards corner cases like empty strings, all-zero inputs, zero seeds, and so on.
 */
void test_hash_equivalence(                                               //
    sz_hash_t hash_base, sz_hash_state_init_t init_base,                  //
    sz_hash_state_update_t stream_base, sz_hash_state_digest_t fold_base, //
    sz_hash_t hash_simd, sz_hash_state_init_t init_simd,                  //
    sz_hash_state_update_t stream_simd, sz_hash_state_digest_t fold_simd) {

    auto test_on_seed = [&](std::string text, sz_u64_t seed) {
        // Compute the entire hash at once, expecting the same output
        sz_u64_t result_base = hash_base(text.data(), text.size(), seed);
        sz_u64_t result_simd = hash_simd(text.data(), text.size(), seed);
        assert(result_base == result_simd);

        // Compare incremental hashing across platforms
        sz_hash_state_t state_base, state_simd;
        init_base(&state_base, seed);
        init_simd(&state_simd, seed);
        assert(sz_hash_state_equal(&state_base, &state_base) == sz_true_k); // Self-equality
        assert(sz_hash_state_equal(&state_simd, &state_simd) == sz_true_k); // Self-equality
        assert(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

        // Let's also create an intentionally misaligned version of the state,
        // assuming some of the SIMD instructions may require alignment.
        sz_align_(64) char state_misaligned_buffer[sizeof(sz_hash_state_t) + 1];
        sz_hash_state_t &state_misaligned = *reinterpret_cast<sz_hash_state_t *>(state_misaligned_buffer + 1);
        init_simd(&state_misaligned, seed);
        assert(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

        // Try breaking those strings into arbitrary chunks, expecting the same output in the streaming mode.
        // The length of each chunk and the number of chunks will be determined with a coin toss.
        iterate_in_random_slices(text, [&](std::string slice) {
            stream_base(&state_base, slice.data(), slice.size());
            stream_simd(&state_simd, slice.data(), slice.size());
            assert(sz_hash_state_equal(&state_base, &state_simd) == sz_true_k); // Same across platforms

            stream_simd(&state_misaligned, slice.data(), slice.size());
            assert(sz_hash_state_equal(&state_base, &state_misaligned) == sz_true_k); // Same across platforms

            result_base = fold_base(&state_base);
            result_simd = fold_simd(&state_simd);
            assert(result_base == result_simd);
            sz_u64_t result_misaligned = fold_simd(&state_misaligned);
            assert(result_base == result_misaligned);
        });
    };

    // Let's try different-length strings repeating a "abc" pattern:
    std::vector<sz_u64_t> seeds = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };
    for (auto seed : seeds)
        for (std::size_t copies = 1; copies != 100; ++copies) //
            test_on_seed(repeat("abc", copies), seed);

    // Let's try truly random inputs of different lengths:
    for (std::size_t length = 0; length != 200; ++length) {
        std::string text(length, '\0');
        randomize_string(&text[0], length);
        for (auto seed : seeds) test_on_seed(text, seed);
    }
}

/**
 *  @brief  Tests Pseudo-Random Number Generators (PRNGs) ensuring that the same nonce
 *          produces exactly the same output across different SIMD implementations.
 */
void test_random_generator_equivalence(sz_fill_random_t generate_base, sz_fill_random_t generate_simd) {

    auto test_on_nonce = [&](std::size_t length, sz_u64_t nonce) {
        std::string text_base(length, '\0');
        std::string text_simd(length, '\0');
        generate_base(&text_base[0], static_cast<sz_size_t>(length), nonce);
        generate_simd(&text_simd[0], static_cast<sz_size_t>(length), nonce);
        assert(text_base == text_simd);
    };

    // Let's try different nonces:
    std::vector<sz_u64_t> nonces = {
        0u,
        42u,                                  //
        std::numeric_limits<sz_u32_t>::max(), //
        std::numeric_limits<sz_u64_t>::max(), //
    };
    std::vector<std::size_t> lengths = {1, 11, 23, 37, 40, 51, 64, 128, 1000};
    for (auto nonce : nonces)
        for (auto length : lengths) //
            test_on_nonce(length, nonce);
}

/**
 *  @brief  Tests SHA256 implementations, comparing serial and SIMD variants
 *          against known FIPS 180-4 test vectors.
 */
void test_sha256_equivalence(                                                                                     //
    sz_sha256_state_init_t init_base, sz_sha256_state_update_t update_base, sz_sha256_state_digest_t digest_base, //
    sz_sha256_state_init_t init_simd, sz_sha256_state_update_t update_simd, sz_sha256_state_digest_t digest_simd) {

    // Test random inputs of various lengths
    for (std::size_t length = 0; length <= 256; ++length) {
        std::string random_text(length, '\0');
        randomize_string(&random_text[0], length);

        sz_sha256_state_t state_base, state_simd;
        sz_u8_t digest_base_result[32], digest_simd_result[32];

        // One-shot hashing
        init_base(&state_base);
        init_simd(&state_simd);
        update_base(&state_base, random_text.data(), length);
        update_simd(&state_simd, random_text.data(), length);
        digest_base(&state_base, digest_base_result);
        digest_simd(&state_simd, digest_simd_result);
        assert(std::memcmp(digest_base_result, digest_simd_result, 32) == 0);

        // Incremental hashing with random chunks
        init_base(&state_base);
        init_simd(&state_simd);
        iterate_in_random_slices(random_text, [&](std::string slice) {
            update_base(&state_base, slice.data(), slice.size());
            update_simd(&state_simd, slice.data(), slice.size());
        });
        digest_base(&state_base, digest_base_result);
        digest_simd(&state_simd, digest_simd_result);
        assert(std::memcmp(digest_base_result, digest_simd_result, 32) == 0);
    }
}

/**
 *  @brief  Tests UTF-8 functions across different SIMD backends against the serial implementation.
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
 */
void test_utf8_equivalence(                                 //
    sz_utf8_count_t count_base, sz_utf8_count_t count_simd, //
    sz_utf8_find_boundary_t newline_base,                   //
    sz_utf8_find_boundary_t newline_simd,                   //
    sz_utf8_find_boundary_t whitespace_base,                //
    sz_utf8_find_boundary_t whitespace_simd,                //
    std::size_t min_text_length = 4000, std::size_t min_iterations = scale_iterations(10000)) {

    auto check = [&](std::string const &text) {
        sz_cptr_t data = text.data();
        sz_size_t len = text.size();

        // Test `sz_utf8_count` equivalence
        sz_size_t count_result_base = count_base(data, len);
        sz_size_t count_result_simd = count_simd(data, len);
        assert(count_result_base == count_result_simd);

        // Test `sz_utf8_find_newline` equivalence by scanning the entire string
        sz_cptr_t pos = data;
        sz_size_t remaining = len;
        while (remaining > 0) {
            sz_size_t matched_base = 0, matched_simd = 0;
            sz_cptr_t found_base = newline_base(pos, remaining, &matched_base);
            sz_cptr_t found_simd = newline_simd(pos, remaining, &matched_simd);
            assert(found_base == found_simd && "Mismatch in newline detection");
            if (found_base == SZ_NULL_CHAR) break;
            assert(matched_base == matched_simd);
            sz_size_t offset = (found_base - pos) + matched_base;
            pos += offset;
            remaining -= offset;
        }

        // Test `sz_utf8_find_whitespace` equivalence by scanning the entire string
        pos = data;
        remaining = len;
        while (remaining > 0) {
            sz_size_t matched_base = 0, matched_simd = 0;
            sz_cptr_t found_base = whitespace_base(pos, remaining, &matched_base);
            sz_cptr_t found_simd = whitespace_simd(pos, remaining, &matched_simd);
            assert(found_base == found_simd && "Mismatched position in whitespace detection");
            if (found_base == SZ_NULL_CHAR) break;
            assert(matched_base == matched_simd);
            sz_size_t offset = (found_base - pos) + matched_base;
            pos += offset;
            remaining -= offset;
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
        std::uniform_int_distribution<std::size_t> pos_dist(0, text.size() - 1);
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t pos = pos_dist(rng);
            text[pos] = '\0';
        }
        check(text);

        // Swap 10% of bytes at random positions, creating malformed UTF-8 sequences
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t pos1 = pos_dist(rng);
            std::size_t pos2 = pos_dist(rng);
            std::swap(text[pos1], text[pos2]);
        }
        check(text);
    }
}

/**
 *  @brief  Tests equivalence of case folding implementations (serial vs SIMD).
 *
 *  Generates random UTF-8 strings containing:
 *  - ASCII text (uppercase and lowercase)
 *  - Multi-byte UTF-8 characters from various scripts (Cyrillic, Greek, Latin Extended)
 *  - Special cases like German ß
 *
 *  For each generated string, compares byte-by-byte output of both implementations.
 */
void test_utf8_case_fold_equivalence(                             //
    sz_utf8_case_fold_t fold_base, sz_utf8_case_fold_t fold_simd, //
    std::size_t min_text_length = 4000, std::size_t min_iterations = 10000) {

    // Output buffers (3x input for worst-case expansion)
    std::vector<char> output_base(min_text_length * 3 + 256);
    std::vector<char> output_simd(min_text_length * 3 + 256);

    auto check = [&](std::string const &text) {
        // Ensure buffers are large enough
        if (output_base.size() < text.size() * 3 + 64) {
            output_base.resize(text.size() * 3 + 64);
            output_simd.resize(text.size() * 3 + 64);
        }

        sz_size_t len_base = fold_base(text.data(), text.size(), output_base.data());
        sz_size_t len_simd = fold_simd(text.data(), text.size(), output_simd.data());

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
        ("Hello \xE1\x82\xA0\xE1\x82\xA1 World"), // Hello ႠႡ World
        ("ABC\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2"
         "DEF"), // ABCႠႡႢdef
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
            std::size_t idx = content_dist(rng);
            text.append(utf8_content[idx]);
        }
        check(text);
    }
}

/**
 *  @brief  Exhaustive fuzz test for UTF-8 case folding using all Unicode codepoints.
 *
 *  - First run: Tests all valid codepoints in order (0x0 to 0x10FFFF).
 *  - Subsequent runs: Shuffles the codepoints to create random sequences.
 *  - Checks for both length and content matches between serial and SIMD implementations.
 */
void test_utf8_case_fold_fuzz(sz_utf8_case_fold_t fold_base, sz_utf8_case_fold_t fold_simd,
                              std::size_t iterations = scale_iterations(100)) {
    std::printf("  - testing case folding fuzz (%zu iterations + ordered check)...\n", iterations);

    // 1. Generate all valid codepoints (ordered initially)
    std::vector<sz_rune_t> all_runes;
    all_runes.reserve(0x10FFFF);
    for (sz_rune_t cp = 0; cp <= 0x10FFFF; ++cp) {
        if (cp >= 0xD800 && cp <= 0xDFFF) continue; // Skip surrogates
        all_runes.push_back(cp);
    }

    // 2. Prepare buffers
    // Max UTF-8 size is 4 bytes per rune.
    std::vector<char> input_buffer(all_runes.size() * 4);
    // Max expansion is handled by the folding functions, usually 3x is safe for worst cases (like µ)
    std::vector<char> output_base(input_buffer.size() * 3 + 64);
    std::vector<char> output_simd(input_buffer.size() * 3 + 64);

    auto &rng = global_random_generator();

    // Iterate: 0 = Ordered, 1..N = Shuffled
    for (std::size_t it = 0; it <= iterations; ++it) {
        if (it > 0) std::shuffle(all_runes.begin(), all_runes.end(), rng);

        // Convert to UTF-8
        char *data_ptr = input_buffer.data();
        for (sz_rune_t cp : all_runes) data_ptr += sz_rune_export(cp, (sz_u8_t *)data_ptr);
        sz_size_t input_len = data_ptr - input_buffer.data();

        // Run tests
        sz_size_t len_base = fold_base(input_buffer.data(), input_len, output_base.data());
        sz_size_t len_simd = fold_simd(input_buffer.data(), input_len, output_simd.data());

        // Validations
        if (len_base != len_simd) {
            std::fprintf(stderr, "Iteration %zu: Length mismatch base=%zu simd=%zu\n", it, len_base, len_simd);
            assert(false);
        }

        // Optimize: Memcmp first, then diagnose
        if (std::memcmp(output_base.data(), output_simd.data(), len_base) != 0) {
            std::fprintf(stderr, "Iteration %zu: Content mismatch\n", it);
            // Find first mismatch for debug
            for (std::size_t i = 0; i < len_base; ++i) {
                if (output_base[i] != output_simd[i]) {
                    std::fprintf(stderr, "Mismatch at byte %zu: 0x%02X vs 0x%02X\n", i, (unsigned char)output_base[i],
                                 (unsigned char)output_simd[i]);

                    // Show context (16 bytes before/after)
                    sz_size_t start = (i > 16) ? i - 16 : 0;
                    sz_size_t end = (i + 16 < len_base) ? i + 16 : len_base;

                    std::fprintf(stderr, "Context (Base): ");
                    for (sz_size_t j = start; j < end; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)output_base[j]);
                    std::fprintf(stderr, "\n");

                    std::fprintf(stderr, "Context (SIMD): ");
                    for (sz_size_t j = start; j < end; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)output_simd[j]);
                    std::fprintf(stderr, "\n");

                    // Try to map back to input (approximate)
                    sz_size_t in_start = (i > 16) ? i - 16 : 0;
                    sz_size_t in_end = (i + 16 < input_len) ? i + 16 : input_len;
                    std::fprintf(stderr, "Input (approx): ");
                    for (sz_size_t j = in_start; j < in_end; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)input_buffer[j]);
                    std::fprintf(stderr, "\n");

                    break;
                }
            }
            assert(false);
        }
    }
    std::printf("    exhaustive fuzzing passed!\n");
}

/**
 *  @brief  Tests case-insensitive UTF-8 substring search equivalence between base and SIMD implementations.
 *
 *  Uses three independent verification methods:
 *  1. Serial case-insensitive find (baseline)
 *  2. SIMD case-insensitive find (under test)
 *  3. Fold haystack + fold needle + regular sz_find (independent oracle)
 *
 *  Static unit tests covering specific edge cases: ASCII, Latin-1, Cyrillic, Greek, Armenian, Vietnamese,
 *  Georgian, Cherokee, Coptic, Glagolitic, ligatures, Eszett ↔ SS, multiplication/division signs,
 *  script transitions, and caseless scripts (CJK, Arabic, Hebrew, etc.).
 */
void test_utf8_ci_find_equivalence(sz_utf8_case_insensitive_find_t find_serial,
                                   sz_utf8_case_insensitive_find_t find_simd, sz_utf8_case_fold_t case_fold,
                                   sz_utf8_count_t utf8_count) {
    std::printf("  - testing case-insensitive find equivalence...\n");

    // Buffers for folded strings (case folding can expand, e.g., ß→ss, so 4x is safe)
    constexpr std::size_t max_folded_size = 4096;
    std::vector<char> folded_haystack_buf(max_folded_size);
    std::vector<char> folded_needle_buf(max_folded_size);

    struct test_case {
        char const *haystack;
        char const *needle;
        char const *description;
    };

    // Static test cases covering all paths
    test_case cases[] = {
        // ASCII path tests
        {"Hello World", "WORLD", "ASCII uppercase needle"},
        {"Hello World", "world", "ASCII lowercase needle"},
        {"the quick brown fox", "QUICK", "ASCII mid-string"},
        {"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "mnop", "ASCII lowercase in uppercase"},
        {"abcdefghijklmnopqrstuvwxyz", "MNOP", "ASCII uppercase in lowercase"},
        {"Hello Hello Hello", "HELLO", "ASCII multiple matches"},

        // Latin1 path tests (C3 lead byte characters: À-ÿ)
        {"Das ist ein sch\xC3\xB6ner Tag",
         "SCH\xC3\x96"
         "NER",
         "German umlaut o"},                                       // schöner
        {"Caf\xC3\xA9 au lait", "CAF\xC3\x89", "French accent e"}, // café
        {"\xC3\x9C"
         "ber allen Gipfeln",
         "\xC3\xBC"
         "BER",
         "German U umlaut"}, // Über
        {"na\xC3\xAFve approach",
         "NA\xC3\x8F"
         "VE",
         "French diaeresis i"}, // naïve
        {"El ni\xC3\xB1o juega",
         "NI\xC3\x91"
         "O",
         "Spanish n tilde"}, // niño
        {"M\xC3\xA4"
         "dchen und B\xC3\xBC"
         "cher",
         "M\xC3\x84"
         "DCHEN",
         "German a umlaut"},                                    // Mädchen
        {"\xC3\x80 la carte", "\xC3\xA0 LA", "French A grave"}, // À la

        // Cyrillic path tests (D0/D1 lead bytes)
        {"\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 \xD0\xBC\xD0\xB8\xD1\x80",
         "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", "Cyrillic lowercase needle"}, // Привет мир
        {"\xD0\x9C\xD0\x9E\xD0\xA1\xD0\x9A\xD0\x92\xD0\x90 \xD1\x81\xD1\x82\xD0\xBE\xD0\xBB\xD0\xB8\xD1\x86"
         "\xD0\xB0",
         "\xD0\xBC\xD0\xbe\xD1\x81\xD0\xba\xD0\xB2\xD0\xB0", "Cyrillic uppercase haystack"}, // МОСКВА
        {"\xD0\x94\xD0\xBE\xD0\xB1\xD1\x80\xD1\x8B\xD0\xB9 \xD0\xB4\xD0\xB5\xD0\xBD\xD1\x8C",
         "\xD0\x94\xD0\x9E\xD0\x91\xD0\xA0\xD0\xAB\xD0\x99", "Cyrillic mixed case"}, // Добрый

        // Cyrillic Extended (D2/D3 lead bytes)
        {"\xD2\x90\xD2\x90", "\xD2\x91\xD2\x91", "Cyrillic Extended Ґ->ґ"},
        {"\xD3\x80\xD3\x80", "\xD3\x8F\xD3\x8F", "Cyrillic Extended Ӏ->ӏ"}, // Palochka
        {"\xD0\x9F\xD2\x90\xD3\x80", "\xD0\xBF\xD2\x91\xD3\x8F", "Cyrillic Mixed D0/D2/D3"},

        // Central European (Latin Extended-A C4/C5)
        {"\xC4\x84\xC4\x86\xC4\x98\xC5\x81\xC5\x83\xC3\x93\xC5\x9A\xC5\xB9\xC5\xBB",
         "\xC4\x85\xC4\x87\xC4\x99\xC5\x82\xC5\x84\xC3\xB3\xC5\x9B\xC5\xBA\xC5\xBC",
         "Polish uppercase to lowercase (ĄĆĘŁŃÓŚŹŻ)"},

        // Turkish
        {"\xC4\x9E\xC5\x9E", "\xC4\x9F\xC5\x9F", "Turkish ĞŞ -> ğş"},

        // Mixed script tests
        {"Hello \xD0\x9C\xD0\xB8\xD1\x80 World", "\xD0\x9C\xD0\x98\xD0\xA0", "Cyrillic in mixed string"}, // Мир
        {"\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0 is beautiful",
         "\xD0\x9C\xD0\x9E\xD0\xA1\xD0\x9A\xD0\x92\xD0\x90", "Cyrillic + ASCII"}, // Москва

        // Greek path tests (CE/CF lead bytes)
        {"\xCE\x9A\xCE\xB1\xCE\xBB\xCE\xB7\xCE\xBC\xCE\xAD\xCF\x81\xCE\xB1",
         "\xCE\xBA\xCE\xB1\xCE\xBB\xCE\xB7\xCE\xBC\xCE\xAD\xCF\x81\xCE\xB1",
         "Greek uppercase to lowercase"}, // Καλημέρα
        {"\xCE\x91\xCE\x92\xCE\x93\xCE\x94\xCE\x95", "\xCE\xB1\xCE\xB2\xCE\xB3\xCE\xB4\xCE\xB5",
         "Greek ΑΒΓΔΕ to αβγδε"},

        // Vietnamese path tests (C3/C6/E1 lead bytes)
        // C6 A0 (Ơ) -> C6 A1 (ơ)
        {"\xC6\xA0\x20", "\xC6\xA1", "Vietnamese Ơ to ơ"},
        {"\xC6\xA1\x20", "\xC6\xA0", "Vietnamese ơ to Ơ"},
        // C6 AF (Ư) -> C6 B0 (ư)
        {"\xC6\xAF\x20", "\xC6\xB0", "Vietnamese Ư to ư"},
        {"\xC6\xB0\x20", "\xC6\xAF", "Vietnamese ư to Ư"},

        // Latin Extended Additional (E1 B8-BB)
        // E1 B8 80 (Ḁ) -> E1 B8 81 (ḁ)
        {"\xE1\xB8\x80", "\xE1\xB8\x81", "Vietnamese Ḁ to ḁ"},
        // E1 B8 BE (Ḿ) -> E1 B8 BF (ḿ)
        {"\xE1\xB8\xBE", "\xE1\xB8\xBF", "Vietnamese Ḿ to ḿ"},

        // Mixed Vietnamese
        // Ơ (C6 A0) + Ḁ (E1 B8 80) + Ư (C6 AF)
        {"\xC6\xA0\xE1\xB8\x80\xC6\xAF", "\xC6\xA1\xE1\xB8\x81\xC6\xB0", "Mixed Vietnamese ƠḀƯ -> ơḁư"},
        {"\xCE\xB1\xCE\xB2\xCE\xB3\xCE\xB4\xCE\xB5", "\xCE\x91\xCE\x92\xCE\x93\xCE\x94\xCE\x95",
         "Greek αβγδε to ΑΒΓΔΕ"},
        // Greek with CE/CF boundary crossing (π-ω are CF 80-89)
        {"\xCE\xA0\xCE\xA1\xCE\xA3\xCE\xA4", "\xCF\x80\xCF\x81\xCF\x83\xCF\x84", "Greek ΠΡΣΤ to πρστ"},
        {"\xCF\x80\xCF\x81\xCF\x83\xCF\x84", "\xCE\xA0\xCE\xA1\xCE\xA3\xCE\xA4", "Greek πρστ to ΠΡΣΤ"},
        // Final sigma tests (ς CF 82 should match σ CF 83 and Σ CE A3)
        {"\xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC\xCE\xBF\xCF\x82", "\xCE\x9A\xCE\x8C\xCE\xA3\xCE\x9C\xCE\x9F\xCE\xA3",
         "Greek κόσμος vs ΚΟΣΜΟΣ"}, // final sigma
        {"\xCE\xBB\xCF\x8C\xCE\xB3\xCE\xBF\xCF\x82", "\xCE\x9B\xCE\x8C\xCE\x93\xCE\x9F\xCE\xA3",
         "Greek λόγος vs ΛΟΓΟΣ"},
        {"\xCF\x83\xCE\xBF\xCF\x86\xCF\x8C\xCF\x82", "\xCE\xA3\xCE\x9F\xCE\xA6\xCE\x8C\xCE\xA3",
         "Greek σοφός vs ΣΟΦΟΣ"},

        // Armenian Fuzz Regression
        // Needle "nԱԲՐԵշ" (Mixed case)
        {"\x6E\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E", "\x6E\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE",
         "Armenian Fuzz Regression"},
        // Ech-Yiwn (U+0587) expansion check
        {"\xD6\x87", "\xD5\xA5\xD6\x82", "Ech-Yiwn U+0587 -> ech yiwn"},
        // Greek + ASCII mixed
        {"Hello \xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC\xCE\xB5 World", "\xCE\x9A\xCE\x8C\xCE\xA3\xCE\x9C\xCE\x95",
         "Greek in ASCII context"},

        // Greek Corner Cases (Anomalies & Danger Zones)
        // ---------------------------------------------
        // Dialytika with Tonos 'ΐ' (CE 90) -> (Standard folding might expand or mapping is complex)
        // Our implementation detects this as anomaly and falls back to serial.
        // Serial usually folds 'ΐ' (U+0390) -> 'ΐ' (U+0390) if simple case folding,
        // or to 'ι' + marks if full. Let's verify identity or simple mapping.
        // Basic check: Find 'ΐ' in 'ΐ' (identity)
        {"\xCE\x90", "\xCE\x90", "Greek ΐ (CE 90) identity"},
        // Check mixed case if applicable (U+0390 is lowercase, U+03AA 'Ϊ' is upper but different)
        // Let's check 'ΰ' (CE B0)
        {"\xCE\xB0", "\xCE\xB0", "Greek ΰ (CE B0) identity"},

        // Greek Symbols (CF 90-96, CF B0-B6)
        // 'ϐ' (CF 90) -> 'β' (CE B2)
        {"\xCF\x90", "\xCE\xB2", "Greek Symbol ϐ to β"},
        {"\xCE\xB2", "\xCF\x90", "Greek Symbol β to ϐ"},
        // 'ϑ' (CF 91) -> 'θ' (CE B8)
        {"\xCF\x91", "\xCE\xB8", "Greek Symbol ϑ to θ"},
        // 'ϕ' (CF 95) -> 'φ' (CF 86) -- Wait, standard 'φ' is CF 86
        // 'ϖ' (CF 96) -> 'π' (CF 80)
        {"\xCF\x96", "\xCF\x80", "Greek Symbol ϖ to π"},

        // Micro Sign (U+00B5, C2 B5) ↔ Greek Mu (U+03BC, CE BC / U+039C, CE 9C)
        // Cross-script folding: Micro Sign folds to Greek lowercase mu
        // This is tricky because Micro Sign is in Latin-1 Supplement (western_group)
        // but must match Greek uppercase/lowercase Mu
        {"\xC2\xB5", "\xCE\xBC", "Micro Sign to Greek lowercase mu"},
        {"\xCE\xBC", "\xC2\xB5", "Greek lowercase mu to Micro Sign"},
        {"\xC2\xB5", "\xCE\x9C", "Micro Sign to Greek uppercase MU"},
        {"\xCE\x9C", "\xC2\xB5", "Greek uppercase MU to Micro Sign"},
        // Multiple occurrences
        {"\xC2\xB5\xC2\xB5", "\xCE\xBC\xCE\xBC", "Two Micro Signs to two Greek mu"},
        {"\xC2\xB5\xC2\xB5", "\xCE\x9C\xCE\x9C", "Two Micro Signs to two Greek MU"},
        // Mixed Micro Sign with Greek context
        {"\xCE\x91\xC2\xB5\xCE\xB1", "\xCE\xB1\xCE\xBC\xCE\xB1", "Greek+Micro+Greek: ΑµΑ to αμα"},
        // Micro Sign in ASCII context (Western Europe kernel selection)
        {"Test \xC2\xB5 unit", "TEST \xCE\xBC UNIT", "Micro Sign in ASCII context"},

        // Armenian Edge Cases
        // -------------------
        // Ligature 'և' (D5 87) -> "եւ" (D5 A5 D6 82) (ech + yiwn)
        // This expands from 2 bytes to 4 bytes.
        {"\xD5\x87", "\xD5\xA5\xD6\x82", "Armenian Ligature և to եւ"},
        {"\xD5\xA5\xD6\x82", "\xD5\x87", "Armenian եւ to Ligature և"},

        // Ligature Men-Now 'ﬓ' (FB 13) -> "մն" (D5 B4 D5 B6)
        {"\xEF\xAC\x93", "\xD5\xB4\xD5\xB6", "Armenian Ligature ﬓ to մն"},
        {"\xD5\xB4\xD5\xB6", "\xEF\xAC\x93", "Armenian մն to Ligature ﬓ"},

        // Ligature Men-Ech 'ﬔ' (FB 14) -> "մե" (D5 B4 D5 A5)
        {"\xEF\xAC\x94", "\xD5\xB4\xD5\xA5", "Armenian Ligature ﬔ to մե"},

        // Armenian path tests (D4/D5/D6 lead bytes)
        // Uppercase: D4 B1-BF (Ա-Խ), D5 80-96 (Ծ-Ֆ) -> Lowercase: D5 A1-BF, D6 80-86
        {"\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E", "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xB5\xD5\xBE",
         "Armenian Բdelays to delays"}, // Barev
        {"\xD5\xA2\xD5\xA1\xD6\x80\xD5\xB5\xD5\xBE", "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E",
         "Armenian delays to ԲDELAYS"},
        {"\xD4\xB1\xD4\xB2\xD4\xB3\xD4\xB4\xD4\xB5", "\xD5\xA1\xD5\xA2\xD5\xA3\xD5\xA4\xD5\xA5",
         "Armenian DELAYS to delays"},
        {"\xD5\xA1\xD5\xA2\xD5\xA3\xD5\xA4\xD5\xA5", "\xD4\xB1\xD4\xB2\xD4\xB3\xD4\xB4\xD4\xB5",
         "Armenian delays to ABGDE"},
        // Armenian with D5/D6 boundary (Ֆ D5 96 -> ֆ D6 86)
        {"\xD5\x94\xD5\x95\xD5\x96", "\xD5\xB4\xD5\xB5\xD6\x86", "Armenian ՔՕՖ to qoف"},
        // Armenian + ASCII mixed
        {"Hello \xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE World", "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E",
         "Armenian in ASCII context"},

        // Vietnamese/Latin Extended Additional path tests (E1 B8-BB lead bytes)
        // Even third byte = uppercase, odd = lowercase
        {"\xE1\xBA\xA0\xE1\xBA\xA2\xE1\xBA\xA4", "\xE1\xBA\xA1\xE1\xBA\xA3\xE1\xBA\xA5", "Vietnamese ẠẢẤ to ạảấ"},
        {"\xE1\xBA\xA1\xE1\xBA\xA3\xE1\xBA\xA5", "\xE1\xBA\xA0\xE1\xBA\xA2\xE1\xBA\xA4", "Vietnamese ạảấ to ẠẢẤ"},
        // Vietnamese with ASCII
        {"Vi\xE1\xBB\x87t Nam", "VI\xE1\xBB\x86T NAM", "Vietnamese Việt to VIỆT"},
        {"VI\xE1\xBB\x86T NAM", "vi\xE1\xBB\x87t nam", "Vietnamese VIỆT to việt"},
        // Latin Extended Additional - Ṁ (E1 B9 80) / ṁ (E1 B9 81)
        {"\xE1\xB9\x80"
         "acedonia",
         "\xE1\xB9\x81"
         "ACEDONIA",
         "Ṁacedonia to ṁACEDONIA"},
        // Mixed Vietnamese + ASCII
        {"Hello \xE1\xBB\x86 World", "\xE1\xBB\x87", "Vietnamese Ệ vs ệ in ASCII"},

        // Latin Extended-B C6 range tests (U+0180-U+01BF)
        // +1 folding patterns (16 chars that stay within C6)
        {"\xC6\x82\xC6\x84", "\xC6\x83\xC6\x85", "C6 +1: Ƃ Ƅ to ƃ ƅ"},
        {"\xC6\x83\xC6\x85", "\xC6\x82\xC6\x84", "C6 +1: ƃ ƅ to Ƃ Ƅ"},
        // Vietnamese Horn letters - important for Vietnamese text
        {"\xC6\xA0\xC6\xAF", "\xC6\xA1\xC6\xB0", "C6 Vietnamese: Ơ Ư to ơ ư"},
        {"\xC6\xA1\xC6\xB0", "\xC6\xA0\xC6\xAF", "C6 Vietnamese: ơ ư to Ơ Ư"},
        {"Vi\xC6\xA1t Nam", "VI\xC6\xA0T NAM", "C6 Vietnamese: Việt Nam mixed"},
        // More +1 patterns
        {"\xC6\x87\xC6\x8B\xC6\x91", "\xC6\x88\xC6\x8C\xC6\x92", "C6 +1 odd: Ƈ Ƌ Ƒ to ƈ ƌ ƒ"},
        {"\xC6\xA7\xC6\xB3\xC6\xB5", "\xC6\xA8\xC6\xB4\xC6\xB6", "C6 +1 odd: Ƨ Ƴ Ƶ to ƨ ƴ ƶ"},
        {"\xC6\x98\xC6\xAC\xC6\xB8\xC6\xBC", "\xC6\x99\xC6\xAD\xC6\xB9\xC6\xBD", "C6 +1 even: Ƙ Ƭ Ƹ Ƽ to ƙ ƭ ƹ ƽ"},

        // C6 cross-block mappings (C6→C9) - West African/Azerbaijani languages
        {"\xC6\x81", "\xC9\x93", "C6→C9: Ɓ to ɓ (Hausa/Fula)"},
        {"\xC9\x93", "\xC6\x81", "C6→C9: ɓ to Ɓ (reverse)"},
        {"\xC6\x86", "\xC9\x94", "C6→C9: Ɔ to ɔ (Akan/Ewe)"},
        {"\xC6\x8F", "\xC9\x99", "C6→C9: Ə to ə (Azerbaijani schwa)"},
        {"\xC9\x99", "\xC6\x8F", "C6→C9: ə to Ə (Azerbaijani reverse)"},
        {"Az\xC9\x99rbaycan", "AZ\xC6\x8FRBAYCAN", "C6 Azerbaijani: Azərbaycan mixed"},
        {"\xC6\x89\xC6\x8A", "\xC9\x96\xC9\x97", "C6→C9: Ɖ Ɗ to ɖ ɗ (Ewe/Hausa)"},
        {"\xC6\x90\xC6\x93\xC6\x94", "\xC9\x9B\xC9\xA0\xC9\xA3", "C6→C9: Ɛ Ɠ Ɣ to ɛ ɠ ɣ (African)"},
        {"\xC6\x96\xC6\x97", "\xC9\xA9\xC9\xA8", "C6→C9: Ɩ Ɨ to ɩ ɨ"},
        {"\xC6\x9C\xC6\x9D\xC6\x9F", "\xC9\xAF\xC9\xB2\xC9\xB5", "C6→C9: Ɯ Ɲ Ɵ to ɯ ɲ ɵ"},

        // C6 cross-block mappings (C6→CA) - IPA/African
        {"\xC6\xA6", "\xCA\x80", "C6→CA: Ʀ to ʀ (Old Norse/IPA)"},
        {"\xCA\x80", "\xC6\xA6", "C6→CA: ʀ to Ʀ (reverse)"},
        {"\xC6\xA9", "\xCA\x83", "C6→CA: Ʃ to ʃ (African/IPA)"},
        {"\xC6\xAE\xC6\xB1\xC6\xB2", "\xCA\x88\xCA\x8A\xCA\x8B", "C6→CA: Ʈ Ʊ Ʋ to ʈ ʊ ʋ"},
        {"\xC6\xB7", "\xCA\x92", "C6→CA: Ʒ to ʒ (Skolt Sami/IPA)"},
        {"\xCA\x92", "\xC6\xB7", "C6→CA: ʒ to Ʒ (reverse)"},

        // C6 cross-block mapping (C6→C7) - just one character
        {"\xC6\x8E", "\xC7\x9D", "C6→C7: Ǝ to ǝ (Pan-Nigerian)"},
        {"\xC7\x9D", "\xC6\x8E", "C6→C7: ǝ to Ǝ (reverse)"},

        // C6 mixed with ASCII
        {"Hello \xC6\x8F World", "\xC9\x99", "C6 schwa in ASCII: Ə vs ə"},
        {"Test \xC6\xA0\xC6\xAF text", "TEST \xC6\xA1\xC6\xB0 TEXT", "C6 Vietnamese in ASCII context"},
        {"Word\xC6\x81word", "WORD\xC9\x93WORD", "C6→C9 in ASCII sandwich"},

        // Georgian path tests (E1 82/83 for Asomtavruli/Mkhedruli, E1 B2 for Mtavruli, E2 B4 for lowercase)
        // Uppercase Asomtavruli Ⴀ (E1 82 A0) -> lowercase ⴀ (E2 B4 80)
        // Mkhedruli ა (E1 83 90) is already lowercase, Mtavruli Ა (E1 B2 90) is uppercase
        {"\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2", "\xE2\xB4\x80\xE2\xB4\x81\xE2\xB4\x82",
         "Georgian Ⴀ Ⴁ Ⴂ to ⴀⴁⴂ (Asomtavruli upper to lower)"},
        {"\xE2\xB4\x80\xE2\xB4\x81\xE2\xB4\x82", "\xE1\x82\xA0\xE1\x82\xA1\xE1\x82\xA2",
         "Georgian ⴀⴁⴂ to ႠႡႢ (lower to Asomtavruli upper)"},
        {"\xE1\xB2\x90\xE1\xB2\x91\xE1\xB2\x92", "\xE1\x83\x90\xE1\x83\x91\xE1\x83\x92",
         "Georgian Ა Ბ Გ to აბგ (Mtavruli upper to Mkhedruli lower)"},
        {"\xE1\x83\x90\xE1\x83\x91\xE1\x83\x92", "\xE1\xB2\x90\xE1\xB2\x91\xE1\xB2\x92",
         "Georgian აბგ to ᲐᲑᲒ (Mkhedruli lower to Mtavruli upper)"},
        // Georgian + ASCII mixed
        {"Hello \xE1\x83\x90\xE1\x83\x91\xE1\x83\x92 World", "\xE1\xB2\x90\xE1\xB2\x91\xE1\xB2\x92",
         "Georgian აბგ in ASCII context"},

        // Cherokee path tests (E1 8E/8F for uppercase, EA AD/AE/AF for lowercase supplement)
        // Cherokee is unusual: lowercase folds TO uppercase (opposite of most scripts!)
        // Uppercase Ꭰ (E1 8E A0) is the fold target of lowercase ꭰ (EA AD B0)
        {"\xE1\x8E\xA0\xE1\x8E\xA1\xE1\x8E\xA2", "\xEA\xAD\xB0\xEA\xAD\xB1\xEA\xAD\xB2",
         "Cherokee Ꭰ Ꭱ Ꭲ vs ꭰꭱꭲ (upper vs lower supplement)"},
        {"\xEA\xAD\xB0\xEA\xAD\xB1\xEA\xAD\xB2", "\xE1\x8E\xA0\xE1\x8E\xA1\xE1\x8E\xA2",
         "Cherokee ꭰꭱꭲ vs ᎠᎡᎢ (lower supplement vs upper)"},
        // Cherokee + ASCII mixed
        {"Hello \xE1\x8E\xA0\xE1\x8E\xA1 World", "\xEA\xAD\xB0\xEA\xAD\xB1", "Cherokee ᎠᎡ in ASCII context"},

        // Coptic path tests (E2 B2/B3 lead bytes) - separate script from Greek
        // Uppercase Ϣ (E2 B2 A2) -> lowercase ϣ (E2 B2 A3)
        {"\xE2\xB2\xA0\xE2\xB2\xA2\xE2\xB2\xA4", "\xE2\xB2\xA1\xE2\xB2\xA3\xE2\xB2\xA5",
         "Coptic Ⲡ Ⲣ Ⲥ to ⲡⲣⲥ (upper to lower)"},
        {"\xE2\xB2\xA1\xE2\xB2\xA3\xE2\xB2\xA5", "\xE2\xB2\xA0\xE2\xB2\xA2\xE2\xB2\xA4",
         "Coptic ⲡⲣⲥ to ⲠⲢⲤ (lower to upper)"},

        // Glagolitic path tests (E2 B0 80-BF for uppercase, E2 B1 80-9F for lowercase)
        // Uppercase Ⰰ (E2 B0 80) -> lowercase ⰰ (E2 B1 80)
        {"\xE2\xB0\x80\xE2\xB0\x81\xE2\xB0\x82", "\xE2\xB1\x80\xE2\xB1\x81\xE2\xB1\x82",
         "Glagolitic Ⰰ Ⰱ Ⰲ to ⰰⰱⰲ (upper to lower)"},
        {"\xE2\xB1\x80\xE2\xB1\x81\xE2\xB1\x82", "\xE2\xB0\x80\xE2\xB0\x81\xE2\xB0\x82",
         "Glagolitic ⰰⰱⰲ to ႠჁჂ (lower to upper)"},

        // Caseless script tests (should trigger fast binary search path)
        // These needles contain NO bicameral characters, so sz_utf8_is_fully_caseless_ returns true
        // Arabic (right-to-left, no textual form in comments)
        {"\xD8\xA7\xD9\x84\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85", "\xD8\xA7\xD9\x84\xD8\xB3\xD9\x84\xD8\xA7\xD9\x85",
         "Arabic caseless exact match"},
        // Hebrew (right-to-left, no textual form in comments)
        {"\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", "\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D", "Hebrew caseless exact match"},
        // CJK - 中文 (zhōngwén)
        {"\xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95", "\xE4\xB8\xAD\xE6\x96\x87",
         "CJK 中文测试 find 中文 (caseless)"},
        // Japanese Hiragana - あいう (aiū)
        {"\xE3\x81\x82\xE3\x81\x84\xE3\x81\x86\xE3\x81\x88\xE3\x81\x8A", "\xE3\x81\x84\xE3\x81\x86",
         "Hiragana あいうえお find いう (caseless)"},
        // Japanese Katakana - アイウ (aiu)
        {"\xE3\x82\xA2\xE3\x82\xA4\xE3\x82\xA6", "\xE3\x82\xA2\xE3\x82\xA4", "Katakana アイウ find アイ (caseless)"},
        // Thai - สวัสดี (sawatdee)
        {"\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5",
         "\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA", "Thai สวัสดี find สวัส (caseless)"},
        // Devanagari - नमस्ते (namaste)
        {"\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87", "\xE0\xA4\xA8\xE0\xA4\xAE",
         "Devanagari नमस्ते find नम (caseless)"},
        // Korean Hangul - 안녕하세요 (annyeonghaseyo)
        {"\xEC\x95\x88\xEB\x85\x95\xED\x95\x98\xEC\x84\xB8\xEC\x9A\x94", "\xEB\x85\x95\xED\x95\x98",
         "Hangul 안녕하세요 find 녕하 (caseless)"},
        // Emoji - caseless symbols
        {"\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82", "\xF0\x9F\x98\x81", "Emoji 😀😁😂 find 😁 (caseless)"},
        // Mixed caseless + cased: CJK haystack with ASCII needle (should NOT match)
        {"\xE4\xB8\xAD\xE6\x96\x87", "abc", "CJK 中文 vs ASCII abc (not found)"},
        // Numbers and punctuation (caseless)
        {"12345!@#$%", "345", "Numbers/punctuation caseless match"},

        // Script-crossing edge cases
        {"\xCE\xB1\xCE\xB2\xCE\xB3"
         "ABC",
         "\xCE\x91\xCE\x92\xCE\x93"
         "abc",
         "Greek to ASCII transition"},
        {"ABC\xCE\xB1\xCE\xB2\xCE\xB3", "abc\xCE\x91\xCE\x92\xCE\x93", "ASCII to Greek transition"},
        {"ABC\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82XYZ",
         "abc\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2xyz", "ASCII-Cyrillic-ASCII sandwich"},

        // Eszett tests (ß ↔ SS)
        {"stra\xC3\x9F"
         "e",
         "STRASSE", "Eszett to SS"}, // straße
        {"STRASSE",
         "stra\xC3\x9F"
         "e",
         "SS to eszett"},                          // reverse
        {"gro\xC3\x9F", "GROSS", "Eszett at end"}, // groß
        {"Fu\xC3\x9F"
         "ball spielen",
         "FUSSBALL", "Eszett mid-word"}, // Fußball

        // Long S tests (ſ U+017F, C5 BF) - folds to regular 's' (U+0073)
        // This is a 2-byte → 1-byte length-changing fold
        {"Progre\xC5\xBF\xC5\xBF", "PROGRESS", "Long S to regular S"},                 // Progreſſ → PROGRESS
        {"PROGRESS", "progre\xC5\xBF\xC5\xBF", "Regular S to Long S"},                 // reverse
        {"\xC5\xBFtring", "STRING", "Long S at start"},                                // ſtring
        {"cla\xC5\xBF\xC5\xBFic", "CLASSIC", "Long S mid-word"},                       // claſſic
        {"\xC5\xBF", "s", "Single Long S to s"},                                       // ſ → s
        {"\xC5\xBF", "S", "Single Long S to S"},                                       // ſ → S
        {"s", "\xC5\xBF", "Single s to Long S"},                                       // s → ſ
        {"Mi\xC5\xBF\xC5\xBFi\xC5\xBF\xC5\xBFippi", "MISSISSIPPI", "Long S multiple"}, // Miſſiſſippi

        // Feminine ordinal indicator (ª U+00AA, C2 AA) - caseless, folds to itself
        {"1\xC2\xAA planta", "1\xC2\xAA PLANTA", "Feminine ordinal with ASCII"}, // 1ª planta
        {"2\xC2\xAA", "2\xC2\xAA", "Feminine ordinal exact match"},              // 2ª
        {"La 3\xC2\xAA vez", "la 3\xC2\xAA VEZ", "Feminine ordinal mixed case"}, // La 3ª vez
        // Masculine ordinal indicator (º U+00BA, C2 BA) - also caseless
        {"1\xC2\xBA lugar", "1\xC2\xBA LUGAR", "Masculine ordinal with ASCII"}, // 1º lugar

        // Multiplication × (C3 97) and Division ÷ (C3 B7) signs - must NOT be folded to each other
        {"2\xC3\x97"
         "3=6",
         "\xC3\x97", "Find multiplication sign"}, // 2×3=6, find ×
        {"6\xC3\xB7"
         "2=3",
         "\xC3\xB7", "Find division sign"}, // 6÷2=3, find ÷
        {"2\xC3\x97"
         "3=6",
         "\xC3\xB7", "Mult not equal to div (not found)"}, // × ≠ ÷
        {"6\xC3\xB7"
         "2=3",
         "\xC3\x97", "Div not equal to mult (not found)"},             // ÷ ≠ ×
        {"\xC3\x97\xC3\xB7", "\xC3\x97", "Mult in mult-div sequence"}, // ×÷, find ×
        {"\xC3\x97\xC3\xB7", "\xC3\xB7", "Div in mult-div sequence"},  // ×÷, find ÷
        {"A\xC3\x97"
         "B",
         "a\xC3\x97"
         "b",
         "Mult with case-insensitive ASCII"}, // A×B vs a×b

        // Not found cases
        {"Hello World", "xyz", "Not found ASCII"},
        {"\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", "xyz", "Not found in Cyrillic"},
        {"Hello World", "\xD0\x9F\xD1\x80\xD0\xB8", "Cyrillic needle not in ASCII"},

        // Regression test: SIMD returned NULL when serial found match
        // Needle contains: ǰ (Latin Extended-B), ẞ (capital sharp S expands to "ss"), Turkish ı, emoji
        // Safe window is "Ithe" at bytes 31-34 which folds to "ithe"
        // Bug triggered by specific prefix content with Armenian, Greek, CJK, ligatures
        {"\x66\x6F\x78\x74\xD0\xB2\x58\x77\x58\x20\x67\x31\x5A\xEF\xAC\x82"
         "\x46\x21\xC3\xA0\x31\x21\xC6\xA0\xEF\xAC\x85\x57\x6F\x72\x6C\x64"
         "\xC4\x91\xE4\xB8\xAD\xE6\x96\x87\x43\xCF\x83\xE3\x81\x82\xE3\x81"
         "\x84\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E\xC4\xB1\x6E\x32\xE4"
         "\xB8\xAD\xE6\x96\x87\x42\x30\x6E\xC3\x9F\x55\xCE\xBA\xCF\x8C\xCF"
         "\x83\xCE\xBC\x30\x62\x72\x6F\x77\x6E\xCF\x83\x67\x66\x6F\x78\x21"
         "\xC2\xB5\x4D\xE4\xB8\xAD\xE6\x96\x87\xC7\xB0\xE1\xBB\x86\xC4\xB0"
         "\x6A\x75\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F"
         "\x98\x80\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3\xB1\x45\x7A\xC3"
         "\xBC\x49\x74\x68\x65\x61\xC5\xBF\xC3\x80\xC3\x85\xD0\x91\xC5\xBF"
         "\x4C\x20\xC4\xB0\xCE\x91\x2C\x67\xE1\xBA\x96\xC3\xA0\x77\xC3\x91"
         "\x4D\x52\xE1\xBA\xA1\x4A\xC6\xA0\xEF\xAC\x85\xE1\xBA\x9E\xF0\x9F"
         "\x98\x80\xEF\xAC\x80\xD0\xB1\xCF\x82\x65\x4B\x7A\xC3\xB1\x65\xC3"
         "\x9C\x64\xC3\xB1\x55\xD0\xB0\xC3\xA4\x67\x41\x7A\xE1\xBB\x87\x5A"
         "\x4A\x71\x76\xC3\x89\xC6\xA0\x45\xCE\x91\x66\x67\x6F\x41\xC3\x85"
         "\x4F\x6B\x58\xC3\xB1\x52\xE1\xBA\x98\xE1\xBA\xA1\x63\x47\xC2\xAA"
         "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E\xC3\x89\x77\x31\x46\xCF"
         "\x82\x76\xCE\xA3\x56\x56\xCA\xBE\xE1\xBA\x96\xD0\x91\x6F\xCE\x92"
         "\x6A\x75\x6D\x70\x73\x33\xE1\xBA\xA1\x6A\x75\x6D\x70\x73\xE1\xBA"
         "\x98\xC3\x9F\xC3\x9C\xC6\xA1\x59\xEF\xAC\x86\x59\x56\x2E\x33\xC3"
         "\xA9\x7A\x4C\x4C",
         "\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80"
         "\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3\xB1\x45\x7A\xC3\xBC\x49"
         "\x74\x68\x65",
         "Regression: complex needle with expansion chars in mixed-script haystack"},

        // Edge cases
        {"", "", "Both empty"},
        {"Hello", "", "Empty needle"},
        {"", "Hello", "Empty haystack"},
        {"A", "a", "Single char match"},
        {"a", "A", "Single char reverse"},
    };

    std::size_t num_cases = sizeof(cases) / sizeof(cases[0]);
    std::size_t passed = 0;

    for (std::size_t i = 0; i < num_cases; ++i) {
        test_case const &tc = cases[i];
        sz_size_t serial_matched_len = 0, simd_matched_len = 0;
        sz_size_t h_len = std::strlen(tc.haystack);
        sz_size_t n_len = std::strlen(tc.needle);

        // Method 1 & 2: Case-insensitive find (serial vs SIMD)
        sz_utf8_case_insensitive_needle_metadata_t serial_metadata = {}, simd_metadata = {};
        sz_cptr_t serial_match =
            find_serial(tc.haystack, h_len, tc.needle, n_len, &serial_metadata, &serial_matched_len);
        sz_cptr_t simd_match = find_simd(tc.haystack, h_len, tc.needle, n_len, &simd_metadata, &simd_matched_len);

        // Method 3: Fold haystack + fold needle + regular sz_find (independent oracle)
        sz_size_t folded_h_len = case_fold(tc.haystack, h_len, folded_haystack_buf.data());
        sz_size_t folded_n_len = case_fold(tc.needle, n_len, folded_needle_buf.data());
        sz_cptr_t oracle_match = (folded_n_len == 0) ? folded_haystack_buf.data()
                                                     : sz_find(folded_haystack_buf.data(), folded_h_len,
                                                               folded_needle_buf.data(), folded_n_len);

        // Compute oracle char offset first (before we overwrite the buffer)
        sz_size_t oracle_char_offset =
            oracle_match ? utf8_count(folded_haystack_buf.data(), oracle_match - folded_haystack_buf.data())
                         : SZ_SIZE_MAX;

        // Convert match positions to normalized character offsets by folding the prefix up to the match
        // and counting characters in the folded result. This accounts for length-changing folds like ß→ss.
        auto compute_folded_char_offset = [&](sz_cptr_t match, sz_cptr_t haystack_start) -> sz_size_t {
            if (!match) return SZ_SIZE_MAX;
            sz_size_t prefix_byte_len = match - haystack_start;
            sz_size_t folded_prefix_len = case_fold(haystack_start, prefix_byte_len, folded_haystack_buf.data());
            return utf8_count(folded_haystack_buf.data(), folded_prefix_len);
        };

        sz_size_t serial_folded_char_offset = compute_folded_char_offset(serial_match, tc.haystack);
        sz_size_t simd_folded_char_offset = compute_folded_char_offset(simd_match, tc.haystack);

        // Check implementations agree on normalized character position and length
        bool implementations_agree = (serial_folded_char_offset == simd_folded_char_offset);
        bool lengths_agree = (serial_matched_len == simd_matched_len);
        // For oracle, we only check existence agreement because one-to-many folds (like ligatures "ﬁ"→"fi")
        // can cause the case-insensitive search and fold+find to locate different matches in the haystack.
        // For example, if haystack has both "ﬁ" and "fi", case-insensitive search finds whichever comes
        // first in the original, while fold+find finds whichever comes first in the folded version.
        bool oracle_agrees = (serial_match != SZ_NULL_CHAR) == (oracle_match != SZ_NULL_CHAR);

        if (!implementations_agree || !lengths_agree || !oracle_agrees) {
            std::fprintf(stderr, "FAIL: %s\n", tc.description);
            std::fprintf(stderr, "  Haystack (%zu bytes): ", h_len);
            for (std::size_t j = 0; j < h_len && j < 50; ++j)
                std::fprintf(stderr, "%02X ", (unsigned char)tc.haystack[j]);
            std::fprintf(stderr, "\n  Needle (%zu bytes): ", n_len);
            for (std::size_t j = 0; j < n_len && j < 50; ++j)
                std::fprintf(stderr, "%02X ", (unsigned char)tc.needle[j]);
            std::fprintf(stderr, "\n");
            std::fprintf(stderr, "  Serial: folded_char_offset=%zu, len=%zu\n",
                         serial_folded_char_offset == SZ_SIZE_MAX ? (sz_size_t)-1 : serial_folded_char_offset,
                         serial_matched_len);
            std::fprintf(stderr, "  SIMD: folded_char_offset=%zu, len=%zu\n",
                         simd_folded_char_offset == SZ_SIZE_MAX ? (sz_size_t)-1 : simd_folded_char_offset,
                         simd_matched_len);
            std::fprintf(stderr, "  SIMD metadata: kernel_id=%u, safe_window.offset=%zu, safe_window.length=%zu\n",
                         simd_metadata.kernel_id, simd_metadata.safe_window.offset, simd_metadata.safe_window.length);
            std::fprintf(stderr, "  SIMD metadata: folded_slice_length=%u, probe_second=%u, probe_third=%u\n",
                         simd_metadata.folded_slice_length, simd_metadata.probe_second, simd_metadata.probe_third);
            std::fprintf(stderr, "  SIMD metadata folded_slice: ");
            for (sz_size_t j = 0; j < simd_metadata.folded_slice_length && j < 16; ++j)
                std::fprintf(stderr, "%02X ", simd_metadata.folded_slice[j]);
            std::fprintf(stderr, "\n");
            std::fprintf(stderr, "  Oracle: char_offset=%zu (folded_h=%zu, folded_n=%zu)\n",
                         oracle_char_offset == SZ_SIZE_MAX ? (sz_size_t)-1 : oracle_char_offset, folded_h_len,
                         folded_n_len);
            if (!oracle_agrees) {
                // Re-fold to show folded content
                case_fold(tc.haystack, h_len, folded_haystack_buf.data());
                std::fprintf(stderr, "  Folded haystack: ");
                for (std::size_t j = 0; j < folded_h_len && j < 50; ++j)
                    std::fprintf(stderr, "%02X ", (unsigned char)folded_haystack_buf[j]);
                std::fprintf(stderr, "\n  Folded needle: ");
                for (std::size_t j = 0; j < folded_n_len && j < 50; ++j)
                    std::fprintf(stderr, "%02X ", (unsigned char)folded_needle_buf[j]);
                std::fprintf(stderr, "\n");
            }
            assert(implementations_agree && "Folded character offset mismatch between serial and SIMD");
            assert(lengths_agree && "Length mismatch between serial and SIMD");
            assert(oracle_agrees && "Oracle character offset disagrees with case-insensitive find");
        }
        ++passed;
    }

    std::printf("    passed %zu/%zu basic equivalence tests\n", passed, num_cases);
}

/**
 *  @brief  Fuzz tests case-insensitive UTF-8 substring search with random haystacks.
 *
 *  Uses three independent verification methods:
 *  1. Serial case-insensitive find (baseline)
 *  2. SIMD case-insensitive find (under test)
 *  3. Fold haystack + fold needle + regular sz_find (independent oracle)
 *
 *  Builds random haystacks from a pool of normal and edge-case Unicode characters, then picks valid UTF-8
 *  slices as needles. Covers ASCII, Latin-1, Cyrillic, Greek, Armenian, Vietnamese, ligatures, and
 *  one-to-many folding cases like Eszett and Turkish dotted I.
 */
void test_utf8_ci_find_fuzz(sz_utf8_case_insensitive_find_t find_serial, sz_utf8_case_insensitive_find_t find_simd,
                            sz_utf8_case_fold_t case_fold, sz_utf8_case_agnostic_t agnostic_serial,
                            sz_utf8_case_agnostic_t agnostic_simd, sz_utf8_find_nth_t utf8_find_nth,
                            sz_utf8_count_t utf8_count, std::size_t iterations = scale_iterations(5000)) {
    std::printf("  - fuzz testing with random haystacks (%zu iterations)...\n", iterations);

    // Buffer for folded needle (case folding can expand, e.g., ß→ss, so 4x is safe)
    constexpr std::size_t max_folded_size = 8192;
    std::vector<char> folded_needle_buf(max_folded_size);

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

        // Latin-1/Extended (Western European)
        "\xC3\x9F", // Eszett ß (folds to ss)
        "\xC3\xB6",
        "\xC3\x96", // o-umlaut
        "\xC3\xBC",
        "\xC3\x9C", // u-umlaut
        "\xC3\xA4",
        "\xC3\x84", // a-umlaut
        "\xC3\xA9",
        "\xC3\x89", // e-accent
        "\xC3\xA0",
        "\xC3\x80", // a-grave
        "\xC3\xB1",
        "\xC3\x91", // n-tilde
        "\xC2\xAA", // FEMININE ORDINAL INDICATOR (caseless)
        "\xC2\xBA", // MASCULINE ORDINAL INDICATOR (caseless)
        "\xC2\xB5", // MICRO SIGN (folds to Greek mu)
        "\xC3\x85",
        "\xC3\xA5", // A-ring (Angstrom target)
        "\xC5\xBF", // LONG S ſ (U+017F, folds to regular s)

        // Kelvin sign (U+212A) - folds to ASCII k
        "\xE2\x84\xAA",
        // Angstrom sign (U+212B) - folds to Latin-1 a-ring
        "\xE2\x84\xAB",
        // Turkish dotted I (U+0130) - expands to i + combining dot
        "\xC4\xB0",
        // Turkish dotless i (U+0131) - folds to itself but maps to I (U+0049)
        "\xC4\xB1",

        // Cyrillic (Russian, Ukrainian)
        "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82", // Cyrillic privet
        "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0", // Cyrillic Moskva
        "\xD0\xB0",
        "\xD0\x90", // a
        "\xD0\xB1",
        "\xD0\x91", // b
        "\xD0\xB2",
        "\xD0\x92", // v

        // Greek (including final sigma)
        "\xCE\xB1",
        "\xCE\x91", // alpha
        "\xCE\xB2",
        "\xCE\x92", // beta
        "\xCF\x83",
        "\xCE\xA3",                         // sigma
        "\xCF\x82",                         // final sigma
        "\xCE\xBA\xCF\x8C\xCF\x83\xCE\xBC", // Greek kosm

        // Armenian
        "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE", // Armenian barev
        "\xD4\xB2\xD4\xB1\xD5\x90\xD4\xB5\xD5\x8E", // Armenian BAREV
        "\xD5\xA5",
        "\xD6\x87", // ech, ech-yiwn ligature

        // Vietnamese/Latin Extended Additional
        "\xE1\xBB\x87",
        "\xE1\xBB\x86", // e with hook + circumflex
        "\xE1\xBA\xA1",
        "\xE1\xBA\xA0", // a with dot below
        "\xC4\x90",
        "\xC4\x91", // D with stroke (Latin Ext-A)
        "\xC6\xA0",
        "\xC6\xA1", // O with horn (Latin Ext-B)

        // Latin ligatures (one-to-many folding)
        "\xEF\xAC\x80", // ff ligature
        "\xEF\xAC\x81", // fi ligature
        "\xEF\xAC\x82", // fl ligature
        "\xEF\xAC\x83", // ffi ligature
        "\xEF\xAC\x84", // ffl ligature
        "\xEF\xAC\x85", // st ligature
        "\xEF\xAC\x86", // st ligature (variant)

        // One-to-many expansions (U+1E96-1E9A)
        "\xE1\xBA\x96", // h with line below -> h + combining line
        "\xE1\xBA\x97", // t with diaeresis -> t + combining diaeresis
        "\xE1\xBA\x98", // w with ring -> w + combining ring
        "\xE1\xBA\x99", // y with ring -> y + combining ring
        "\xE1\xBA\x9A", // a with right half ring -> a + modifier letter

        // Capital Eszett (U+1E9E) - folds to ss
        "\xE1\xBA\x9E",
        // Afrikaans n-apostrophe (U+0149) -> 'n
        "\xC5\x89",
        // J-caron (U+01F0) -> j + combining caron
        "\xC7\xB0",
        // Modifier letter apostrophe (U+02BC) - context for n
        "\xCA\xBC",
        // Modifier letter right half ring (U+02BE) - context for a
        "\xCA\xBE",

        // Caseless scripts (for coverage)
        "\xE4\xB8\xAD\xE6\x96\x87", // CJK zhongwen
        "\xE3\x81\x82\xE3\x81\x84", // Hiragana ai
        "\xF0\x9F\x98\x80",         // Emoji grinning face
    };
    std::size_t const pool_size = sizeof(char_pool) / sizeof(char_pool[0]);

    std::uniform_int_distribution<std::size_t> pool_dist(0, pool_size - 1);
    std::uniform_int_distribution<std::size_t> haystack_len_dist(50, 200);

    std::size_t fuzz_passed = 0;

    // Scale iterations proportionally, with reasonable bounds
    std::size_t const max_iters_per_haystack = 500;
    std::size_t const min_rounds = 5;
    std::size_t const num_rounds = std::max(min_rounds, iterations / max_iters_per_haystack);
    std::size_t const iters_per_round = (iterations + num_rounds - 1) / num_rounds; // Round up

    for (std::size_t round = 0; round < num_rounds; ++round) {
        // Build random haystack by concatenating pool entries
        std::size_t target_chars = haystack_len_dist(rng);
        std::string haystack;
        for (std::size_t i = 0; i < target_chars; ++i) { haystack += char_pool[pool_dist(rng)]; }

        // Count actual characters in haystack (may differ from target due to multi-byte chars)
        sz_size_t haystack_char_count = utf8_count(haystack.data(), haystack.size());
        if (haystack_char_count < 2) continue;

        std::uniform_int_distribution<sz_size_t> char_dist(0, haystack_char_count);

        for (std::size_t iter = 0; iter < iters_per_round; ++iter) {
            // Pick random character range for needle
            sz_size_t start_char = char_dist(rng);
            sz_size_t end_char = char_dist(rng);
            if (start_char > end_char) std::swap(start_char, end_char);
            if (start_char == end_char) {
                if (end_char < haystack_char_count) ++end_char;
                else if (start_char > 0)
                    --start_char;
                else
                    continue;
            }

            // Find byte positions using sz_utf8_find_nth (no dynamic allocation)
            sz_cptr_t needle_start_ptr =
                (start_char == 0) ? haystack.data() : utf8_find_nth(haystack.data(), haystack.size(), start_char);
            sz_cptr_t needle_end_ptr = (end_char == haystack_char_count)
                                           ? haystack.data() + haystack.size()
                                           : utf8_find_nth(haystack.data(), haystack.size(), end_char);

            if (!needle_start_ptr || !needle_end_ptr || needle_end_ptr <= needle_start_ptr) continue;

            sz_size_t needle_byte_len = needle_end_ptr - needle_start_ptr;
            sz_cptr_t needle_data = needle_start_ptr;

            // Verify case-agnostic agreement between implementations
            sz_bool_t serial_agnostic_h = agnostic_serial(haystack.data(), haystack.size());
            sz_bool_t simd_agnostic_h = agnostic_simd(haystack.data(), haystack.size());
            sz_bool_t serial_agnostic_n = agnostic_serial(needle_data, needle_byte_len);
            sz_bool_t simd_agnostic_n = agnostic_simd(needle_data, needle_byte_len);

            if (serial_agnostic_h != simd_agnostic_h || serial_agnostic_n != simd_agnostic_n) {
                std::fprintf(stderr, "CASE-AGNOSTIC MISMATCH round=%zu iter=%zu\n", round, iter);
                if (serial_agnostic_h != simd_agnostic_h) {
                    std::fprintf(stderr, "  Haystack: serial=%d, simd=%d (len=%zu)\n", serial_agnostic_h,
                                 simd_agnostic_h, haystack.size());
                    std::fprintf(stderr, "  Haystack bytes: ");
                    for (std::size_t j = 0; j < haystack.size() && j < 50; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)haystack[j]);
                    std::fprintf(stderr, "...\n");
                }
                if (serial_agnostic_n != simd_agnostic_n) {
                    std::fprintf(stderr, "  Needle: serial=%d, simd=%d (len=%zu)\n", serial_agnostic_n, simd_agnostic_n,
                                 needle_byte_len);
                    std::fprintf(stderr, "  Needle bytes: ");
                    for (sz_size_t j = 0; j < needle_byte_len && j < 50; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)needle_data[j]);
                    std::fprintf(stderr, "\n");
                }
                assert(serial_agnostic_h == simd_agnostic_h && "Case-agnostic mismatch on haystack");
                assert(serial_agnostic_n == simd_agnostic_n && "Case-agnostic mismatch on needle");
            }

            // If needle is case-agnostic, folded version must equal original
            if (serial_agnostic_n) {
                sz_size_t folded_len = case_fold(needle_data, needle_byte_len, folded_needle_buf.data());
                if (folded_len != needle_byte_len ||
                    std::memcmp(needle_data, folded_needle_buf.data(), needle_byte_len) != 0) {
                    std::fprintf(stderr, "CASE-AGNOSTIC FOLD MISMATCH round=%zu iter=%zu\n", round, iter);
                    std::fprintf(stderr, "  Needle (len=%zu): ", needle_byte_len);
                    for (sz_size_t j = 0; j < needle_byte_len && j < 50; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)needle_data[j]);
                    std::fprintf(stderr, "\n  Folded (len=%zu): ", folded_len);
                    for (std::size_t j = 0; j < folded_len && j < 50; ++j)
                        std::fprintf(stderr, "%02X ", (unsigned char)folded_needle_buf[j]);
                    std::fprintf(stderr, "\n");
                    assert(false && "Case-agnostic string changed when folded");
                }
            }

            // Compare serial vs SIMD implementations
            // Both operate on the same haystack, so we can compare pointers directly
            sz_size_t serial_matched_len = 0, simd_matched_len = 0;
            sz_utf8_case_insensitive_needle_metadata_t serial_metadata = {}, simd_metadata = {};
            sz_cptr_t serial_match = find_serial(haystack.data(), haystack.size(), needle_data, needle_byte_len,
                                                 &serial_metadata, &serial_matched_len);
            sz_cptr_t simd_match = find_simd(haystack.data(), haystack.size(), needle_data, needle_byte_len,
                                             &simd_metadata, &simd_matched_len);

            // Check implementations agree on position (pointer) and matched length
            // Note: We don't use an oracle here because the needle is always a substring of the haystack,
            // so existence is guaranteed. The oracle (fold+find) can also find different matches than
            // case-insensitive search due to one-to-many folds like ligatures.
            bool implementations_agree = (serial_match == simd_match);
            bool lengths_agree = (serial_matched_len == simd_matched_len);

            if (!implementations_agree || !lengths_agree) {
                std::fprintf(stderr, "FUZZ FAIL round=%zu iter=%zu\n", round, iter);
                std::fprintf(stderr, "  Haystack len=%zu, needle len=%zu\n", haystack.size(), needle_byte_len);
                std::fprintf(stderr, "  Needle bytes: ");
                for (sz_size_t j = 0; j < needle_byte_len && j < 50; ++j)
                    std::fprintf(stderr, "%02X ", (unsigned char)needle_data[j]);
                std::fprintf(stderr, "\n");
                sz_size_t serial_offset = serial_match ? (sz_size_t)(serial_match - haystack.data()) : SZ_SIZE_MAX;
                sz_size_t simd_offset = simd_match ? (sz_size_t)(simd_match - haystack.data()) : SZ_SIZE_MAX;
                std::fprintf(stderr, "  Serial: offset=%zu, len=%zu\n",
                             serial_offset == SZ_SIZE_MAX ? (sz_size_t)-1 : serial_offset, serial_matched_len);
                std::fprintf(stderr, "  SIMD: offset=%zu, len=%zu\n",
                             simd_offset == SZ_SIZE_MAX ? (sz_size_t)-1 : simd_offset, simd_matched_len);
                std::fprintf(stderr, "  SIMD metadata: kernel_id=%u, safe_window.offset=%zu, safe_window.length=%zu\n",
                             simd_metadata.kernel_id, simd_metadata.safe_window.offset,
                             simd_metadata.safe_window.length);
                std::fprintf(stderr, "  SIMD metadata: folded_slice_length=%u, probe_second=%u, probe_third=%u\n",
                             simd_metadata.folded_slice_length, simd_metadata.probe_second, simd_metadata.probe_third);
                std::fprintf(stderr, "  SIMD metadata folded_slice: ");
                for (sz_size_t j = 0; j < simd_metadata.folded_slice_length && j < 16; ++j)
                    std::fprintf(stderr, "%02X ", simd_metadata.folded_slice[j]);
                std::fprintf(stderr, "\n");
                assert(implementations_agree && "Fuzz byte offset mismatch");
                assert(lengths_agree && "Fuzz length mismatch");
            }
            ++fuzz_passed;
        }
    }

    std::printf("    passed %zu random slice tests\n", fuzz_passed);
}

void test_equivalence() {

    // Ensure the seed affects hash results
    assert(sz_hash_serial("abc", 3, 100) != sz_hash_serial("abc", 3, 200));
    assert(sz_hash_serial("abcdefgh", 8, 0) != sz_hash_serial("abcdefgh", 8, 7));

#if SZ_USE_WESTMERE
    test_hash_equivalence(                                        //
        sz_hash_serial, sz_hash_state_init_serial,                //
        sz_hash_state_update_serial, sz_hash_state_digest_serial, //
        sz_hash_westmere, sz_hash_state_init_westmere,            //
        sz_hash_state_update_westmere, sz_hash_state_digest_westmere);
    test_random_generator_equivalence(sz_fill_random_serial, sz_fill_random_westmere);
#endif
#if SZ_USE_SKYLAKE
    test_hash_equivalence(                                        //
        sz_hash_serial, sz_hash_state_init_serial,                //
        sz_hash_state_update_serial, sz_hash_state_digest_serial, //
        sz_hash_skylake, sz_hash_state_init_skylake,              //
        sz_hash_state_update_skylake, sz_hash_state_digest_skylake);
    test_random_generator_equivalence(sz_fill_random_serial, sz_fill_random_skylake);
#endif
#if SZ_USE_ICE
    test_hash_equivalence(                                        //
        sz_hash_serial, sz_hash_state_init_serial,                //
        sz_hash_state_update_serial, sz_hash_state_digest_serial, //
        sz_hash_ice, sz_hash_state_init_ice,                      //
        sz_hash_state_update_ice, sz_hash_state_digest_ice);
    test_random_generator_equivalence(sz_fill_random_serial, sz_fill_random_ice);
#endif
#if SZ_USE_NEON_AES
    test_hash_equivalence(                                        //
        sz_hash_serial, sz_hash_state_init_serial,                //
        sz_hash_state_update_serial, sz_hash_state_digest_serial, //
        sz_hash_neon, sz_hash_state_init_neon,                    //
        sz_hash_state_update_neon, sz_hash_state_digest_neon);
    test_random_generator_equivalence(sz_fill_random_serial, sz_fill_random_neon);
#endif
#if SZ_USE_SVE2_AES
    test_hash_equivalence(                                        //
        sz_hash_serial, sz_hash_state_init_serial,                //
        sz_hash_state_update_serial, sz_hash_state_digest_serial, //
        sz_hash_sve2, sz_hash_state_init_sve2,                    //
        sz_hash_state_update_sve2, sz_hash_state_digest_sve2);
    test_random_generator_equivalence(sz_fill_random_serial, sz_fill_random_sve2);
#endif

    // Test SHA256 implementations
#if SZ_USE_ICE
    test_sha256_equivalence(                                                                       //
        sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial, //
        sz_sha256_state_init_ice, sz_sha256_state_update_ice, sz_sha256_state_digest_ice           //
    );
#endif
#if SZ_USE_GOLDMONT
    test_sha256_equivalence(                                                                            //
        sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial,      //
        sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont, sz_sha256_state_digest_goldmont //
    );
#endif
#if SZ_USE_NEON_SHA
    test_sha256_equivalence(                                                                       //
        sz_sha256_state_init_serial, sz_sha256_state_update_serial, sz_sha256_state_digest_serial, //
        sz_sha256_state_init_neon, sz_sha256_state_update_neon, sz_sha256_state_digest_neon        //
    );
#endif

    // Test UTF-8 functions
#if SZ_USE_HASWELL
    test_utf8_equivalence(                           //
        sz_utf8_count_serial, sz_utf8_count_haswell, //
        sz_utf8_find_newline_serial,                 //
        sz_utf8_find_newline_haswell,                //
        sz_utf8_find_whitespace_serial,              //
        sz_utf8_find_whitespace_haswell);
#endif
#if SZ_USE_ICE
    test_utf8_equivalence(                       //
        sz_utf8_count_serial, sz_utf8_count_ice, //
        sz_utf8_find_newline_serial,             //
        sz_utf8_find_newline_ice,                //
        sz_utf8_find_whitespace_serial,          //
        sz_utf8_find_whitespace_ice);

    test_utf8_case_fold_equivalence(sz_utf8_case_fold_serial, sz_utf8_case_fold_ice);
    test_utf8_case_fold_fuzz(sz_utf8_case_fold_serial, sz_utf8_case_fold_ice);
    test_utf8_ci_find_equivalence(sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice,
                                  sz_utf8_case_fold_serial, sz_utf8_count_serial);
    test_utf8_ci_find_fuzz(sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice,
                           sz_utf8_case_fold_serial, sz_utf8_case_agnostic_serial, sz_utf8_case_agnostic_ice,
                           sz_utf8_find_nth_serial, sz_utf8_count_serial);
#endif
#if SZ_USE_NEON
    test_utf8_equivalence(                        //
        sz_utf8_count_serial, sz_utf8_count_neon, //
        sz_utf8_find_newline_serial,              //
        sz_utf8_find_newline_neon,                //
        sz_utf8_find_whitespace_serial,           //
        sz_utf8_find_whitespace_neon);
#endif
#if SZ_USE_SVE2
    test_utf8_equivalence(                        //
        sz_utf8_count_serial, sz_utf8_count_sve2, //
        sz_utf8_find_newline_serial,              //
        sz_utf8_find_newline_sve2,                //
        sz_utf8_find_whitespace_serial,           //
        sz_utf8_find_whitespace_sve2);
#endif
};

/**
 *  @brief  Tests various ASCII-based methods (e.g., `is_alpha`, `is_digit`)
 *          provided by `sz::string` and `sz::string_view`.
 */
template <typename string_type>
void test_ascii_utilities() {

    using str = string_type;

    assert("aaa"_bs.size() == 1ull);
    assert("\0\0"_bs.size() == 1ull);
    assert("abc"_bs.size() == 3ull);
    assert("a\0bc"_bs.size() == 4ull);

    assert(!"abc"_bs.contains('\0'));
    assert(str("bca").contains_only("abc"_bs));

    assert(!str("").is_alpha());
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ").is_alpha());
    assert(!str("abc9").is_alpha());

    assert(!str("").is_alnum());
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").is_alnum());
    assert(!str("abc!").is_alnum());

    assert(str("").is_ascii());
    assert(str("\x00x7F").is_ascii());
    assert(!str("abc123🔥").is_ascii());

    assert(!str("").is_digit());
    assert(str("0123456789").is_digit());
    assert(!str("012a").is_digit());

    assert(!str("").is_lower());
    assert(str("abcdefghijklmnopqrstuvwxyz").is_lower());
    assert(!str("abcA").is_lower());
    assert(!str("abc\n").is_lower());

    assert(!str("").is_space());
    assert(str(" \t\n\r\f\v").is_space());
    assert(!str(" \t\r\na").is_space());

    assert(!str("").is_upper());
    assert(str("ABCDEFGHIJKLMNOPQRSTUVWXYZ").is_upper());
    assert(!str("ABCa").is_upper());

    assert(str("").is_printable());
    assert(str("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+").is_printable());
    assert(!str("012🔥").is_printable());

    assert(str("").contains_only("abc"_bs));
    assert(str("abc").contains_only("abc"_bs));
    assert(!str("abcd").contains_only("abc"_bs));
}

inline void expect_equality(char const *a, char const *b, std::size_t size) {
    if (std::memcmp(a, b, size) == 0) return;
    std::size_t mismatch_position = 0;
    for (; mismatch_position < size; ++mismatch_position)
        if (a[mismatch_position] != b[mismatch_position]) break;
    std::fprintf(stderr, "Mismatch at position %zu: %c != %c\n", mismatch_position, a[mismatch_position],
                 b[mismatch_position]);
    assert(false);
}

/**
 *  @brief  Validates that `sz::memcpy`, `sz::memset`, and `sz::memmove` work similar to their `std::` counterparts.
 *
 *  Uses a large heap-allocated buffer to ensure that operations optimized for @b larger-than-L2-cache memory
 *  regions are tested. Covers various chunk sizes, overlapping regions, and both forward and backward traversals.
 */
void test_memory_utilities(std::size_t max_l2_size = 1024ull * 1024ull) {

    // We will be mirroring the operations on both standard and StringZilla strings.
    std::string text_stl(max_l2_size, '-');
    std::string text_sz(max_l2_size, '-');
    expect_equality(text_stl.data(), text_sz.data(), max_l2_size);

    // The traditional `memset` and `memcpy` functions are undefined for zero-length buffers and NULL pointers
    // for older C standards.  However, with the N3322 proposal for C2y, that issue has been resolved.
    // https://developers.redhat.com/articles/2024/12/11/making-memcpynull-null-0-well-defined
    //
    // Let's make sure, that our versions don't trigger any undefined behavior.
    sz::memset(NULL, 0, 0);
    sz::memcpy(NULL, NULL, 0);
    sz::memmove(NULL, NULL, 0);

    // First start with simple deterministic tests.
    // Let's use `memset` to fill the strings with a pattern like "122333444455555...00000000000011111111111..."
    std::size_t count_groups = 0;
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size;
         offset += fill_length, ++fill_length, ++count_groups) {
        char fill_value = '0' + fill_length % 10;
        fill_length = offset + fill_length > max_l2_size ? max_l2_size - offset : fill_length;
        std::memset((void *)(text_stl.data() + offset), fill_value, fill_length);
        sz::memset((void *)(text_sz.data() + offset), fill_value, fill_length);
        expect_equality(text_stl.data(), text_sz.data(), max_l2_size);
    }

    // Let's copy those chunks to an empty buffer one by one, validating the overall equivalency after every copy.
    std::string copy_stl(max_l2_size, '-');
    std::string copy_sz(max_l2_size, '-');
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size; offset += fill_length, ++fill_length) {
        fill_length = offset + fill_length > max_l2_size ? max_l2_size - offset : fill_length;
        std::memcpy((void *)(copy_stl.data() + offset), (void *)(text_stl.data() + offset), fill_length);
        sz::memcpy((void *)(copy_sz.data() + offset), (void *)(text_sz.data() + offset), fill_length);
        expect_equality(copy_stl.data(), copy_sz.data(), max_l2_size);
    }
    expect_equality(text_stl.data(), copy_stl.data(), max_l2_size);
    expect_equality(text_sz.data(), copy_sz.data(), max_l2_size);

    // Let's simulate a realistic `memmove` workloads, compacting parts of this buffer, removing all odd values,
    // so the buffer will look like "224444666666..."
    for (std::size_t offset = 0, fill_length = 1; offset < max_l2_size; offset += fill_length, ++fill_length) {
        if (fill_length % 2 == 0) continue;             // Skip even chunks
        if (offset + fill_length >= max_l2_size) break; // This is the last & there are no more even chunks to shift

        // Make sure we don't overflow the buffer
        std::size_t next_offset = offset + fill_length;
        std::size_t next_fill_length = fill_length + 1;
        next_fill_length = next_offset + next_fill_length > max_l2_size ? max_l2_size - next_offset : next_fill_length;

        std::memmove((void *)(text_stl.data() + offset), (void *)(text_stl.data() + next_offset), next_fill_length);
        sz::memmove((void *)(text_sz.data() + offset), (void *)(text_sz.data() + next_offset), next_fill_length);
        expect_equality(text_stl.data(), text_sz.data(), max_l2_size);
    }

    // Now the opposite workload, expanding the buffer, inserting a dash "-" before every group of equal characters.
    // We will need to navigate right-to left to avoid overwriting the groups.
    std::size_t dashed_capacity = copy_stl.size() + count_groups;
    std::size_t dashed_length = 0;
    copy_stl.resize(dashed_capacity);
    copy_sz.resize(dashed_capacity);
    for (std::size_t reverse_offset = 0; reverse_offset < max_l2_size;) {

        // Walk backwards to find the length of the current group
        std::size_t offset = max_l2_size - reverse_offset - 1;
        std::size_t fill_length = 1;
        while (offset > 0 && copy_stl[offset - 1] == copy_stl[offset]) --offset, ++fill_length;

        std::size_t new_offset = dashed_capacity - dashed_length - fill_length;
        std::memmove((void *)(copy_stl.data() + new_offset), (void *)(copy_stl.data() + offset), fill_length);
        sz::memmove((void *)(copy_sz.data() + new_offset), (void *)(copy_sz.data() + offset), fill_length);
        expect_equality(copy_stl.data(), copy_sz.data(), max_l2_size);

        // Put the delimiter
        copy_stl[new_offset] = '-';
        copy_sz[new_offset] = '-';
        dashed_length += fill_length + 1;
        reverse_offset += fill_length;
    }
}

/**
 *  @brief  Tests memory utilities on large buffers (>1MB) that trigger special code paths
 *          in AVX2/AVX512 implementations. This specifically tests the bidirectional
 *          traversal optimization used for huge buffers.
 */
void test_large_memory_utilities() {
    // Test sizes that trigger the "huge buffer" path (> 1MB)
    std::vector<std::size_t> test_sizes = {
        1024ull * 1024ull + 1,       // Just over 1MB
        1024ull * 10ull * 103ull,    // From GitHub issue #228: 1,055,360 bytes
        2ull * 1024ull * 1024ull,    // 2MB
        3ull * 1024ull * 1024ull + 7 // 3MB + 7 (unaligned size)
    };

    for (std::size_t size : test_sizes) {
        // Test memcpy with aligned buffers
        {
            std::vector<char> src(size);
            std::vector<char> dst_std(size);
            std::vector<char> dst_sz(size);

            // Fill source with pattern to detect copying errors
            for (std::size_t i = 0; i < size; i++) { src[i] = static_cast<char>('A' + (i % 26)); }

            std::memcpy(dst_std.data(), src.data(), size);
            sz::memcpy(dst_sz.data(), src.data(), size);

            expect_equality(dst_std.data(), dst_sz.data(), size);
        }

        // Test memcpy with unaligned buffers
        {
            std::vector<char> src_buf(size + 64);
            std::vector<char> dst_std_buf(size + 64);
            std::vector<char> dst_sz_buf(size + 64);

            // Use unaligned pointers
            char *src = src_buf.data() + 7;
            char *dst_std = dst_std_buf.data() + 11;
            char *dst_sz = dst_sz_buf.data() + 11;

            for (std::size_t i = 0; i < size; i++) { src[i] = static_cast<char>('a' + (i % 26)); }

            std::memcpy(dst_std, src, size);
            sz::memcpy(dst_sz, src, size);

            expect_equality(dst_std, dst_sz, size);
        }

        // Test memset
        {
            std::vector<char> buf_std(size);
            std::vector<char> buf_sz(size);

            std::memset(buf_std.data(), 'Z', size);
            sz::memset(buf_sz.data(), 'Z', size);

            expect_equality(buf_std.data(), buf_sz.data(), size);
        }

        // Test memmove with overlapping regions
        {
            std::vector<char> buf_std(size);
            std::vector<char> buf_sz(size);

            // Initialize both buffers identically
            for (std::size_t i = 0; i < size; i++) { buf_std[i] = buf_sz[i] = static_cast<char>('0' + (i % 10)); }

            // Move overlapping region forward
            std::size_t overlap_size = size / 2;
            std::memmove(buf_std.data() + 100, buf_std.data(), overlap_size);
            sz::memmove(buf_sz.data() + 100, buf_sz.data(), overlap_size);

            expect_equality(buf_std.data(), buf_sz.data(), size);
        }
    }
}

#define scope_assert(init, operation, condition) \
    do {                                         \
        init;                                    \
        operation;                               \
        assert(condition);                       \
    } while (0)

#define let_assert(init, condition) \
    do {                            \
        init;                       \
        assert(condition);          \
    } while (0)

#define assert_throws(expression, exception_type) \
    do {                                          \
        bool threw = false;                       \
        try {                                     \
            sz_unused_(expression);               \
        }                                         \
        catch (exception_type const &) {          \
            threw = true;                         \
        }                                         \
        assert(threw);                            \
    } while (0)

/**
 *  @brief  Invokes different C++ member methods of immutable strings to cover all STL APIs.
 *          This test guarantees API @b compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
void test_stl_compatibility_for_reads() {

    using str = string_type;

    // Constructors.
    assert(str().empty());             // Test default constructor
    assert(str().size() == 0);         // Test default constructor
    assert(str("").empty());           // Test default constructor
    assert(str("").size() == 0);       // Test default constructor
    assert(str("hello").size() == 5);  // Test constructor with c-string
    assert(str("hello", 4) == "hell"); // Construct from substring

    // Element access.
    assert(str("rest")[0] == 'r');
    assert(str("rest").at(1) == 'e');
    assert(*str("rest").data() == 'r');
    assert(str("front").front() == 'f');
    assert(str("back").back() == 'k');

    // Iterators.
    assert(*str("begin").begin() == 'b' && *str("cbegin").cbegin() == 'c');
    assert(*str("rbegin").rbegin() == 'n' && *str("crbegin").crbegin() == 'n');
    assert(str("size").size() == 4 && str("length").length() == 6);

    // Slices... out-of-bounds exceptions are asymmetric!
    // Moreover, `std::string` has no `remove_prefix` and `remove_suffix` methods.
    // scope_assert(str s = "hello", s.remove_prefix(1), s == "ello");
    // scope_assert(str s = "hello", s.remove_suffix(1), s == "hell");
    assert(str("hello world").substr(0, 5) == "hello");
    assert(str("hello world").substr(6, 5) == "world");
    assert(str("hello world").substr(6) == "world");
    assert(str("hello world").substr(6, 100) == "world"); // 106 is beyond the length of the string, but its OK
    assert_throws(str("hello world").substr(100), std::out_of_range);   // 100 is beyond the length of the string
    assert_throws(str("hello world").substr(20, 5), std::out_of_range); // 20 is beyond the length of the string
#if defined(__GNUC__) && !defined(__NVCC__) // -1 casts to unsigned without warnings on GCC, but not NVCC
    assert_throws(str("hello world").substr(-1, 5), std::out_of_range);
    assert(str("hello world").substr(0, -1) == "hello world");
#endif

    // Character search in normal and reverse directions.
    assert(str("hello").find('e') == 1);
    assert(str("hello").find('e', 1) == 1);
    assert(str("hello").find('e', 2) == str::npos);
    assert(str("hello").rfind('l') == 3);
    assert(str("hello").rfind('l', 2) == 2);
    assert(str("hello").rfind('l', 1) == str::npos);

    // Substring search in normal and reverse directions.
    assert(str("hello").find("ell") == 1);
    assert(str("hello").find("ell", 1) == 1);
    assert(str("hello").find("ell", 2) == str::npos);
    assert(str("hello").find("el", 1) == 1);
    assert(str("hello").find("ell", 1, 2) == 1);
    assert(str("hello").rfind("l") == 3);
    assert(str("hello").rfind("l", 2) == 2);
    assert(str("hello").rfind("l", 1) == str::npos);

    // The second argument is the last possible value of the returned offset.
    assert(str("hello").rfind("el", 1) == 1);
    assert(str("hello").rfind("ell", 1) == 1);
    assert(str("hello").rfind("ello", 1) == 1);
    assert(str("hello").rfind("ell", 1, 2) == 1);

    // More complex queries.
    assert(str("abbabbaaaaaa").find("aa") == 6);
    assert(str("abbabbaaaaaa").find("ba") == 2);
    assert(str("abbabbaaaaaa").find("bb") == 1);
    assert(str("abbabbaaaaaa").find("bab") == 2);
    assert(str("abbabbaaaaaa").find("babb") == 2);
    assert(str("abbabbaaaaaa").find("babba") == 2);
    assert(str("abcdabcd").substr(2, 4).find("abc") == str::npos);
    assert(str("hello, world!").substr(0, 11).find("world") == str::npos);
    assert(str("axabbcxcaaabbccc").find("aaabbccc") == 8);
    assert(str("abcdabcdabc________").find("abcd") == 0);
    assert(str("________abcdabcdabc").find("abcd") == 8);

    // Cover every SWAR case for unique string sequences.
    auto lowercase_alphabet = str("abcdefghijklmnopqrstuvwxyz");
    for (std::size_t one_byte_offset = 0; one_byte_offset + 1 <= lowercase_alphabet.size(); ++one_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(one_byte_offset, 1)) == one_byte_offset);
    for (std::size_t two_byte_offset = 0; two_byte_offset + 2 <= lowercase_alphabet.size(); ++two_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(two_byte_offset, 2)) == two_byte_offset);
    for (std::size_t four_byte_offset = 0; four_byte_offset + 4 <= lowercase_alphabet.size(); ++four_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(four_byte_offset, 4)) == four_byte_offset);
    for (std::size_t three_byte_offset = 0; three_byte_offset + 3 <= lowercase_alphabet.size(); ++three_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(three_byte_offset, 3)) == three_byte_offset);
    for (std::size_t five_byte_offset = 0; five_byte_offset + 5 <= lowercase_alphabet.size(); ++five_byte_offset)
        assert(lowercase_alphabet.find(lowercase_alphabet.substr(five_byte_offset, 5)) == five_byte_offset);

    // Simple repeating patterns - with one "almost match" before an actual match in each direction.
    assert(str("_ab_abc_").find("abc") == 4);
    assert(str("_abc_ab_").rfind("abc") == 1);
    assert(str("_abc_abcd_").find("abcd") == 5);
    assert(str("_abcd_abc_").rfind("abcd") == 1);
    assert(str("_abcd_abcde_").find("abcde") == 6);
    assert(str("_abcde_abcd_").rfind("abcde") == 1);
    assert(str("_abcde_abcdef_").find("abcdef") == 7);
    assert(str("_abcdef_abcde_").rfind("abcdef") == 1);
    assert(str("_abcdef_abcdefg_").find("abcdefg") == 8);
    assert(str("_abcdefg_abcdef_").rfind("abcdefg") == 1);

    // ! `rfind` and `find_last_of` are not consistent in meaning of their arguments.
    assert(str("hello").find_first_of("le") == 1);
    assert(str("hello").find_first_of("le", 1) == 1);
    assert(str("hello").find_last_of("le") == 3);
    assert(str("hello").find_last_of("le", 2) == 2);
    assert(str("hello").find_first_not_of("hel") == 4);
    assert(str("hello").find_first_not_of("hel", 1) == 4);
    assert(str("hello").find_last_not_of("hel") == 4);
    assert(str("hello").find_last_not_of("hel", 4) == 4);

    // Try longer strings to enforce SIMD.
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('x') == 23);  // first byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find('X') == 49);  // first byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('x') == 23); // last byte
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind('X') == 49); // last byte

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xy") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XY") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("yz") == 24);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("YZ") == 50);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xy") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XY") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyz") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyz") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("xyzA") == 23);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find("XYZ0") == 49);  // first match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("xyzA") == 23); // last match
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").rfind("XYZ0") == 49); // last match

    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("xyz") == 23); // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_first_of("XYZ") == 49); // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("xyz") == 25);  // sets
    assert(str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-").find_last_of("XYZ") == 51);  // sets

    // clang-format off
    // Using single-byte non-ASCII values, e.g., À (0xC0), Æ (0xC6)
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_first_of("\xC6\xC7") == 9);  // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_first_of("\xC0\xC1") == 54); // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_last_of("\xC6\xC7") == 9);   // sets
    assert(str("abcdefgh" "\x01" "\xC6" "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "\xC0" "\xFA" "0123456789+-", 68).find_last_of("\xC0\xC1") == 54);  // sets
    // clang-format on

    // Boundary conditions.
    assert(str("hello").find_first_of("ox", 4) == 4);
    assert(str("hello").find_first_of("ox", 5) == str::npos);
    assert(str("hello").find_last_of("ox", 4) == 4);
    assert(str("hello").find_last_of("ox", 5) == 4);
    assert(str("hello").find_first_of("hx", 0) == 0);
    assert(str("hello").find_last_of("hx", 0) == 0);

    // More complex relative patterns
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0223456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") <=
           str("0213456789012345678901234567890123456789012345678901234567890123"));
    assert(str("12341234") <= str("12341234"));
    assert(str("12341234") > str("12241224"));
    assert(str("12341234") < str("13241324"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") ==
           str("0123456789012345678901234567890123456789012345678901234567890123"));
    assert(str("0123456789012345678901234567890123456789012345678901234567890123") !=
           str("0223456789012345678901234567890123456789012345678901234567890123"));

    // Comparisons.
    assert(str("a") != str("b"));
    assert(str("a") < str("b"));
    assert(str("a") <= str("b"));
    assert(str("b") > str("a"));
    assert(str("b") >= str("a"));
    assert(str("a") < str("aa"));

#if SZ_IS_CPP20_ && defined(__cpp_lib_three_way_comparison)
    // Spaceship operator instead of conventional comparions.
    assert((str("a") <=> str("b")) == std::strong_ordering::less);
    assert((str("b") <=> str("a")) == std::strong_ordering::greater);
    assert((str("b") <=> str("b")) == std::strong_ordering::equal);
    assert((str("a") <=> str("aa")) == std::strong_ordering::less);
#endif

    // Compare with another `str`.
    assert(str("test").compare(str("test")) == 0);   // Equal strings
    assert(str("apple").compare(str("banana")) < 0); // "apple" is less than "banana"
    assert(str("banana").compare(str("apple")) > 0); // "banana" is greater than "apple"

    // Compare with a C-string.
    assert(str("test").compare("test") == 0); // Equal to C-string "test"
    assert(str("alpha").compare("beta") < 0); // "alpha" is less than C-string "beta"
    assert(str("beta").compare("alpha") > 0); // "beta" is greater than C-string "alpha"

    // Compare substring with another `str`.
    assert(str("hello world").compare(0, 5, str("hello")) == 0); // Substring "hello" is equal to "hello"
    assert(str("hello world").compare(6, 5, str("earth")) > 0);  // Substring "world" is greater than "earth"
    assert(str("hello world").compare(6, 5, str("worlds")) < 0); // Substring "world" is less than "worlds"
    assert_throws(str("hello world").compare(20, 5, str("worlds")), std::out_of_range);

    // Compare substring with another `str`'s substring.
    assert(str("hello world").compare(0, 5, str("say hello"), 4, 5) == 0);      // Substring "hello" in both strings
    assert(str("hello world").compare(6, 5, str("world peace"), 0, 5) == 0);    // Substring "world" in both strings
    assert(str("hello world").compare(6, 5, str("a better world"), 9, 5) == 0); // Both substrings are "world"

    // Out of bounds cases for both compared strings.
    assert_throws(str("hello world").compare(20, 5, str("a better world"), 9, 5), std::out_of_range);
    assert_throws(str("hello world").compare(6, 5, str("a better world"), 90, 5), std::out_of_range);

    // Compare substring with a C-string.
    assert(str("hello world").compare(0, 5, "hello") == 0); // Substring "hello" is equal to C-string "hello"
    assert(str("hello world").compare(6, 5, "earth") > 0);  // Substring "world" is greater than C-string "earth"
    assert(str("hello world").compare(6, 5, "worlds") < 0); // Substring "world" is greater than C-string "worlds"

    // Compare substring with a C-string's prefix.
    assert(str("hello world").compare(0, 5, "hello Ash", 5) == 0); // Substring "hello" in both strings
    assert(str("hello world").compare(6, 5, "worlds", 5) == 0);    // Substring "world" in both strings
    assert(str("hello world").compare(6, 5, "worlds", 6) < 0);     // Substring "world" is less than "worlds"

#if SZ_IS_CPP20_ && defined(__cpp_lib_starts_ends_with)
    // Prefix and suffix checks against strings.
    assert(str("https://cppreference.com").starts_with(str("http")) == true);
    assert(str("https://cppreference.com").starts_with(str("ftp")) == false);
    assert(str("https://cppreference.com").ends_with(str("com")) == true);
    assert(str("https://cppreference.com").ends_with(str("org")) == false);

    // Prefix and suffix checks against characters.
    assert(str("C++20").starts_with('C') == true);
    assert(str("C++20").starts_with('J') == false);
    assert(str("C++20").ends_with('0') == true);
    assert(str("C++20").ends_with('3') == false);

    // Prefix and suffix checks against C-style strings.
    assert(str("string_view").starts_with("string") == true);
    assert(str("string_view").starts_with("String") == false);
    assert(str("string_view").ends_with("view") == true);
    assert(str("string_view").ends_with("View") == false);
#endif

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_contains)
    // Checking basic substring presence.
    assert(str("hello").contains(str("ell")) == true);
    assert(str("hello").contains(str("oll")) == false);
    assert(str("hello").contains('l') == true);
    assert(str("hello").contains('x') == false);
    assert(str("hello").contains("lo") == true);
    assert(str("hello").contains("lx") == false);
#endif

    // Exporting the contents of the string using the `str::copy` method.
    scope_assert(char buf[5 + 1] = {0}, str("hello").copy(buf, 5), std::strcmp(buf, "hello") == 0);
    scope_assert(char buf[4 + 1] = {0}, str("hello").copy(buf, 4, 1), std::strcmp(buf, "ello") == 0);
    assert_throws(str("hello").copy((char *)"", 1, 100), std::out_of_range);

    // Swaps.
    for (str const first : {"", "hello", "hellohellohellohellohellohellohellohellohellohellohellohello"}) {
        for (str const second : {"", "world", "worldworldworldworldworldworldworldworldworldworldworldworld"}) {
            str first_copy = first;
            str second_copy = second;
            first_copy.swap(second_copy);
            assert(first_copy == second && second_copy == first);
            first_copy.swap(first_copy); // Swapping with itself.
            assert(first_copy == second);
        }
    }

    // Make sure the standard hash and function-objects instantiate just fine.
    assert(std::hash<str> {}("hello") != 0);
    scope_assert(std::ostringstream os, os << str("hello"), os.str() == "hello");

#if SZ_IS_CPP14_
    // Comparison function objects are a C++14 feature.
    assert(std::equal_to<str> {}("hello", "world") == false);
    assert(std::less<str> {}("hello", "world") == true);
#endif
}

/**
 *  @brief  Invokes different C++ member methods of the memory-owning string class to make sure they all pass
 *          compilation. This test guarantees API compatibility with STL `std::basic_string` template.
 */
template <typename string_type>
void test_stl_compatibility_for_updates() {

    using str = string_type;

    // Constructors.
    assert(str().empty());                             // Test default constructor
    assert(str().size() == 0);                         // Test default constructor
    assert(str("").empty());                           // Test default constructor
    assert(str("").size() == 0);                       // Test default constructor
    assert(str("hello").size() == 5);                  // Test constructor with c-string
    assert(str("hello", 4) == "hell");                 // Construct from substring
    assert(str(5, 'a') == "aaaaa");                    // Construct with count and character
    assert(str({'h', 'e', 'l', 'l', 'o'}) == "hello"); // Construct from initializer list
    assert(str(str("hello"), 2) == "llo");             // Construct from another string suffix
    assert(str(str("hello"), 2, 2) == "ll");           // Construct from another string range

    // Corner case constructors and search behaviors for long strings
    assert(str(258, '0').find(str(256, '1')) == str::npos);

    // Assignments.
    scope_assert(str s = "obsolete", s = "hello", s == "hello");
    scope_assert(str s = "obsolete", s.assign("hello"), s == "hello");
    scope_assert(str s = "obsolete", s.assign("hello", 4), s == "hell");
    scope_assert(str s = "obsolete", s.assign(5, 'a'), s == "aaaaa");
    scope_assert(str s = "obsolete", s.assign(32, 'a'), s == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    scope_assert(str s = "obsolete", s.assign({'h', 'e', 'l', 'l', 'o'}), s == "hello");
    scope_assert(str s = "obsolete", s.assign(str("hello")), s == "hello");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2), s == "llo");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_assert(str s = "obsolete", s.assign(str("hello"), 2, 2), s == "ll");
    scope_assert(str s = "obsolete", s.assign(s), s == "obsolete");                  // Self-assignment
    scope_assert(str s = "obsolete", s.assign(s.begin(), s.end()), s == "obsolete"); // Self-assignment
    scope_assert(str s = "obsolete", s.assign(s, 4), s == "lete");                   // Partial self-assignment
    scope_assert(str s = "obsolete", s.assign(s, 4, 3), s == "let");                 // Partial self-assignment

    // Self-assignment is a special case of assignment.
    scope_assert(str s = "obsolete", s = s, s == "obsolete");
    scope_assert(str s = "obsolete", s.assign(s), s == "obsolete");
    scope_assert(str s = "obsolete", s.assign(s.data(), 2), s == "ob");
    scope_assert(str s = "obsolete", s.assign(s.data(), s.size()), s == "obsolete");

    // Allocations, capacity and memory management.
    scope_assert(str s, s.reserve(10), s.capacity() >= 10);
    scope_assert(str s, s.resize(10), s.size() == 10);
    scope_assert(str s, s.resize(10, 'a'), s.size() == 10 && s == "aaaaaaaaaa");
    assert(str().max_size() > 0);
    assert(str().get_allocator() == std::allocator<char>());
    assert(std::strcmp(str("c_str").c_str(), "c_str") == 0);

#if SZ_IS_CPP23_ && defined(__cpp_lib_string_resize_and_overwrite)
    // Test C++23 resize and overwrite functionality
    scope_assert(str s("hello"),
                 s.resize_and_overwrite(10,
                                        [](char *p, std::size_t count) noexcept {
                                            std::memset(p, 'X', count);
                                            return count;
                                        }),
                 s.size() == 10 && s == "XXXXXXXXXX");

    scope_assert(str s("test"),
                 s.resize_and_overwrite(8,
                                        [](char *p, std::size_t) noexcept {
                                            std::strcpy(p, "ABCDE");
                                            return 5;
                                        }),
                 s.size() == 5 && s == "ABCDE");

    scope_assert(str s("orig"),
                 s.try_resize_and_overwrite(6,
                                            [](char *p, std::size_t count) noexcept {
                                                std::strcpy(p, "works!");
                                                return count;
                                            }),
                 s.size() == 6 && s == "works!");
#endif

    // On 32-bit systems the base capacity can be larger than our `z::string::min_capacity`.
    // It's true for MSVC: https://github.com/ashvardanian/StringZilla/issues/168
    if (SZ_IS_64BIT_) scope_assert(str s = "hello", s.shrink_to_fit(), s.capacity() <= sz::string::min_capacity);

    // Concatenation.
    // Following are missing in strings, but are present in vectors.
    // scope_assert(str s = "!?", s.push_front('a'), s == "a!?");
    // scope_assert(str s = "!?", s.pop_front(), s == "?");
    assert(str().append("test") == "test");
    assert(str("test") + "ing" == "testing");
    assert(str("test") + str("ing") == "testing");
    assert(str("test") + str("ing") + str("123") == "testing123");
    scope_assert(str s = "!?", s.push_back('a'), s == "!?a");
    scope_assert(str s = "!?", s.pop_back(), s == "!");

    // Incremental construction.
    assert(str("__").insert(1, "test") == "_test_");
    assert(str("__").insert(1, "test", 2) == "_te_");
    assert(str("__").insert(1, 5, 'a') == "_aaaaa_");
    assert(str("__").insert(1, str("test")) == "_test_");
    assert(str("__").insert(1, str("test"), 2) == "_st_");
    assert(str("__").insert(1, str("test"), 2, 1) == "_s_");

    // Inserting at a given iterator position yields back an iterator.
    scope_assert(str s = "__", s.insert(s.begin() + 1, 5, 'a'), s == "_aaaaa_");
    scope_assert(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}), s == "_abc_");
    let_assert(str s = "__", s.insert(s.begin() + 1, 5, 'a') == (s.begin() + 1));
    let_assert(str s = "__", s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1));

    // Handle exceptions.
    // The `length_error` might be difficult to catch due to a large `max_size()`.
    // assert_throws(large_string.insert(large_string.size() - 1, large_number_of_chars, 'a'), std::length_error);
    assert_throws(str("hello").insert(6, "world"), std::out_of_range);         // `index > size()` case from STL
    assert_throws(str("hello").insert(5, str("world"), 6), std::out_of_range); // `s_index > str.size()` case from STL

    // Erasure.
    assert(str("").erase(0, 3) == "");
    assert(str("test").erase(1, 2) == "tt");
    assert(str("test").erase(1) == "t");
    scope_assert(str s = "test", s.erase(s.begin() + 1), s == "tst");
    scope_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 2), s == "tst");
    scope_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 3), s == "tt");
    let_assert(str s = "test", s.erase(s.begin() + 1) == (s.begin() + 1));
    let_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 2) == (s.begin() + 1));
    let_assert(str s = "test", s.erase(s.begin() + 1, s.begin() + 3) == (s.begin() + 1));

    // Substitutions.
    assert(str("hello").replace(1, 2, "123") == "h123lo");
    assert(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
    assert(str("hello").replace(1, 2, "123", 1) == "h1lo");
    assert(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
    assert(str("hello").replace(1, 2, 3, 'a') == "haaalo");

    // Substitutions with iterators.
    scope_assert(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, 3, 'a'), s == "haaalo");
    scope_assert(str s = "hello", s.replace(s.begin() + 1, s.begin() + 3, {'a', 'b'}), s == "hablo");

    // Some nice "tweetable" examples :)
    assert(str("Loose").replace(2, 2, str("vath"), 1) == "Loathe");
    assert(str("Loose").replace(2, 2, "vath", 1) == "Love");

    // Insertion is a special case of replacement.
    // Appending and assigning are special cases of insertion.
    // Still, we test them separately to make sure they are not broken.
    assert(str("hello").append("123") == "hello123");
    assert(str("hello").append(str("123")) == "hello123");
    assert(str("hello").append(str("123"), 1) == "hello23");
    assert(str("hello").append(str("123"), 1, 1) == "hello2");
    assert(str("hello").append({'1', '2'}) == "hello12");
    assert(str("hello").append(2, '!') == "hello!!");
    let_assert(str s = "123", str("hello").append(s.begin(), s.end()) == "hello123");
}

/**
 *  @brief  Constructs StringZilla classes from STL and vice-versa to ensure that the conversions are working.
 */
void test_stl_conversions() {
    // From a mutable STL string to StringZilla and vice-versa.
    {
        std::string stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        sz::string_span szs = stl;
        stl = sz;
        stl = szv;
        stl = szs;
    }
    // From an immutable STL string to StringZilla.
    {
        std::string const stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        sz_unused_(sz);
        sz_unused_(szv);
    }
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    // From STL `string_view` to StringZilla and vice-versa.
    {
        std::string_view stl {"hello"};
        sz::string sz = stl;
        sz::string_view szv = stl;
        stl = sz;
        stl = szv;
    }
#endif
}

/**
 *  @brief The sum of an arithmetic progression.
 *  @see https://en.wikipedia.org/wiki/Arithmetic_progression
 */
inline std::size_t arithmetic_sum(std::size_t first, std::size_t last, std::size_t step = 1) {
    std::size_t n = (last >= first) ? ((last - first) / step + 1) : 0;
    // Return 0 if there are no terms
    if (n == 0) return 0;
    // Compute the sum using the arithmetic sequence formula
    std::size_t sum = n / 2 * (2 * first + (n - 1) * step);
    // If n is odd, handle the remaining term separately to avoid overflow
    if (n % 2 == 1) sum += (2 * first + (n - 1) * step) / 2;
    return sum;
}

/**
 *  @brief  Invokes different C++ member methods of immutable strings to cover
 *          extensions beyond the STL API.
 */
template <typename string_type>
void test_non_stl_extensions_for_reads() {
    using str = string_type;

    // Signed offset lookups and slices.
    assert(str("hello").sat(0) == 'h');
    assert(str("hello").sat(-1) == 'o');
    assert(str("rest").sat(1) == 'e');
    assert(str("rest").sat(-1) == 't');
    assert(str("rest").sat(-4) == 'r');

    assert(str("front").front() == 'f');
    assert(str("front").front(1) == "f");
    assert(str("front").front(2) == "fr");
    assert(str("front").front(2) == "fr");
    assert(str("front").front(-2) == "fro");
    assert(str("front").front(0) == "");
    assert(str("front").front(5) == "front");
    assert(str("front").front(-5) == "");

    assert(str("back").back() == 'k');
    assert(str("back").back(1) == "ack");
    assert(str("back").back(2) == "ck");
    assert(str("back").back(-1) == "k");
    assert(str("back").back(-2) == "ck");
    assert(str("back").back(-4) == "back");
    assert(str("back").back(4) == "");

    assert(str("hello").sub(1) == "ello");
    assert(str("hello").sub(-1) == "o");
    assert(str("hello").sub(1, 2) == "e");
    assert(str("hello").sub(1, 100) == "ello");
    assert(str("hello").sub(100, 100) == "");
    assert(str("hello").sub(-2, -1) == "l");
    assert(str("hello").sub(-2, -2) == "");
    assert(str("hello").sub(100, -100) == "");

    // Passing initializer lists to `operator[]`.
    // Put extra braces to correctly estimate the number of macro arguments :)
    assert((str("hello")[{1, 2}] == "e"));
    assert((str("hello")[{1, 100}] == "ello"));
    assert((str("hello")[{100, 100}] == ""));
    assert((str("hello")[{100, -100}] == ""));
    assert((str("hello")[{-100, -100}] == ""));

    // Checksums
    auto accumulate_bytes = [](str const &s) -> std::size_t {
        return std::accumulate(s.begin(), s.end(), (std::size_t)0,
                               [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
    };
    assert(str("a").bytesum() == (std::size_t)'a');
    assert(str("0").bytesum() == (std::size_t)'0');
    assert(str("0123456789").bytesum() == arithmetic_sum('0', '9'));
    assert(str("abcdefghijklmnopqrstuvwxyz").bytesum() == arithmetic_sum('a', 'z'));
    assert(str("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz").bytesum() ==
           arithmetic_sum('a', 'z') * 3);
    let_assert(
        str s =
            "近来，加文出席微博之夜时对着镜头频繁摆出假笑表情、一度累瘫睡倒在沙发上的照片被广泛转发，引发对他失去童年、"
            "被过度消费的担忧。八岁的加文，已当网红近六年了，可以说，自懂事以来，他没有过过一天没有名气的日子。",
        s.bytesum() == accumulate_bytes(s));
}

void test_non_stl_extensions_for_updates() {
    using str = sz::string;

    // Try methods.
    assert(str("obsolete").try_assign("hello"));
    assert(str().try_reserve(10));
    assert(str().try_resize(10));
    assert(str("__").try_insert(1, "test"));
    assert(str("test").try_erase(1, 2));
    assert(str("test").try_clear());
    assert(str("test").try_replace(1, 2, "aaaa"));
    assert(str("test").try_push_back('a'));
    assert(str("test").try_shrink_to_fit());

    // Self-referencing methods.
    scope_assert(str s = "test", s.try_assign(s.view()), s == "test");
    scope_assert(str s = "test", s.try_assign(s.view().sub(1, 2)), s == "e");
    scope_assert(str s = "test", s.try_append(s.view().sub(1, 2)), s == "teste");

    // Try methods going beyond and beneath capacity threshold.
    scope_assert(str s = "0123456789012345678901234567890123456789012345678901234567890123", // 64 symbols at start
                 s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_append(s) && s.try_clear() &&
                     s.try_shrink_to_fit(),
                 s.capacity() < sz::string::min_capacity);

    // Same length replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "xx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", "1"), s == "he11o");
    scope_assert(str s = "hello", s.replace_all("he", "al"), s == "alllo");
    scope_assert(str s = "hello", s.replace_all("x"_bs, "!"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("o"_bs, "!"), s == "hell!");
    scope_assert(str s = "hello", s.replace_all("ho"_bs, "!"), s == "!ell!");

    // Shorter replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "x"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", ""), s == "heo");
    scope_assert(str s = "hello", s.replace_all("h", ""), s == "ello");
    scope_assert(str s = "hello", s.replace_all("o", ""), s == "hell");
    scope_assert(str s = "hello", s.replace_all("llo", "!"), s == "he!");
    scope_assert(str s = "hello", s.replace_all("x"_bs, ""), s == "hello");
    scope_assert(str s = "hello", s.replace_all("lo"_bs, ""), s == "he");

    // Longer replacements.
    scope_assert(str s = "hello", s.replace_all("xx", "xxx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("l", "ll"), s == "hellllo");
    scope_assert(str s = "hello", s.replace_all("h", "hh"), s == "hhello");
    scope_assert(str s = "hello", s.replace_all("o", "oo"), s == "helloo");
    scope_assert(str s = "hello", s.replace_all("llo", "llo!"), s == "hello!");
    scope_assert(str s = "hello", s.replace_all("x"_bs, "xx"), s == "hello");
    scope_assert(str s = "hello", s.replace_all("lo"_bs, "lo"), s == "helololo");

    // Directly mapping bytes using a Look-Up Table.
    sz::look_up_table invert_case = sz::look_up_table::identity();
    for (char c = 'a'; c <= 'z'; c++) invert_case[c] = c - 'a' + 'A';
    for (char c = 'A'; c <= 'Z'; c++) invert_case[c] = c - 'A' + 'a';
    scope_assert(str s = "hello", s.lookup(invert_case), s == "HELLO");
    scope_assert(str s = "HeLLo", s.lookup(invert_case), s == "hEllO");
    scope_assert(str s = "H-lL0", s.lookup(invert_case), s == "h-Ll0");

    // Concatenation.
    assert(str(str("a") | str("b")) == "ab");
    assert(str(str("a") | str("b") | str("ab")) == "abab");

    assert(str(sz::concatenate("a"_sv, "b"_sv)) == "ab");
    assert(str(sz::concatenate("a"_sv, "b"_sv, "c"_sv)) == "abc");

    // Randomization.
    assert(str::random(0).empty());
    assert(str::random(4).size() == 4);
    assert(str::random(4, 42).size() == 4);
}

/**
 *  @brief  Tests copy constructor and copy-assignment constructor of `sz::string`.
 */
void test_constructors() {
    std::string alphabet {sz::ascii_printables(), sizeof(sz::ascii_printables())};
    std::vector<sz::string> strings;
    for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
        strings.push_back(alphabet.substr(0, alphabet_slice));
    std::vector<sz::string> copies {strings};
    assert(copies.size() == strings.size());
    for (size_t i = 0; i < copies.size(); ++i) {
        assert(copies[i].size() == strings[i].size());
        assert(copies[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(copies[i][j] == strings[i][j]); }
    }
    std::vector<sz::string> assignments = strings;
    for (size_t i = 0; i < assignments.size(); ++i) {
        assert(assignments[i].size() == strings[i].size());
        assert(assignments[i] == strings[i]);
        for (size_t j = 0; j < strings[i].size(); j++) { assert(assignments[i][j] == strings[i][j]); }
    }
    assert(std::equal(strings.begin(), strings.end(), copies.begin()));
    assert(std::equal(strings.begin(), strings.end(), assignments.begin()));
}

/**
 *  @brief  Helper structure that counts the number of allocations and deallocations.
 */
struct accounting_allocator : public std::allocator<char> {
    inline static bool &verbose_ref() {
        static bool global_value = false;
        return global_value;
    }
    inline static std::size_t &counter_ref() {
        static std::size_t global_value = 0ul;
        return global_value;
    }

    template <typename... args_types_>
    static void print_if_verbose(char const *fmt, args_types_... args) {
        if (!verbose_ref()) return;
        std::printf(fmt, args...);
    }

    char *allocate(std::size_t n) {
        counter_ref() += n;
        print_if_verbose("alloc %zd -> %zd\n", n, counter_ref());
        return std::allocator<char>::allocate(n);
    }

    void deallocate(char *val, std::size_t n) {
        assert(n <= counter_ref());
        counter_ref() -= n;
        print_if_verbose("dealloc: %zd -> %zd\n", n, counter_ref());
        std::allocator<char>::deallocate(val, n);
    }

    template <typename callback_type>
    static std::size_t account_block(callback_type callback) {
        auto before = accounting_allocator::counter_ref();
        print_if_verbose("starting block: %zd\n", before);
        callback();
        auto after = accounting_allocator::counter_ref();
        print_if_verbose("ending block: %zd\n", after);
        return after - before;
    }
};

template <typename callback_type>
void assert_balanced_memory(callback_type callback) {
    auto bytes = accounting_allocator::account_block(callback);
    assert(bytes == 0);
}

/**
 *  @brief  Checks for memory leaks in the string class using the `accounting_allocator`.
 *
 *  @note   The baseline iteration count (100) is scaled by `SZ_TEST_ITERATIONS_MULTIPLIER`.
 */
void test_memory_stability_for_length(std::size_t len = 1ull << 10, std::size_t iterations = scale_iterations(100)) {

    assert(accounting_allocator::counter_ref() == 0);
    using string = sz::basic_string<char, accounting_allocator>;
    string base;

    for (std::size_t i = 0; i < len; ++i) base.push_back('c');
    assert(base.length() == len);

    // Do copies leak?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy(base);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // How about assignments?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string copy;
            copy = base;
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // How about the move constructor?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            assert(unique_item.length() == len);
            assert(unique_item == base);
            string copy(std::move(unique_item));
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // And the move assignment operator with an empty target payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            copy = std::move(unique_item);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // And move assignment where the target had a payload?
    assert_balanced_memory([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            string unique_item(base);
            string copy;
            for (std::size_t j = 0; j < 317; j++) copy.push_back('q');
            copy = std::move(unique_item);
            assert(copy.length() == len);
            assert(copy == base);
        }
    });

    // Now let's clear the base and check that we're back to zero
    base = string();
    assert(accounting_allocator::counter_ref() == 0);
}

/**
 *  @brief  Tests the correctness of the string class update methods, such as `push_back` and `erase`.
 */
void test_updates(std::size_t repetitions = 1024) {
    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    for (std::size_t repetition = 0; repetition != repetitions; ++repetition) {
        std::string stl_string;
        sz::string sz_string;
        for (std::size_t length = 1; length != 200; ++length) {
            char c = alphabet_chars[std::rand() % 26];
            stl_string.push_back(c);
            sz_string.push_back(c);
            assert(sz::string_view(stl_string) == sz::string_view(sz_string));
        }

        // Compare STL and StringZilla strings erase functionality.
        while (stl_string.length()) {
            std::size_t offset_to_erase = std::rand() % stl_string.length();
            std::size_t chars_to_erase = std::rand() % (stl_string.length() - offset_to_erase) + 1;
            stl_string.erase(offset_to_erase, chars_to_erase);
            sz_string.erase(offset_to_erase, chars_to_erase);
            assert(sz::string_view(stl_string) == sz::string_view(sz_string));
        }
    }
}

/**
 *  @brief  Tests the correctness of the string class comparison methods, such as `compare` and `operator==`.
 */
void test_comparisons() {
    // Comparing relative order of the strings
    assert("a"_sv.compare("a") == 0);
    assert("a"_sv.compare("ab") == -1);
    assert("ab"_sv.compare("a") == 1);
    assert("a"_sv.compare("a\0"_sv) == -1);
    assert("a\0"_sv.compare("a") == 1);
    assert("a\0"_sv.compare("a\0"_sv) == 0);
    assert("a"_sv == "a"_sv);
    assert("a"_sv != "a\0"_sv);
    assert("a\0"_sv == "a\0"_sv);
}

/**
 *  @brief  Tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *          This covers haystacks and needles of different lengths, as well as character-sets.
 */
void test_search() {

    // Searching for a set of characters
    assert(sz::string_view("a").find_first_of("az") == 0);
    assert(sz::string_view("a").find_last_of("az") == 0);
    assert(sz::string_view("a").find_first_of("xz") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_of("xz") == sz::string_view::npos);

    assert(sz::string_view("a").find_first_not_of("xz") == 0);
    assert(sz::string_view("a").find_last_not_of("xz") == 0);
    assert(sz::string_view("a").find_first_not_of("az") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_not_of("az") == sz::string_view::npos);

    assert(sz::string_view("aXbYaXbY").find_first_of("XY") == 1);
    assert(sz::string_view("axbYaxbY").find_first_of("Y") == 3);
    assert(sz::string_view("YbXaYbXa").find_last_of("XY") == 6);
    assert(sz::string_view("YbxaYbxa").find_last_of("Y") == 4);
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("_") == sz::string_view::npos);
    assert(sz::string_view(sz::base64(), sizeof(sz::base64())).find_first_of("+") == 62);
    assert(sz::string_view(sz::ascii_printables(), sizeof(sz::ascii_printables())).find_first_of("~") !=
           sz::string_view::npos);

    assert("aabaa"_sv.remove_prefix("a") == "abaa");
    assert("aabaa"_sv.remove_suffix("a") == "aaba");
    assert("aabaa"_sv.lstrip("a"_bs) == "baa");
    assert("aabaa"_sv.rstrip("a"_bs) == "aab");
    assert("aabaa"_sv.strip("a"_bs) == "b");

    // Check more advanced composite operations
    assert("abbccc"_sv.partition('b').before.size() == 1);
    assert("abbccc"_sv.partition("bb").before.size() == 1);
    assert("abbccc"_sv.partition("bb").match.size() == 2);
    assert("abbccc"_sv.partition("bb").after.size() == 3);
    assert("abbccc"_sv.partition("bb").before == "a");
    assert("abbccc"_sv.partition("bb").match == "bb");
    assert("abbccc"_sv.partition("bb").after == "ccc");
    assert("abb ccc"_sv.partition(sz::whitespaces_set()).after == "ccc");

    // Check ranges of search matches
    assert("hello"_sv.find_all("l").size() == 2);
    assert("hello"_sv.rfind_all("l").size() == 2);

    assert(""_sv.find_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sv.find_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sv.find_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sv.find_all(".", sz::exclude_overlaps_type {}).size() == 2);
    assert(""_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 0);
    assert(""_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 0);
    assert("."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 1);
    assert("."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 1);
    assert(".."_sv.rfind_all(".", sz::include_overlaps_type {}).size() == 2);
    assert(".."_sv.rfind_all(".", sz::exclude_overlaps_type {}).size() == 2);

    assert("a.b.c.d"_sv.find_all(".").size() == 3);
    assert("a.,b.,c.,d"_sv.find_all(".,").size() == 3);
    assert("a.,b.,c.,d"_sv.rfind_all(".,").size() == 3);
    assert("a.b,c.d"_sv.find_all(".,"_bs).size() == 3);
    assert("a...b...c"_sv.rfind_all("..").size() == 4);
    assert("a...b...c"_sv.rfind_all("..", sz::include_overlaps_type {}).size() == 4);
    assert("a...b...c"_sv.rfind_all("..", sz::exclude_overlaps_type {}).size() == 2);

    auto finds = "a.b.c"_sv.find_all("abcd"_bs).template to<std::vector<std::string>>();
    assert(finds.size() == 3);
    assert(finds[0] == "a");

    auto rfinds = "a.b.c"_sv.rfind_all("abcd"_bs).template to<std::vector<std::string>>();
    assert(rfinds.size() == 3);
    assert(rfinds[0] == "c");

    // Test propagating strings and their non-owning views into temporary ranges and iterators
    assert(sz::find_all("abc"_sv, "b"_sv).size() == 1);
    assert(sz::find_all("hello"_sv, "l"_sv).size() == 2);
    assert(sz::rfind_all("abc"_sv, "b"_sv).size() == 1);

    {
        sz::string h("abc"), n("b");
        assert(sz::find_all(h, n).size() == 1);
    }
    {
        sz::string h("hello"), n("l");
        assert(sz::find_all(h, n).size() == 2);
    }
    {
        sz::string h("abc"), n("b");
        assert(sz::rfind_all(h, n).size() == 1);
    }

    assert(sz::find_all(sz::string("abc"), sz::string("b")).size() == 1);
    assert(sz::find_all(sz::string("hello"), sz::string("l")).size() == 2);
    assert(sz::rfind_all(sz::string("abc"), sz::string("b")).size() == 1);

    // Check splitting - the inverse of `find_all` ranges
    {
        auto splits = ".a..c."_sv.split("."_bs).template to<std::vector<std::string>>();
        assert(splits.size() == 5);
        assert(splits[0] == "");
        assert(splits[1] == "a");
        assert(splits[4] == "");
    }

    {
        auto splits = "line1\nline2\nline3"_sv.split("line3").template to<std::vector<std::string>>();
        assert(splits.size() == 2);
        assert(splits[0] == "line1\nline2\n");
        assert(splits[1] == "");
    }

    assert(""_sv.split(".").size() == 1);
    assert(""_sv.rsplit(".").size() == 1);

    assert("hello"_sv.split("l").size() == 3);
    assert("hello"_sv.rsplit("l").size() == 3);
    assert(*advanced("hello"_sv.split("l").begin(), 0) == "he");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 0) == "o");
    assert(*advanced("hello"_sv.split("l").begin(), 1) == "");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 1) == "");
    assert(*advanced("hello"_sv.split("l").begin(), 2) == "o");
    assert(*advanced("hello"_sv.rsplit("l").begin(), 2) == "he");

    assert("a.b.c.d"_sv.split(".").size() == 4);
    assert("a.b.c.d"_sv.rsplit(".").size() == 4);
    assert(*("a.b.c.d"_sv.split(".").begin()) == "a");
    assert(*("a.b.c.d"_sv.rsplit(".").begin()) == "d");
    assert(*advanced("a.b.c.d"_sv.split(".").begin(), 1) == "b");
    assert(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 1) == "c");
    assert(*advanced("a.b.c.d"_sv.split(".").begin(), 3) == "d");
    assert(*advanced("a.b.c.d"_sv.rsplit(".").begin(), 3) == "a");
    assert("a.b.,c,d"_sv.split(".,").size() == 2);
    assert("a.b,c.d"_sv.split(".,"_bs).size() == 4);

    auto rsplits = ".a..c."_sv.rsplit("."_bs).template to<std::vector<std::string>>();
    assert(rsplits.size() == 5);
    assert(rsplits[0] == "");
    assert(rsplits[1] == "c");
    assert(rsplits[4] == "");
}

/**
 *  @brief Tests UTF-8 specific functionality, including character counting, nth character finding,
 *         character iteration, and Unicode-aware splitting.
 */
void test_utf8() {

    // Test utf8_count() - character counting vs byte length
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
        let_assert(auto c = chars("a\u200Bb"), c.size() == 3 && c[0] == 'a' && c[1] == 0x200B && c[2] == 'b');
        let_assert(auto c = chars("\uFEFF"), c.size() == 1 && c[0] == 0xFEFF); // BOM

        // Combining diacritics (é as e + combining acute)
        let_assert(auto c = chars("e\u0301"), c.size() == 2 && c[0] == 'e' && c[1] == 0x0301);

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
        let_assert(auto l = lines("a\u2028b"), l.size() >= 1);
        let_assert(auto l = lines("a\u2029b"), l.size() >= 1);

        // Use `_sv` literals for size-aware NUL-containing strings
        let_assert(auto l = lines("a\x00"
                                  "b"_sv),
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
        let_assert(auto w = words("a\u0085b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0085 NEL (Next Line)
        let_assert(auto w = words("a\u00A0b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+00A0 NBSP (No-Break Space)

        // Triple-byte whitespace (17 chars) - various space widths
        let_assert(auto w = words("a\u1680b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+1680 OGHAM SPACE MARK
        let_assert(auto w = words("a\u2000b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2000 EN QUAD
        let_assert(auto w = words("a\u2001b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2001 EM QUAD
        let_assert(auto w = words("a\u2002b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2002 EN SPACE
        let_assert(auto w = words("a\u2003b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2003 EM SPACE
        let_assert(auto w = words("a\u2004b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2004 THREE-PER-EM SPACE
        let_assert(auto w = words("a\u2005b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2005 FOUR-PER-EM SPACE
        let_assert(auto w = words("a\u2006b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2006 SIX-PER-EM SPACE
        let_assert(auto w = words("a\u2007b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2007 FIGURE SPACE
        let_assert(auto w = words("a\u2008b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2008 PUNCTUATION SPACE
        let_assert(auto w = words("a\u2009b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2009 THIN SPACE
        let_assert(auto w = words("a\u200Ab"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+200A HAIR SPACE
        let_assert(auto w = words("a\u2028b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2028 LINE SEPARATOR
        let_assert(auto w = words("a\u2029b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2029 PARAGRAPH SEPARATOR
        let_assert(auto w = words("a\u202Fb"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+202F NARROW NO-BREAK SPACE
        let_assert(auto w = words("a\u205Fb"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+205F MEDIUM MATHEMATICAL SPACE
        let_assert(auto w = words("a\u3000b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+3000 IDEOGRAPHIC SPACE

        // Mixed byte-length whitespace patterns
        let_assert(auto w = words("a \u00A0\u2000b"), w.size() == 4);             // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("a\t\u0085\u3000b"), w.size() == 4);            // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("Hello\u2000世界\u00A0Привет"), w.size() == 3); // Unicode content + spaces

        // Edge cases
        let_assert(auto w = words(""), w.size() == 1 && w[0] == "");
        let_assert(auto w = words("   "), w.size() == 4);                // "" "" "" ""
        let_assert(auto w = words("\t\n\r\v\f"), w.size() == 6);         // All single-byte whitespace
        let_assert(auto w = words("\u0085\u00A0"), w.size() == 3);       // All double-byte whitespace
        let_assert(auto w = words("\u2000\u2001\u3000"), w.size() == 4); // All triple-byte whitespace
        let_assert(auto w = words("NoSpaces"), w.size() == 1 && w[0] == "NoSpaces");

        // Non-ASCII content with regular spaces
        let_assert(auto w = words("Hello 世界 Привет 😀"),
                   w.size() == 4 && w[1] == "世界" && w[2] == "Привет" && w[3] == "😀");
        let_assert(auto w = words("مرحبا بك"), w.size() == 2);
        let_assert(auto w = words("שלום עולם"), w.size() == 2);

        // U+001C-U+001F are separators, not whitespace
        let_assert(auto w = words("a\u001Cb"), w.size() == 1); // FILE SEPARATOR - correctly NOT split
        let_assert(auto w = words("a\u001Db"), w.size() == 1); // GROUP SEPARATOR - correctly NOT split
        let_assert(auto w = words("a\u001Eb"), w.size() == 1); // RECORD SEPARATOR - correctly NOT split
        let_assert(auto w = words("a\u001Fb"), w.size() == 1); // UNIT SEPARATOR - correctly NOT split

        // Use `_sv` literals for size-aware NUL-containing strings
        let_assert(auto w = words("a\x00"
                                  "b"_sv),
                   w.size() == 1);                                      // NUL in middle - NOT split
        let_assert(auto w = words("\x00\x00\x00"_sv), w.size() == 1);   // Only NULs - one "word"
        let_assert(auto w = words("hello\x00world"_sv), w.size() == 1); // NUL between words - NOT split
        let_assert(auto w = words("\x00 a"_sv), w.size() == 2);         // NUL before space - yields 2 segments
        let_assert(auto w = words("a \x00"_sv), w.size() == 2);         // Space before NUL - yields 2 segments

        // U+200B-U+200D are format characters per Unicode, but implementation treats them as whitespace
        // Note: This may be intentional for compatibility, but differs from Unicode "White_Space" property
        let_assert(auto w = words("a\u200Bb"), w.size() == 2); // ZERO WIDTH SPACE
        let_assert(auto w = words("a\u200Cb"), w.size() == 2); // ZERO WIDTH NON-JOINER
        let_assert(auto w = words("a\u200Db"), w.size() == 2); // ZERO WIDTH JOINER

        // Consecutive different whitespace types - N delimiters yield N+1 segments
        let_assert(auto w = words("a \t\n\r\vb"), w.size() == 6);                // 5 whitespace chars between a and b
        let_assert(auto w = words("a\u0020\u00A0\u2000\u3000b"), w.size() == 5); // 1+2+3+3 byte: 4 delims -> 5 segs

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
        auto case_fold = [](sz::string s) -> sz::string {
            assert(s.try_utf8_case_fold());
            return s;
        };

        // ASCII uppercase to lowercase
        assert(case_fold("HELLO WORLD") == "hello world");
        assert(case_fold("ABC") == "abc");
        assert(case_fold("abc") == "abc"); // already lowercase
        assert(case_fold("123") == "123"); // no change for digits
        assert(case_fold("") == "");       // empty string

        // German Eszett - one-to-many expansion
        assert(case_fold("\xC3\x9F") == "ss");    // U+00DF -> ss
        assert(case_fold("STRAẞE") == "strasse"); // Capital U+1E9E -> ss

        // Cyrillic uppercase to lowercase
        assert(case_fold("ПРИВЕТ") == "привет");

        // Greek uppercase to lowercase
        assert(case_fold("ΑΒΓΔ") == "αβγδ");

        // Latin Extended characters
        assert(case_fold("ÀÁÂ") == "àáâ");

        // Armenian
        assert(case_fold("Ա") == "ա"); // U+0531 -> U+0561

        // Mixed case preservation for non-alphabetic
        assert(case_fold("Hello 123 World!") == "hello 123 world!");

        // Unicode characters without case folding should pass through unchanged
        assert(case_fold("日本語") == "日本語"); // Japanese (no case)
        assert(case_fold("中文") == "中文");     // Chinese (no case)
    }

    // Test case-insensitive string comparison
    {
        // Equal strings (ASCII)
        let_assert(auto r = sz_utf8_case_insensitive_order("hello", 5, "HELLO", 5), r == sz_equal_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("abc", 3, "ABC", 3), r == sz_equal_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("HeLLo WoRLd", 11, "hello world", 11), r == sz_equal_k);

        // Less than
        let_assert(auto r = sz_utf8_case_insensitive_order("abc", 3, "abd", 3), r == sz_less_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("ab", 2, "abc", 3), r == sz_less_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("ABC", 3, "abd", 3), r == sz_less_k);

        // Greater than
        let_assert(auto r = sz_utf8_case_insensitive_order("abd", 3, "abc", 3), r == sz_greater_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("abcd", 4, "abc", 3), r == sz_greater_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("ABD", 3, "abc", 3), r == sz_greater_k);

        // German Eszett: "straße" (7 bytes) vs "STRASSE" (7 bytes) should be equal
        let_assert(auto r = sz_utf8_case_insensitive_order("straße", 7, "STRASSE", 7), r == sz_equal_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("STRASSE", 7, "straße", 7), r == sz_equal_k);

        // Multiplication × (C3 97) and Division ÷ (C3 B7) must NOT be treated as case pairs
        let_assert(auto r = sz_utf8_case_insensitive_order("×", 2, "×", 2), r == sz_equal_k); // × == ×
        let_assert(auto r = sz_utf8_case_insensitive_order("÷", 2, "÷", 2), r == sz_equal_k); // ÷ == ÷
        let_assert(auto r = sz_utf8_case_insensitive_order("×", 2, "÷", 2), r != sz_equal_k); // × ≠ ÷
        let_assert(auto r = sz_utf8_case_insensitive_order("a×b", 4, "A×B", 4),
                   r == sz_equal_k); // a×b == A×B

        // Empty strings
        let_assert(auto r = sz_utf8_case_insensitive_order("", 0, "", 0), r == sz_equal_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("a", 1, "", 0), r == sz_greater_k);
        let_assert(auto r = sz_utf8_case_insensitive_order("", 0, "a", 1), r == sz_less_k);

        // Greek case folding
        let_assert(auto r = sz_utf8_case_insensitive_order("αβγδ", 8, "ΑΒΓΔ", 8), r == sz_equal_k);

        // Cyrillic case folding
        let_assert(auto r = sz_utf8_case_insensitive_order("привет", 12, "ПРИВЕТ", 12), r == sz_equal_k);
    }

    // Test case-insensitive substring search
    {
        sz_size_t matched_len;
        sz_cptr_t haystack;
        sz_cptr_t result;
        sz_utf8_case_insensitive_needle_metadata_t metadata = {};

        // Basic ASCII search
        haystack = "Hello World";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 11, "WORLD", 5, &metadata, &matched_len);
        assert(result == haystack + 6 && matched_len == 5);

        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 11, "world", 5, &metadata, &matched_len);
        assert(result == haystack + 6 && matched_len == 5);

        // Search at start
        haystack = "HELLO";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 5, "hello", 5, &metadata, &matched_len);
        assert(result == haystack && matched_len == 5);

        // Not found
        metadata = {};
        result = sz_utf8_case_insensitive_find("Hello", 5, "xyz", 3, &metadata, &matched_len);
        assert(result == SZ_NULL_CHAR);

        // Empty needle
        haystack = "Hello";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 5, "", 0, &metadata, &matched_len);
        assert(result == haystack && matched_len == 0);

        // Eszett matching: search for "ss" in "straße"
        haystack = "straße";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 7, "SS", 2, &metadata, &matched_len);
        assert(result != SZ_NULL_CHAR);

        // Search for "straße" in "STRASSE"
        haystack = "STRASSE";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 7, "straße", 7, &metadata, &matched_len);
        assert(result == haystack);

        // Greek case folding search
        haystack = "αβγδ";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 8, "ΑΒΓΔ", 8, &metadata, &matched_len);
        assert(result == haystack);

        // Cyrillic search
        haystack = "привет мир";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 19, "ПРИВЕТ", 12, &metadata, &matched_len);
        assert(result == haystack);

        // Mixed case in middle
        haystack = "foo BAR baz";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 11, "bar", 3, &metadata, &matched_len);
        assert(result == haystack + 4 && matched_len == 3);

        // Turkish I test (İ -> i + dot)
        // İ (U+0130, C4 B0) should match "i\u0307" (69 CC 87)
        haystack = "İstanbul";
        metadata = {};
        // Search for "istanbul" (default verify might fail if it expects strict folding?)
        // Actually, "İ" folds to "i\u0307". "i" folds to "i".
        // So "İ" != "i".
        // "İ" matches "i\u0307".
        result = sz_utf8_case_insensitive_find(haystack, 9, "i\xcc\x87stanbul", 10, &metadata, &matched_len);
        assert(result == haystack);

        // German Maße -> MASSE
        haystack = "Maße";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 5, "MASSE", 5, &metadata, &matched_len);
        // "ß" folds to "ss". "SS" folds to "ss". Match!
        assert(result != SZ_NULL_CHAR);
        assert(matched_len == 5); // Matched "Maße" (5 bytes) against "MASSE"

        // Weird combination: "ß" vs "ss"
        // Haystack "Fuss" (4 bytes), Needle "Fuß" (4 bytes)
        // "Fuss" folds to "fuss". "Fuß" folds to "fuss".
        haystack = "Fuss";
        metadata = {};
        result = sz_utf8_case_insensitive_find(haystack, 4, "Fu\xc3\x9f", 4, &metadata, &matched_len);
        assert(result == haystack);
    }

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
}

#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurrences of the `needle_stl`
 *  in a haystack formed of `haystack_pattern` repeated from one to `max_repeats` times.
 *
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;

    // Allocate a buffer to store the haystack with enough padding to mis-align it.
    std::size_t haystack_buffer_length = max_repeats * haystack_pattern.size() + 2 * SZ_CACHE_LINE_WIDTH;
    std::vector<char> haystack_buffer(haystack_buffer_length, 'x');
    char *haystack = haystack_buffer.data();

    // Skip the misaligned part.
    while (reinterpret_cast<std::uintptr_t>(haystack) % SZ_CACHE_LINE_WIDTH != misalignment) ++haystack;

    /// Helper container to store the offsets of the matches. Useful during debugging :)
    std::vector<std::size_t> offsets_stl, offsets_sz;

    for (std::size_t repeats = 0; repeats != max_repeats; ++repeats) {
        std::size_t haystack_length = (repeats + 1) * haystack_pattern.size();

#if defined(__SANITIZE_ADDRESS__)
        // Let's manually poison the prefix and the suffix.
        std::size_t poisoned_prefix_length = haystack - haystack_buffer.data();
        std::size_t poisoned_suffix_length = haystack_buffer_length - haystack_length - poisoned_prefix_length;
        ASAN_POISON_MEMORY_REGION(haystack_buffer.data(), poisoned_prefix_length);
        ASAN_POISON_MEMORY_REGION(haystack + haystack_length, poisoned_suffix_length);
#endif

        // Append the new repetition to our buffer.
        std::memcpy(haystack + repeats * haystack_pattern.size(), haystack_pattern.data(), haystack_pattern.size());

        // Convert to string views
        auto haystack_stl = std::string_view(haystack, haystack_length);
        auto haystack_sz = sz::string_view(haystack, haystack_length);
        auto needle_sz = sz::string_view(needle_stl.data(), needle_stl.size());

        // Wrap into ranges
        auto matches_stl = stl_matcher_(haystack_stl, {needle_stl});
        auto matches_sz = sz_matcher_(haystack_sz, {needle_sz});
        auto begin_stl = matches_stl.begin();
        auto begin_sz = matches_sz.begin();
        auto end_stl = matches_stl.end();
        auto end_sz = matches_sz.end();
        auto count_stl = std::distance(begin_stl, end_stl);
        auto count_sz = std::distance(begin_sz, end_sz);

        // To simplify debugging, let's first export all the match offsets, and only then compare them
        std::transform(begin_stl, end_stl, std::back_inserter(offsets_stl),
                       [&](auto const &match) { return match.data() - haystack_stl.data(); });
        std::transform(begin_sz, end_sz, std::back_inserter(offsets_sz),
                       [&](auto const &match) { return match.data() - haystack_sz.data(); });
        auto print_all_matches = [&]() {
            std::printf("Breakdown of found matches:\n");
            std::printf("- STL (%zu): ", offsets_stl.size());
            for (auto offset : offsets_stl) std::printf("%zu ", offset);
            std::printf("\n");
            std::printf("- StringZilla (%zu): ", offsets_sz.size());
            for (auto offset : offsets_sz) std::printf("%zu ", offset);
            std::printf("\n");
        };

        // Compare results
        for (std::size_t match_idx = 0; begin_stl != end_stl && begin_sz != end_sz;
             ++begin_stl, ++begin_sz, ++match_idx) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            if (match_stl.data() != match_sz.data()) {
                std::printf("Mismatch at index #%zu: %zu != %zu\n", match_idx, match_stl.data() - haystack_stl.data(),
                            match_sz.data() - haystack_sz.data());
                print_all_matches();
                assert(false);
            }
        }

        // If one range is not finished, assert failure
        if (count_stl != count_sz) {
            print_all_matches();
            assert(false);
        }
        assert(begin_stl == end_stl && begin_sz == end_sz);

        offsets_stl.clear();
        offsets_sz.clear();

#if defined(__SANITIZE_ADDRESS__)
        // Don't forget to manually unpoison the prefix and the suffix.
        ASAN_UNPOISON_MEMORY_REGION(haystack_buffer.data(), poisoned_prefix_length);
        ASAN_UNPOISON_MEMORY_REGION(haystack + haystack_length, poisoned_suffix_length);
#endif
    }
}

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurrences of the `needle_stl`,
 *  as a substring, as a set of allowed characters, or as a set of disallowed characters, in a haystack.
 */
void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl,
                                             std::size_t misalignment) {

    test_search_with_misaligned_repetitions<                                     //
        sz::range_matches<std::string_view, sz::matcher_find<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                       //
        sz::range_rmatches<std::string_view, sz::matcher_rfind<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_rfind<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                              //
        sz::range_matches<std::string_view, sz::matcher_find_first_of<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                              //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_of<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                                  //
        sz::range_matches<std::string_view, sz::matcher_find_first_not_of<std::string_view>>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);

    test_search_with_misaligned_repetitions<                                                  //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_not_of<std::string_view>>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_not_of<sz::string_view>>>(  //
        haystack_pattern, needle_stl, misalignment);
}

void test_search_with_misaligned_repetitions(std::string_view haystack_pattern, std::string_view needle_stl) {
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 0);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 1);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 2);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 3);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 63);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 24);
    test_search_with_misaligned_repetitions(haystack_pattern, needle_stl, 33);
}

/**
 *  @brief  Extensively tests the correctness of the string class search methods, such as `find` and `find_first_of`.
 *          Covers different alignment cases within a cache line, repetitive patterns, and overlapping matches.
 */
void test_search_with_misaligned_repetitions() {
    // When haystack is only formed of needles:
    test_search_with_misaligned_repetitions("a", "a");
    test_search_with_misaligned_repetitions("ab", "ab");
    test_search_with_misaligned_repetitions("abc", "abc");
    test_search_with_misaligned_repetitions("abcd", "abcd");
    test_search_with_misaligned_repetitions({sz::base64(), sizeof(sz::base64())}, {sz::base64(), sizeof(sz::base64())});
    test_search_with_misaligned_repetitions({sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())},
                                            {sz::ascii_lowercase(), sizeof(sz::ascii_lowercase())});
    test_search_with_misaligned_repetitions({sz::ascii_printables(), sizeof(sz::ascii_printables())},
                                            {sz::ascii_printables(), sizeof(sz::ascii_printables())});

    // When we are dealing with NULL characters inside the string
    test_search_with_misaligned_repetitions("\0", "\0");
    test_search_with_misaligned_repetitions("a\0", "a\0");
    test_search_with_misaligned_repetitions("ab\0", "ab");
    test_search_with_misaligned_repetitions("ab\0", "ab\0");
    test_search_with_misaligned_repetitions("abc\0", "abc");
    test_search_with_misaligned_repetitions("abc\0", "abc\0");
    test_search_with_misaligned_repetitions("abcd\0", "abcd");

    // When haystack is formed of equidistant needles:
    test_search_with_misaligned_repetitions("ab", "a");
    test_search_with_misaligned_repetitions("abc", "a");
    test_search_with_misaligned_repetitions("abcd", "a");

    // When matches occur in between pattern words:
    test_search_with_misaligned_repetitions("ab", "ba");
    test_search_with_misaligned_repetitions("abc", "ca");
    test_search_with_misaligned_repetitions("abcd", "da");

    // Examples targeted exactly against the Raita heuristic,
    // which matches the first, the last, and the middle characters with SIMD.
    test_search_with_misaligned_repetitions("aaabbccc", "aaabbccc");
    test_search_with_misaligned_repetitions("axabbcxc", "aaabbccc");
    test_search_with_misaligned_repetitions("axabbcxcaaabbccc", "aaabbccc");
}

#endif

/**
 *  Evaluates the correctness of look-up table transforms using random lookup tables.
 *
 *  @param lookup_tables_to_try The number of random lookup tables to try.
 *  @param slices_per_table The number of random inputs to test per lookup table.
 */
void test_replacements(std::size_t lookup_tables_to_try = 32, std::size_t slices_per_table = 16) {

    std::string body, transformed;
    body.resize(1024 * 1024); // 1MB
    transformed.resize(1024 * 1024);
    std::generate(body.begin(), body.end(), []() { return (char)(std::rand() % 256); });

    for (std::size_t lookup_table_variation = 0; lookup_table_variation != lookup_tables_to_try;
         ++lookup_table_variation) {
        sz::look_up_table lut;
        for (std::size_t i = 0; i < 256; ++i) lut[(char)i] = (char)(std::rand() % 256);

        for (std::size_t slice_idx = 0; slice_idx != slices_per_table; ++slice_idx) {
            std::size_t slice_offset = std::rand() % (body.length());
            std::size_t slice_length = std::rand() % (body.length() - slice_offset);

            sz::lookup<char>(sz::string_view(body.data() + slice_offset, slice_length), lut,
                             const_cast<char *>(transformed.data()) + slice_offset);
            for (std::size_t i = 0; i != slice_length; ++i) {
                assert(transformed[slice_offset + i] == lut[body[slice_offset + i]]);
            }
        }
    }
}

/**
 *  @brief  Tests array sorting functionality, such as `argsort`, `sort`, and `sorted`.
 *
 *  Tries to sort incrementally complex inputs, such as strings of varying lengths, with many equal inputs.
 *  1. Basic tests with predetermined orders.
 *  2. Test on long strings of identical length.
 *  3. Test on random very small strings of varying lengths, likely with many equal inputs.
 *  4. Test on random strings of varying lengths.
 *  5. Test on random strings of varying lengths with zero characters.
 */
void test_sorting_algorithms() {
    using strs_t = std::vector<std::string>;
    using order_t = std::vector<sz::sorted_idx_t>;

    // Basic tests with predetermined orders.
    let_assert(strs_t x({"a", "b", "c", "d"}), sz::argsort(x) == order_t({0u, 1u, 2u, 3u}));
    let_assert(strs_t x({"b", "c", "d", "a"}), sz::argsort(x) == order_t({3u, 0u, 1u, 2u}));
    let_assert(strs_t x({"b", "a", "d", "c"}), sz::argsort(x) == order_t({1u, 0u, 3u, 2u}));

    // Single character vs multi-character strings
    let_assert(strs_t x({"aa", "a", "aaa", "aa"}), sz::argsort(x) == order_t({1u, 0u, 3u, 2u}));

    // Mix of short and long strings with common prefixes
    let_assert(strs_t x({"test", "t", "testing", "te", "tests", "testify", "tea", "team"}),
               sz::argsort(x) == order_t({1u, 3u, 6u, 7u, 0u, 5u, 2u, 4u}));

    // Single character vs multi-character strings with varied patterns
    let_assert(strs_t x({"zebra", "z", "zoo", "zip", "zap", "a", "apple", "ant", "ark", "mango", "m", "maple"}),
               sz::argsort(x) == order_t({5u, 7u, 6u, 8u, 10u, 9u, 11u, 1u, 4u, 0u, 3u, 2u}));

    // Numeric-like strings of varying lengths
    let_assert(strs_t x({"100", "1", "10", "1000", "11", "111", "101", "110"}),
               sz::argsort(x) == order_t({1u, 2u, 0u, 3u, 6u, 4u, 7u, 5u}));

    // Real names with varied lengths and prefixes (this one is already correct)
    let_assert(strs_t x({"Anna", "Andrew", "Alex", "Bob", "Bobby", "Charlie", "Chris", "David", "Dan"}),
               sz::argsort(x) == order_t({2u, 1u, 0u, 3u, 4u, 5u, 6u, 8u, 7u}));

    // Test on long strings of identical length.
    for (std::size_t string_length : {5u, 25u}) {
        for (std::size_t dataset_size : {10u, 100u, 1000u, 10000u}) {
            strs_t dataset;
            dataset.reserve(dataset_size);
            for (std::size_t i = 0; i < dataset_size; ++i)
                dataset.push_back(sz::scripts::random_string(string_length, "ab", 2));

            // Run several iterations of fuzzy tests.
            for (std::size_t experiment_idx = 0; experiment_idx < 10; ++experiment_idx) {
                std::shuffle(dataset.begin(), dataset.end(), global_random_generator());
                auto order = sz::argsort(dataset);
                for (std::size_t i = 1; i < dataset.size(); ++i) assert(dataset[order[i - 1]] <= dataset[order[i]]);
            }
        }
    }

    // Test on random very small strings of varying lengths, likely with many equal inputs.
    for (std::size_t dataset_size : {10u, 100u, 1000u, 10000u}) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        for (std::size_t i = 0; i < dataset_size; ++i) dataset.push_back(sz::scripts::random_string(i % 6, "ab", 2));

        // Run several iterations of fuzzy tests.
        for (std::size_t experiment_idx = 0; experiment_idx < 10; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), global_random_generator());
            auto order = sz::argsort(dataset);
            for (std::size_t i = 1; i < dataset_size; ++i) { assert(dataset[order[i - 1]] <= dataset[order[i]]); }
        }
    }

    // Test on random strings of varying lengths.
    for (std::size_t dataset_size : {10u, 100u, 1000u, 10000u}) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        constexpr std::size_t min_length = 6;
        for (std::size_t i = 0; i < dataset_size; ++i)
            dataset.push_back(sz::scripts::random_string(min_length + i % 32, "ab", 2));

        // Run several iterations of fuzzy tests.
        for (std::size_t experiment_idx = 0; experiment_idx < 10; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), global_random_generator());
            auto order = sz::argsort(dataset);
            for (std::size_t i = 1; i < dataset_size; ++i) { assert(dataset[order[i - 1]] <= dataset[order[i]]); }
        }
    }

    // Test on random strings of varying lengths with zero characters.
    for (std::size_t dataset_size : {10u, 100u, 1000u, 10000u}) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        for (std::size_t i = 0; i < dataset_size; ++i) dataset.push_back(sz::scripts::random_string(i % 32, "ab\0", 3));

        // Run several iterations of fuzzy tests.
        for (std::size_t experiment_idx = 0; experiment_idx < 10; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), global_random_generator());
            auto order = sz::argsort(dataset);
            for (std::size_t i = 1; i < dataset_size; ++i) { assert(dataset[order[i - 1]] <= dataset[order[i]]); }
        }
    }
}

/**
 *  @brief  Tests array intersection functionality.
 */
void test_intersecting_algorithms() {
    using strs_t = std::vector<std::string>;
    using result_t = sz::intersect_result_t;

    // The mapping aren't guaranteed to be in any specific order, so we will sort them for comparisons.
    using idx_pair_t = std::pair<std::size_t, std::size_t>;
    using idx_pairs_t = std::set<idx_pair_t>;
    auto to_pairs = [](result_t const &result) -> idx_pairs_t {
        idx_pairs_t pairs;
        for (std::size_t i = 0; i < result.first_offsets.size(); ++i)
            pairs.insert({result.first_offsets[i], result.second_offsets[i]});
        return pairs;
    };

    // Predetermined simple cases
    {
        strs_t abcd({"a", "b", "c", "d"});
        strs_t dcba({"d", "c", "b", "a"});
        strs_t abs({"a", "b", "s"});
        strs_t empty;
        result_t result;
        // Empty sets
        {
            result = sz::intersect(empty, empty);
            assert(result.first_offsets.size() == 0 && result.second_offsets.size() == 0);
            result = sz::intersect(abcd, empty);
            assert(result.first_offsets.size() == 0 && result.second_offsets.size() == 0);
        }
        // Identity check
        {
            result = sz::intersect(abcd, abcd);
            assert(result.first_offsets.size() == 4 && result.second_offsets.size() == 4);
            assert(to_pairs(result) == idx_pairs_t({{0u, 0u}, {1u, 1u}, {2u, 2u}, {3u, 3u}}));
        }
        // Identical size, different order
        {
            result = sz::intersect(abcd, dcba);
            assert(result.first_offsets.size() == 4 && result.second_offsets.size() == 4);
            assert(to_pairs(result) == idx_pairs_t({{0u, 3u}, {1u, 2u}, {2u, 1u}, {3u, 0u}}));
        }
        // Different sets
        {
            result = sz::intersect(abcd, abs);
            assert(result.first_offsets.size() == 2 && result.second_offsets.size() == 2);
            assert(to_pairs(result) == idx_pairs_t({{0u, 0u}, {1u, 1u}}));
        }
    }

    // Generate random strings
    struct {
        std::size_t min_length;
        std::size_t max_length;
        std::size_t count_strings;
    } experiments[] = {
        {10, 10, 100},
        {15, 15, 1000},
        {5, 30, 2000},
    };
    for (auto experiment : experiments) {
        std::unordered_set<std::string> random_strings;
        while (random_strings.size() < experiment.count_strings)
            random_strings.insert(sz::scripts::random_string(
                experiment.min_length + std::rand() % (experiment.max_length - experiment.min_length + 1), //
                "ab", 2));

        strs_t all_strings(random_strings.begin(), random_strings.end());
        strs_t first_half(all_strings.begin(), all_strings.begin() + all_strings.size() / 2);

        // Try different joins
        result_t result;
        result = sz::intersect(all_strings, first_half);
        assert(result.first_offsets.size() == first_half.size() && result.second_offsets.size() == first_half.size());
    }
}

/**
 *  @brief  Tests constructing STL containers with StringZilla strings.
 */
void test_stl_containers() {
    std::map<sz::string, int> sorted_words_sz;
    std::unordered_map<sz::string, int> words_sz;
    assert(sorted_words_sz.empty());
    assert(words_sz.empty());

    std::map<std::string, int, sz::less> sorted_words_stl;
    std::unordered_map<std::string, int, sz::hash, sz::equal_to> words_stl;
    assert(sorted_words_stl.empty());
    assert(words_stl.empty());
}

int main(int argc, char const **argv) {

    // Let's greet the user nicely
    sz_unused_(argc && argv);
    std::printf("Hi, dear tester! You look nice today!\n");
    std::printf("- Uses Westmere: %s \n", SZ_USE_WESTMERE ? "yes" : "no");
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Goldmont: %s \n", SZ_USE_GOLDMONT ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses NEON AES: %s \n", SZ_USE_NEON_AES ? "yes" : "no");
    std::printf("- Uses NEON SHA: %s \n", SZ_USE_NEON_SHA ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses SVE2 AES: %s \n", SZ_USE_SVE2_AES ? "yes" : "no");
    std::printf("- Uses CUDA: %s \n", SZ_USE_CUDA ? "yes" : "no");
    print_test_environment();

#if SZ_USE_CUDA
    cudaError_t cuda_error = cudaFree(0); // Force context initialization
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA initialization error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    int device_count = 0;
    cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    std::printf("CUDA device count: %d\n", device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices found.\n");
        return 1;
    }
    std::printf("- CUDA devices:\n");
    cudaDeviceProp prop;
    for (int i = 0; i < device_count; ++i) {
        cuda_error = cudaGetDeviceProperties(&prop, i);
        std::printf("  - %s\n", prop.name);
    }
    std::printf("- CUDA managed memory support: %s\n", prop.managedMemory == 1 ? "yes" : "no");
    std::printf("- CUDA unified memory support: %s\n", prop.unifiedAddressing == 1 ? "yes" : "no");
#endif

    std::printf("\n=== Basic Utilities ===\n");
    std::printf("- test_arithmetical_utilities...\n");
    test_arithmetical_utilities();
    std::printf("- test_sequence_struct...\n");
    test_sequence_struct();
    std::printf("- test_memory_allocator_struct...\n");
    test_memory_allocator_struct();
    std::printf("- test_byteset_struct...\n");
    test_byteset_struct();
    std::printf("- test_equivalence...\n");
    test_equivalence();

    std::printf("\n=== Sequence Algorithms ===\n");
    std::printf("- test_sorting_algorithms...\n");
    test_sorting_algorithms();
    std::printf("- test_intersecting_algorithms...\n");
    test_intersecting_algorithms();

    std::printf("\n=== Core APIs ===\n");
    std::printf("- test_ascii_utilities<sz::string>...\n");
    test_ascii_utilities<sz::string>();
    std::printf("- test_ascii_utilities<sz::string_view>...\n");
    test_ascii_utilities<sz::string_view>();
    std::printf("- test_memory_utilities...\n");
    test_memory_utilities();
    std::printf("- test_large_memory_utilities...\n");
    test_large_memory_utilities();
    std::printf("- test_replacements...\n");
    test_replacements();

    std::printf("\n=== STL Compatibility ===\n");
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    std::printf("- test_stl_compatibility_for_reads<std::string_view>...\n");
    test_stl_compatibility_for_reads<std::string_view>();
#endif
    std::printf("- test_stl_compatibility_for_reads<std::string>...\n");
    test_stl_compatibility_for_reads<std::string>();
    std::printf("- test_stl_compatibility_for_reads<sz::string_view>...\n");
    test_stl_compatibility_for_reads<sz::string_view>();
    std::printf("- test_stl_compatibility_for_reads<sz::string>...\n");
    test_stl_compatibility_for_reads<sz::string>();
    std::printf("- test_stl_compatibility_for_updates<std::string>...\n");
    test_stl_compatibility_for_updates<std::string>();
    std::printf("- test_stl_compatibility_for_updates<sz::string>...\n");
    test_stl_compatibility_for_updates<sz::string>();
    std::printf("- test_stl_conversions...\n");
    test_stl_conversions();
    std::printf("- test_stl_containers...\n");
    test_stl_containers();

    std::printf("\n=== StringZilla Extensions ===\n");
    std::printf("- test_non_stl_extensions_for_reads<sz::string_view>...\n");
    test_non_stl_extensions_for_reads<sz::string_view>();
    std::printf("- test_non_stl_extensions_for_reads<sz::string>...\n");
    test_non_stl_extensions_for_reads<sz::string>();
    std::printf("- test_non_stl_extensions_for_updates...\n");
    test_non_stl_extensions_for_updates();

    std::printf("\n=== String Class Implementation ===\n");
    std::printf("- test_constructors...\n");
    test_constructors();
    std::printf("- test_memory_stability_for_length(1024)...\n");
    test_memory_stability_for_length(1024);
    std::printf("- test_memory_stability_for_length(14)...\n");
    test_memory_stability_for_length(14);
    std::printf("- test_updates...\n");
    test_updates();

    std::printf("\n=== Search and Comparison ===\n");
    std::printf("- test_comparisons...\n");
    test_comparisons();
    std::printf("- test_search...\n");
    test_search();
    std::printf("- test_utf8...\n");
    test_utf8();
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    std::printf("- test_search_with_misaligned_repetitions...\n");
    test_search_with_misaligned_repetitions();
#endif

    std::printf("\nAll tests passed!\n");
    return 0;
}
