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
 *  @brief  Fuzz tests case-insensitive UTF-8 substring search with controlled haystack sizes.
 *
 *  Uses two verification modes:
 *  - Exhaustive (max_needles_per_haystack == 0): Tests ALL N*(N+1)/2 substrings of each folded haystack
 *  - Sampled (max_needles_per_haystack > 0): Tests up to that many random substrings per haystack
 *
 *  Algorithm:
 *  1. Generate random haystack of ~haystack_length runes from character pool
 *  2. Case-fold the haystack
 *  3. Extract needles from folded haystack (guarantees needle exists case-insensitively)
 *  4. Search needle in ORIGINAL (unfolded) haystack with both serial and SIMD
 *  5. Both must return identical positions
 *
 *  @param haystack_length Target number of bytes in each haystack
 *  @param max_needles_per_haystack  0 = exhaustive, >0 = sample this many per haystack
 *  @param total_queries Total needle searches to perform across all haystacks
 */
void test_utf8_ci_find_fuzz(sz_utf8_case_insensitive_find_t find_serial, sz_utf8_case_insensitive_find_t find_simd,
                            sz_utf8_case_fold_t case_fold, sz_utf8_find_nth_t utf8_find_nth, sz_utf8_count_t utf8_count,
                            std::size_t haystack_length, std::size_t max_needles_per_haystack,
                            std::size_t total_queries) {

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
        "\xE2\x84\xAA", // 'K' (U+212A, E2 84 AA) - Kelvin Sign (folds to ASCII k)
        "\xE2\x84\xAB", // 'Å' (U+212B, E2 84 AB) - Angstrom Sign (folds to Latin-1 a-ring)

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

        // One-to-many expansions (U+1E96-1E9A)
        "\xE1\xBA\x96", // 'ẖ' (U+1E96, E1 BA 96) - Latin Small Letter H with Line Below
        "\xE1\xBA\x97", // 'ẗ' (U+1E97, E1 BA 97) - Latin Small Letter T with Diaeresis
        "\xE1\xBA\x98", // 'ẘ' (U+1E98, E1 BA 98) - Latin Small Letter W with Ring Above
        "\xE1\xBA\x99", // 'ẙ' (U+1E99, E1 BA 99) - Latin Small Letter Y with Ring Above
        "\xE1\xBA\x9A", // 'ẚ' (U+1E9A, E1 BA 9A) - Latin Small Letter A with Right Half Ring

        // Capital Eszett (U+1E9E) - folds to ss
        "\xE1\xBA\x9E", // 'ẞ' (U+1E9E, E1 BA 9E) - Latin Capital Letter Sharp S

        // Afrikaans n-apostrophe (U+0149) -> 'n
        "\xC5\x89", // 'ŉ' (U+0149, C5 89) - Latin Small Letter N Preceded by Apostrophe

        // J-caron (U+01F0) -> j + combining caron
        "\xC7\xB0", // 'ǰ' (U+01F0, C7 B0) - Latin Small Letter J with Caron

        // Modifier letter apostrophe (U+02BC) - context for n
        "\xCA\xBC", // 'ʼ' (U+02BC, CA BC) - Modifier Letter Apostrophe

        // Modifier letter right half ring (U+02BE) - context for a
        "\xCA\xBE", // 'ʾ' (U+02BE, CA BE) - Modifier Letter Right Half Ring

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
        haystack_folded.resize(case_fold(haystack.data(), haystack.size(), haystack_folded.data()));
        if (haystack_folded.empty()) continue;

        // 3. Count runes in folded haystack
        sz_size_t runes_in_folded_haystack = utf8_count(haystack_folded.data(), haystack_folded.size());
        if (runes_in_folded_haystack < 2) continue;

        // 4. Calculate needles for this haystack
        std::size_t const needles_in_this_haystack =
            max_needles_per_haystack == 0 ? ((runes_in_folded_haystack * (runes_in_folded_haystack + 1)) / 2)
                                          : std::min(max_needles_per_haystack, queries_remaining);

        // Helper to extract needle from folded haystack and test both implementations
        auto test_needle = [&](sz_size_t start, sz_size_t len) -> bool {
            sz_cptr_t needle_start = (start == 0)
                                         ? haystack_folded.data()
                                         : utf8_find_nth(haystack_folded.data(), haystack_folded.size(), start);
            sz_cptr_t needle_end = ((start + len) == runes_in_folded_haystack)
                                       ? haystack_folded.data() + haystack_folded.size()
                                       : utf8_find_nth(haystack_folded.data(), haystack_folded.size(), start + len);
            if (!needle_start || !needle_end || needle_end <= needle_start) return false;

            sz_size_t needle_bytes = needle_end - needle_start;
            sz_size_t serial_matched = 0, simd_matched = 0;
            sz_utf8_case_insensitive_needle_metadata_t serial_meta = {}, simd_meta = {};

            sz_cptr_t serial_result = find_serial(haystack.data(), haystack.size(), needle_start, needle_bytes,
                                                  &serial_meta, &serial_matched);
            sz_cptr_t simd_result =
                find_simd(haystack.data(), haystack.size(), needle_start, needle_bytes, &simd_meta, &simd_matched);

            if (serial_result != simd_result || serial_matched != simd_matched) {
                std::fprintf(stderr, "FUZZ FAIL haystack=%zu start=%zu len=%zu\n", haystacks_tested, start, len);
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
                assert(serial_result == simd_result && "Fuzz offset mismatch");
                assert(serial_matched == simd_matched && "Fuzz length mismatch");
            }
            return true;
        };

        // 5. Generate and test needles
        if (max_needles_per_haystack == 0) {
            // Exhaustive mode: every possible (start, length) pair from folded haystack
            for (sz_size_t start = 0; start < runes_in_folded_haystack && queries_remaining > 0; ++start) {
                for (sz_size_t len = 1; len <= runes_in_folded_haystack - start && queries_remaining > 0; ++len) {
                    if (test_needle(start, len)) {
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
                std::uniform_int_distribution<sz_size_t> len_dist(1, runes_in_folded_haystack - start);
                if (test_needle(start, len_dist(rng))) {
                    ++total_passed;
                    --queries_remaining;
                }
            }
        }

        ++haystacks_tested;
    }

    std::printf("    passed %zu fuzz tests across %zu haystacks\n", total_passed, haystacks_tested);
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

    // Fuzz testing with different haystack sizes and sampling strategies:
    // - (16, 0, N): Exhaustive on tiny haystacks (~136 needles each)
    // - (32, 0, N): Exhaustive on small haystacks (~528 needles each)
    // - (100, 100, N): Sampled on medium haystacks (100 random needles)
    // - (200, 100, N): Sampled on larger haystacks (100 random needles)
    std::size_t fuzz_queries = scale_iterations(100000);
    test_utf8_ci_find_fuzz( //
        sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice, sz_utf8_case_fold_serial,
        sz_utf8_find_nth_serial, sz_utf8_count_serial, 16, 0, fuzz_queries);
    test_utf8_ci_find_fuzz( //
        sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice, sz_utf8_case_fold_serial,
        sz_utf8_find_nth_serial, sz_utf8_count_serial, 32, 0, fuzz_queries);
    test_utf8_ci_find_fuzz( //
        sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice, sz_utf8_case_fold_serial,
        sz_utf8_find_nth_serial, sz_utf8_count_serial, 100, 100, fuzz_queries);
    test_utf8_ci_find_fuzz( //
        sz_utf8_case_insensitive_find_serial, sz_utf8_case_insensitive_find_ice, sz_utf8_case_fold_serial,
        sz_utf8_find_nth_serial, sz_utf8_count_serial, 200, 100, fuzz_queries);
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
}

void test_utf8_case() {

    using str = sz::string_view;

    // Equal strings (ASCII)
    assert(str("hello").utf8_case_insensitive_order("HELLO") == sz_equal_k);
    assert(str("abc").utf8_case_insensitive_order("ABC") == sz_equal_k);
    assert(str("HeLLo WoRLd").utf8_case_insensitive_order("hello world") == sz_equal_k);

    // ASCII Extensions
    let_assert(auto m = str("prefixhello").utf8_case_insensitive_find("HELLO"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("hello_suffix").utf8_case_insensitive_find("HELLO"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("mid_hello_mid").utf8_case_insensitive_find("HELLO"), m.offset == 4 && m.length == 5);

    // Less than
    assert(str("abc").utf8_case_insensitive_order("abd") == sz_less_k);
    assert(str("ab").utf8_case_insensitive_order("abc") == sz_less_k);
    assert(str("ABC").utf8_case_insensitive_order("abd") == sz_less_k);

    // Greater than
    assert(str("abd").utf8_case_insensitive_order("abc") == sz_greater_k);
    assert(str("abcd").utf8_case_insensitive_order("abc") == sz_greater_k);
    assert(str("ABD").utf8_case_insensitive_order("abc") == sz_greater_k);

    // Latin-1 Supplement & Latin Extended-A
    // German Umlauts
    assert(str("schöner").utf8_case_insensitive_order("SCHÖNER") == sz_equal_k);
    let_assert(auto m = str("Das ist ein schöner Tag").utf8_case_insensitive_find("SCHÖNER"),
               m.offset == 12 && m.length == 8); // 'ö' (U+00F6, C3 B6) is 2 bytes

    // French Accents
    assert(str("café").utf8_case_insensitive_order("CAFÉ") == sz_equal_k);
    assert(str("naïve").utf8_case_insensitive_order("NAÏVE") == sz_equal_k);
    assert(str("À la carte").utf8_case_insensitive_order("à la CARTE") == sz_equal_k);

    // Spanish/Portuguese
    assert(str("niño").utf8_case_insensitive_order("NIÑO") == sz_equal_k);

    // Polish / Central European (Latin Extended-A)
    // "ĄĆĘŁŃÓŚŹŻ" -> "ąćęłńóśźż"
    // "Zaółć gęślą jaźń" (classic Polish pangram fragment)
    assert(str("Zaółć gęślą jaźń").utf8_case_insensitive_order("ZAÓŁĆ GĘŚLĄ JAŹŃ") == sz_equal_k);

    // German (Eszett 'ß')
    // 'ß' (U+00DF, C3 9F) -> "ss"
    // "straße" -> "strasse"
    // "STRASSE" -> "strasse"
    assert(str("straße").utf8_case_insensitive_order("STRASSE") == sz_equal_k);
    assert(str("STRASSE").utf8_case_insensitive_order("straße") == sz_equal_k);

    // Uppercase 'ẞ' (U+1E9E, E1 BA 9E) -> "ss" or "ß" depending on fold
    // StringZilla generally folds to lowercase first. 'ẞ' -> 'ss'.
    // Haystack uses 'ß' (2 bytes), Needle "SS".
    let_assert(auto m = str("straße").utf8_case_insensitive_find("SS"),
               m.offset == 4 && m.length == 2); // Matches 'ß' (2 bytes)

    // Eszett Context Extensions
    let_assert(auto m = str("Eine straße").utf8_case_insensitive_find("SS"),
               m.offset == 9 && m.length == 2); // "Eine " is 5 chars -> 5 bytes + "stra" (4) = 9
    let_assert(auto m = str("straßebahn").utf8_case_insensitive_find("SS"), m.offset == 4 && m.length == 2);
    let_assert(auto m = str("Eine straßebahn").utf8_case_insensitive_find("SS"), m.offset == 9 && m.length == 2);

    // Same case-folding, but different relation
    let_assert(auto m = str("HelloäeßHelloL").utf8_case_insensitive_find("helloäesshellol"),
               m.offset == 0 && m.length == 16);
    let_assert(auto m = str("helloäesshellol").utf8_case_insensitive_find("HelloäeßHelloL"),
               m.offset == 0 && m.length == 16);

    // Same case-folding, but different relation and needle length due to uppercase tripple-byte 'ẞ' (U+1E9E, E1 BA 9E)
    let_assert(auto m = str("HelloäeẞHelloL").utf8_case_insensitive_find("helloäesshellol"),
               m.offset == 0 && m.length == 17);
    let_assert(auto m = str("helloäesshellol").utf8_case_insensitive_find("HelloäeẞHelloL"),
               m.offset == 0 && m.length == 16);

    // Haystack "STRASSE", Needle "straße"
    let_assert(auto m = str("STRASSE").utf8_case_insensitive_find("straße"),
               m.offset == 0 && m.length == 7); // Matches "STRASSE" (7 bytes)

    // "Maße" -> "MASSE"
    let_assert(auto m = str("Maße").utf8_case_insensitive_find("MASSE"),
               m.offset == 0 && m.length == 5); // Matches "Maße" (5 bytes)

    // Haystack: "Fuss" (4 bytes) "u", "s", "s"
    // Needle: "Fuß" (4 bytes) "u", "ß"
    // They are equal in order, and searching "Fuß" in "Fuss" works.
    let_assert(auto m = str("Fuss").utf8_case_insensitive_find("Fuß"),
               m.offset == 0 && m.length == 4); // Matches "Fuss"

    // Math Symbols
    // Multiplication × (U+00D7, C3 97) and Division ÷ (U+00F7, C3 B7)
    // Often confusable with 'x' and '+'/'=', but strictly they are distinct.
    // They should equal themselves but not each other.
    assert(str("×").utf8_case_insensitive_order("×") == sz_equal_k); // × == ×
    assert(str("÷").utf8_case_insensitive_order("÷") == sz_equal_k); // ÷ == ÷
    assert(str("×").utf8_case_insensitive_order("÷") != sz_equal_k); // × ≠ ÷
    assert(str("a×b").utf8_case_insensitive_order("A×B") == sz_equal_k);

    // Math Context Extensions
    let_assert(auto m = str("2×3=6").utf8_case_insensitive_find("×"), m.offset == 1 && m.length == 2);
    let_assert(auto m = str("6÷2=3").utf8_case_insensitive_find("÷"), m.offset == 1 && m.length == 2);

    // Empty strings
    assert(str("").utf8_case_insensitive_order("") == sz_equal_k);
    assert(str("a").utf8_case_insensitive_order("") == sz_greater_k);
    assert(str("").utf8_case_insensitive_order("a") == sz_less_k);

    // Greek
    // Basic casing: "αβγδ" vs "ΑΒΓΔ"
    assert(str("αβγδ").utf8_case_insensitive_order("ΑΒΓΔ") == sz_equal_k);
    let_assert(auto m = str("αβγδ").utf8_case_insensitive_find("ΑΒΓΔ"),
               m.offset == 0 && m.length == 8); // 4 * 2 bytes = 8 bytes

    // Greek Context Extensions
    // "prefix " is 7 bytes.
    let_assert(auto m = str("prefix αβγδ").utf8_case_insensitive_find("ΑΒΓΔ"), m.offset == 7 && m.length == 8);
    // " suffix" is 7 bytes. "αβγδ" is 8 bytes.
    let_assert(auto m = str("αβγδ suffix").utf8_case_insensitive_find("ΑΒΓΔ"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("prefix αβγδ suffix").utf8_case_insensitive_find("ΑΒΓΔ"), m.offset == 7 && m.length == 8);

    // Sigma: 'Σ' (U+03A3, CE A3) matches both 'σ' (U+03C3, CF 83, medial) and 'ς' (U+03C2, CF 82, final)
    // Haystack: "ΟΔΥΣΣΕΥΣ" (Odysseus uppercase)
    // Needle: "οδυσσευς" (lowercase with final sigma)
    // Lengths match byte-for-byte in this case.
    let_assert(auto m = str("ΟΔΥΣΣΕΥΣ").utf8_case_insensitive_find("οδυσσευς"),
               m.offset == 0 && m.length == 16); // 8 chars * 2 bytes

    // Micro Sign 'µ' (U+00B5) vs Greek Mu 'μ' (U+03BC) vs 'Μ' (U+039C)
    // These should all fold to the same canonical representation.
    let_assert(auto m = str("µ").utf8_case_insensitive_find("μ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("μ").utf8_case_insensitive_find("µ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("µ").utf8_case_insensitive_find("Μ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("Μ").utf8_case_insensitive_find("µ"), m.offset == 0 && m.length == 2);
    // Context: Head/Tail/Middle
    let_assert(auto m = str("123µ456").utf8_case_insensitive_find("123μ456"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("LongPrefix Μ Suffix").utf8_case_insensitive_find("Prefix µ Suf"),
               m.offset == 4 && m.length == 13);

    // Greek Lunate Epsilon 'ϵ' (U+03F5) -> 'ε' (U+03B5)
    let_assert(auto m = str("ϵ").utf8_case_insensitive_find("ε"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("start ϵ end").utf8_case_insensitive_find("start ε end"), m.offset == 0 && m.length == 12);
    let_assert(auto m = str("...ϵ...").utf8_case_insensitive_find(".ε."), m.offset == 2 && m.length == 4);
    // Greek Kappa Symbol 'ϰ' (U+03F0) -> 'κ' (U+03BA)
    let_assert(auto m = str("ϰ").utf8_case_insensitive_find("κ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("text ϰ").utf8_case_insensitive_find("text κ"), m.offset == 0 && m.length == 7); // 5 + 2
    let_assert(auto m = str("ϰ text").utf8_case_insensitive_find("κ text"), m.offset == 0 && m.length == 7);

    // Greek Symbols & Anomalies
    // 'ϐ' (CF 90) -> 'β' (CE B2)
    let_assert(auto m = str("ϐ").utf8_case_insensitive_find("β"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("alpha ϐ").utf8_case_insensitive_find("alpha β"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("ϐ beta").utf8_case_insensitive_find("β beta"), m.offset == 0 && m.length == 7);
    // 'ϑ' (CF 91) -> 'θ' (CE B8)
    let_assert(auto m = str("ϑ").utf8_case_insensitive_find("θ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("1ϑ2").utf8_case_insensitive_find("1θ2"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("prefix ϑ suffix").utf8_case_insensitive_find("fix θ suf"),
               m.offset == 3 && m.length == 10);
    // 'ϖ' (CF 96) -> 'π' (CF 80)
    let_assert(auto m = str("ϖ").utf8_case_insensitive_find("π"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("AϖB").utf8_case_insensitive_find("AπB"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("Long string with ϖ in it").utf8_case_insensitive_find("th π in"),
               m.offset == 14 && m.length == 8);

    // Greek Context Extensions (Symbols)
    let_assert(auto m = str("alpha ϖ omega").utf8_case_insensitive_find("π"), m.offset == 6 && m.length == 2);

    // Dialytika with Tonos 'ΐ' (CE 90) -> Identity check mostly
    assert(str("ΐ").utf8_case_insensitive_order("ΐ") == sz_equal_k);

    // Greek in Mixed Scripts (boundary checks)
    let_assert(auto m = str("ABCαβγ").utf8_case_insensitive_find("abcΑΒΓ"),
               m.offset == 0 && m.length == 9); // 3 + 3*2 bytes

    // Cyrillic
    // Basic: "привет" vs "ПРИВЕТ"
    assert(str("привет").utf8_case_insensitive_order("ПРИВЕТ") == sz_equal_k);
    let_assert(auto m = str("привет мир").utf8_case_insensitive_find("ПРИВЕТ"),
               m.offset == 0 && m.length == 12); // 6 chars * 2 bytes

    // Cyrillic Context Extensions
    // "Check " is 6 bytes.
    let_assert(auto m = str("Check привет").utf8_case_insensitive_find("ПРИВЕТ"), m.offset == 6 && m.length == 12);
    let_assert(auto m = str("привет check").utf8_case_insensitive_find("ПРИВЕТ"), m.offset == 0 && m.length == 12);

    // Palochka 'Ӏ' (U+04C0, D3 80) -> 'ӏ' (U+04CF, D3 8F)
    // Used in Caucasian languages. Case agnostic.
    let_assert(auto m = str("Ӏ").utf8_case_insensitive_find("ӏ"), m.offset == 0 && m.length == 2);
    let_assert(auto m = str("ӏ").utf8_case_insensitive_find("Ӏ"), m.offset == 0 && m.length == 2);

    // Ukrainian Ґ (U+0490) -> ґ (U+0491)
    let_assert(auto m = str("Ґ").utf8_case_insensitive_find("ґ"), m.offset == 0 && m.length == 2);

    // Mixed Cyrillic
    let_assert(auto m = str("Москва is beautiful").utf8_case_insensitive_find("МОСКВА"),
               m.offset == 0 && m.length == 12); // 6 chars * 2

    // Turkish
    // Dotted 'İ' (U+0130, C4 B0) -> 'i' (ASCII) + combining dot (U+0307, CC 87)
    // "İstanbul" (starts with İ) vs "i̇stanbul" (starts with i + dot)
    // StringZilla finds canonical equivalence. 'İ' (2 bytes) matches 'i̇' (3 bytes).
    let_assert(auto m = str("İstanbul").utf8_case_insensitive_find("i̇stanbul"), // "i" + dot
               m.offset == 0 && m.length == 9); // Haystack length is 2 (İ) + 7 (stanbul) = 9

    // Turkish Context Extensions
    // "Welcome to " is 11 bytes.
    let_assert(auto m = str("Welcome to İstanbul").utf8_case_insensitive_find("i̇stanbul"),
               m.offset == 11 && m.length == 9);
    // "İstanbul city"
    let_assert(auto m = str("İstanbul city").utf8_case_insensitive_find("i̇stanbul"), m.offset == 0 && m.length == 9);

    // Undotted 'ı' (U+0131)
    // Typically 'I' (ASCII) folds to 'i' (ASCII).
    // 'ı' folds to... itself? Or 'I' if we are in Turkish mode?
    // Default fold often treats 'ı' as distinct from 'i'.
    // 'I' -> 'i'. 'ı' -> 'ı'. So 'I' != 'ı'.
    let_assert(auto m = str("I").utf8_case_insensitive_find("ı"), m.offset == str::npos);

    // Turkish Ğ (U+011E) -> ğ (U+011F) and Ş (U+015E) -> ş (U+015F)
    let_assert(auto m = str("ĞŞ").utf8_case_insensitive_find("ğş"), m.offset == 0 && m.length == 4);

    // Armenian
    // Ligature: 'և' (U+0587, D6 87) -> 'ե' (U+0565, D5 A5) + 'ւ' (U+0582, D6 82)
    // Haystack: "և" (2 bytes). Needle: "եւ" (2 + 2 = 4 bytes).
    // Match should return haystack slice (2 bytes).
    let_assert(auto m = str("և").utf8_case_insensitive_find("եւ"), m.offset == 0 && m.length == 2);

    // Armenian Context Extensions
    let_assert(auto m = str("abcև").utf8_case_insensitive_find("եւ"), m.offset == 3 && m.length == 2);
    let_assert(auto m = str("ևabc").utf8_case_insensitive_find("եւ"), m.offset == 0 && m.length == 2);
    // Reverse: Haystack "եւ" (4 bytes). Needle "և" (2 bytes).
    // Match should return haystack slice (4 bytes).
    let_assert(auto m = str("եւ").utf8_case_insensitive_find("և"), m.offset == 0 && m.length == 4);

    // Armenian Context Extensions Reverse
    let_assert(auto m = str("abcեւ").utf8_case_insensitive_find("և"), m.offset == 3 && m.length == 4);

    // Ligature: 'ﬓ' (U+FB13 Men-Now) -> 'մ' (U+0574) + 'ն' (U+0576)
    // Haystack 3 bytes (EF AC 93). Needle 4 bytes (D5 B4 D5 B6).
    let_assert(auto m = str("ﬓ").utf8_case_insensitive_find("մն"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("abcﬓdef").utf8_case_insensitive_find("մն"), m.offset == 3 && m.length == 3);
    let_assert(auto m = str("ﬓ start").utf8_case_insensitive_find("մն start"), m.offset == 0 && m.length == 9);

    // Ligature: 'ﬔ' (U+FB14 Men-Ech) -> 'մ' (U+0574) + 'ե' (U+0565)
    let_assert(auto m = str("ﬔ").utf8_case_insensitive_find("մե"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Some ﬔ text").utf8_case_insensitive_find("մե"), m.offset == 5 && m.length == 3);
    let_assert(auto m = str("End ﬔ").utf8_case_insensitive_find("End մե"), m.offset == 0 && m.length == 7);

    // Ligature: 'ﬕ' (U+FB15 Men-Ini) -> 'մ' (U+0574) + 'ի' (U+056B)
    let_assert(auto m = str("ﬕ").utf8_case_insensitive_find("մի"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("123 ﬕ 456").utf8_case_insensitive_find("123 մի 456"), m.offset == 0 && m.length == 11);
    let_assert(auto m = str("prefixﬕ").utf8_case_insensitive_find("մի"), m.offset == 6 && m.length == 3);

    // Ligature: 'ﬖ' (U+FB16 Vew-Now) -> 'վ' (U+057E) + 'ն' (U+0576)
    let_assert(auto m = str("ﬖ").utf8_case_insensitive_find("վն"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Test ﬖ Case").utf8_case_insensitive_find("Test վն Case"), m.offset == 0 && m.length == 13);
    let_assert(auto m = str("ﬖ").utf8_case_insensitive_find("վն"),
               m.offset == 0 && m.length == 3); // Redundant but safe

    // Ligature: 'ﬗ' (U+FB17 Men-Xeh) -> 'մ' (U+0574) + 'խ' (U+056D)
    let_assert(auto m = str("ﬗ").utf8_case_insensitive_find("մխ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Mid ﬗ dle").utf8_case_insensitive_find("մխ"), m.offset == 4 && m.length == 3);
    let_assert(auto m = str("Start ﬗ").utf8_case_insensitive_find("Start մխ"), m.offset == 0 && m.length == 9);

    // Vietnamese / Latin Extended Additional
    // 'Ạ' (U+1EA0, E1 BA A0) -> 'ạ' (U+1EA1, E1 BA A1)
    let_assert(auto m = str("Ạ").utf8_case_insensitive_find("ạ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Word Ạ End").utf8_case_insensitive_find("Word ạ End"), m.offset == 0 && m.length == 12);
    let_assert(auto m = str("PrefixẠ").utf8_case_insensitive_find("ạ"), m.offset == 6 && m.length == 3);

    // 'Ấ' (U+1EA4, E1 BA A4) -> 'ấ' (U+1EA5, E1 BA A5)
    let_assert(auto m = str("Ấ").utf8_case_insensitive_find("ấ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Ấ Start").utf8_case_insensitive_find("ấ Start"), m.offset == 0 && m.length == 9);
    let_assert(auto m = str("Mid Ấ dle").utf8_case_insensitive_find("Mid ấ dle"), m.offset == 0 && m.length == 11);

    // Horn letters: Ơ (U+01A0, C6 A0) -> ơ (U+01A1, C6 A1), Ư (U+01AF, C6 AF) -> ư (U+01B0, C6 B0)
    let_assert(auto m = str("ƠƯ").utf8_case_insensitive_find("ơư"), m.offset == 0 && m.length == 4);
    let_assert(auto m = str("Big ƠƯ Horns").utf8_case_insensitive_find("Big ơư Horns"),
               m.offset == 0 && m.length == 14);
    let_assert(auto m = str("Prefix ƠƯ").utf8_case_insensitive_find("ơư"), m.offset == 7 && m.length == 4);

    // Latin Extended Additional: Ḁ (U+1E80, E1 BA 80) -> ḁ (U+1E81, E1 BA 81)
    let_assert(auto m = str("Ḁ").utf8_case_insensitive_find("ḁ"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Code Ḁ").utf8_case_insensitive_find("Code ḁ"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("StartḀ").utf8_case_insensitive_find("Startḁ"), m.offset == 0 && m.length == 8);

    // Vietnamese Context Extensions
    let_assert(auto m = str("xin chào Ḁ").utf8_case_insensitive_find("ḁ"), m.offset == 10 && m.length == 3);

    // Special Symbols (Latin)
    // Kelvin Sign 'K' (U+212A, E2 84 AA) -> 'k' (1 byte). Match length includes 3-byte 'K'.
    let_assert(auto m = str("273 K").utf8_case_insensitive_find("273 k"), m.offset == 0 && m.length == 7);

    // Reverse: Haystack "273 k" (5 bytes). Needle "273 K".
    // Should match.
    let_assert(auto m = str("273 k").utf8_case_insensitive_find("273 K"), m.offset == 0 && m.length == 5);

    // Angstrom Sign 'Å' (U+212B) -> 'å' (U+00E5)
    let_assert(auto m = str("Å").utf8_case_insensitive_find("å"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Å").utf8_case_insensitive_find("Å"), m.offset == 0 && m.length == 3);

    // Context Extensions (Special Symbols)
    let_assert(auto m = str("Temp: 273 K").utf8_case_insensitive_find("k"), m.offset == 10 && m.length == 3);
    let_assert(auto m = str("Unit: Å").utf8_case_insensitive_find("å"), m.offset == 6 && m.length == 3);

    // Long S 'ſ' (U+017F) -> 's'
    // "Messer" vs "Meſſer"
    // Haystack "Meſſer": M(1) e(1) ſ(2) ſ(2) e(1) r(1) = 8 bytes.
    // Needle "MESSER": 6 bytes.
    let_assert(auto m = str("Meſſer").utf8_case_insensitive_find("MESSER"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("Ein Meſſer").utf8_case_insensitive_find("MESSER"), m.offset == 4 && m.length == 8);
    let_assert(auto m = str("Meſſer block").utf8_case_insensitive_find("MESSER"), m.offset == 0 && m.length == 8);

    // Ligature 'ﬅ' (U+FB05 "st") -> "st"
    // Haystack "ﬅ" (3 bytes). Needle "st" (2 bytes).
    let_assert(auto m = str("ﬅ").utf8_case_insensitive_find("st"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("Test ﬅ").utf8_case_insensitive_find("Test st"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("ﬅart").utf8_case_insensitive_find("start"), m.offset == 0 && m.length == 6);

    // Ligature 'ﬆ' (U+FB06, EF AC 86) -> "st"
    let_assert(auto m = str("ﬆ").utf8_case_insensitive_find("st"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("My ﬆyle").utf8_case_insensitive_find("My style"), m.offset == 0 && m.length == 9);
    let_assert(auto m = str("Faﬆ").utf8_case_insensitive_find("Fast"), m.offset == 0 && m.length == 5);

    // Extended Ligature Contexts
    // "Messer" vs "Meſſer" ('ſ' is U+017F, C5 BF)
    let_assert(auto m = str("Das Meſſer schneidet").utf8_case_insensitive_find("MESSER"),
               m.offset == 4 && m.length == 8); // "Das " (4) + "Meſſer" (8) = 12, start at 4
    let_assert(auto m = str("Meſſer").utf8_case_insensitive_find("MESSER"), m.offset == 0 && m.length == 8);
    let_assert(auto m = str("Großes Meſſer").utf8_case_insensitive_find("MESSER"),
               m.offset == 8 && m.length == 8); // "Großes " (4+2+1+1+1 = 9 bytes? No. 'ß' is 2 bytes.
                                                // G(1)r(1)o(1)ß(2)e(1)s(1) (1) = 8 bytes. So offset 8.

    // 'ﬅ' (U+FB05, EF AC 85)
    let_assert(auto m = str("Ligature ﬅ check").utf8_case_insensitive_find("st"), m.offset == 9 && m.length == 3);
    let_assert(auto m = str("end with ﬅ").utf8_case_insensitive_find("st"), m.offset == 9 && m.length == 3);

    // More complex ligatures
    let_assert(auto m = str("ﬃJaCä").utf8_case_insensitive_find("fija"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("ﬃJaCä").utf8_case_insensitive_find("ﬁja"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("alﬃJaCä").utf8_case_insensitive_find("fija"), m.offset == 2 && m.length == 5);
    let_assert(auto m = str("alﬃJaCä").utf8_case_insensitive_find("ﬁja"), m.offset == 2 && m.length == 5);

    // 'ﬆ' (U+FB06, EF AC 86)
    let_assert(auto m = str("Big ﬆ").utf8_case_insensitive_find("st"), m.offset == 4 && m.length == 3);

    // Georgian
    // Mtavruli (Upper) -> Mkhedruli (Lower)
    // 'Ა' (U+1C90, E1 B2 90) -> 'ა' (U+10D0, E1 83 90)
    // Both are 3 bytes in UTF-8.
    // Georgian Context
    let_assert(auto m = str("Text Ა").utf8_case_insensitive_find("ა"), m.offset == 5 && m.length == 3);

    // Cherokee
    // Cherokee Supplement (Lower, U+AB70, EA AD B0, 'ꭰ') -> Cherokee (Upper, U+13A0, E1 8E A0, 'Ꭰ')
    // Both 3 bytes.
    let_assert(auto m = str("ꭰ").utf8_case_insensitive_find("Ꭰ"), m.offset == 0 && m.length == 3);

    // Cherokee Context
    let_assert(auto m = str("Syllable ꭰ").utf8_case_insensitive_find("Ꭰ"), m.offset == 9 && m.length == 3);

    // Coptic (Extended)
    // Coptic Ⲡ (U+2C80, E2 B2 80) -> ⲡ (U+2C81, E2 B2 81)
    let_assert(auto m = str("Ⲡ").utf8_case_insensitive_find("ⲡ"), m.offset == 0 && m.length == 3);

    // Glagolitic
    // Ⰰ (U+2C00, E2 B0 80) -> ⰰ (U+2C30, E2 B0 B0)
    let_assert(auto m = str("Ⰰ").utf8_case_insensitive_find("ⰰ"), m.offset == 0 && m.length == 3);

    // Glagolitic Context
    let_assert(auto m = str("Letter Ⰰ").utf8_case_insensitive_find("ⰰ"), m.offset == 7 && m.length == 3);

    // Caseless Scripts (CJK, Arabic, Hebrew, Emoji)
    // These generally don't fold, so they must match exactly or effectively be case-insensitive by identity.

    // Arabic "Salam"
    assert(str("السلام").utf8_case_insensitive_order("السلام") == sz_equal_k);

    // Hebrew "Shalom"
    assert(str("שלום").utf8_case_insensitive_order("שלום") == sz_equal_k);

    // Numbers & Punctuation
    let_assert(auto m = str("12345!@#$%").utf8_case_insensitive_find("345"), m.offset == 2 && m.length == 3);

    // Negative Tests
    // Not found in Cyrillic
    let_assert(auto m = str("Привет").utf8_case_insensitive_find("xyz"), m.offset == str::npos);
    // Not found Cyrillic in ASCII
    let_assert(auto m = str("Hello World").utf8_case_insensitive_find("При"), m.offset == str::npos);

    // CJK "Chinese"
    let_assert(auto m = str("中文测试").utf8_case_insensitive_find("中文"), m.offset == 0 && m.length == 6);

    // Emoji
    let_assert(auto m = str("😀😁😂").utf8_case_insensitive_find("😁"), m.offset == 4 && m.length == 4);

    // Emoji Context
    let_assert(auto m = str("smile 😀😁😂").utf8_case_insensitive_find("😁"), m.offset == 10 && m.length == 4);

    // Regressions & Complex Cases
    // "Fuzz Regression": Needle "nԱԲՐԵշ" (Mixed case Armenian + ASCII)
    let_assert(auto m = str("nԱԲՐԵշ").utf8_case_insensitive_find("nաբրեշ"), m.offset == 0 && m.length == 11);

    // Complex SIMD Regression Trigger
    // Needle includes: ǰ (Latin B), ẞ (Sharp S), Turkish ı, Emoji
    std::string complex_haystack = "\x66\x6F\x78\x74\xD0\xB2\x58\x77\x58\x20\x67\x31\x5A\xEF\xAC\x82"
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
                                   "\xA9\x7A\x4C\x4C";

    std::string complex_needle = "\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80"
                                 "\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3\xB1\x45\x7A\xC3\xBC\x49"
                                 "\x74\x68\x65";

    let_assert(auto m = str(complex_haystack).utf8_case_insensitive_find(complex_needle), m.length != 0);

    // ==========================================================================
    // Cross-Script Mixed Needles (Regression tests for kernel selection issues)
    // ==========================================================================

    // Capital Eszett (U+1E9E, E1 BA 9E) - folds to "ss"
    // Single Capital Eszett
    let_assert(auto m = str("\xE1\xBA\x9E").utf8_case_insensitive_find("ss"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("ss").utf8_case_insensitive_find("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Capital Eszett vs lowercase ß (C3 9F)
    let_assert(auto m = str("\xE1\xBA\x9E").utf8_case_insensitive_find("\xC3\x9F"), m.offset == 0 && m.length == 3);
    let_assert(auto m = str("\xC3\x9F").utf8_case_insensitive_find("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Double Capital Eszett
    let_assert(auto m = str("\xE1\xBA\x9E\xE1\xBA\x9E").utf8_case_insensitive_find("ssss"),
               m.offset == 0 && m.length == 6);

    // Capital Eszett at boundaries
    let_assert(auto m = str("prefix\xE1\xBA\x9E"
                            "suffix")
                            .utf8_case_insensitive_find("xss"),
               m.offset == 5 && m.length == 4); // 'x'(1) + ẞ(3) = 4

    // Capital Eszett + Vietnamese (Western + Vietnamese kernels)
    // ẞ (E1 BA 9E) + ệ (E1 BB 87) - the exact failing pattern from fuzz tests
    let_assert(auto m = str("test\xE1\xBA\x9E\xE1\xBB\x87"
                            "end")
                            .utf8_case_insensitive_find("ss\xE1\xBB\x86"),
               m.offset == 4 && m.length == 6); // ẞ(3) + ệ(3) searched as ss + Ệ

    // Micro Sign + Greek (Western + Greek kernels)
    // µ (C2 B5) surrounded by Greek α (CE B1) and β (CE B2)
    let_assert(auto m = str("\xCE\xB1\xC2\xB5\xCE\xB2").utf8_case_insensitive_find("\xCE\xB1\xCE\xBC\xCE\xB2"),
               m.offset == 0 && m.length == 6); // αµβ vs αμβ

    // Long S (C5 BF) + non-ASCII context
    let_assert(auto m = str("me\xC5\xBF\xC5\xBF"
                            "age")
                            .utf8_case_insensitive_find("MESSAGE"),
               m.offset == 0 && m.length == 9); // meſſage (9 bytes)

    // One-to-Many Expansions (U+1E96-1E9A range)
    // h with line below (U+1E96, E1 BA 96) -> h + combining line below (CC B1)
    let_assert(auto m = str("\xE1\xBA\x96").utf8_case_insensitive_find("h\xCC\xB1"), m.offset == 0 && m.length == 3);

    // t with diaeresis (U+1E97, E1 BA 97) -> t + combining diaeresis (CC 88)
    let_assert(auto m = str("\xE1\xBA\x97").utf8_case_insensitive_find("t\xCC\x88"), m.offset == 0 && m.length == 3);

    // w with ring above (U+1E98, E1 BA 98) -> w + combining ring above (CC 8A)
    let_assert(auto m = str("\xE1\xBA\x98").utf8_case_insensitive_find("w\xCC\x8A"), m.offset == 0 && m.length == 3);

    // y with ring above (U+1E99, E1 BA 99) -> y + combining ring above (CC 8A)
    let_assert(auto m = str("\xE1\xBA\x99").utf8_case_insensitive_find("y\xCC\x8A"), m.offset == 0 && m.length == 3);

    // Kelvin Sign (E2 84 AA) in mixed context
    let_assert(auto m = str("273 \xE2\x84\xAA test").utf8_case_insensitive_find("273 k"),
               m.offset == 0 && m.length == 7); // K is 3 bytes

    // Angstrom Sign (E2 84 AB) with accented chars
    let_assert(auto m = str("10 \xE2\x84\xAB unit").utf8_case_insensitive_find("10 \xC3\xA5"),
               m.offset == 0 && m.length == 6); // Å (3) vs å (2)

    // ==========================================================================
    // 64-byte Boundary Stress Tests
    // ==========================================================================

    // Capital Eszett at position 63 (just at SIMD boundary)
    {
        std::string prefix(63, 'x');
        let_assert(auto m = str((prefix + "\xE1\xBA\x9E"
                                          "end")
                                    .c_str())
                                .utf8_case_insensitive_find("xss"),
                   m.offset == 62 && m.length == 4); // last 'x' + ẞ(3)
    }

    // Vietnamese char at position 62
    {
        std::string prefix(62, 'a');
        let_assert(auto m = str((prefix + "\xE1\xBB\x87"
                                          "b")
                                    .c_str())
                                .utf8_case_insensitive_find("\xE1\xBB\x86"
                                                            "B"),
                   m.offset == 62 && m.length == 4); // ệ(3) + b(1)
    }

    // Micro Sign at position 64 (just past SIMD boundary)
    {
        std::string prefix(64, 'z');
        let_assert(auto m = str((prefix + "\xC2\xB5"
                                          "test")
                                    .c_str())
                                .utf8_case_insensitive_find("\xCE\xBC"),
                   m.offset == 64 && m.length == 2); // µ matches μ
    }

    // Basic ASCII search
    let_assert(auto m = str("Hello World").utf8_case_insensitive_find("WORLD"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("Hello World").utf8_case_insensitive_find("world"), m.offset == 6 && m.length == 5);
    let_assert(auto m = str("HELLO").utf8_case_insensitive_find("hello"), m.offset == 0 && m.length == 5);
    let_assert(auto m = str("Hello").utf8_case_insensitive_find("xyz"), m.offset == str::npos);
    let_assert(auto m = str("Hello").utf8_case_insensitive_find(""), m.offset == 0 && m.length == 0);

    // ==========================================================================
    // Fuzz-Discovered Regressions (Serial vs SIMD mismatches)
    // These patterns were discovered by test_utf8_ci_find_fuzz() and expose
    // disagreements between serial and SIMD implementations.
    // ==========================================================================

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
        let_assert(auto m = str("test\xEF\xAC\x85\xC2\xBA"
                                "end")
                                .utf8_case_insensitive_find("st\xC2\xBA"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + º (2)

        // "st" followed by ñ
        let_assert(auto m = str("test\xEF\xAC\x85\xC3\xB1"
                                "end")
                                .utf8_case_insensitive_find("st\xC3\xB1"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + ñ (2)

        // "st" followed by Greek α
        let_assert(auto m = str("prefix\xEF\xAC\x85\xCE\xB1"
                                "suffix")
                                .utf8_case_insensitive_find("st\xCE\xB1"),
                   m.offset == 6 && m.length == 5); // st ligature (3) + α (2)
    }

    // Pattern 2: "ss" + Latin-1/Greek (Eszett expansion)
    // Needle: 73 73 CE B1 = "ss" + Greek α
    // Needle: 73 73 C3 A5 = "ss" + å
    {
        // "ss" followed by Greek α - should match ß + α
        let_assert(auto m = str("test\xC3\x9F\xCE\xB1"
                                "end")
                                .utf8_case_insensitive_find("ss\xCE\xB1"),
                   m.offset == 4 && m.length == 4); // ß (2) + α (2)

        // "ss" followed by å
        let_assert(auto m = str("prefix\xC3\x9F\xC3\xA5"
                                "suffix")
                                .utf8_case_insensitive_find("ss\xC3\xA5"),
                   m.offset == 6 && m.length == 4); // ß (2) + å (2)
    }

    // Pattern 3: ASCII + combining diacritical + other char
    // Needle: 68 CC B1 D5 A5 = "h" + combining macron below + Armenian ե
    // Needle: 77 CC 8A CE B2 = "w" + combining ring above + Greek β
    // Needle: 6A CC 8C D5 A2 = "j" + combining caron + Armenian բ
    // These test one-to-many expansions (U+1E96 range) mixed with other scripts
    {
        // h + combining macron below should match ẖ (U+1E96)
        let_assert(auto m = str("\xE1\xBA\x96\xD5\xA5").utf8_case_insensitive_find("h\xCC\xB1\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẖ (3) + ե (2)

        // w + combining ring above should match ẘ (U+1E98)
        let_assert(auto m = str("\xE1\xBA\x98\xCE\xB2").utf8_case_insensitive_find("w\xCC\x8A\xCE\xB2"),
                   m.offset == 0 && m.length == 5); // ẘ (3) + β (2)

        // j + combining caron should match ǰ (U+01F0)
        let_assert(auto m = str("\xC7\xB0\xD5\xA2").utf8_case_insensitive_find("j\xCC\x8C\xD5\xA2"),
                   m.offset == 0 && m.length == 4); // ǰ (2) + բ (2)
    }

    // Pattern 4: Modifier letters + other chars
    // Needle: CA BC 6E CE BC = modifier apostrophe + "n" + Greek μ
    // Needle: 61 CA BE D5 A5 = "a" + modifier right half ring + Armenian ե
    // These test n-apostrophe (U+0149) and a-right-half-ring (U+1E9A) expansions
    {
        // 'n (U+0149) expands to modifier apostrophe + n
        let_assert(auto m = str("\xC5\x89\xCE\xBC")
                                .utf8_case_insensitive_find("\xCA\xBC"
                                                            "n\xCE\xBC"),
                   m.offset == 0 && m.length == 4); // ʼn (2) + μ (2)

        // a + modifier right half ring should match ẚ (U+1E9A)
        let_assert(auto m = str("\xE1\xBA\x9A\xD5\xA5").utf8_case_insensitive_find("a\xCA\xBE\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẚ (3) + ե (2)
    }

    // Pattern 5: Armenian + combining chars / ligatures
    // Needle: D5 A5 D6 82 CE B2 = Armenian delays + Greek β
    {
        // Armenian delays character followed by Greek
        let_assert(auto m = str("\xD5\xA5\xD6\x82\xCE\xB2").utf8_case_insensitive_find("\xD5\xA5\xD6\x82\xCE\xB2"),
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
        let_assert(auto m = str(haystack).utf8_case_insensitive_find(needle), m.offset == 0 && m.length == 16);
    }
}

void test_utf8_words() {

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

    // Call it once again at start - temporary measure to debug Ice Lake vs Serial differences
    test_utf8_case();

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
    std::printf("- test_utf8_case...\n");
    test_utf8_case();
#if SZ_IS_CPP17_ && defined(__cpp_lib_string_view)
    std::printf("- test_search_with_misaligned_repetitions...\n");
    test_search_with_misaligned_repetitions();
#endif

    std::printf("\nAll tests passed!\n");
    return 0;
}
