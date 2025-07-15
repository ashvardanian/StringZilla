/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_fingerprint.cuh
 *  @author  Ash Vardanian
 */
#include <cstring> // `std::memcmp`
#include <thread>  // `std::thread::hardware_concurrency`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include "stringzillas/fingerprint.hpp"

#if SZ_USE_CUDA && 0
#include "stringzillas/fingerprint.cuh"
#endif

#if !_SZ_IS_CPP17
#error "This test requires C++17 or later."
#endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

namespace fu = fork_union;
using namespace stringzilla;
using namespace stringzilla::scripts;

template <typename hasher_type_>
void test_rolling_hasher(hasher_type_ &&hasher, std::vector<std::string> const &strs) {

    // Let's make sure that all slice hashes are the same as rolling hashes
    std::size_t window_width = hasher.window_width();
    using hasher_t = typename std::decay<hasher_type_>::type;
    using hash_t = typename hasher_t::hash_t;

    for (std::size_t i = 0; i != strs.size(); ++i) {
        auto const &str = strs[i];
        if (str.size() <= window_width) continue; // Skip very short inputs

        // Compute the hash of the slice
        std::size_t count_hashes = str.size() - window_width + 1;
        std::vector<hash_t> hashes(count_hashes);
        for (std::size_t j = 0; j < count_hashes; ++j) {
            hash_t slice_hash = 0;
            for (std::size_t k = 0; k < window_width; ++k) slice_hash = hasher.update(slice_hash, str[j + k]);
            hashes[j] = slice_hash;
        }

        // Pre-populate the rolling-hash state until the first window ends
        hash_t rolling_hash = 0;
        for (std::size_t j = 0; j < window_width; ++j) rolling_hash = hasher.update(rolling_hash, str[j]);
        _sz_assert(rolling_hash == hashes[0]);

        // Now compute the rolling hash and compare it to the slice hashes
        for (std::size_t j = window_width; j < str.size(); ++j) {
            rolling_hash = hasher.update(rolling_hash, str[j - window_width], str[j]);
            _sz_assert(rolling_hash == hashes[j - window_width + 1]);
        }
    }
}

std::vector<std::string> rolling_hasher_basic_inputs() {
    std::vector<std::string> strings;

    strings.emplace_back("his");
    strings.emplace_back("is");
    strings.emplace_back("she");
    strings.emplace_back("her");
    strings.emplace_back("this is");
    strings.emplace_back("That is a test string");
    strings.emplace_back("ahishers");
    strings.emplace_back("hishishersherishis");
    strings.emplace_back("si siht si a tset gnirts; reh ton si ehs, tub sih ti si.");
    strings.emplace_back("his\0is\r\nshe\0her");

    // Unicode variants:
    strings.emplace_back("école"), strings.emplace_back("école");                   // decomposed
    strings.emplace_back("Schön"), strings.emplace_back("Scho\u0308n");             // combining diaeresis
    strings.emplace_back("naïve"), strings.emplace_back("naive");                   // stripped diaeresis
    strings.emplace_back("façade"), strings.emplace_back("facade");                 // no cedilla
    strings.emplace_back("office"), strings.emplace_back("ofﬁce");                  // “fi” ligature
    strings.emplace_back("Straße"), strings.emplace_back("Strasse");                // ß vs ss
    strings.emplace_back("ABBA"), strings.emplace_back("\u0410\u0412\u0412\u0410"); // Latin vs Cyrillic
    strings.emplace_back("中国"), strings.emplace_back("中國");                     // simplified vs traditional
    strings.emplace_back("🙂"), strings.emplace_back("☺️");                          // emoji variants
    strings.emplace_back("€100"), strings.emplace_back("EUR 100");                  // currency symbol vs abbreviation

    return strings;
}

std::vector<std::string> rolling_hasher_dna_like_inputs() {
    std::vector<std::string> strings;

    fuzzy_config_t config;
    config.alphabet = "ACGT";
    config.batch_size = 100;
    config.min_string_length = 100;
    config.max_string_length = 100 * 1024;

    randomize_strings(config, strings);
    return strings;
}

std::vector<std::string> rolling_hasher_dna_like_inputs() {
    std::vector<std::string> strings;

    fuzzy_config_t config;
    config.alphabet = "ACGT";
    config.batch_size = 100;
    config.min_string_length = 100;
    config.max_string_length = 100 * 1024;

    randomize_strings(config, strings);
    return strings;
}

std::vector<std::string> rolling_hasher_inconvenient_inputs() {
    std::vector<std::string> strings;

    static std::uint8_t const inconvenient_chars[4] = {0x00, 0x01, 0x7F, 0xFF};

    fuzzy_config_t config;
    config.alphabet = {reinterpret_cast<char const *>(&inconvenient_chars[0]), 4};
    config.batch_size = 100;
    config.min_string_length = 100;
    config.max_string_length = 100 * 1024;

    randomize_strings(config, strings);
    return strings;
}

void test_rolling_hasher() {

    // Some very basic variants:
    auto unit_strings = rolling_hasher_basic_inputs();
    auto dna_like_strings = rolling_hasher_dna_like_inputs();
    auto inconvenient_strings = rolling_hasher_inconvenient_inputs();

    using u16u32_hasher_t = rabin_karp_rolling_hasher<u16_t, u32_t>;
    using u32u64_hasher_t = rabin_karp_rolling_hasher<u32_t, u64_t>;
    using u32mul_hasher_t = multiplying_rolling_hasher<u32_t>;
    using i32mul_hasher_t = multiplying_rolling_hasher<i32_t>;
    using u64mul_hasher_t = multiplying_rolling_hasher<u64_t>;
    using u32buz_hasher_t = buz_rolling_hasher<u32_t>;
    using u64buz_hasher_t = buz_rolling_hasher<u64_t>;
    using f32u32_hasher_t = floating_rolling_hasher<float>;
    using f64u64_hasher_t = floating_rolling_hasher<double>;

    std::vector<u16u32_hasher_t> u16u32_hashers;
    u16u32_hashers.emplace_back(3, 31, 65521);
    u16u32_hashers.emplace_back(5, 31, 65521);
    u16u32_hashers.emplace_back(7, 31, 65521);
    for (auto hasher : u16u32_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<u32u64_hasher_t> u32u64_hashers;
    u32u64_hashers.emplace_back(3, 31, 65521);
    u32u64_hashers.emplace_back(5, 31, 65521);
    u32u64_hashers.emplace_back(4, 257, SZ_U32_MAX_PRIME);
    u32u64_hashers.emplace_back(7, 257, SZ_U32_MAX_PRIME);
    for (auto hasher : u32u64_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<u32mul_hasher_t> u32mul_hashers;
    u32mul_hashers.emplace_back(3);
    u32mul_hashers.emplace_back(5);
    u32mul_hashers.emplace_back(4);
    u32mul_hashers.emplace_back(7);
    u32mul_hashers.emplace_back(3, 31);
    u32mul_hashers.emplace_back(5, 65521);
    u32mul_hashers.emplace_back(4, 257);
    u32mul_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    for (auto hasher : u32mul_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<i32mul_hasher_t> i32mul_hashers;
    i32mul_hashers.emplace_back(3);
    i32mul_hashers.emplace_back(5);
    i32mul_hashers.emplace_back(4);
    i32mul_hashers.emplace_back(7);
    i32mul_hashers.emplace_back(3, 31);
    i32mul_hashers.emplace_back(5, 65521);
    i32mul_hashers.emplace_back(4, 257);
    i32mul_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    for (auto hasher : i32mul_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<u64mul_hasher_t> u64mul_hashers;
    u64mul_hashers.emplace_back(3, 31);
    u64mul_hashers.emplace_back(5, 65521);
    u64mul_hashers.emplace_back(4, 257);
    u64mul_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    u64mul_hashers.emplace_back(4, 257);
    u64mul_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    u64mul_hashers.emplace_back(4, 257);
    u64mul_hashers.emplace_back(7, SZ_U64_MAX_PRIME);
    u64mul_hashers.emplace_back(32, 257);
    for (auto hasher : u64mul_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<u32buz_hasher_t> u32buz_hashers;
    u32buz_hashers.emplace_back(3);
    u32buz_hashers.emplace_back(5);
    u32buz_hashers.emplace_back(4);
    u32buz_hashers.emplace_back(7);
    u32buz_hashers.emplace_back(3, 31);
    u32buz_hashers.emplace_back(5, 65521);
    u32buz_hashers.emplace_back(4, 257);
    u32buz_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    for (auto hasher : u32buz_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<u64buz_hasher_t> u64buz_hashers;
    u64buz_hashers.emplace_back(3, 31);
    u64buz_hashers.emplace_back(5, 65521);
    u64buz_hashers.emplace_back(4, 257);
    u64buz_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    u64buz_hashers.emplace_back(4, 257);
    u64buz_hashers.emplace_back(7, SZ_U32_MAX_PRIME);
    u64buz_hashers.emplace_back(4, 257);
    u64buz_hashers.emplace_back(7, SZ_U64_MAX_PRIME);
    u64buz_hashers.emplace_back(32, 257);
    for (auto hasher : u64buz_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<f32u32_hasher_t> f32u32_hashers;
    f32u32_hashers.emplace_back(3, 31);
    f32u32_hashers.emplace_back(5, 65521);
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(32, 257);
    f32u32_hashers.emplace_back(32, 65521);
    f32u32_hashers.emplace_back(3);
    f32u32_hashers.emplace_back(32);
    f32u32_hashers.emplace_back(65);
    f32u32_hashers.emplace_back(257);   // Super-wide window
    f32u32_hashers.emplace_back(1000);  // Super-wide window
    f32u32_hashers.emplace_back(30000); // Super-wide window
    for (auto hasher : f32u32_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);

    std::vector<f64u64_hasher_t> f64u64_hashers;
    f64u64_hashers.emplace_back(3, 31);
    f64u64_hashers.emplace_back(5, 65521);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(32, 257);
    f64u64_hashers.emplace_back(32, 65521);
    f64u64_hashers.emplace_back(3);
    f64u64_hashers.emplace_back(32);
    f64u64_hashers.emplace_back(65);
    f64u64_hashers.emplace_back(257);   // Super-wide window
    f64u64_hashers.emplace_back(1000);  // Super-wide window
    f64u64_hashers.emplace_back(30000); // Super-wide window
    for (auto hasher : f64u64_hashers)
        test_rolling_hasher(hasher, unit_strings), test_rolling_hasher(hasher, dna_like_strings),
            test_rolling_hasher(hasher, inconvenient_strings);
}

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian
