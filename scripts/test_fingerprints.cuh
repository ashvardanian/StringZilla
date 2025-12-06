/**
 *  @brief   Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_fingerprints.cuh
 *  @author  Ash Vardanian
 */
#include <cstring> // `std::memcmp`
#include <thread>  // `std::thread::hardware_concurrency`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include "stringzillas/fingerprints.hpp"

#if SZ_USE_CUDA
#include "stringzillas/fingerprints.cuh"
#include "stringzillas/types.cuh" // `unified_alloc`
#endif

#if !SZ_IS_CPP17_
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

    using hasher_t = typename std::decay<hasher_type_>::type;
    using state_t = typename hasher_t::state_t;
    using hash_t = typename hasher_t::hash_t;

    // Let's make sure that all slice hashes are the same as rolling hashes
    std::size_t window_width = hasher.window_width();

    for (std::size_t i = 0; i != strs.size(); ++i) {
        auto const &str = strs[i];
        if (str.size() <= window_width) continue; // Skip very short inputs

        // Compute the hash of the slice
        std::size_t count_hashes = str.size() - window_width + 1;
        std::vector<hash_t> hashes(count_hashes);
        for (std::size_t j = 0; j < count_hashes; ++j) {
            state_t slice_state = 0;
            for (std::size_t k = 0; k < window_width; ++k) slice_state = hasher.push(slice_state, str[j + k]);
            hashes[j] = hasher.digest(slice_state);
        }

        // Pre-populate the rolling-hash state until the first window ends
        state_t rolling_state = 0;
        for (std::size_t j = 0; j < window_width; ++j) rolling_state = hasher.push(rolling_state, str[j]);
        hash_t rolling_hash = hasher.digest(rolling_state);
        sz_assert_(rolling_hash == hashes[0]);

        // Now compute the rolling hash and compare it to the slice hashes
        for (std::size_t j = window_width; j < str.size(); ++j) {
            rolling_state = hasher.roll(rolling_state, str[j - window_width], str[j]);
            rolling_hash = hasher.digest(rolling_state);
            sz_assert_(rolling_hash == hashes[j - window_width + 1]);
        }
    }
}

template <typename hasher_type_, typename baseline_hasher_type_>
void test_rolling_hasher(hasher_type_ &&hasher, baseline_hasher_type_ &&baseline_hasher,
                         std::vector<std::string> const &strs) {

    using hasher_t = typename std::decay<hasher_type_>::type;
    using state_t = typename hasher_t::state_t;
    using hash_t = typename hasher_t::hash_t;

    using baseline_hasher_t = typename std::decay<baseline_hasher_type_>::type;
    using baseline_state_t = typename baseline_hasher_t::state_t;
    using baseline_hash_t = typename baseline_hasher_t::hash_t;

    // Let's make sure that all slice hashes are the same as rolling hashes
    std::size_t window_width = hasher.window_width();

    for (std::size_t i = 0; i != strs.size(); ++i) {
        auto const &str = strs[i];
        if (str.size() <= window_width) continue; // Skip very short inputs

        // Compute the hash of the slice
        std::size_t count_hashes = str.size() - window_width + 1;
        std::vector<hash_t> hashes(count_hashes);
        std::vector<baseline_hash_t> baseline_hashes(count_hashes);
        for (std::size_t j = 0; j < count_hashes; ++j) {
            state_t slice_state = 0;
            baseline_state_t baseline_slice_state = 0;
            for (std::size_t k = 0; k < window_width; ++k) {
                slice_state = hasher.push(slice_state, str[j + k]);
                baseline_slice_state = baseline_hasher.push(baseline_slice_state, str[j + k]);
            }
            hashes[j] = hasher.digest(slice_state);
            baseline_hashes[j] = baseline_hasher.digest(baseline_slice_state);
            sz_assert_(hashes[j] == baseline_hashes[j] && "Slice hashes do not match baseline hashes");
        }

        // Pre-populate the rolling-hash state until the first window ends
        state_t rolling_state = 0;
        baseline_state_t baseline_rolling_state = 0;
        for (std::size_t j = 0; j < window_width; ++j) {
            rolling_state = hasher.push(rolling_state, str[j]);
            baseline_rolling_state = baseline_hasher.push(baseline_rolling_state, str[j]);
        }
        hash_t rolling_hash = hasher.digest(rolling_state);
        baseline_hash_t baseline_rolling_hash = baseline_hasher.digest(baseline_rolling_state);
        sz_assert_(rolling_hash == baseline_rolling_hash && "Rolling hashes do not match baseline hashes");

        // Now compute the rolling hash and compare it to the slice hashes
        for (std::size_t j = window_width; j < str.size(); ++j) {
            rolling_state = hasher.roll(rolling_state, str[j - window_width], str[j]);
            rolling_hash = hasher.digest(rolling_state);

            baseline_rolling_state = baseline_hasher.roll(baseline_rolling_state, str[j - window_width], str[j]);
            baseline_rolling_hash = baseline_hasher.digest(baseline_rolling_state);

            sz_assert_(rolling_hash == baseline_rolling_hash && "Rolling hashes do not match baseline rolling hashes");
            sz_assert_(rolling_hash == hashes[j - window_width + 1]);
            sz_assert_(baseline_rolling_hash == baseline_hashes[j - window_width + 1]);
        }
    }
}

std::vector<std::string> rolling_hasher_basic_inputs() {
    std::vector<std::string> strings;

    // strings.emplace_back("his");
    // strings.emplace_back("is");
    // strings.emplace_back("she");
    // strings.emplace_back("her");
    strings.emplace_back("this is");
    strings.emplace_back("That is a test string");
    strings.emplace_back("ahishers");
    strings.emplace_back("hishishersherishis");
    strings.emplace_back("si siht si a tset gnirts; reh ton si ehs, tub sih ti si.");
    strings.emplace_back("his\0is\r\nshe\0her");

    // Repetitive patterns to check min-counts
    strings.emplace_back("ab ab ab ab ab");
    strings.emplace_back("ababababab");
    strings.emplace_back("abcabcabcabc");
    strings.emplace_back("a a a a a");
    strings.emplace_back("ab ab ab ab ab ab ab ab ab ab");
    strings.emplace_back("abc abc abc abc abc abc abc abc");

    // Unicode variants
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

    // Try longer strings that will trigger some loop-unrolled optimizations
    strings.emplace_back( //
        "This is a longer string that will be used to test the rolling hasher. "
        "It should be long enough to cover multiple windows and provide a good test case for the "
        "rolling hasher implementation. Let's see how it performs with this longer input string.");

    return strings;
}

std::vector<std::string> rolling_hasher_dna_like_inputs() {
    std::vector<std::string> strings;

    fuzzy_config_t config;
    config.alphabet = "ACGT";
    config.batch_size = scale_iterations(100);
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
    config.batch_size = scale_iterations(100);
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
    using f64u64_hasher_t = floating_rolling_hasher<f64_t>;

    test_rolling_hasher(f64u64_hasher_t(4, 257, 65521), u32u64_hasher_t(4, 257, 65521), unit_strings);
    test_rolling_hasher(f64u64_hasher_t(4, 257, 65521), u32u64_hasher_t(4, 257, 65521), dna_like_strings);
    test_rolling_hasher(f64u64_hasher_t(4, 257, 65521), u32u64_hasher_t(4, 257, 65521), inconvenient_strings);

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
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(4, 257);
    f32u32_hashers.emplace_back(32, 257);
    f32u32_hashers.emplace_back(5, 257, 7001);
    f32u32_hashers.emplace_back(32, 71, 7001);
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
    f64u64_hashers.emplace_back(5, 31, 65521);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(4, 257);
    f64u64_hashers.emplace_back(32, 257);
    f64u64_hashers.emplace_back(32, 257, 65521);
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

template <std::size_t dims_, typename texts_type_, typename baseline_hasher_type_, typename accelerated_hasher_type_>
void test_rolling_hashers_equivalence_against_baseline(texts_type_ const &texts,
                                                       baseline_hasher_type_ const &baseline_hasher,
                                                       accelerated_hasher_type_ const &accelerated_hasher) {
    constexpr std::size_t dims_k = dims_;
    using min_hashes_t = safe_array<u32_t, dims_k>;
    using min_counts_t = safe_array<u32_t, dims_k>;

    arrow_strings_tape_t texts_tape;
    unified_vector<min_hashes_t> serial_hashes_per_text, accelerated_hashes_per_text;
    unified_vector<min_counts_t> serial_counts_per_text, accelerated_counts_per_text;

    sz_assert_(texts_tape.try_assign(texts.begin(), texts.end()) == status_t::success_k);
    serial_hashes_per_text.resize(texts.size());
    accelerated_hashes_per_text.resize(texts.size());
    serial_counts_per_text.resize(texts.size());
    accelerated_counts_per_text.resize(texts.size());

    // Compute the fingerprints
    for (size_t text_index = 0; text_index < texts.size(); ++text_index) {
        auto text = texts_tape[text_index];
        min_hashes_t &serial_hashes = serial_hashes_per_text[text_index];
        min_counts_t &serial_counts = serial_counts_per_text[text_index];
        min_hashes_t &accelerated_hashes = accelerated_hashes_per_text[text_index];
        min_counts_t &accelerated_counts = accelerated_counts_per_text[text_index];
        baseline_hasher.template try_fingerprint<dims_k>(text.template cast<byte_t const>(), serial_hashes,
                                                         serial_counts);
        accelerated_hasher.try_fingerprint(text.template cast<byte_t const>(), accelerated_hashes, accelerated_counts);

        // Compare the results
        std::size_t const first_mismatch_index =
            std::mismatch(serial_hashes.begin(), serial_hashes.end(), accelerated_hashes.begin()).first -
            serial_hashes.begin();

        if (first_mismatch_index != serial_hashes.size()) {
            std::printf("Fingerprint mismatch at index %zu:\n", first_mismatch_index);
            std::printf("  String: \"%.*s\"\n", static_cast<int>(text.size()), text.data());
            std::printf("  Serial hash:      %u\n", serial_hashes[first_mismatch_index]);
            std::printf("  Accelerated hash: %u\n", accelerated_hashes[first_mismatch_index]);
            std::printf("  Serial count:     %u\n", serial_counts[first_mismatch_index]);
            std::printf("  Accelerated count:%u\n", accelerated_counts[first_mismatch_index]);
            for (std::size_t i = 0; i < serial_hashes.size(); ++i) {
                std::printf("  [%zu] serial=%u accelerated=%u\n", i, serial_hashes[i], accelerated_hashes[i]);
            }
        }
        sz_assert_(first_mismatch_index == serial_hashes.size() && "Fingerprints do not match");

        // Counters can't be zero, if the input string is at least the size of a window
        for (std::size_t i = 0; i < serial_counts.size(); ++i) {
            if (text.size() >= baseline_hasher.window_width(i)) {
                sz_assert_(serial_counts[i] > 0 && "Serial fingerprint count is zero");
                sz_assert_(accelerated_counts[i] > 0 && "Accelerated fingerprint count is zero");
            }
            else {
                sz_assert_(serial_counts[i] == 0 && "Serial fingerprint should be zero");
                sz_assert_(accelerated_counts[i] == 0 && "Accelerated fingerprint should be zero");
            }
        }

        // Compare the counts
        std::size_t const first_counts_mismatch_index =
            std::mismatch(serial_counts.begin(), serial_counts.end(), accelerated_counts.begin()).first -
            serial_counts.begin();
        if (first_counts_mismatch_index != serial_counts.size()) {
            std::printf("Fingerprint counts mismatch at index %zu:\n", first_counts_mismatch_index);
            std::printf("  String: \"%.*s\"\n", static_cast<int>(text.size()), text.data());
            std::printf("  Serial count:      %u\n", serial_counts[first_counts_mismatch_index]);
            std::printf("  Accelerated count: %u\n", accelerated_counts[first_counts_mismatch_index]);
            for (std::size_t i = 0; i < serial_counts.size(); ++i) {
                std::printf("  [%zu] serial=%u accelerated=%u\n", i, serial_counts[i], accelerated_counts[i]);
            }
        }
        sz_assert_(first_counts_mismatch_index == serial_counts.size() && "Fingerprint counts do not match");
    }
}

/**
 *  Compares the equivalence of SIMD backends to @b `floating_rolling_hashers<sz_cap_serial_k>`
 *  and the simpler `basic_rolling_hashers<floating_rolling_hasher<f64_t>, ..., u32_t>`.
 */
template <std::size_t window_width_, std::size_t dims_>
void test_rolling_hashers_equivalence_for_width() {

    constexpr std::size_t window_width_k = window_width_;
    constexpr std::size_t dims_k = dims_;

    // Define hasher classes
    using rolling_f64_t = basic_rolling_hashers<floating_rolling_hasher<f64_t>, u32_t>;
    rolling_f64_t rolling_f64;
    sz_assert_(rolling_f64.try_extend(window_width_k, dims_k) == status_t::success_k);

    // Test on each individual dataset
    auto unit_strings = rolling_hasher_basic_inputs();
    auto dna_like_strings = rolling_hasher_dna_like_inputs();
    auto inconvenient_strings = rolling_hasher_inconvenient_inputs();

    using rolling_serial_t = floating_rolling_hashers<sz_cap_serial_k, dims_k>;
    rolling_serial_t rolling_serial;
    sz_assert_(rolling_serial.try_seed(window_width_k) == status_t::success_k);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(unit_strings, rolling_f64, rolling_serial);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(dna_like_strings, rolling_f64, rolling_serial);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(inconvenient_strings, rolling_f64, rolling_serial);

#if SZ_USE_HASWELL
    using rolling_haswell_t = floating_rolling_hashers<sz_cap_haswell_k, dims_k>;
    rolling_haswell_t rolling_haswell;
    sz_assert_(rolling_haswell.try_seed(window_width_k) == status_t::success_k);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(unit_strings, rolling_f64, rolling_haswell);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(dna_like_strings, rolling_f64, rolling_haswell);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(inconvenient_strings, rolling_f64, rolling_haswell);
#endif

#if SZ_USE_SKYLAKE
    using rolling_skylake_t = floating_rolling_hashers<sz_cap_skylake_k, dims_k>;
    rolling_skylake_t rolling_skylake;
    sz_assert_(rolling_skylake.try_seed(window_width_k) == status_t::success_k);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(unit_strings, rolling_f64, rolling_skylake);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(dna_like_strings, rolling_f64, rolling_skylake);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(inconvenient_strings, rolling_f64, rolling_skylake);
#endif

#if SZ_USE_CUDA
    using rolling_cuda_t = floating_rolling_hashers<sz_cap_cuda_k, dims_k>;
    rolling_cuda_t rolling_cuda;
    sz_assert_(rolling_cuda.try_seed(window_width_k) == status_t::success_k);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(unit_strings, rolling_f64, rolling_cuda);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(dna_like_strings, rolling_f64, rolling_cuda);
    test_rolling_hashers_equivalence_against_baseline<dims_k>(inconvenient_strings, rolling_f64, rolling_cuda);
#endif
}

void test_rolling_hashers_equivalence() {
    // Just 2 hashes per input
    // test_rolling_hashers_equivalence_for_width<3, 2>();
    test_rolling_hashers_equivalence_for_width<7, 2>();

    // 32 hashes per input
    test_rolling_hashers_equivalence_for_width<3, 32>();
    test_rolling_hashers_equivalence_for_width<7, 32>();
    test_rolling_hashers_equivalence_for_width<33, 32>();
    test_rolling_hashers_equivalence_for_width<64, 32>();

    // 32 hashes per input with windows divisible by 4
    test_rolling_hashers_equivalence_for_width<4, 32>();
    test_rolling_hashers_equivalence_for_width<8, 32>();
    test_rolling_hashers_equivalence_for_width<12, 32>();
    test_rolling_hashers_equivalence_for_width<16, 32>();
}

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian
