/**
 *  @brief  Sequence sort equivalence/backends/algorithms and intersection tests.
 *  @file   scripts/test_sort.cpp
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

#pragma region Helpers

/** @brief Wraps a `std::vector<std::string>` as an `sz_sequence_t` so C backends can be called directly. */
static sz_cptr_t sort_sequence_get_start_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].data();
}
static sz_size_t sort_sequence_get_length_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].size();
}

/** @brief Fills an `sz_sequence_t` view over a `std::vector<std::string>` via the shared accessor helpers. */
static sz_sequence_t sort_sequence_from_(std::vector<std::string> const &strings) {
    sz_sequence_t sequence;
    sequence.handle = &strings;
    sequence.count = (sz_size_t)strings.size();
    sequence.get_start = sort_sequence_get_start_;
    sequence.get_length = sort_sequence_get_length_;
    return sequence;
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer + coverage for the sequence sort & intersect family on hand-verifiable inputs.
 *
 *  Exercises each function through the dispatched C API (automatic kernel resolution), through the
 *  natively-compiled backend kernels directly (manual propagation to a specific kernel), and through the
 *  C++ `sz::argsort` / `sz::intersect` wrappers, so a regression that the serial-vs-SIMD agreement tests
 *  would miss - because both share the same wrong ordering - is still caught against an external ground truth.
 *  Then sorts incrementally complex inputs against `std::stable_sort` references:
 *  1. Basic tests with predetermined orders.
 *  2. Test on long strings of identical length.
 *  3. Test on random very small strings of varying lengths, likely with many equal inputs.
 *  4. Test on random strings of varying lengths.
 *  5. Test on random strings of varying lengths with zero characters.
 */
void test_sort_unit() {
    using strs_t = std::vector<std::string>;
    using order_t = std::vector<sz::sorted_idx_t>;

    std::printf("  - testing sequence sort & intersect known-answer vectors...\n");

    // Byte arg-sort: {"banana","apple","cherry"} sorts lexicographically to {"apple","banana","cherry"},
    // so the permutation is {1, 0, 2}.
    {
        std::vector<std::string> const fruits = {"banana", "apple", "cherry"};
        sz_sequence_t const sequence = sort_sequence_from_(fruits);
        sz_sorted_idx_t const expected[3] = {1u, 0u, 2u};

        sz_sorted_idx_t order[3];
        assert(sz_sequence_argsort(&sequence, nullptr, order, 0, sz_false_k) == sz_success_k); // Dispatched
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);
        assert(sz_sequence_argsort_serial(&sequence, nullptr, order, 0, sz_false_k) == sz_success_k); // Serial kernel
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);
#if SZ_USE_HASWELL
        assert(sz_sequence_argsort_haswell(&sequence, nullptr, order, 0, sz_false_k) == sz_success_k); // Haswell kernel
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);
#endif
#if SZ_USE_SKYLAKE
        assert(sz_sequence_argsort_skylake(&sequence, nullptr, order, 0, sz_false_k) == sz_success_k); // Skylake kernel
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);
#endif

        // C++ wrapper must agree with the dispatched C API on the same permutation.
        assert(sz::argsort(fruits) == std::vector<sz::sorted_idx_t>({1u, 0u, 2u}));
    }

    // Uncased UTF-8 arg-sort: {"Banana","apple"} case-folds to {"banana","apple"}, ordering them {1, 0}.
    {
        std::vector<std::string> const words = {"Banana", "apple"};
        sz_sequence_t const sequence = sort_sequence_from_(words);
        sz_sorted_idx_t const expected[2] = {1u, 0u};

        sz_sorted_idx_t order[2];
        assert(sz_sequence_argsort_utf8_uncased(&sequence, nullptr, order, 0, sz_false_k) == sz_success_k); // Dispatched
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);
        assert(sz_sequence_argsort_utf8_uncased_serial( // Serial kernel
                   &sequence, nullptr, order, 0, sz_false_k) == sz_success_k);
        assert(std::memcmp(order, expected, sizeof(expected)) == 0);

        // C++ wrapper must agree on the case-folded ordering.
        assert(sz::argsort_utf8_uncased(words) == std::vector<sz::sorted_idx_t>({1u, 0u}));
    }

    // Intersection: {"apple","banana","cherry"} vs {"cherry","date","banana"} share {"banana","cherry"}.
    {
        std::vector<std::string> const first = {"apple", "banana", "cherry"};
        std::vector<std::string> const second = {"cherry", "date", "banana"};
        sz_sequence_t const first_sequence = sort_sequence_from_(first);
        sz_sequence_t const second_sequence = sort_sequence_from_(second);

        // The matched pairs by (first index, second index): banana=(1,2) and cherry=(2,0). Output order is
        // unspecified, so collect the pairs into a set before comparing against the known intersection.
        using idx_pairs_t = std::set<std::pair<std::size_t, std::size_t>>;
        idx_pairs_t const expected_pairs = {{1u, 2u}, {2u, 0u}};

        auto check_intersect = [&](sz_status_t status, sz_size_t intersection_size, //
                                   sz_sorted_idx_t const *first_positions, sz_sorted_idx_t const *second_positions) {
            assert(status == sz_success_k);
            assert(intersection_size == 2u);
            idx_pairs_t produced;
            for (sz_size_t i = 0; i != intersection_size; ++i)
                produced.insert({(std::size_t)first_positions[i], (std::size_t)second_positions[i]});
            assert(produced == expected_pairs);
        };

        sz_size_t intersection_size = 0;
        sz_sorted_idx_t first_positions[3], second_positions[3];
        check_intersect(sz_sequence_intersect( // Dispatched (automatic kernel)
                            &first_sequence, &second_sequence, nullptr, 0u, &intersection_size, //
                            first_positions, second_positions),
                        intersection_size, first_positions, second_positions);
        check_intersect(sz_sequence_intersect_serial( // Manual propagation to the serial kernel
                            &first_sequence, &second_sequence, nullptr, 0u, &intersection_size, //
                            first_positions, second_positions),
                        intersection_size, first_positions, second_positions);
#if SZ_USE_ICELAKE
        check_intersect(sz_sequence_intersect_icelake( // Manual propagation to the icelake kernel
                            &first_sequence, &second_sequence, nullptr, 0u, &intersection_size, //
                            first_positions, second_positions),
                        intersection_size, first_positions, second_positions);
#endif

        // C++ wrapper must surface the same matched pairs.
        sz::intersect_result_t const result = sz::intersect(first, second);
        assert(result.first_offsets.size() == 2u && result.second_offsets.size() == 2u);
        idx_pairs_t wrapper_pairs;
        for (std::size_t i = 0; i != result.first_offsets.size(); ++i)
            wrapper_pairs.insert({result.first_offsets[i], result.second_offsets[i]});
        assert(wrapper_pairs == expected_pairs);
    }

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

    // Stability, reverse, top-K, and uncased coverage against `std::stable_sort` references.
    auto compare_bytes = [](std::string const &a, std::string const &b) -> int {
        std::size_t const min_length = a.size() < b.size() ? a.size() : b.size();
        for (std::size_t i = 0; i < min_length; ++i) {
            unsigned char const ca = (unsigned char)a[i], cb = (unsigned char)b[i];
            if (ca != cb) return ca < cb ? -1 : 1;
        }
        return a.size() == b.size() ? 0 : (a.size() < b.size() ? -1 : 1);
    };
    auto fold_string = [](std::string const &s) -> std::string {
        std::vector<char> destination(s.size() * 3 + 4);
        std::size_t const folded_length = sz_utf8_uncased_fold(s.data(), s.size(), destination.data());
        return std::string(destination.data(), folded_length);
    };

    // A duplicate-heavy, mixed-case, multi-script dataset exercising every new knob.
    strs_t mixed;
    {
        char const *seed_words[] = {"apple",  "Apple",   "APPLE",  "banana", "BANANA", "ab",     "AB", "Ab", "aB",
                                    "straße", "STRASSE", "Straße", "Привет", "привет", "ПРИВЕТ", "",   "a",  "A"};
        for (std::size_t repeat = 0; repeat < 200; ++repeat)
            for (char const *word : seed_words) mixed.push_back(word);
        for (std::size_t i = 0; i < 4000; ++i) mixed.push_back(sz::scripts::random_string(i % 6, "abc", 3));
        std::shuffle(mixed.begin(), mixed.end(), global_random_generator());
    }
    std::size_t const mixed_count = mixed.size();
    auto is_permutation = [&](order_t const &order) {
        std::vector<char> seen(mixed_count, 0);
        for (auto idx : order) {
            if (idx >= mixed_count || seen[idx]) return false;
            seen[idx] = 1;
        }
        return order.size() == mixed_count;
    };
    auto reference_order = [&](std::vector<std::string> const &keys, bool reverse) {
        order_t reference(mixed_count);
        std::iota(reference.begin(), reference.end(), 0u);
        std::stable_sort(reference.begin(), reference.end(), [&](sz::sorted_idx_t a, sz::sorted_idx_t b) {
            int const ordering = compare_bytes(keys[a], keys[b]);
            if (ordering != 0) return reverse ? ordering > 0 : ordering < 0;
            return a < b; // Equal keys stay ascending by original index in both directions.
        });
        return reference;
    };

    // Ascending and descending must match a byte-key stable sort exactly.
    assert(is_permutation(sz::argsort(mixed)) && sz::argsort(mixed) == reference_order(mixed, false));
    assert(sz::argsort(mixed, 0, true) == reference_order(mixed, true));

    // Top-K must reproduce the value-prefix of the full sort and stay a permutation.
    for (std::size_t top_count : {std::size_t(1), std::size_t(50), std::size_t(777), mixed_count}) {
        for (bool reverse : {false, true}) {
            order_t const got = sz::argsort(mixed, top_count, reverse);
            order_t const reference = reference_order(mixed, reverse);
            assert(is_permutation(got));
            std::size_t const head = top_count < mixed_count ? top_count : mixed_count;
            for (std::size_t i = 0; i < head; ++i) assert(mixed[got[i]] == mixed[reference[i]]);
        }
    }

    // Uncased sort must match folding every string then byte-stable-sorting.
    std::vector<std::string> folded(mixed_count);
    for (std::size_t i = 0; i < mixed_count; ++i) folded[i] = fold_string(mixed[i]);
    assert(sz::argsort_utf8_uncased(mixed) == reference_order(folded, false));
    assert(sz::argsort_utf8_uncased(mixed, 0, true) == reference_order(folded, true));
}

/**
 *  @brief Tests array intersection functionality.
 */
void test_intersect_unit() {
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

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps a sequence argsort backend (byte or uncased) by its kernel pointer. */
template <sz_sequence_argsort_t argsort_>
struct sequence_argsort_from_sz_ {
    sz_status_t operator()(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc, sz_sorted_idx_t *order,
                           sz_size_t top, sz_bool_t reverse) const noexcept {
        return argsort_(sequence, alloc, order, top, reverse);
    }
};

/** @brief Wraps a pgram integer-sort backend by its kernel pointer. */
template <sz_pgrams_sort_t pgrams_sort_>
struct pgrams_sort_from_sz_ {
    sz_status_t operator()(sz_pgram_t *keys, sz_size_t count, sz_memory_allocator_t *alloc,
                           sz_sorted_idx_t *order) const noexcept {
        return pgrams_sort_(keys, count, alloc, order);
    }
};

/** @brief Bundles a backend's byte arg-sort, uncased arg-sort, and pgram sort under a single name. */
template <sz_sequence_argsort_t argsort_, sz_sequence_argsort_t argsort_uncased_, sz_pgrams_sort_t pgrams_sort_>
struct sequence_sort_from_sz_ {
    sequence_argsort_from_sz_<argsort_> argsort;
    sequence_argsort_from_sz_<argsort_uncased_> argsort_uncased;
    pgrams_sort_from_sz_<pgrams_sort_> pgrams_sort;
};

/**
 *  @brief Demands a candidate sort backend produce results identical to the reference backend, across many inputs.
 *
 *  Both the byte and uncased arg-sorts are @b stable, so for any input the permutation is unique - the candidate
 *  and reference `order` arrays must match exactly across ascending, descending, and top-K modes. The pgram integer
 *  sort is not stable, so only the @b sorted key arrays are compared (equal keys may tie-break differently between
 *  backends), with the permutation checked for validity.
 *
 *  Mirrors `test_hash_equivalence` & friends: one generic checker, invoked once per ISA under `SZ_USE_*`.
 *
 *  @param reference The serial backend treated as ground truth (a `sequence_sort_from_sz_` instance).
 *  @param candidate The SIMD backend under scrutiny (a `sequence_sort_from_sz_` instance).
 *  @param inputs Unused dataset-size hint, kept for signature parity with the other equivalence checkers.
 */
template <typename reference_, typename candidate_>
void test_sort_equivalence(reference_ reference, candidate_ candidate, sz_size_t inputs) {
    [[maybe_unused]] sz_size_t const inputs_hint = inputs;

    using strs_t = std::vector<std::string>;
    auto &generator = global_random_generator();

    // Datasets spanning the vectorized block, the scalar tail, and the slack-region boundaries; sizes are
    // deliberately > 32 so the QuickSort partition (not the insertion-sort fallback) is exercised.
    std::vector<strs_t> datasets;
    for (std::size_t count : {33u, 64u, 100u, 1000u, 5000u}) {
        strs_t fixed_dups; // Short strings over a tiny alphabet => many exact duplicates (fills the equal region).
        for (std::size_t i = 0; i < count; ++i) fixed_dups.push_back(sz::scripts::random_string(i % 5, "ab", 2));
        datasets.push_back(fixed_dups);
        strs_t varied; // Longer, common-prefix strings => deep pgram recursion.
        for (std::size_t i = 0; i < count; ++i) varied.push_back(sz::scripts::random_string(6 + i % 40, "abc", 3));
        datasets.push_back(varied);
    }
    { // Deterministic mixed-case / multi-script set so the uncased path sees real folds.
        char const *seed[] = {"Apple",   "apple",  "BANANA", "banana", "Straße",
                              "STRASSE", "Привет", "ПРИВЕТ", "Ab",     "aB"};
        strs_t mixed;
        for (std::size_t r = 0; r < 50; ++r)
            for (char const *word : seed) mixed.push_back(word);
        std::shuffle(mixed.begin(), mixed.end(), generator);
        datasets.push_back(mixed);
    }

    for (strs_t const &dataset : datasets) {
        std::size_t const count = dataset.size();
        sz_sequence_t sequence;
        sequence.handle = &dataset;
        sequence.count = count;
        sequence.get_start = sort_sequence_get_start_;
        sequence.get_length = sort_sequence_get_length_;

        std::vector<sz_sorted_idx_t> order_reference(count), order_candidate(count);
        for (sz_size_t top : {sz_size_t(0), sz_size_t(1), (sz_size_t)(count / 3), (sz_size_t)count}) {
            for (sz_bool_t reverse : {sz_false_k, sz_true_k}) {
                std::size_t const head = (top != 0 && top < count) ? top : count;

                // Byte arg-sort: stable, so the permutations must match exactly over the ordered prefix.
                reference.argsort(&sequence, nullptr, order_reference.data(), top, reverse);
                candidate.argsort(&sequence, nullptr, order_candidate.data(), top, reverse);
                for (std::size_t i = 0; i < head; ++i)
                    assert(order_reference[i] == order_candidate[i] && "SIMD byte arg-sort disagrees with serial");

                // Uncased arg-sort: also stable, same exact-match requirement.
                reference.argsort_uncased(&sequence, nullptr, order_reference.data(), top, reverse);
                candidate.argsort_uncased(&sequence, nullptr, order_candidate.data(), top, reverse);
                for (std::size_t i = 0; i < head; ++i)
                    assert(order_reference[i] == order_candidate[i] && "SIMD uncased arg-sort disagrees with serial");
            }
        }
    }

    // Pgram integer sort: compare the sorted key arrays (the sort is not stable, so permutations may differ).
    for (std::size_t count : {33u, 100u, 1000u, 5000u}) {
        std::vector<sz_pgram_t> keys(count);
        for (auto &key : keys) key = (sz_pgram_t)(generator() % 50); // Heavy duplication exercises the equal region.
        std::vector<sz_pgram_t> keys_reference = keys, keys_candidate = keys;
        std::vector<sz_sorted_idx_t> order_reference(count), order_candidate(count);
        reference.pgrams_sort(keys_reference.data(), count, nullptr, order_reference.data());
        candidate.pgrams_sort(keys_candidate.data(), count, nullptr, order_candidate.data());
        assert(keys_reference == keys_candidate && "SIMD pgram sort produced a different sorted order than serial");
        for (std::size_t i = 0; i < count; ++i)
            assert(keys[order_candidate[i]] == keys_candidate[i] && "SIMD pgram sort permutation is invalid");
    }
}

#pragma endregion // Equivalence

#pragma region Drivers

/** @brief Runs `test_sort_equivalence` (serial reference vs. every enabled SIMD backend candidate). */
void test_sort_all() {
    sequence_sort_from_sz_<sz_sequence_argsort_serial, sz_sequence_argsort_utf8_uncased_serial, sz_pgrams_sort_serial>
        reference;
#if SZ_USE_HASWELL
    test_sort_equivalence(reference,
                          sequence_sort_from_sz_<sz_sequence_argsort_haswell, sz_sequence_argsort_utf8_uncased_haswell,
                                                 sz_pgrams_sort_haswell> {},
                          0);
#endif
#if SZ_USE_SKYLAKE
    test_sort_equivalence(reference,
                          sequence_sort_from_sz_<sz_sequence_argsort_skylake, sz_sequence_argsort_utf8_uncased_skylake,
                                                 sz_pgrams_sort_skylake> {},
                          0);
#endif
#if SZ_USE_SVE
    test_sort_equivalence(
        reference,
        sequence_sort_from_sz_<sz_sequence_argsort_sve, sz_sequence_argsort_utf8_uncased_sve, sz_pgrams_sort_sve> {}, 0);
#endif
#if SZ_USE_NEON
    test_sort_equivalence(
        reference,
        sequence_sort_from_sz_<sz_sequence_argsort_neon, sz_sequence_argsort_utf8_uncased_neon, sz_pgrams_sort_neon> {},
        0);
#endif
#if SZ_USE_RVV
    test_sort_equivalence(
        reference,
        sequence_sort_from_sz_<sz_sequence_argsort_rvv, sz_sequence_argsort_utf8_uncased_rvv, sz_pgrams_sort_rvv> {}, 0);
#endif
}

#pragma endregion // Drivers
