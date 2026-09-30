/**
 *  @file test/sort.cpp
 *  @author Ash Vardanian
 *  @date February 6, 2024
 *  @brief Sequence sort equivalence/backends/algorithms and intersection tests.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Include the StringZilla headers before anything else, to intercept missing @c #include
 *  directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cstdint> // `std::uintptr_t`
#include <cstdio>  // `stderr`
#include <cstring> // `std::memcpy`

#include <algorithm>     // `std::transform`
#include <iterator>      // `std::distance`
#include <map>           // `std::map`
#include <memory>        // `std::allocator`
#include <numeric>       // `std::accumulate`
#include <random>        // `std::random_device`
#include <set>           // `std::set`
#include <sstream>       // `std::ostringstream`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <vector>        // `std::vector`

#include <fmt/format.h>

#include "cross.hpp"   // `sort_backend_t`, `check_sort_unit_`, `check_intersect_unit_`
#include "harness.hpp" // `random_string`, `test_context_t`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;
using sz::literals::operator""_sv; // for `sz::string_view_t`
using sz::literals::operator""_bs; // for `sz::byteset_t`

using namespace std::literals; // for ""sv

#pragma region Helpers

/** The dispatched byte and uncased arg-sorts, in the shape of their capability kernels. */
static sort_backend_t const sort_dispatched {"dispatched", cpu_best<sz_sequence_argsort_best>,
                                             cpu_best<sz_sequence_argsort_uncased_best>};

#pragma endregion Helpers

#pragma region Unit

/**
 *  @brief Known-answer and coverage tests for the sequence sort and intersect family.
 *
 *  Exercises each function through the dispatched C API with automatic kernel resolution and
 *  through the C++ @c sz::argsort and @c sz::intersect wrappers, so a regression that the
 *  serial-vs-SIMD agreement tests would miss - because both share the same wrong ordering - is
 *  still caught against an external ground truth. The randomized sweeps against @c std::stable_sort
 *  live in @c test_sort_reference_equivalence, as this tier costs the same at any multiplier.
 */
void test_sort_unit() {
    using strs_t = std::vector<std::string>;
    using order_t = std::vector<sz::sorted_idx_t>;

    // Byte arg-sort: {"banana","apple","cherry"} sorts lexicographically to {1, 0, 2}, and uncased
    // {"Banana","apple"} case-folds to {"banana","apple"}, ordering them {1, 0}.
    check_sort_unit_(sort_dispatched);
    verify(sz::argsort(strs_t {"banana", "apple", "cherry"}).value == order_t({1u, 0u, 2u}));
    verify(sz::argsort_utf8_uncased(strs_t {"Banana", "apple"}).value == order_t({1u, 0u}));

    // Intersection: {"apple","banana","cherry"} vs {"cherry","date","banana"} share {"banana","cherry"}.
    {
        std::vector<std::string> const first = {"apple", "banana", "cherry"};
        std::vector<std::string> const second = {"cherry", "date", "banana"};

        // The matched pairs by (first index, second index) are banana=(1,2) and cherry=(2,0), in an
        // unspecified order, so they are compared as a set.
        std::set<intersect_match_t> const expected_pairs = {{1u, 2u}, {2u, 0u}};

        auto const [result, status] = sz::intersect(first, second);
        verify(sz::succeeded(status));
        verify(result.first_offsets.size() == 2u && result.second_offsets.size() == 2u);
        std::set<intersect_match_t> wrapper_pairs;
        for (std::size_t index = 0; index != result.first_offsets.size(); ++index)
            wrapper_pairs.insert({result.first_offsets[index], result.second_offsets[index]});
        verify(wrapper_pairs == expected_pairs);
    }

    // The span overloads fill caller-sized outputs; `intersect` also returns its match count.
    {
        auto as_view = [](std::string const &s) -> sz::string_view_t { return {s.data(), s.size()}; };

        std::vector<std::string> const fruits = {"banana", "apple", "cherry"};
        order_t order(fruits.size());
        verify(sz::argsort(fruits, as_view, {order.data(), order.size()}) == sz::status_t::success_k);
        verify(order == order_t({1u, 0u, 2u}));
        verify(sz::argsort_utf8_uncased(fruits, as_view, {order.data(), order.size()}) == sz::status_t::success_k);
        verify(order == order_t({1u, 0u, 2u}));

        // An output shorter than the collection is refused in release builds too, not written past.
        order_t short_order(fruits.size() - 1);
        verify(sz::argsort(fruits, as_view, {short_order.data(), short_order.size()}) ==
               sz::status_t::unexpected_dimensions_k);

        std::vector<std::string> const first = {"apple", "banana", "cherry"};
        std::vector<std::string> const second = {"cherry", "date", "banana"};
        std::size_t const capacity = (std::min)(first.size(), second.size());
        order_t first_positions(capacity), second_positions(capacity);
        auto const [matched, status] = sz::intersect( //
            first, as_view, second, as_view, /*seed*/ 0u, {first_positions.data(), first_positions.size()},
            {second_positions.data(), second_positions.size()});
        verify(sz::succeeded(status));
        verify(matched == 2u);
        std::set<intersect_match_t> low_level_pairs;
        for (std::size_t index = 0; index != matched; ++index)
            low_level_pairs.insert({first_positions[index], second_positions[index]});
        std::set<intersect_match_t> const expected_low_level = {{1u, 2u}, {2u, 0u}};
        verify(low_level_pairs == expected_low_level);
    }

    // Basic tests with predetermined orders.
    let_verify(auto result = sz::argsort(strs_t({"a", "b", "c", "d"})).value, result == order_t({0u, 1u, 2u, 3u}));
    let_verify(auto result = sz::argsort(strs_t({"b", "c", "d", "a"})).value, result == order_t({3u, 0u, 1u, 2u}));
    let_verify(auto result = sz::argsort(strs_t({"b", "a", "d", "c"})).value, result == order_t({1u, 0u, 3u, 2u}));

    // Single character vs multi-character strings
    let_verify(auto result = sz::argsort(strs_t({"aa", "a", "aaa", "aa"})).value, result == order_t({1u, 0u, 3u, 2u}));

    // Mix of short and long strings with common prefixes
    let_verify(
        auto result = sz::argsort(strs_t({"test", "t", "testing", "te", "tests", "testify", "tea", "team"})).value,
        result == order_t({1u, 3u, 6u, 7u, 0u, 5u, 2u, 4u}));

    // Single character vs multi-character strings with varied patterns
    let_verify(auto result = sz::argsort(strs_t({"zebra", "z", "zoo", "zip", "zap", "a", "apple", "ant", "ark", "mango",
                                                 "m", "maple"}))
                                 .value,
               result == order_t({5u, 7u, 6u, 8u, 10u, 9u, 11u, 1u, 4u, 0u, 3u, 2u}));

    // Numeric-like strings of varying lengths
    let_verify(auto result = sz::argsort(strs_t({"100", "1", "10", "1000", "11", "111", "101", "110"})).value,
               result == order_t({1u, 2u, 0u, 3u, 6u, 4u, 7u, 5u}));

    // Real names with varied lengths and prefixes
    let_verify(
        auto result =
            sz::argsort(strs_t({"Anna", "Andrew", "Alex", "Bob", "Bobby", "Charlie", "Chris", "David", "Dan"})).value,
        result == order_t({2u, 1u, 0u, 3u, 4u, 5u, 6u, 8u, 7u}));
}

/** Known-answer intersection pairs through the dispatched API and the C++ wrapper. */
void test_intersect_unit() {
    check_intersect_unit_(cpu_best<sz_sequence_intersect_best>);

    using strs_t = std::vector<std::string>;
    using result_t = sz::intersect_result_t;

    // The mapping aren't guaranteed to be in any specific order, so we will sort them for comparisons.
    using idx_pairs_t = std::set<intersect_match_t>;
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
            result = sz::intersect(empty, empty).value;
            verify(result.first_offsets.size() == 0 && result.second_offsets.size() == 0);
            result = sz::intersect(abcd, empty).value;
            verify(result.first_offsets.size() == 0 && result.second_offsets.size() == 0);
        }
        // Identity check
        {
            result = sz::intersect(abcd, abcd).value;
            verify(result.first_offsets.size() == 4 && result.second_offsets.size() == 4);
            verify(to_pairs(result) == idx_pairs_t({{0u, 0u}, {1u, 1u}, {2u, 2u}, {3u, 3u}}));
        }
        // Identical size, different order
        {
            result = sz::intersect(abcd, dcba).value;
            verify(result.first_offsets.size() == 4 && result.second_offsets.size() == 4);
            verify(to_pairs(result) == idx_pairs_t({{0u, 3u}, {1u, 2u}, {2u, 1u}, {3u, 0u}}));
        }
        // Different sets
        {
            result = sz::intersect(abcd, abs).value;
            verify(result.first_offsets.size() == 2 && result.second_offsets.size() == 2);
            verify(to_pairs(result) == idx_pairs_t({{0u, 0u}, {1u, 1u}}));
        }
    }
}

/**
 *  @brief Randomized intersection sizes against @c sz::intersect, across dataset sizes and shapes.
 *
 *  Lives here rather than in @c test_intersect_unit because it draws fresh corpora every run: the
 *  @c _unit tier has to cost the same at every multiplier, and these sweeps are exactly the part
 *  that does not.
 */
void test_intersect_equivalence(test_context_t &context) {
    using strs_t = std::vector<std::string>;
    using result_t = sz::intersect_result_t;

    struct {
        std::size_t min_length;
        std::size_t max_length;
        std::size_t count_strings;
    } experiments[] = {
        {10, 10, 100},
        {15, 15, 1000},
        {5, 30, 2000},
    };
    std::mt19937 &generator = context.generator;
    for (auto experiment : experiments) {
        std::unordered_set<std::string> random_strings;
        while (random_strings.size() < experiment.count_strings)
            random_strings.insert(sz::test::random_string(
                generator, experiment.min_length + generator() % (experiment.max_length - experiment.min_length + 1),
                "ab"));

        strs_t all_strings(random_strings.begin(), random_strings.end());
        strs_t first_half(all_strings.begin(), all_strings.begin() + all_strings.size() / 2);

        // Try different joins
        result_t const result = sz::intersect(all_strings, first_half).value;
        verify(result.first_offsets.size() == first_half.size() && result.second_offsets.size() == first_half.size() &&
               "A subset intersected with its superset must recover the whole subset");
    }
}

#pragma endregion Unit

/**
 *  @brief Randomized sorting against a @c std::stable_sort reference, over many sizes and shapes.
 *
 *  Lives here rather than in @c test_sort_unit because it draws fresh corpora every run: the
 *  @c _unit tier has to cost the same at every multiplier, and these sweeps are exactly the part
 *  that does not.
 */
void test_sort_reference_equivalence(test_context_t &context) {
    using strs_t = std::vector<std::string>;
    using order_t = std::vector<sz::sorted_idx_t>;

    // Sizes scale with the multiplier, as this tier's contract requires. The largest crosses 100k strings
    // to exercise the large-input partitioning path.
    std::size_t const dataset_sizes[] = {10u, 100u, 1000u, 10000u, 100000u};
    std::mt19937 &generator = context.generator;
    std::size_t const experiment_count = context.iterations(10);

    // Test on long strings of identical length.
    for (std::size_t string_length : {5u, 25u}) {
        for (std::size_t dataset_size : dataset_sizes) {
            strs_t dataset;
            dataset.reserve(dataset_size);
            for (std::size_t i = 0; i < dataset_size; ++i)
                dataset.push_back(sz::test::random_string(generator, string_length, "ab"));

            for (std::size_t experiment_idx = 0; experiment_idx < experiment_count; ++experiment_idx) {
                std::shuffle(dataset.begin(), dataset.end(), generator);
                auto order = sz::argsort(dataset).value;
                for (std::size_t i = 1; i < dataset.size(); ++i)
                    verify(dataset[order[i - 1]] <= dataset[order[i]] && "argsort output is not sorted");
            }
        }
    }

    // Test on random very small strings of varying lengths, likely with many equal inputs.
    for (std::size_t dataset_size : dataset_sizes) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        for (std::size_t i = 0; i < dataset_size; ++i)
            dataset.push_back(sz::test::random_string(generator, i % 6, "ab"));

        for (std::size_t experiment_idx = 0; experiment_idx < experiment_count; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), generator);
            auto order = sz::argsort(dataset).value;
            for (std::size_t i = 1; i < dataset_size; ++i) {
                verify(dataset[order[i - 1]] <= dataset[order[i]] && "argsort output is not sorted");
            }
        }
    }

    // Test on random strings of varying lengths.
    for (std::size_t dataset_size : dataset_sizes) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        constexpr std::size_t min_length = 6;
        for (std::size_t i = 0; i < dataset_size; ++i)
            dataset.push_back(sz::test::random_string(generator, min_length + i % 32, "ab"));

        for (std::size_t experiment_idx = 0; experiment_idx < experiment_count; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), generator);
            auto order = sz::argsort(dataset).value;
            for (std::size_t i = 1; i < dataset_size; ++i) {
                verify(dataset[order[i - 1]] <= dataset[order[i]] && "argsort output is not sorted");
            }
        }
    }

    // Test on random strings of varying lengths with zero characters.
    for (std::size_t dataset_size : dataset_sizes) {
        strs_t dataset;
        dataset.reserve(dataset_size);
        for (std::size_t i = 0; i < dataset_size; ++i)
            dataset.push_back(sz::test::random_string(generator, i % 32, std::string_view("ab\0", 3)));

        for (std::size_t experiment_idx = 0; experiment_idx < experiment_count; ++experiment_idx) {
            std::shuffle(dataset.begin(), dataset.end(), generator);
            auto order = sz::argsort(dataset).value;
            for (std::size_t i = 1; i < dataset_size; ++i) {
                verify(dataset[order[i - 1]] <= dataset[order[i]] && "argsort output is not sorted");
            }
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
        std::size_t const folded_length = kernel_result<sz_size_t>(cpu_best<sz_utf8_uncased_fold_best>, s.data(),
                                                                   s.size(), destination.data());
        return std::string(destination.data(), folded_length);
    };

    // A duplicate-heavy, mixed-case, multi-script dataset exercising every new knob.
    strs_t mixed;
    {
        char const *seed_words[] = {"apple",  "Apple",   "APPLE",  "banana", "BANANA", "ab",     "AB", "Ab", "aB",
                                    "straße", "STRASSE", "Straße", "Привет", "привет", "ПРИВЕТ", "",   "a",  "A"};
        for (std::size_t repeat = 0; repeat < 200; ++repeat)
            for (char const *word : seed_words) mixed.push_back(word);
        for (std::size_t i = 0; i < 4000; ++i) mixed.push_back(sz::test::random_string(generator, i % 6, "abc"));
        std::shuffle(mixed.begin(), mixed.end(), generator);
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
    enum class sort_direction_t : bool { ascending_k, descending_k };
    auto reference_order = [&](std::vector<std::string> const &keys, sort_direction_t direction) {
        order_t reference(mixed_count);
        std::iota(reference.begin(), reference.end(), 0u);
        std::stable_sort(reference.begin(), reference.end(), [&](sz::sorted_idx_t a, sz::sorted_idx_t b) {
            int const ordering = compare_bytes(keys[a], keys[b]);
            if (ordering != 0) return direction == sort_direction_t::descending_k ? ordering > 0 : ordering < 0;
            return a < b; // Equal keys stay ascending by original index in both directions.
        });
        return reference;
    };

    // Ascending and descending must match a byte-key stable sort exactly.
    verify(is_permutation(sz::argsort(mixed).value) && "argsort output is not a permutation");
    verify(sz::argsort(mixed).value == reference_order(mixed, sort_direction_t::ascending_k) &&
           "Ascending argsort disagrees with the stable-sort reference");
    verify(sz::argsort(mixed, 0, true).value == reference_order(mixed, sort_direction_t::descending_k) &&
           "Descending argsort disagrees with the stable-sort reference");

    // Top-K must reproduce the value-prefix of the full sort and stay a permutation.
    for (std::size_t top_count : {std::size_t(1), std::size_t(50), std::size_t(777), mixed_count}) {
        for (sort_direction_t direction : {sort_direction_t::ascending_k, sort_direction_t::descending_k}) {
            bool const reverse = direction == sort_direction_t::descending_k;
            order_t const got = sz::argsort(mixed, top_count, reverse).value;
            order_t const reference = reference_order(mixed, direction);
            verify(is_permutation(got) && "Top-K argsort output is not a permutation");
            std::size_t const head = top_count < mixed_count ? top_count : mixed_count;
            for (std::size_t i = 0; i < head; ++i)
                verify(mixed[got[i]] == mixed[reference[i]] && "Top-K prefix disagrees with the full sort");
        }
    }

    // Uncased sort must match folding every string then byte-stable-sorting.
    std::vector<std::string> folded(mixed_count);
    for (std::size_t i = 0; i < mixed_count; ++i) folded[i] = fold_string(mixed[i]);
    verify(sz::argsort_utf8_uncased(mixed).value == reference_order(folded, sort_direction_t::ascending_k) &&
           "Ascending uncased argsort disagrees with the folded stable-sort reference");
    verify(sz::argsort_utf8_uncased(mixed, 0, true).value == reference_order(folded, sort_direction_t::descending_k) &&
           "Descending uncased argsort disagrees with the folded stable-sort reference");
}

#pragma region Safety

/** Degenerate sequences through the dispatched arg-sorts, whose output must stay a permutation. */
void test_sort_safety() { check_sort_safety_(sort_dispatched); }

#pragma endregion Safety

#pragma region Drivers

/** Holds the dispatched arg-sorts to serial across the ascending, descending, and top-K modes. */
void test_sort_all(test_context_t &context) { check_sort_equivalence_(context, sort_dispatched); }

#pragma endregion Drivers
