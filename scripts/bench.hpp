/**
 *  @brief  Helper structures and functions for C++ benchmarks.
 */
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <stringzilla/stringzilla.h>
#include <stringzilla/stringzilla.hpp>

#ifdef NDEBUG // Make debugging faster
#define default_seconds_m 10
#else
#define default_seconds_m 10
#endif

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using seconds_t = double;

struct benchmark_result_t {
    std::size_t iterations = 0;
    std::size_t bytes_passed = 0;
    seconds_t seconds = 0;
};

using unary_function_t = std::function<sz_ssize_t(sz_string_view_t)>;
using binary_function_t = std::function<sz_ssize_t(sz_string_view_t, sz_string_view_t)>;

/**
 *  @brief  Wrapper for a single execution backend.
 */
template <typename function_at>
struct tracked_function_gt {
    std::string name {""};
    function_at function {nullptr};
    bool needs_testing {false};

    std::size_t failed_count {0};
    std::vector<std::string> failed_strings;
    benchmark_result_t results;

    void print() const {
        char const *format;
        // Now let's print in the format:
        //  - name, up to 20 characters
        //  - throughput in GB/s with up to 3 significant digits, 10 characters
        //  - call latency in ns with up to 1 significant digit, 10 characters
        //  - number of failed tests, 10 characters
        //  - first example of a failed test, up to 20 characters
        if constexpr (std::is_same<function_at, binary_function_t>())
            format = "- %-20s %15.4f GB/s %15.1f ns %10zu errors in %10zu iterations %s %s\n";
        else
            format = "- %-20s %15.4f GB/s %15.1f ns %10zu errors in %10zu iterations %s\n";
        std::printf(format, name.c_str(), results.bytes_passed / results.seconds / 1.e9,
                    results.seconds * 1e9 / results.iterations, failed_count, results.iterations,
                    failed_strings.size() ? failed_strings[0].c_str() : "",
                    failed_strings.size() ? failed_strings[1].c_str() : "");
    }
};

using tracked_unary_functions_t = std::vector<tracked_function_gt<unary_function_t>>;
using tracked_binary_functions_t = std::vector<tracked_function_gt<binary_function_t>>;

/**
 *  @brief  Stops compilers from optimizing out the expression.
 *          Shamelessly stolen from Google Benchmark.
 */
template <typename value_at>
inline void do_not_optimize(value_at &&value) {
    asm volatile("" : "+r"(value) : : "memory");
}

inline sz_string_view_t sz_string_view(std::string_view str) { return {str.data(), str.size()}; };
inline sz_string_view_t sz_string_view(std::string const &str) { return {str.data(), str.size()}; };

/**
 *  @brief  Rounds the number down to the preceding power of two.
 *          Equivalent to `std::bit_ceil`.
 */
inline std::size_t bit_floor(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_siginificant_bit_position = 0;
    while (n > 1) n >>= 1, most_siginificant_bit_position++;
    return static_cast<std::size_t>(1) << most_siginificant_bit_position;
}

inline std::string read_file(std::string path) {
    std::ifstream stream(path);
    if (!stream.is_open()) { throw std::runtime_error("Failed to open file: " + path); }
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

/**
 *  @brief  Splits a string into words,using newlines, tabs, and whitespaces as delimiters.
 */
inline std::vector<std::string_view> tokenize(std::string_view str) {
    std::vector<std::string_view> words;
    std::size_t start = 0;
    for (std::size_t end = 0; end <= str.length(); ++end) {
        if (end == str.length() || std::isspace(str[end])) {
            if (start < end) words.push_back({&str[start], end - start});
            start = end + 1;
        }
    }
    return words;
}

template <typename string_type>
inline std::vector<string_type> filter_by_length(std::vector<string_type> tokens, std::size_t n) {
    std::vector<string_type> result;
    for (auto const &str : tokens)
        if (str.length() == n) result.push_back(str);
    return result;
}

struct dataset_t {
    std::string text;
    std::vector<std::string_view> tokens;
};

/**
 *  @brief  Loads a dataset from a file.
 */
inline dataset_t make_dataset_from_path(std::string path) {
    dataset_t data;
    data.text = read_file(path);
    data.text.resize(bit_floor(data.text.size()));
    data.tokens = tokenize(data.text);
    data.tokens.resize(bit_floor(data.tokens.size()));

#ifdef NDEBUG // Shuffle only in release mode
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::shuffle(data.tokens.begin(), data.tokens.end(), random_generator);
#endif

    // Report some basic stats about the dataset
    std::size_t mean_bytes = 0;
    for (auto const &str : data.tokens) mean_bytes += str.size();
    mean_bytes /= data.tokens.size();
    std::printf("Parsed the file with %zu words of %zu mean length!\n", data.tokens.size(), mean_bytes);

    return data;
}

/**
 *  @brief  Loads a dataset, depending on the passed CLI arguments.
 */
inline dataset_t make_dataset(int argc, char const *argv[]) {
    if (argc != 2) { throw std::runtime_error("Usage: " + std::string(argv[0]) + " <path>"); }
    return make_dataset_from_path(argv[1]);
}

/**
 *  @brief  Loop over all elements in a dataset in somewhat random order, benchmarking the function cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  function Function to be applied to each `sz_string_view_t`. Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename function_at>
benchmark_result_t bench_on_tokens(strings_at &&strings, function_at &&function,
                                   seconds_t max_time = default_seconds_m) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    benchmark_result_t result;
    std::size_t lookup_mask = bit_floor(strings.size()) - 1;

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask]);
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  Loop over all elements in a dataset, benchmarking the function cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  function Function to be applied to pairs of `sz_string_view_t`.
 *                   Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_at, typename function_at>
benchmark_result_t bench_on_token_pairs(strings_at &&strings, function_at &&function,
                                        seconds_t max_time = default_seconds_m) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    benchmark_result_t result;
    std::size_t lookup_mask = bit_floor(strings.size()) - 1;
    std::size_t largest_prime = 18446744073709551557ull;

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask],
                                            strings[(result.iterations * largest_prime) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask],
                                            strings[(result.iterations * largest_prime) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask],
                                            strings[(result.iterations * largest_prime) & lookup_mask]);
            result.bytes_passed += function(strings[(++result.iterations) & lookup_mask],
                                            strings[(result.iterations * largest_prime) & lookup_mask]);
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > max_time) break;
    }

    return result;
}

/**
 *  @brief  Evaluation for unary string operations: hashing.
 */
template <typename strings_at>
void evaluate_unary_functions(strings_at &&strings, tracked_unary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](auto str) {
                auto baseline = variants[0].function(str);
                auto result = variant.function(str);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) { variant.failed_strings.push_back({str.start, str.length}); }
                }
                return str.length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_tokens(strings, [&](auto str) {
                do_not_optimize(variant.function(str));
                return str.length;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for binary string operations: equality, ordering, prefix, suffix, distance.
 */
template <typename strings_at>
void bench_binary_functions(strings_at &&strings, tracked_binary_functions_t &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_token_pairs(strings, [&](auto str_a, auto str_b) {
                auto baseline = variants[0].function(str_a, str_b);
                auto result = variant.function(str_a, str_b);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) {
                        variant.failed_strings.push_back({str_a.start, str_a.length});
                        variant.failed_strings.push_back({str_b.start, str_b.length});
                    }
                }
                return str_a.length + str_b.length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_token_pairs(strings, [&](auto str_a, auto str_b) {
                do_not_optimize(variant.function(str_a, str_b));
                return str_a.length + str_b.length;
            });
        }

        variant.print();
    }
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian