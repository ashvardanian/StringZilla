/**
 *  @brief  Helper structures and functions for C++ benchmarks.
 */
#pragma once
#include <algorithm>
#include <chrono>     // `std::chrono::high_resolution_clock`
#include <clocale>    // `std::setlocale`
#include <cstring>    // `std::memcpy`
#include <functional> // `std::equal_to`
#include <limits>     // `std::numeric_limits`
#include <random>     // `std::random_device`, `std::mt19937`
#include <string>     // `std::hash`
#include <vector>

#include <string_view> // Requires C++17

#include <stringzilla/stringzilla.h>
#include <stringzilla/stringzilla.hpp>

#include "test.hpp" // `read_file`

namespace sz = ashvardanian::stringzilla;

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using seconds_t = double;

struct benchmark_result_t {
    std::size_t iterations = 0;
    std::size_t bytes_passed = 0;
    seconds_t seconds = 0;
};

using unary_function_t = std::function<std::size_t(std::string_view)>;
using binary_function_t = std::function<std::size_t(std::string_view, std::string_view)>;

/**
 *  @brief  Wrapper for a single execution backend.
 */
template <typename function_type>
struct tracked_function_gt {
    std::string name {""};
    function_type function {nullptr};
    bool needs_testing {false};

    std::size_t failed_count;
    std::vector<std::string> failed_strings;
    benchmark_result_t results;

    tracked_function_gt(std::string name = "", function_type function = nullptr, bool needs_testing = false)
        : name(name), function(function), needs_testing(needs_testing), failed_count(0), failed_strings(), results() {}

    tracked_function_gt(tracked_function_gt const &) = default;
    tracked_function_gt &operator=(tracked_function_gt const &) = default;

    void print() const {
        bool is_binary = std::is_same<function_type, binary_function_t>();

        // If failures have occured, output them to file tos implify the debugging process.
        bool contains_failures = !failed_strings.empty();
        if (contains_failures) {
            // The file name is made of the string hash and the function name.
            for (std::size_t fail_index = 0; fail_index != failed_strings.size();) {
                std::string const &first_argument = failed_strings[fail_index];
                std::string file_name =
                    "failed_" + name + "_" + std::to_string(std::hash<std::string> {}(first_argument));
                if (is_binary) {
                    std::string const &second_argument = failed_strings[fail_index + 1];
                    write_file(file_name + ".first.txt", first_argument);
                    write_file(file_name + ".second.txt", second_argument);
                    fail_index += 2;
                }
                else {
                    write_file(file_name + ".txt", first_argument);
                    fail_index += 1;
                }
            }
        }

        // Now let's print in the format:
        //  - name, up to 32 characters
        //  - throughput in GB/s with up to 3 significant digits, 10 characters
        //  - call latency in ns with up to 1 significant digit, 10 characters
        //  - number of failed tests, 10 characters
        //  - first example of a failed test, up to 20 characters
        char const *format;
        if (is_binary) { format = "- %-32s %15.4f GB/s %15.1f ns %10zu errors in %10zu iterations %-20s %-20s\n"; }
        else { format = "- %-32s %15.4f GB/s %15.1f ns %10zu errors in %10zu iterations %-20s\n"; }

        std::printf(format, name.c_str(), results.bytes_passed / results.seconds / 1.e9,
                    results.seconds * 1e9 / results.iterations, failed_count, results.iterations,
                    failed_strings.size() ? failed_strings[0].c_str() : "",
                    failed_strings.size() >= 2 && is_binary ? failed_strings[1].c_str() : "");
    }
};

using tracked_unary_functions_t = std::vector<tracked_function_gt<unary_function_t>>;
using tracked_binary_functions_t = std::vector<tracked_function_gt<binary_function_t>>;

/**
 *  @brief  Stops compilers from optimizing out the expression.
 *          Shamelessly stolen from Google Benchmark.
 */
template <typename argument_type>
inline void do_not_optimize(argument_type &&value) {
#if defined(_MSC_VER) // MSVC
    using plain_type = typename std::remove_reference<argument_type>::type;
    // Use the `volatile` keyword and a memory barrier to prevent optimization
    volatile plain_type *p = &value;
    _ReadWriteBarrier();
#else // Other compilers (GCC, Clang, etc.)
    asm volatile("" : "+r,m"(value) : : "memory");
#endif
}

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

template <typename is_separator_callback_type>
inline std::vector<std::string_view> tokenize(std::string_view str, is_separator_callback_type &&is_separator) {
    std::vector<std::string_view> words;
    std::size_t start = 0;
    for (std::size_t end = 0; end <= str.length(); ++end) {
        if (end == str.length() || is_separator(str[end])) {
            if (start < end) words.push_back({&str[start], end - start});
            start = end + 1;
        }
    }
    return words;
}

/**
 *  @brief  Splits a string into words, using newlines, tabs, and whitespaces as delimiters.
 */
inline std::vector<std::string_view> tokenize(std::string_view str) {
    return tokenize(str, [](char c) { return std::isspace(c); });
}

template <typename result_string_type = std::string_view, typename from_string_type = result_string_type,
          typename comparator_type = std::equal_to<std::size_t>>
inline std::vector<result_string_type> filter_by_length(std::vector<from_string_type> tokens, std::size_t n,
                                                        comparator_type &&comparator = {}) {
    std::vector<result_string_type> result;
    for (auto const &str : tokens)
        if (comparator(str.length(), n)) result.push_back({str.data(), str.length()});
    return result;
}

inline static std::size_t seconds_per_benchmark = 5;

struct dataset_t {
    std::string text;
    std::vector<std::string_view> tokens;
    std::vector<std::string_view> lines;
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
    data.lines = tokenize(data.text, [](char c) { return c == '\n'; });
    data.lines.resize(bit_floor(data.lines.size()));

#if !SZ_DEBUG // Shuffle only in release mode
    auto &generator = global_random_generator();
    std::shuffle(data.tokens.begin(), data.tokens.end(), generator);
    std::shuffle(data.lines.begin(), data.lines.end(), generator);
#endif

    // Report some basic stats about the dataset
    double mean_token_bytes = 0, mean_line_bytes = 0;
    for (auto const &str : data.tokens) mean_token_bytes += str.size();
    for (auto const &str : data.lines) mean_line_bytes += str.size();
    mean_token_bytes /= data.tokens.size();
    mean_line_bytes /= data.lines.size();

    std::setlocale(LC_NUMERIC, "");
    std::printf(                                     //
        "Parsed the dataset with:\n"                 //
        "- %zu words of mean length ~ %.2f bytes\n"  //
        "- %zu lines of mean length ~ %.2f bytes\n", //
        data.tokens.size(), mean_token_bytes, data.lines.size(), mean_line_bytes);

    return data;
}

/**
 *  @brief  Loads a dataset, depending on the passed CLI arguments.
 */
inline dataset_t prepare_benchmark_environment(int argc, char const *argv[]) {
    if (argc < 2 || argc > 3)
        throw std::runtime_error("Usage: " + std::string(argv[0]) + " <path> [seconds_per_benchmark]");

    dataset_t data = make_dataset_from_path(argv[1]);

    // If the seconds_per_benchmark argument is provided, update the value in the dataset
    if (argc == 3) {
        seconds_per_benchmark = std::stoi(argv[2]);
        if (seconds_per_benchmark == 0)
            throw std::invalid_argument("The number of seconds per task must be greater than 0.");
    }

    return data;
}

inline sz_string_view_t to_c(std::string_view str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(std::string const &str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz::string_view str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz::string const &str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz_string_view_t str) noexcept { return str; }

/**
 *  @brief  Loop over all elements in a dataset in somewhat random order, benchmarking the function cost.
 *  @param  strings Strings to loop over. Length must be a power of two.
 *  @param  function Function to be applied to each `sz_string_view_t`. Must return the number of bytes processed.
 *  @return Number of seconds per iteration.
 */
template <typename strings_type, typename function_type>
benchmark_result_t bench_on_tokens(strings_type &&strings, function_type &&function) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    benchmark_result_t result;
    std::size_t lookup_mask = bit_floor(strings.size()) - 1;

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            result.bytes_passed += function(strings[(result.iterations + 0) & lookup_mask]) +
                                   function(strings[(result.iterations + 1) & lookup_mask]) +
                                   function(strings[(result.iterations + 2) & lookup_mask]) +
                                   function(strings[(result.iterations + 3) & lookup_mask]);
            result.iterations += 4;
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > seconds_per_benchmark) break;
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
template <typename strings_type, typename function_type>
benchmark_result_t bench_on_token_pairs(strings_type &&strings, function_type &&function) {

    namespace stdc = std::chrono;
    using stdcc = stdc::high_resolution_clock;
    stdcc::time_point t1 = stdcc::now();
    benchmark_result_t result;
    std::size_t lookup_mask = bit_floor(strings.size()) - 1;
    std::size_t largest_prime = static_cast<std::size_t>(18446744073709551557ull);

    while (true) {
        // Unroll a few iterations, to avoid some for-loops overhead and minimize impact of time-tracking
        {
            auto second = (result.iterations * largest_prime) & lookup_mask;
            result.bytes_passed += function(strings[(result.iterations + 0) & lookup_mask], strings[second]) +
                                   function(strings[(result.iterations + 1) & lookup_mask], strings[second]) +
                                   function(strings[(result.iterations + 2) & lookup_mask], strings[second]) +
                                   function(strings[(result.iterations + 3) & lookup_mask], strings[second]);
            result.iterations += 4;
        }

        stdcc::time_point t2 = stdcc::now();
        result.seconds = stdc::duration_cast<stdc::nanoseconds>(t2 - t1).count() / 1.e9;
        if (result.seconds > seconds_per_benchmark) break;
    }

    return result;
}

/**
 *  @brief  Evaluation for unary string operations: hashing.
 */
template <typename strings_type, typename functions_type>
void bench_unary_functions(strings_type &&strings, functions_type &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_tokens(strings, [&](auto str) -> std::size_t {
                auto baseline = variants[0].function(str);
                auto result = variant.function(str);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) {
                        variant.failed_strings.push_back({to_c(str).start, to_c(str).length});
                    }
                }
                return to_c(str).length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_tokens(strings, [&](auto str) -> std::size_t {
                do_not_optimize(variant.function(str));
                return to_c(str).length;
            });
        }

        variant.print();
    }
}

/**
 *  @brief  Evaluation for binary string operations: equality, ordering, prefix, suffix, distance.
 */
template <typename strings_type, typename functions_type>
void bench_binary_functions(strings_type &&strings, functions_type &&variants) {

    for (std::size_t variant_idx = 0; variant_idx != variants.size(); ++variant_idx) {
        auto &variant = variants[variant_idx];

        // Tests
        if (variant.function && variant.needs_testing) {
            bench_on_token_pairs(strings, [&](auto str_a, auto str_b) -> std::size_t {
                auto baseline = variants[0].function(str_a, str_b);
                auto result = variant.function(str_a, str_b);
                if (result != baseline) {
                    ++variant.failed_count;
                    if (variant.failed_strings.empty()) {
                        variant.failed_strings.push_back({to_c(str_a).start, to_c(str_a).length});
                        variant.failed_strings.push_back({to_c(str_b).start, to_c(str_b).length});
                    }
                }
                return to_c(str_a).length + to_c(str_b).length;
            });
        }

        // Benchmarks
        if (variant.function) {
            variant.results = bench_on_token_pairs(strings, [&](auto str_a, auto str_b) -> std::size_t {
                do_not_optimize(variant.function(str_a, str_b));
                return to_c(str_a).length + to_c(str_b).length;
            });
        }

        variant.print();
    }
}

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian