/**
 *  @file   bench.hpp
 *  @brief  Helper structures and functions for C++ benchmarks.
 *
 *  The StringZilla benchmarking suite doesn't use any external frameworks like Criterion or Google Benchmark.
 *  There are several reasons for that:
 *
 *  1.  Reduce the number of @b dependencies and the complexity of the build system.
 *
 *  2.  Combine @b "stress-testing" with benchmarks to deduplicate logic.
 *      As we work with often large datasets, with complex preprocessing, and many different backends,
 *      we want to minimize the surface area we debug and maintain, keeping track of string-specific
 *      properties, like:
 *
 *      -   Is the string start aligned in memory?
 *      -   Does it take more than one cache line? Is it's length a multiple of the SIMD vector size?
 *      -   Is the string cached in the L1 or L2 cache? Can the dataset fit in L3?
 *
 *      As part of that stress-testing, on failure, those properties will be persisted in a file on disk.
 *
 *  3.  Use cheaper profiling methods like @b CPU-counter instructions, as opposed to wall-clock time.
 *      Assuming we can clearly isolate single-threaded workloads and are more interested in the number
 *      of retired instructions, CPU counters can be more accurate and less noisy.
 *
 *  4.  Integrate with Linux @b `perf` and other tools for more detailed analysis.
 *      We can isolate the relevant pieces of code, excluding the preprocessing costs from the actual workload.
 *
 *  5.  Visualize the results differently, with a compact output for both generic workloads and special cases.
 */
#pragma once
#include <algorithm>
#include <chrono>     // `std::chrono::high_resolution_clock`
#include <clocale>    // `std::setlocale`
#include <cstring>    // `std::memcpy`
#include <exception>  // `std::invalid_argument`
#include <functional> // `std::equal_to`
#include <limits>     // `std::numeric_limits`
#include <random>     // `std::random_device`, `std::mt19937`
#include <string>     // `std::hash`
#include <vector>     // `std::vector`
#include <regex>      // `std::regex`, `std::regex_search`
#include <thread>     // `std::this_thread::sleep_for`

#include <string_view> // Requires C++17

#include <stringzilla/stringzilla.h>
#include <stringzilla/stringzilla.hpp>

#include "test.hpp" // `read_file`

namespace sz = ashvardanian::stringzilla;
namespace stdc = std::chrono;

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using accurate_clock_t = stdc::high_resolution_clock;

template <std::size_t multiple>
std::size_t round_up_to_multiple(std::size_t n) {
    return n == 0 ? multiple : ((n + multiple - 1) / multiple) * multiple;
}

struct call_result_t {
    /** @brief Number of input bytes processed. */
    std::size_t bytes_passed = 0;
    /** @brief Some value used to compare execution result between the baseline and accelerated backend. */
    std::size_t check_value = 0;
    /** @brief For some operations with non-linear complexity, the throughput should be measured differently. */
    std::size_t operations = 0;

    call_result_t() = default;
    call_result_t(std::size_t bytes_passed, std::size_t check_value = 0, std::size_t operations = 0)
        : bytes_passed(bytes_passed), check_value(check_value), operations(operations) {}
};

struct callable_no_op_t {
    call_result_t operator()(std::size_t) const { return {}; }
};

using profiled_function_t = std::function<call_result_t(std::size_t)>;

/**
 *  @brief  Cross-platform function to get the number of CPU cycles elapsed @b only on the current core.
 *          Used as a more efficient alternative to `std::chrono::high_resolution_clock`.
 */
inline std::uint64_t cpu_cycle_counter() {
#if defined(__i386__) || defined(__x86_64__)
    // Use x86 inline assembly for `rdtsc` only if actually compiling for x86.
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__) || defined(_SZ_IS_ARM64)
    // On ARM64, read the virtual count register (CNTVCT_EL0) which provides cycle count.
    std::uint64_t cnt;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#else
    return 0;
#endif
}

/** @brief Measures the approximate number of CPU cycles per second. */
inline std::uint64_t cpu_cycles_per_second() {
    std::uint64_t start = cpu_cycle_counter();
    std::this_thread::sleep_for(stdc::seconds(1));
    std::uint64_t end = cpu_cycle_counter();
    return end - start;
}

/** @brief Measures the duration of a single call to the given function. */
template <typename function_type_>
double seconds_per_call(function_type_ &&function) {
    accurate_clock_t::time_point start = accurate_clock_t::now();
    function();
    accurate_clock_t::time_point end = accurate_clock_t::now();
    return stdc::duration_cast<stdc::nanoseconds>(end - start).count() / 1.e9;
}

/**
 *  @brief  Allows time-limited for-loop iteration, similar to Google Benchmark's `for (auto _ : state)`.
 *          Use as `for (auto running_seconds : repeat_up_to(5.0)) { ... }`.
 */
struct repeat_up_to {
    double max_seconds = 0;
    double passed_seconds = 0;

    struct end_sentinel {};
    class iterator {
        accurate_clock_t::time_point start_time_;
        double max_seconds_ = 0;
        double &passed_seconds_;

      public:
        inline iterator(double max_seconds, double &passed_seconds)
            : start_time_(accurate_clock_t::now()), max_seconds_(max_seconds), passed_seconds_(passed_seconds) {}
        inline bool operator!=(end_sentinel) const {
            accurate_clock_t::time_point current_time = accurate_clock_t::now();
            passed_seconds_ = stdc::duration_cast<stdc::nanoseconds>(current_time - start_time_).count() / 1.e9;
            return passed_seconds_ < max_seconds_;
        }
        inline double operator*() const { return passed_seconds_; }
        constexpr void operator++() {} // No-op
    };

    inline repeat_up_to(double max_seconds) : max_seconds(max_seconds) {}
    inline iterator begin() { return {max_seconds, passed_seconds}; }
    inline end_sentinel end() const noexcept { return {}; }
    inline double seconds() const noexcept { return passed_seconds; }
};

/**
 *  @brief  Stops compilers from optimizing out the expression.
 *          Shamelessly stolen from Google Benchmark's @b `DoNotOptimize`.
 */
template <typename argument_type>
static void do_not_optimize(argument_type &&value) noexcept {

#if defined(_MSC_VER) // MSVC
    using plain_type = typename std::remove_reference<argument_type>::type;
    // Use the `volatile` keyword and a memory barrier to prevent optimization
    volatile plain_type *p = &value;
    _ReadWriteBarrier();
#else // Other compilers (GCC, Clang, etc.)
    __asm__ __volatile__("" : "+g"(value) : : "memory");
#endif
}

/**
 *  @brief Rounds the number @b down to the preceding power of two.
 *  @see Equivalent to `std::bit_floor`: https://en.cppreference.com/w/cpp/numeric/bit_floor
 */
inline std::size_t bit_floor(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_siginificant_bit_position = 0;
    while (n > 1) n >>= 1, most_siginificant_bit_position++;
    return static_cast<std::size_t>(1) << most_siginificant_bit_position;
}

/**
 *  @brief Tokenizes a string with the given separator predicate.
 *  @see For faster ways to tokenize a string with STL: https://ashvardanian.com/posts/splitting-strings-cpp/
 */
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

/** @brief Splits a string into words, using newlines, tabs, and whitespaces as delimiters using @b `std::isspace`. */
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

/**
 *  @brief  Environment for the benchmarking scripts pulled from the CLI arguments.
 *
 *  The original CLI arguments include the @p path to the dataset file and the number of @p seconds per benchmark,
 *  the Regex @p filter to select only the backends that match the given pattern, as well as the @p tokenization
 *  mode to convert the loaded textual @p dataset to a @p tokens array.
 *
 *  In the RELEASE mode, the tokens will be shuffled to avoid any bias in the benchmarking process.
 *  The @p seed is used to guarantee reproducibility of the results between different runs.
 */
struct environment_t {
    enum tokenization_t : unsigned char {
        file_k = 255,
        lines_k = 254,
        words_k = 253,
    };

    /** @brief Absolute path of the textual input file on disk. */
    std::string path;
    /** @brief Stress-testing results directory. */
    std::string stress_dir;

    /** @brief Tokenization mode to convert the @p dataset to @p tokens. */
    tokenization_t tokenization = tokenization_t::words_k;
    /** @brief Regular expression to filter the backends. */
    std::string filter;

    /** @brief Whether to stress-test the backends. */
    bool stress = true;
    /** @brief Upper time bound on a duration of the stress-test for a single callable. */
    std::size_t stress_seconds = SZ_DEBUG ? 1 : 10;
    /** @brief Upper time bound on a duration of a single callable. */
    std::size_t benchmark_seconds = SZ_DEBUG ? 1 : 10;
    /** @brief Seed for the random number generator. */
    std::size_t seed = 0;
    /** @brief Upper bound on the number of stress test failures on a callable. */
    std::size_t stress_limit = 1;

    /** @brief Textual content of the dataset file, fully loaded into memory. */
    std::string dataset;
    /** @brief Array of tokens extracted from the @p dataset. */
    std::vector<std::string_view> tokens;

    bool allow(std::string const &benchmark_name) const {
        return filter.empty() || std::regex_search(benchmark_name, std::regex(filter));
    }
};

/**
 *  @brief  Prepares the environment for benchmarking based on environment variables and default settings.
 *          It's expected that different workloads may use different default datasets and tokenization modes,
 *          but time limits and seeds are usually consistent across all benchmarks.
 *
 *  @param[in] argc Number of command-line string arguments. Not used in reality.
 *  @param[in] argv Array of command-line string arguments. Not used in reality.
 *
 *  @param[in] default_dataset Path to the default dataset file, if the @b `STRINGWARS_DATASET` is not set.
 *  @param[in] default_tokens Tokenization mode, if the @b `STRINGWARS_TOKENS` is not set.
 *  @param[in] default_duration Time limit per benchmark, if the @b `STRINGWARS_DURATION` is not set.
 *
 *  @param[in] default_stress Whether to stress-test the backends, if the @b `STRINGWARS_STRESS` is not set.
 *  @param[in] default_stress_dir Directory for stress-testing logs, if the @b `STRINGWARS_STRESS_DIR` is not set.
 *  @param[in] default_stress_limit Max number of failures to tolerate, if the @b `STRINGWARS_STRESS_LIMIT` is not set.
 *  @param[in] default_stress_duration Time limit per stress-test, if the @b `STRINGWARS_STRESS_DURATION` is not set.
 *
 *  @param[in] default_filter Regular expression to filter the backends, if the @b `STRINGWARS_FILTER` is not set.
 *  @param[in] default_seed Seed for reproducibility, if the @b `STRINGWARS_SEED` is not set.
 */
inline environment_t build_environment(                                        //
    int argc, char const *argv[],                                              //< Ignored
    std::string default_dataset, environment_t::tokenization_t default_tokens, //< Mandatory
    std::size_t default_duration = SZ_DEBUG ? 1 : 10,                          //< Optional
    bool default_stress = true,                                                //
    std::string default_stress_dir = ".tmp",                                   //
    std::size_t default_stress_limit = 1,                                      //
    std::size_t default_stress_duration = SZ_DEBUG ? 1 : 10,                   //
    std::string default_filter = "",                                           //
    std::size_t default_seed = 0                                               //
    ) noexcept(false) {

    sz_unused(argc && argv); // Unused in this context
    environment_t env;

    // Use `STRINGWARS_DATASET` if set, otherwise `default_dataset`
    if (char const *env_var = std::getenv("STRINGWARS_DATASET")) { env.path = env_var; }
    else { env.path = default_dataset; }

    // Use `STRINGWARS_FILTER` if set, otherwise `default_filter`
    if (char const *env_var = std::getenv("STRINGWARS_FILTER")) { env.filter = env_var; }
    else { env.filter = default_filter; }

    // Use `STRINGWARS_DURATION` if set, otherwise `default_duration`
    if (char const *env_var = std::getenv("STRINGWARS_DURATION")) {
        env.benchmark_seconds = std::stoul(env_var);
        if (env.benchmark_seconds == 0) throw std::invalid_argument("The time limit must be greater than 0.");
    }
    else { env.benchmark_seconds = default_duration; }

    // Use `STRINGWARS_SEED` if set, otherwise `default_seed`
    if (char const *env_var = std::getenv("STRINGWARS_SEED")) {
        env.seed = std::stoul(env_var);
        if (env.seed == 0) throw std::invalid_argument("The seed must be a positive integer.");
    }
    else { env.seed = default_seed; }

    // Use `STRINGWARS_TOKENS` if set, otherwise `default_tokens`
    if (char const *env_var = std::getenv("STRINGWARS_TOKENS")) {
        std::string token_arg(env_var);
        if (token_arg == "file") { env.tokenization = environment_t::file_k; }
        else if (token_arg == "lines") { env.tokenization = environment_t::lines_k; }
        else if (token_arg == "words") { env.tokenization = environment_t::words_k; }
        else {
            // If it's not one of the known strings, assume it's an unsigned integer (for N-grams).
            env.tokenization = static_cast<environment_t::tokenization_t>(std::stoul(token_arg));
            if (env.tokenization == 0)
                throw std::invalid_argument(
                    "The tokenization mode must be 'file', 'line', 'word', or a positive integer.");
        }
    }
    else { env.tokenization = default_tokens; }

    // Extract the stress-testing settings
    if (char const *env_var = std::getenv("STRINGWARS_STRESS")) {
        bool is_zero = std::strcmp(env_var, "0") != 0 || std::strcmp(env_var, "false") != 0;
        bool is_one = std::strcmp(env_var, "1") != 0 || std::strcmp(env_var, "true") != 0;
        env.stress = is_one;
        if (!is_zero && !is_one) throw std::invalid_argument("The stress-testing flag must be '0' or '1'.");
    }
    else { env.stress = default_stress; }
    if (char const *env_var = std::getenv("STRINGWARS_STRESS_DURATION")) {
        env.stress_seconds = std::stoul(env_var);
        if (env.stress_seconds == 0)
            throw std::invalid_argument("The stress-testing time limit must be greater than 0.");
    }
    else { env.stress_seconds = default_stress_duration; }
    if (char const *env_var = std::getenv("STRINGWARS_STRESS_DIR")) { env.stress_dir = env_var; }
    else { env.stress_dir = default_stress_dir; }
    if (char const *env_var = std::getenv("STRINGWARS_STRESS_LIMIT")) {
        env.stress_limit = std::stoul(env_var);
        if (env.stress_limit == 0) throw std::invalid_argument("The stress-testing limit must be greater than 0.");
    }
    else { env.stress_limit = default_stress_limit; }

    env.dataset = read_file(env.path);
    env.dataset.resize(bit_floor(env.dataset.size())); // Shrink to the nearest power of two

    // Tokenize the dataset according to the tokenization mode
    if (env.tokenization == environment_t::file_k) { env.tokens.push_back(env.dataset); }
    else if (env.tokenization == environment_t::lines_k) {
        env.tokens = tokenize(env.dataset, [](char c) { return c == '\n'; });
    }
    else if (env.tokenization == environment_t::words_k) { env.tokens = tokenize(env.dataset); }
    else {
        std::size_t n = static_cast<std::size_t>(env.tokenization);
        env.tokens = filter_by_length(tokenize(env.dataset), n, std::equal_to<std::size_t>());
    }
    env.tokens.resize(bit_floor(env.tokens.size())); // Shrink to the nearest power of two

    // In "RELEASE" mode, shuffle tokens to avoid bias.
    char const *seed_message = " (not used in DEBUG mode)";
#if !defined(SZ_DEBUG)
    std::mt19937 generator(static_cast<unsigned int>(env.seed));
    std::shuffle(env.tokens.begin(), env.tokens.end(), generator);
    seed_message = "";
#endif

    auto const mean_token_length =
        std::accumulate(env.tokens.begin(), env.tokens.end(), 0,
                        [](std::size_t sum, std::string_view token) { return sum + token.size(); }) *
        1.0 / env.tokens.size();

    // Group integer decimal separators by 3
    // https://www.ibm.com/docs/en/i/7.4?topic=categories-lc-numeric-category
    std::setlocale(LC_NUMERIC, "en_US.UTF-8");
    std::printf("Environment built with the following settings:\n");
    std::printf(" - Dataset path: %s\n", env.path.c_str());
    std::printf(" - Time limit: %zu seconds per benchmark (%zu per stress-test)\n", env.benchmark_seconds,
                env.stress_seconds);
    if (!env.filter.empty()) std::printf(" - Algorithm filter: %s\n", env.filter.c_str());
    std::printf(" - Tokenization mode: ");
    switch (env.tokenization) {
    case environment_t::file_k: std::printf("file\n"); break;
    case environment_t::lines_k: std::printf("line\n"); break;
    case environment_t::words_k: std::printf("word\n"); break;
    default: std::printf("%zu-grams\n", static_cast<std::size_t>(env.tokenization)); break;
    }
    std::printf(" - Seed: %zu%s\n", env.seed, seed_message);
    std::printf(" - Loaded dataset size: %zu bytes\n", env.dataset.size());
    std::printf(" - Number of tokens: %zu\n", env.tokens.size());
    std::printf(" - Mean token length: %.2f bytes\n", mean_token_length);

    return env;
}

/**
 *  @brief  Uses C-style file IO to save information about the most recent stress test failure.
 *          Files can be found in: "$STRINGWARS_STRESS_DIR/failed_$time_$name.txt".
 */
inline void log_stress_failure(environment_t const &env, std::string const &name, std::size_t input_index,
                               std::size_t expected_check_value, std::size_t actual_check_value) noexcept(false) {

    std::string file_name = "failed_" + name + "_" + std::to_string(input_index) + ".txt";
    std::string file_path = env.stress_dir + "/" + file_name;
    std::FILE *file = std::fopen(file_path.c_str(), "w");
    if (!file) throw std::runtime_error("Failed to open file for writing: " + file_name);

    std::fprintf(file, "Expected: %zu\n", expected_check_value);
    std::fprintf(file, "Actual: %zu\n", actual_check_value);
    std::fclose(file);
}

struct benchmark_result_t {
    std::string name;
    bool skipped = false;

    std::size_t stress_calls = 0;
    std::size_t profiled_calls = 0;
    double profiled_seconds = 0;
    std::uint64_t profiled_cpu_cycles = 0;

    std::size_t bytes_passed = 0; //< Pulled from the `call_result_t`
    std::size_t operations = 0;   //< Pulled from the `call_result_t`
    std::size_t errors = 0;       //< Pulled from the `call_result_t`

    inline benchmark_result_t &operator+=(call_result_t const &run) noexcept {
        bytes_passed += run.bytes_passed;
        operations += run.operations;
        return *this;
    }

    /**
     *  @brief  Logs the benchmark results to the console, including the throughput and latency.
     *
     *  Example output:
     *
     *  @code{.unparsed}
     *  Benchmarking sz_find_serial:
     *  - Performance: 0.00 TOps/s @ 0.00 ns/call
     *  - Errors: 1 in 1 calls
     *  @endcode
     */
    benchmark_result_t const &log() const {
        benchmark_result_t const &result = *this;
        if (result.skipped) return result;
        std::printf("Benchmarking %s:\n", result.name.c_str());

        // Infer the latency from the number of calls and the total time
        auto duration = result.profiled_seconds * 1e9 / result.profiled_calls;
        auto duration_unit = "ns";
        if (duration > 1e3) duration /= 1e3, duration_unit = "us";
        if (duration > 1e3) duration /= 1e3, duration_unit = "ms";
        if (duration > 1e3) duration /= 1e3, duration_unit = "s";

        // We may want to analyze the call latency distribution:
        // auto cpu_frequency = result.profiled_cpu_cycles / result.profiled_seconds;
        // auto cpu_frequency_unit = "Hz";
        // if (cpu_frequency > 1e3) cpu_frequency /= 1e3, cpu_frequency_unit = "KHz";
        // if (cpu_frequency > 1e3) cpu_frequency /= 1e3, cpu_frequency_unit = "MHz";
        // if (cpu_frequency > 1e3) cpu_frequency /= 1e3, cpu_frequency_unit = "GHz";

        // Infer the throughput from the number of operations and the total time
        auto throughput = (result.operations ? result.operations : result.bytes_passed) / result.profiled_seconds;
        auto throughput_unit = result.operations ? "Ops/s" : "B/s";
        if (throughput > 1e3) throughput /= 1e3, throughput_unit = result.operations ? "KOps/s" : "KB/s";
        if (throughput > 1e3) throughput /= 1e3, throughput_unit = result.operations ? "MOps/s" : "MB/s";
        if (throughput > 1e3) throughput /= 1e3, throughput_unit = result.operations ? "GOps/s" : "GB/s";

        // Print to console
        std::printf(" - Performance: %.2f %s @ %.2f %s/call\n", throughput, throughput_unit, duration, duration_unit);
        if (result.errors) std::printf(" - Errors: %zu in %zu calls\n", result.errors, result.stress_calls);

        return result;
    }

    /**
     *  @brief  Logs @b relative results to the console, comparing @p this to a @p base result.
     *
     *  Example output:
     *
     *  @code{.unparsed}
     *  Benchmarking sz_find_skylake:
     *  - Performance: 0.00 TOps/s @ 0.00 ns/call
     *  - Errors: 1 in 1 calls
     *  - Relative performance: +25% vs sz_find_serial
     *  @endcode
     */
    benchmark_result_t const &log(benchmark_result_t const &base) const {
        benchmark_result_t const &new_ = *this;
        new_.log();

        if (new_.skipped || base.skipped) return new_; //? Nothing to compare to
        auto base_throughput = (base.operations ? base.operations : base.bytes_passed) / base.profiled_seconds;
        auto new_throughput = (new_.operations ? new_.operations : new_.bytes_passed) / new_.profiled_seconds;
        auto relative_throughput = new_throughput / base_throughput;

        // Now format the relative improvement as a percentage for small changes and as a multiplier for large ones,
        // formatting it with a plus and a green color for improvements and a minus and a red color for regressions.
        auto relative_color = relative_throughput > 1 ? "\033[32m" : "\033[31m";
        auto relative_sign = relative_throughput > 1 ? "+" : "-";
        auto relative_unit = relative_throughput > 2 ? "x" : "%";
        if (relative_throughput < 0.5) relative_throughput = 1 / relative_throughput, relative_unit = "x";
        if (std::strcmp(relative_unit, "%") == 0) relative_throughput *= 100;
        std::printf(" - Relative performance: %s%s%.0f %s\033[0m vs. %s\n", relative_color, relative_sign,
                    relative_throughput, relative_unit, base.name.c_str());
        return new_;
    }
};

/**
 *  @brief Loops over all tokens (in loop-unrolled batches) in environment and applies the given unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] baseline Optional serial analog, against which the accelerated function will be stress-tested.
 *  @param[in] callable Unary function taking a @b `std::size_t` token index and returning a @b `call_result_t`.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_, typename baseline_type_ = callable_no_op_t>
benchmark_result_t benchmark(environment_t const &env, std::string const &name, baseline_type_ &&baseline,
                             callable_type_ &&callable) {

    benchmark_result_t result;
    result.name = name;
    if (!env.allow(name)) {
        result.skipped = true;
        return result;
    }

    std::size_t const lookup_mask = bit_floor(env.tokens.size()) - 1;
    if constexpr (!std::is_same<baseline_type_, callable_no_op_t>())
        for (auto running_seconds : repeat_up_to(env.stress_seconds)) {
            std::size_t const input_index = (result.stress_calls++) & lookup_mask;
            call_result_t const accelerated_result = callable(input_index);
            call_result_t const baseline_result = baseline(input_index);
            if (accelerated_result.check_value == baseline_result.check_value) continue; // No failures

            // If we got here, the error needs to be reported and investigated.
            ++result.errors;
            if (result.errors > env.stress_limit) {
                std::printf("Too many errors in %s after %.3f seconds. Stopping the test.\n", name.c_str(),
                            running_seconds);
                std::terminate();
            }
            log_stress_failure(env, name, input_index, baseline_result.check_value, accelerated_result.check_value);
        }

    // For profiling, we will first run the benchmark just once to get a rough estimate of the time.
    // But then we will repeat it in an unrolled fashion for a more accurate measurement.
    result.profiled_seconds += seconds_per_call([&] {
        std::uint64_t start_cycle = cpu_cycle_counter();
        result += callable(0); // First input for debugging
        std::uint64_t end_cycle = cpu_cycle_counter();
        result.profiled_calls += 1;
        result.profiled_cpu_cycles += end_cycle - start_cycle;
    });
    if (result.profiled_seconds >= env.benchmark_seconds) return result;

    // Repeat the benchmarks in unrolled batches until the time limit is reached.
    for (auto running_seconds : repeat_up_to(env.benchmark_seconds - result.profiled_seconds)) {
        std::uint64_t start_cycle = cpu_cycle_counter();
        call_result_t r0 = callable((result.profiled_calls + 0) & lookup_mask);
        call_result_t r1 = callable((result.profiled_calls + 1) & lookup_mask);
        call_result_t r2 = callable((result.profiled_calls + 2) & lookup_mask);
        call_result_t r3 = callable((result.profiled_calls + 3) & lookup_mask);
        std::uint64_t end_cycle = cpu_cycle_counter();

        // Aggregate all of them:
        result += r0;
        result += r1;
        result += r2;
        result += r3;
        result.profiled_calls += 4;
        result.profiled_cpu_cycles += end_cycle - start_cycle;
        result.profiled_seconds = running_seconds;
    }

    return result;
}

/**
 *  @brief Loops over all tokens (in loop-unrolled batches) in environment and applies the given unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] callable Unary function taking a @b `std::size_t` token index and returning a @b `call_result_t`.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_>
benchmark_result_t benchmark(environment_t const &env, std::string const &name, callable_type_ &&callable) {
    return benchmark(env, name, callable_no_op_t {}, callable);
}

inline sz_string_view_t to_c(std::string_view str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(std::string const &str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz::string_view str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz::string const &str) noexcept { return {str.data(), str.size()}; }
inline sz_string_view_t to_c(sz_string_view_t str) noexcept { return str; }

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian