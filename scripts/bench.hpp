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
 *      We can also track individual hardware counters, including platform-specific `PERF_TYPE_RAW` ones,
 *      that are not handled by most tools.
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
#include <numeric>    // `std::accumulate`
#include <random>     // `std::random_device`, `std::mt19937`
#include <string>     // `std::hash`
#include <vector>     // `std::vector`
#include <regex>      // `std::regex`, `std::regex_search`
#include <thread>     // `std::this_thread::sleep_for`
#include <optional>   // `std::optional`

#include <string_view> // Requires C++17
#include <span>        // Requires C++20, used to pass info to batch-capable parallel backends

#if defined(_MSC_VER)
#include <intrin.h> // `__rdtsc`, `_ReadStatusReg`
#if defined(_M_ARM64)
#include <arm64intr.h> // `ARM64_CNTVCT`
#endif
#endif

#if defined(__linux__)
#include <unistd.h>      // syscall, close
#include <sys/syscall.h> // __NR_perf_event_open
#include <sys/ioctl.h>   // ioctl
#include <linux/perf_event.h>
#endif

#include "stringzilla/stringzilla.h"
#include "stringzilla/stringzilla.hpp"

#if SZ_USE_CUDA
#include "stringzillas/types.cuh" // `unified_alloc`
#endif

#include "test_stringzilla.hpp" // `read_file`

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

using check_value_t = std::uint64_t;

struct call_result_t {
    /** @brief Number of input bytes processed. */
    std::size_t bytes_passed = 0;
    /** @brief Some value used to compare execution result between the baseline and accelerated backend. */
    check_value_t check_value = 0;
    /** @brief For some operations with non-linear complexity, the throughput should be measured differently. */
    std::size_t operations = 0;
    /** @brief Equal to 1 for most inputs, but can be larger for batch-capable functions. */
    std::size_t inputs_processed = 1;

    call_result_t() = default;
    call_result_t(std::size_t bytes_passed, std::size_t check_value = 0, std::size_t operations = 0)
        : bytes_passed(bytes_passed), check_value(check_value), operations(operations), inputs_processed(1) {}
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
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    // Use MSVC intrinsics for `rdtsc`
    return __rdtsc();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    // Use MSVC ARM64 intrinsics to read the virtual count register - CNTVCT_EL0.
    // This register is more reliably accessible in user mode than PMCCNTR_EL0.
#if !defined(ARM64_CNTVCT)
#define ARM64_CNTVCT ARM64_SYSREG(3, 3, 14, 0, 1)
#endif
    return _ReadStatusReg(ARM64_CNTVCT);
#elif defined(__i386__) || defined(__x86_64__)
    // Use x86 inline assembly for `rdtsc` only if actually compiling for x86.
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__) || defined(SZ_IS_64BIT_ARM_)
    // On ARM64, read the virtual count register `CNTVCT_EL0` which provides cycle count.
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

#if defined(__linux__)

/**
 *  @brief Wrapper for Linux `perf_event_open` system call.
 *         Provides precise hardware counter measurements without overhead from preprocessing.
 */
class perf_event_t {
    int fd_ = -1;

  public:
    perf_event_t() = default;

    /**
     *  @brief Initialize perf event for CPU cycles counting.
     *  @return true if successfully initialized, false otherwise.
     */
    bool init_cycles() {
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(pe));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.config = PERF_COUNT_HW_CPU_CYCLES;
        pe.disabled = 1;       // Start disabled
        pe.exclude_kernel = 1; // Don't count kernel cycles
        pe.exclude_hv = 1;     // Don't count hypervisor cycles

        fd_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));
        return fd_ >= 0;
    }

    /**
     *  @brief Initialize perf event for instruction counting.
     *  @return true if successfully initialized, false otherwise.
     */
    bool init_instructions() {
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(pe));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.config = PERF_COUNT_HW_INSTRUCTIONS;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        fd_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));
        return fd_ >= 0;
    }

    /**
     *  @brief Initialize perf event with custom configuration.
     *  @param type Event type (PERF_TYPE_HARDWARE, PERF_TYPE_RAW, etc.)
     *  @param config Event configuration
     *  @return true if successfully initialized, false otherwise.
     */
    bool init_custom(std::uint32_t type, std::uint64_t config) {
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(pe));
        pe.type = type;
        pe.size = sizeof(pe);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        fd_ = static_cast<int>(syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0));
        return fd_ >= 0;
    }

    /**
     *  @brief Reset the counter to zero.
     */
    inline void reset() {
        if (fd_ >= 0) { ioctl(fd_, PERF_EVENT_IOC_RESET, 0); }
    }

    /**
     *  @brief Enable the counter (start counting).
     */
    inline void enable() {
        if (fd_ >= 0) { ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0); }
    }

    /**
     *  @brief Disable the counter (stop counting).
     */
    inline void disable() {
        if (fd_ >= 0) { ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0); }
    }

    /**
     *  @brief Read the current counter value.
     *  @return Counter value, or 0 if reading failed.
     */
    inline std::uint64_t read() const {
        std::uint64_t count = 0;
        if (fd_ >= 0) {
            ssize_t bytes = ::read(fd_, &count, sizeof(count));
            if (bytes != sizeof(count)) return 0;
        }
        return count;
    }

    /**
     *  @brief Check if the perf event is valid.
     */
    inline bool is_valid() const { return fd_ >= 0; }

    /**
     *  @brief Close the perf event file descriptor.
     */
    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ~perf_event_t() { close(); }

    // Disable copy
    perf_event_t(perf_event_t const &) = delete;
    perf_event_t &operator=(perf_event_t const &) = delete;

    // Enable move
    perf_event_t(perf_event_t &&other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    perf_event_t &operator=(perf_event_t &&other) noexcept {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }
};

/**
 *  @brief RAII wrapper to automatically enable/disable a perf event around a code section.
 */
class scoped_perf_measurement_t {
    perf_event_t &event_;

  public:
    explicit scoped_perf_measurement_t(perf_event_t &event) : event_(event) {
        event_.reset();
        event_.enable();
    }

    ~scoped_perf_measurement_t() { event_.disable(); }

    scoped_perf_measurement_t(scoped_perf_measurement_t const &) = delete;
    scoped_perf_measurement_t &operator=(scoped_perf_measurement_t const &) = delete;
};

#endif // __linux__

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
            return max_seconds_ != 0 && passed_seconds_ < max_seconds_;
        }
        inline double operator*() const { return passed_seconds_; }
        constexpr void operator++() {} // No-op
    };

    inline repeat_up_to(double max_seconds) : max_seconds(max_seconds) {}
    inline repeat_up_to(std::size_t max_seconds) : max_seconds(static_cast<double>(max_seconds)) {}
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
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else // GCC
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/**
 *  @brief Rounds the number @b down to the preceding power of two.
 *  @see Equivalent to `std::bit_floor`: https://en.cppreference.com/w/cpp/numeric/bit_floor
 */
inline std::size_t bit_floor(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_significant_bit_position = 0;
    while (n > 1) n >>= 1, most_significant_bit_position++;
    return static_cast<std::size_t>(1) << most_significant_bit_position;
}

#if !SZ_USE_CUDA
using dataset_t = std::string;
using token_view_t = std::string_view;
using tokens_t = std::vector<token_view_t>;
#else
using dataset_t = std::basic_string<char, std::char_traits<char>, stringzillas::unified_alloc<char>>;
using token_view_t = stringzilla::span<char const>;
using tokens_t = std::vector<token_view_t, stringzillas::unified_alloc<token_view_t>>;
#endif

/**
 *  @brief Tokenizes a string with the given separator predicate.
 *  @see For faster ways to tokenize a string with STL: https://ashvardanian.com/posts/splitting-strings-cpp/
 */
template <typename is_separator_callback_type_>
tokens_t tokenize(std::string_view str, is_separator_callback_type_ &&is_separator) {

    // First, let's count the number of separators to minimize the number of allocations.
    std::size_t separator_count = 0;
    for (std::size_t i = 0; i < str.length(); ++i)
        if (is_separator(str[i])) separator_count++;

    // Now, let's allocate the vector with the right size.
    std::size_t const token_upper_bound = separator_count + 1;
    tokens_t tokens(token_upper_bound);

    // Now, let's split the string into non-empty tokens.
    std::size_t tokens_found = 0;
    for (std::size_t start = 0, end = 0; end <= str.length(); ++end)
        if (end == str.length() || is_separator(str[end])) {
            if (start < end) tokens[tokens_found] = {&str[start], end - start}, ++tokens_found;
            start = end + 1;
        }

    // Now, let's resize the vector to the actual number of tokens found.
    tokens.resize(tokens_found);
    return tokens;
}

/** @brief Splits a string into words, using newlines, tabs, and whitespaces as delimiters using @b `std::isspace`. */
inline tokens_t tokenize(std::string_view str) {
    return tokenize(str, [](char c) { return std::isspace(c); });
}

template <typename result_string_type_ = std::string_view, typename from_string_type_ = result_string_type_,
          typename comparator_type_ = std::equal_to<std::size_t>, typename allocator_type_ = std::allocator<char>>
std::vector<result_string_type_, allocator_type_> filter_by_length(
    std::vector<from_string_type_, allocator_type_> const &tokens, //
    std::size_t n, comparator_type_ &&comparator = {}) {

    std::vector<result_string_type_, allocator_type_> result;
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
    std::uint64_t seed = 0;
    /** @brief Upper bound on the number of stress test failures on a callable. */
    std::size_t stress_limit = 1;

    /** @brief Textual content of the dataset file, fully loaded into memory. */
    dataset_t dataset;
    /** @brief Array of tokens extracted from the @p dataset. */
    tokens_t tokens;

    bool allow(std::string const &benchmark_name) const {
        return filter.empty() || std::regex_search(benchmark_name, std::regex(filter));
    }

    std::string_view operator[](std::size_t i) const {
        if (i >= tokens.size()) throw std::out_of_range("Index out of range");
        return {tokens[i].data(), tokens[i].size()};
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

    sz_unused_(argc && argv); // Unused in this context
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
        bool is_zero = std::strcmp(env_var, "0") == 0 || std::strcmp(env_var, "false") == 0;
        bool is_one = std::strcmp(env_var, "1") == 0 || std::strcmp(env_var, "true") == 0;
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
    if (env.tokenization == environment_t::file_k) { env.tokens.push_back({env.dataset.data(), env.dataset.size()}); }
    else if (env.tokenization == environment_t::lines_k) {
        env.tokens = tokenize(env.dataset, [](char c) { return c == '\n'; });
    }
    else if (env.tokenization == environment_t::words_k) { env.tokens = tokenize(env.dataset); }
    else {
        std::size_t n = static_cast<std::size_t>(env.tokenization);
        env.tokens = filter_by_length<token_view_t>(tokenize(env.dataset), n, std::equal_to<std::size_t>());
    }
    env.tokens.resize(bit_floor(env.tokens.size())); // Shrink to the nearest power of two

    // In "RELEASE" mode, shuffle tokens to avoid bias.
    char const *seed_message = " (will avoid shuffling)";
    if (env.seed != 0) {
        std::mt19937_64 generator(static_cast<unsigned long>(env.seed));
        std::shuffle(env.tokens.begin(), env.tokens.end(), generator);
        seed_message = " (will shuffle tokens)";
    }

    auto const mean_token_length =
        std::accumulate(env.tokens.begin(), env.tokens.end(), (std::size_t)0u,
                        [](std::size_t sum, token_view_t token) -> std::size_t { return sum + token.size(); }) *
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
    std::printf(" - Seed: %zu%s\n", static_cast<std::size_t>(env.seed), seed_message);
    std::printf(" - Stress-testing: %s\n", env.stress ? "yes" : "no");
    std::printf(" - Loaded dataset size: %zu bytes\n", env.dataset.size());
    std::printf(" - Number of tokens: %zu\n", env.tokens.size());
    std::printf(" - Mean token length: %.2f bytes\n", mean_token_length);

    std::printf("Compile-time capabilities:\n");
    std::printf("- Uses Westmere: %s \n", SZ_USE_WESTMERE ? "yes" : "no");
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");

    return env;
}

/**
 *  @brief  Uses C-style file IO to save information about the most recent stress test failure.
 *          Files can be found in: "$STRINGWARS_STRESS_DIR/failed_$time_$name.txt".
 */
inline void log_failure(                                              //
    environment_t const &env, std::string const &name,                //
    std::size_t expected_check_value, std::size_t actual_check_value, //
    std::optional<std::size_t> token_index) noexcept(false) {

    std::string timestamp = std::to_string(std::time(nullptr));
    std::string file_name = "failed_" + timestamp + "_" + name + "_" + ".txt";
    std::string file_path = env.stress_dir + "/" + file_name;
    std::FILE *file = std::fopen(file_path.c_str(), "w");
    if (!file) throw std::runtime_error("Failed to open file for writing: " + file_name);

    std::fprintf(file, "Dataset path: %s\n", env.path.c_str());
    std::fprintf(file, "Tokenization mode: %d\n", env.tokenization);
    std::fprintf(file, "Seed: %zu\n", static_cast<std::size_t>(env.seed));
    if (token_index) std::fprintf(file, "Token index: %zu\n", *token_index);
    std::fprintf(file, "Expected: %zu\n", expected_check_value);
    std::fprintf(file, "Actual: %zu\n", actual_check_value);
    std::fclose(file);
}

/**
 *  @brief  Light-weight structure to construct a histogram of function call durations for a very
 *          wide range of floating point values using logarithmic binning. TODO:
 */
template <std::size_t slots_ = 128>
struct duration_histogram {
    using count_t = std::uint32_t;
    std::array<count_t, slots_> bins = {};
    static constexpr double max_seconds = 1000; // Hard to imagine a single call taking more than 15-ish minutes
    static constexpr double min_seconds = 1e-9; // A single nanosecond is just 3-ish CPU cycles on modern hardware

    inline count_t &operator[](double seconds) {
        auto bin_float = std::log(seconds / min_seconds) / std::log(max_seconds / min_seconds) * bins.size();
        std::size_t bin = std::min(bins.size(), static_cast<std::size_t>(bin_float));
        return bins[bin];
    }
};

using duration_histogram_t = duration_histogram<>;

struct bench_result_t {
    std::string name;
    bool skipped = false;

    std::size_t stress_calls = 0;   //< Number of calls to the callable for stress-testing
    std::size_t profiled_calls = 0; //< Number of calls to the callable for profiling/benchmarking

    std::size_t stress_inputs = 0;   //< Can be larger than `stress_calls` for batch-capable functions
    std::size_t profiled_inputs = 0; //< Can be larger than `profiled_calls` for batch-capable functions

    std::size_t profiled_cpu_cycles = 0; //< Number of CPU cycles used in the benchmark by the main thread
    double profiled_seconds = 0;         //< Wall clock duration of the benchmark

    duration_histogram_t cpu_cycles_histogram;

    std::size_t bytes_passed = 0; //< Pulled from the `call_result_t`
    std::size_t operations = 0;   //< Pulled from the `call_result_t`
    std::size_t errors = 0;       //< Pulled from the `call_result_t`

    /**
     *  @brief  Logs the benchmark results to the console, including the throughput and latency,
     *          comparing against one or more baselines.
     *
     *  Example output:
     *
     *  @code{.unparsed}
     *  Benchmarking `sz_find_skylake`:
     *  > Throughput: 0.00 TB/s @ 0.00 ns/call
     *  > Efficiency: 0.00 TOps/s @ 0.00 ops/cycle
     *  > Errors: 0 in 10 calls
     *  > + 3.5 x against `sz_find_serial`
     *  > + 70 % against `memmem`
     *  @endcode
     *
     *  When running on Linux, additional hardware counters can be sampled using `perf`:
     *
     *  @code{.unparsed}
     *  > Instructions retired: ... ~ 3.2 per cycle
     *  > L1 cache misses: ...
     *  > L2 cache misses: ...
     *  > L3 cache misses: ...
     *  > Branch misses: ... ~ 3% of all branches
     *  > Branch instructions: ... ~ 20% of all instructions
     *  > Frontend stall cycles: %
     *  > Backend stall cycles: %
     *  > Port 0 cycles: ... progress bar showing its share of the total
     *  > Port 3 cycles: ... progress bar showing its share of the total
     *  ...
     *  @endcode
     *
     *  After a section of benchmarks is completed, you can use other functionality to visualize the results
     *  in a more structured way, like a table or a graph or a set of progress bars.
     */
    template <typename... baselines_types_>
    bench_result_t const &log(baselines_types_ const &...bases) const {
        if (skipped) return *this;
        std::printf("\n"); // Let's add some spacing between separate benchmarks
        std::printf("Benchmarking \033[1m`%s`\033[0m:\n", name.c_str());

        // Print the number of errors, if any
        if (errors) std::printf("> Errors: %zu in %zu calls\n", errors, stress_calls);

        // Compute average call latency.
        auto seconds_printable = profiled_seconds * 1e9 / profiled_calls;
        char const *seconds_printable_unit = "ns";
        if (seconds_printable > 1e3) seconds_printable /= 1e3, seconds_printable_unit = "us";
        if (seconds_printable > 1e3) seconds_printable /= 1e3, seconds_printable_unit = "ms";
        if (seconds_printable > 1e3) seconds_printable /= 1e3, seconds_printable_unit = "s";

        // Compute throughput based on operations.
        // Assuming we normalize by a power of 2, we use "Ki", "Mi", "Gi" prefixes over "K", "M", "G".
        auto bytes_printable = bytes_passed / profiled_seconds;
        char const *bytes_printable_unit = "B/s";
        if (bytes_printable > 1024) bytes_printable /= 1024, bytes_printable_unit = "KiB/s";
        if (bytes_printable > 1024) bytes_printable /= 1024, bytes_printable_unit = "MiB/s";
        if (bytes_printable > 1024) bytes_printable /= 1024, bytes_printable_unit = "GiB/s";
        std::printf("> Throughput: %.2f %s @ %.2f %s/call\n", //
                    bytes_printable, bytes_printable_unit,    //
                    seconds_printable, seconds_printable_unit);

        // Print the number of operations, if there was a separate tracking mechanism for those.
        if (operations) {
            auto ops_printable = operations * 1.0 / profiled_seconds;
            auto ops_per_cycle = operations * 1.0 / profiled_cpu_cycles;
            char const *ops_printable_unit = (operations ? "Ops/s" : "B/s");
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "KOps/s";
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "MOps/s";
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "GOps/s";
            std::printf("> Efficiency: %.2f %s @ %.2f ops/cycle\n", ops_printable, ops_printable_unit, ops_per_cycle);
        }

        // Define a helper lambda to log relative performance with folding expressions.
        auto log_relative = [this](bench_result_t const &base) {
            if (skipped || base.skipped) return;
            auto relative_throughput = (bytes_passed / profiled_seconds) / (base.bytes_passed / base.profiled_seconds);
            if (operations)
                relative_throughput = (operations / profiled_seconds) / (base.operations / base.profiled_seconds);

            // Format relative improvements: green and a plus for improvements, red and a minus for regressions.
            char const *relative_color = (relative_throughput > 1) ? "\033[32m" : "\033[31m";
            char const *relative_sign = (relative_throughput > 1) ? "+" : "-";
            char const *relative_unit = (relative_throughput > 2) ? "x" : "%";
            if (relative_throughput < 0.5) relative_throughput = 1 / relative_throughput, relative_unit = "x";
            if (std::strcmp(relative_unit, "%") == 0) relative_throughput = (relative_throughput - 1) * 100;
            std::printf("> %s%s %.1f %s\033[0m against `%s`\n",       //
                        relative_color,                               //
                        relative_sign, std::abs(relative_throughput), //
                        relative_unit,                                //
                        base.name.c_str());
        };

        // Expand over all provided baselines.
        (void)std::initializer_list<int> {(log_relative(bases), 0)...};
        sz_unused_(log_relative); // In case no `bases` were provided

        return *this;
    }
};

/**
 *  @brief Repeatedly calls and profiles a given @b nullary function, comparing it against a baseline.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] baseline Optional serial analog, against which the accelerated function will be stress-tested.
 *  @param[in] callable Nullary function taking no arguments and returning a @b `call_result_t`.
 *  @param[in] check_validator Optional function to validate the results of the benchmark.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <                                                        //
    typename callable_type_,                                      //
    typename baseline_type_ = callable_no_op_t,                   //
    typename preprocessing_type_ = callable_no_op_t,              //
    typename check_validator_type_ = std::equal_to<check_value_t> //
    >
bench_result_t bench_nullary(  //
    environment_t const &env,  //
    std::string const &name,   //
    baseline_type_ &&baseline, //
    callable_type_ &&callable, //
    preprocessing_type_ &&preprocessing = preprocessing_type_ {},
    check_validator_type_ &&check_validator = check_validator_type_ {}) {

    bench_result_t result;
    result.name = name;
    if (!env.allow(name)) {
        result.skipped = true;
        return result;
    }

    // Pre-process before testing
    if constexpr (!is_same_type<preprocessing_type_, callable_no_op_t>::value) preprocessing();

    // Perform the testing against the baseline, if provided.
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value)
        for (auto running_seconds : repeat_up_to(env.stress ? env.stress_seconds : 0)) {
            call_result_t const accelerated_result = callable();
            call_result_t const baseline_result = baseline();
            ++result.stress_calls;
            result.stress_inputs += accelerated_result.inputs_processed;
            if (check_validator(accelerated_result.check_value, baseline_result.check_value)) continue; // No failures

            // If we got here, the error needs to be reported and investigated.
            ++result.errors;
            if (result.errors > env.stress_limit) {
                std::printf("Too many errors in %s after %.3f seconds. Stopping the test.\n", name.c_str(),
                            running_seconds);
                std::terminate();
            }
            log_failure(env, name, baseline_result.check_value, accelerated_result.check_value, {});
        }

    // Repeat the benchmark of the unary function. Assume most of them are applied to the entire
    // dataset and take a lot of time, so we don't unroll much, unlike `bench_unary`.
    for (auto running_seconds : repeat_up_to(env.benchmark_seconds)) {
        std::uint64_t cpu_cycles_at_start = cpu_cycle_counter();
        call_result_t call_result = callable();
        std::uint64_t cpu_cycles_at_end = cpu_cycle_counter();

        // Aggregate:
        result.operations += call_result.operations;
        result.bytes_passed += call_result.bytes_passed;
        result.profiled_inputs += call_result.inputs_processed;
        result.profiled_seconds = running_seconds;
        result.profiled_calls += 1;
        result.profiled_cpu_cycles += cpu_cycles_at_end - cpu_cycles_at_start;
        result.cpu_cycles_histogram[static_cast<double>(cpu_cycles_at_end - cpu_cycles_at_start)] += 1;
    }

    return result;
}

/**
 *  @brief Loops over all tokens (in loop-unrolled batches) in environment and applies the given @b unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] baseline Optional serial analog, against which the accelerated function will be stress-tested.
 *  @param[in] callable Unary function taking a @b `std::size_t` token index and returning a @b `call_result_t`.
 *  @param[in] preprocessing Optional function to pre-process the data after the prediction.
 *  @param[in] check_validator Optional function to validate the results of the benchmark.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <                                                        //
    typename callable_type_,                                      //
    typename baseline_type_ = callable_no_op_t,                   //
    typename preprocessing_type_ = callable_no_op_t,              //
    typename check_validator_type_ = std::equal_to<check_value_t> //
    >
bench_result_t bench_unary(    //
    environment_t const &env,  //
    std::string const &name,   //
    baseline_type_ &&baseline, //
    callable_type_ &&callable, //
    preprocessing_type_ &&preprocessing = preprocessing_type_ {},
    check_validator_type_ &&check_validator = check_validator_type_ {}) {

    bench_result_t result;
    result.name = name;
    if (!env.allow(name)) {
        result.skipped = true;
        return result;
    }

    // Pre-process before testing
    if constexpr (!is_same_type<preprocessing_type_, callable_no_op_t>::value) preprocessing();

    std::size_t const lookup_mask = bit_floor(env.tokens.size()) - 1;
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value)
        for (auto running_seconds : repeat_up_to(env.stress ? env.stress_seconds : 0)) {
            std::size_t const token_index = (result.stress_calls++) & lookup_mask;
            call_result_t const accelerated_result = callable(token_index);
            call_result_t const baseline_result = baseline(token_index);
            result.stress_calls += accelerated_result.inputs_processed;
            if (check_validator(accelerated_result.check_value, baseline_result.check_value)) continue; // No failures

            // If we got here, the error needs to be reported and investigated.
            ++result.errors;
            if (result.errors > env.stress_limit) {
                std::printf("Too many errors in %s after %.3f seconds. Stopping the test.\n", name.c_str(),
                            running_seconds);
                std::terminate();
            }
            log_failure(env, name, baseline_result.check_value, accelerated_result.check_value, token_index);
        }

    // For profiling, we will first run the benchmark just once to get a rough estimate of the time.
    // But then we will repeat it in an unrolled fashion for a more accurate measurement.
    auto const first_call_duration = seconds_per_call([&] {
        std::uint64_t cpu_cycles_at_start = cpu_cycle_counter();
        call_result_t const call_result = callable((std::size_t)0); //? Use the first token
        std::uint64_t cpu_cycles_at_end = cpu_cycle_counter();

        result.operations += call_result.operations;
        result.bytes_passed += call_result.bytes_passed;
        result.profiled_inputs += call_result.inputs_processed;
        result.profiled_calls += 1;
        result.profiled_cpu_cycles += cpu_cycles_at_end - cpu_cycles_at_start;
        result.cpu_cycles_histogram[static_cast<double>(cpu_cycles_at_end - cpu_cycles_at_start)] += 1;
    });
    result.profiled_seconds = first_call_duration;
    if (first_call_duration >= env.benchmark_seconds) return result;

    // Repeat the benchmarks in unrolled batches until the time limit is reached.
    for (auto running_seconds : repeat_up_to(env.benchmark_seconds - first_call_duration)) {
        std::uint64_t t0 = cpu_cycle_counter();
        call_result_t r0 = callable((result.profiled_calls + 0) & lookup_mask);
        call_result_t r1 = callable((result.profiled_calls + 1) & lookup_mask);
        call_result_t r2 = callable((result.profiled_calls + 2) & lookup_mask);
        call_result_t r3 = callable((result.profiled_calls + 3) & lookup_mask);
        call_result_t r4 = callable((result.profiled_calls + 4) & lookup_mask);
        call_result_t r5 = callable((result.profiled_calls + 5) & lookup_mask);
        call_result_t r6 = callable((result.profiled_calls + 6) & lookup_mask);
        call_result_t r7 = callable((result.profiled_calls + 7) & lookup_mask);
        std::uint64_t t7 = cpu_cycle_counter();

        // Aggregate all of them:
        result.operations += r0.operations, result.operations += r1.operations,                           //
            result.operations += r2.operations, result.operations += r3.operations,                       //
            result.operations += r4.operations, result.operations += r5.operations,                       //
            result.operations += r6.operations, result.operations += r7.operations;                       //
        result.bytes_passed += r0.bytes_passed, result.bytes_passed += r1.bytes_passed,                   //
            result.bytes_passed += r2.bytes_passed, result.bytes_passed += r3.bytes_passed,               //
            result.bytes_passed += r4.bytes_passed, result.bytes_passed += r5.bytes_passed,               //
            result.bytes_passed += r6.bytes_passed, result.bytes_passed += r7.bytes_passed;               //
        result.profiled_inputs += r0.inputs_processed, result.profiled_inputs += r1.inputs_processed,     //
            result.profiled_inputs += r2.inputs_processed, result.profiled_inputs += r3.inputs_processed, //
            result.profiled_inputs += r4.inputs_processed, result.profiled_inputs += r5.inputs_processed, //
            result.profiled_inputs += r6.inputs_processed, result.profiled_inputs += r7.inputs_processed; //
        result.profiled_calls += 8;

        result.profiled_seconds = running_seconds;
        result.profiled_cpu_cycles += t7 - t0;
        result.cpu_cycles_histogram[static_cast<double>(t7 - t0)] += 8;
    }

    result.profiled_seconds += first_call_duration;
    return result;
}

/**
 *  @brief Loops over all tokens (in loop-unrolled batches) in environment and applies the given @b nullary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] callable Nullary function taking no arguments and returning a @b `call_result_t`.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_>
bench_result_t bench_nullary(environment_t const &env, std::string const &name, callable_type_ &&callable) {
    return bench_nullary(env, name, callable_no_op_t {}, callable);
}

/**
 *  @brief Loops over all tokens (in loop-unrolled batches) in environment and applies the given @b unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] callable Unary function taking a @b `std::size_t` token index and returning a @b `call_result_t`.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_>
bench_result_t bench_unary(environment_t const &env, std::string const &name, callable_type_ &&callable) {
    return bench_unary(env, name, callable_no_op_t {}, callable);
}

template <typename value_type_>
struct arrays_equality {
    using vector_t = unified_vector<value_type_>;
    bool operator()(check_value_t const &a, check_value_t const &b) const noexcept {
        vector_t const &a_ = *reinterpret_cast<vector_t const *>(a);
        vector_t const &b_ = *reinterpret_cast<vector_t const *>(b);
        if (a_.size() != b_.size()) return false;
        for (std::size_t i = 0; i < a_.size(); ++i)
            if (a_[i] != b_[i]) {
                std::printf("Mismatch at index %zu\n", i);
                return false;
            }
        return true;
    }
};

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian