/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date January 4, 2024
 *  @brief Helper structures and functions for C++ benchmarks.
 *
 *  The StringZilla benchmarking suite doesn't use any external frameworks like Criterion or Google
 *  Benchmark. There are several reasons for that:
 *
 *  1. Reduce the number of @b dependencies and the complexity of the build system.
 *
 *  2. Combine @b "stress-testing" with benchmarks to deduplicate logic.
 *
 *  As we work with often large datasets, with complex preprocessing, and many different backends,
 *  we want to minimize the surface area we debug and maintain, tracking string-specific properties:
 *
 *  - Is the string start aligned in memory?
 *  - Does it take more than one cache line? Is it's length a multiple of the SIMD vector size?
 *  - Is the string cached in the L1 or L2 cache? Can the dataset fit in L3?
 *
 *  As part of that stress-testing, on failure, those properties are persisted in a file on disk.
 *
 *  3. Use cheaper profiling methods like @b CPU-counter instructions, as opposed to wall-clock
 *     time. Assuming we can clearly isolate single-threaded workloads and are more interested in
 *     the number of retired instructions, CPU counters can be more accurate and less noisy.
 *
 *  4. Integrate with Linux @b perf and other tools for more detailed analysis.
 *
 *  We can isolate the relevant pieces of code, excluding the preprocessing costs from the actual
 *  workload. We can also track individual hardware counters, including platform-specific
 *  @c PERF_TYPE_RAW ones, that are not handled by most tools.
 *
 *  5. Visualize results differently, with compact output for generic workloads and special cases.
 */
#pragma once
#include <cctype>  // `std::isalnum`
#include <clocale> // `std::setlocale`
#include <cmath>   // `std::ceil`, `std::log`, `std::pow`
#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`
#include <cstdio>  // `std::fopen`, `std::fclose`, `std::FILE`
#include <cstdlib> // `std::abort`, `std::getenv`, `std::strtod`
#include <cstring> // `std::memcpy`, `std::strcmp`, `std::strlen`

#include <algorithm>
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::high_resolution_clock`
#include <exception>    // `std::invalid_argument`
#include <functional>   // `std::equal_to`
#include <limits>       // `std::numeric_limits`
#include <numeric>      // `std::accumulate`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`, `std::mt19937`
#include <regex>        // `std::regex`, `std::regex_search`
#include <span>         // `std::span`, `std::as_bytes`
#include <string>       // `std::hash`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <thread>       // `std::this_thread::sleep_for`, `std::thread::hardware_concurrency`
#include <type_traits>  // `std::invoke_result_t`
#include <vector>       // `std::vector`

#if defined(_MSC_VER)
#include <intrin.h> // `__rdtsc`, `_ReadStatusReg`
#if defined(_M_ARM64)
#include <arm64intr.h> // `ARM64_CNTVCT`
#endif
#endif
#if defined(_WIN32)
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h> // `std::byte`

#include "stringzilla/stringzilla.h"
#include "stringzilla/stringzilla.hpp"

namespace sz = ashvardanian::stringzilla;
namespace stdc = std::chrono;

namespace ashvardanian::stringzilla::bench {

using accurate_clock_t = stdc::high_resolution_clock;

#if !STRINGZILLA_TARGET_CUDA
template <typename value_type_>
using unified_vector = std::vector<value_type_, std::allocator<value_type_>>;
#else
template <typename value_type_>
using unified_vector = std::vector<value_type_, unified_alloc<value_type_>>;

/** Page-locked host memory, which the driver reports as host and every engine refuses. */
template <typename value_type_>
using pinned_vector = std::vector<value_type_, pinned_alloc<value_type_>>;

/**
 *  @brief Plain device memory a kernel can write and the host cannot touch.
 *
 *  @c safe_vector is what the engines already store device-resident scratch in, and its
 *  @c try_resize_uninitialized is the only growth a non-host-accessible allocator admits.
 */
template <typename value_type_>
using device_vector = safe_vector<value_type_, device_alloc<value_type_>>;

/**
 *  @brief Drains a device-resident buffer into @p destination, forwarding the driver's status.
 *  @param[out] destination At least as many elements as @p source holds; only that prefix is set.
 */
template <typename value_type_>
inline CUresult copy_device_to_host(device_vector<value_type_> const &source, span<value_type_> destination) {
    if (source.size() == 0) return CUDA_SUCCESS;
    if (destination.size() < source.size()) return CUDA_ERROR_INVALID_VALUE;
    return cuMemcpyDtoH(destination.data(), (CUdeviceptr)source.data(), source.size() * sizeof(value_type_));
}
#endif // STRINGZILLA_TARGET_CUDA

/** Reads a file into a string via LibC @c <cstdio>. A non-zero @p max_bytes stops the read after
 *  that many bytes, so the file tail is never touched. */
inline std::string read_file(std::string path, std::size_t max_bytes = 0) noexcept(false) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    std::size_t capacity = max_bytes;
    if (capacity == 0) {
        std::fseek(file, 0, SEEK_END);
        long const size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        capacity = size > 0 ? static_cast<std::size_t>(size) : 0;
    }
    std::string content(capacity, '\0');
    std::size_t const read_bytes = std::fread(&content[0], 1, capacity, file);
    std::fclose(file);
    content.resize(read_bytes);
    return content;
}

/** Reads the environment variable @p name as @p value_type_, or returns @p fallback when it is
 *  unset or empty; aborts naming the variable when the text does not parse, so a typo never becomes
 *  a silent default. */
template <typename value_type_>
[[nodiscard]] value_type_ env_variable(char const *name, value_type_ fallback) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return fallback;
    if constexpr (std::is_same_v<value_type_, char const *>) return text;
    else if constexpr (std::is_same_v<value_type_, bool>) return std::strcmp(text, "0") && std::strcmp(text, "false");
    else {
        value_type_ value {};
        char *stop = nullptr;
        if constexpr (std::is_floating_point_v<value_type_>) value = static_cast<value_type_>(std::strtod(text, &stop));
        else {
            auto const [end, error] = std::from_chars(text, text + std::strlen(text), value);
            stop = error == std::errc {} ? const_cast<char *>(end) : const_cast<char *>(text);
        }
        if (stop != text && *stop == '\0') return value;
        fmt::println(stderr, "{}=\"{}\" does not parse", name, text);
        std::abort();
    }
}

/** Prints the lines every benchmark opens with: the version and both capability lists. */
inline void log_environment() {
    fmt::println("StringZilla {}.{}.{}", STRINGZILLA_H_VERSION_MAJOR, STRINGZILLA_H_VERSION_MINOR,
                 STRINGZILLA_H_VERSION_PATCH);
    fmt::println("- Compiled for: {}", sz_capabilities_to_string(sz_capabilities_comptime()));
    fmt::println("- This machine: {}", sz_capabilities_to_string(sz_capabilities_runtime()));
}

#if STRINGZILLA_TARGET_CUDA

/**
 *  @brief Prints the "- CUDA:" line naming the first visible device and its compute capability,
 *      or "- CUDA: no device".
 *  @return Whether a device is visible; without one, the GPU benchmarks skip.
 */
inline bool log_cuda_device() {
    // The device is asked directly rather than through `sz_capabilities`: that verb answers for the
    // library this binary links, and `define_stringzilla_library` compiles the core without CUDA.
    int device_count = 0;
    cudaDeviceProp properties;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
        cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        fmt::println("- CUDA: no device");
        return false;
    }
    fmt::println("- CUDA: {} sm_{}{}", properties.name, properties.major, properties.minor);
    return true;
}
#endif // STRINGZILLA_TARGET_CUDA

/** Prints a backtrace on a fatal signal, so a crashing kernel self-localizes instead of dying
 *  silently under output redirection. Writes raw, since the crashing thread may already hold the
 *  stdio lock that @c fmt::println takes. */
inline void bench_fatal_signal_handler(int signal_number) noexcept {
    std::string_view constexpr message = "\n*** Fatal signal - backtrace follows ***\n";
#if defined(_WIN32)
    [[maybe_unused]] auto const written = _write(2, message.data(), static_cast<unsigned>(message.size()));
#else
    [[maybe_unused]] auto const written = ::write(STDERR_FILENO, message.data(), message.size());
#endif
#if defined(__linux__) && defined(__GLIBC__)
    void *frames[64];
    int const frames_count = backtrace(frames, sizeof(frames) / sizeof(frames[0]));
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

/** Installs the fatal-signal backtrace handler and line-buffers stdout; call once from @c main. */
inline void install_bench_signal_handlers() noexcept {
    // Size must be nonzero: Windows ucrt fast-fails on a zero-sized buffering mode.
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    for (int signal_number : {SIGSEGV, SIGABRT, SIGILL, SIGFPE}) std::signal(signal_number, bench_fatal_signal_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, bench_fatal_signal_handler);
#endif
}

template <std::size_t multiple_>
std::size_t round_up_to_multiple(std::size_t n) {
    return n == 0 ? multiple_ : ((n + multiple_ - 1) / multiple_) * multiple_;
}

using check_value_t = std::uint64_t;

struct call_result_t {

    /** Number of input bytes processed. */
    std::size_t bytes_passed = 0;

    /** Some value used to compare execution result between the baseline and accelerated backend. */
    check_value_t check_value = 0;

    /** For operations with non-linear complexity, the throughput should be measured differently. */
    std::size_t operations = 0;

    /** Equal to 1 for most inputs, but can be larger for batch-capable functions. */
    std::size_t inputs_processed = 1;

    call_result_t() = default;
    call_result_t(std::size_t bytes_passed, std::size_t check_value = 0, std::size_t operations = 0)
        : bytes_passed(bytes_passed), check_value(check_value), operations(operations), inputs_processed(1) {}
};

struct callable_no_op_t {
    call_result_t operator()(std::size_t) const { return {}; }
};

/** Cross-platform function to get the number of CPU cycles elapsed @b only on the current core.
 *  Used as a more efficient alternative to @c std::chrono::high_resolution_clock. */
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
    unsigned int low, high;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (static_cast<std::uint64_t>(high) << 32) | low;
#elif defined(__aarch64__) || STRINGZILLA_ARCH_ARM64_
    // On ARM64, read the virtual count register `CNTVCT_EL0` which provides cycle count.
    // (`STRINGZILLA_ARCH_ARM64_` is a 0/1 value macro — testing `defined()` of it would be true everywhere,
    // wrongly selecting this branch on non-ARM targets such as wasm32.)
    std::uint64_t counter;
    asm volatile("mrs %0, cntvct_el0" : "=r"(counter));
    return counter;
#else
    return 0;
#endif
}

/** Measures the duration of a single call to the given function. */
template <typename function_type_>
double seconds_per_call(function_type_ &&function) {
    accurate_clock_t::time_point start = accurate_clock_t::now();
    function();
    accurate_clock_t::time_point end = accurate_clock_t::now();
    return stdc::duration_cast<stdc::nanoseconds>(end - start).count() / 1.e9;
}

/**
 *  @brief Allows time-limited for-loop iteration, like `for (auto _ : state)` in Google Benchmark.
 *      Use as `for (auto call_index : repeat_up_to(5.0)) { ... }`, then read `repeat.seconds()`.
 *
 *  The loop body receives the @b iteration index — the quantity it actually needs to rotate over
 *  tokens and count calls. The elapsed time and iteration count are deliberately @b not yielded
 *  per-iteration; they are exposed as `seconds()` and `count()`, computed live from the owning
 *  object. Nothing is cached, so a read after the loop is as valid as one inside it — there is no
 *  stale snapshot to get the denominator out of sync with the numerator.
 */
struct repeat_up_to_t {
    double max_seconds_ = 0;
    accurate_clock_t::time_point start_ = accurate_clock_t::now();
    std::size_t count_ = 0;

    struct end_sentinel_t {};
    struct iterator_t {
        repeat_up_to_t *parent_;
        inline bool operator!=(end_sentinel_t) const { return parent_->keep_running(); }
        inline std::size_t operator*() const { return parent_->count_; }
        inline void operator++() const { ++parent_->count_; }
    };

    inline explicit repeat_up_to_t(double max_seconds) : max_seconds_(max_seconds) {}
    inline explicit repeat_up_to_t(std::size_t max_seconds) : max_seconds_(static_cast<double>(max_seconds)) {}
    inline iterator_t begin() { return start_ = accurate_clock_t::now(), count_ = 0, iterator_t {this}; }
    inline end_sentinel_t end() const noexcept { return {}; }

    /** Live wall-clock seconds since `begin()` — computed on demand, never cached. */
    inline double seconds() const noexcept {
        return stdc::duration_cast<stdc::nanoseconds>(accurate_clock_t::now() - start_).count() / 1.e9;
    }

    /** Number of completed iterations — authoritative both during and after the loop. */
    inline std::size_t count() const noexcept { return count_; }
    inline bool keep_running() const noexcept { return max_seconds_ != 0 && seconds() < max_seconds_; }
};

inline repeat_up_to_t repeat_up_to(double max_seconds) noexcept { return repeat_up_to_t {max_seconds}; }

/** Stops compilers from optimizing out the expression, like Google Benchmark's @b DoNotOptimize. */
template <typename argument_type_>
static void do_not_optimize(argument_type_ &&value) noexcept {

#if defined(_MSC_VER) // MSVC
    using plain_type = typename std::remove_reference<argument_type_>::type;
    // Use the `volatile` keyword and a memory barrier to prevent optimization
    volatile plain_type *p = &value;
    _ReadWriteBarrier();
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else // GCC
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** The device-measured kernel time an engine reports, in milliseconds, or @b 0 for CPU engines,
 *  whose plain @c status_t carries no such timer. */
template <typename status_type_>
float engine_elapsed_milliseconds(status_type_ const &status) noexcept {
    if constexpr (requires { status.elapsed_milliseconds; }) return status.elapsed_milliseconds;
    else return 0.0f;
}

/** A status paired with the device-measured kernel time, both read out of one materialized copy. */
struct engine_timing_t {
    sz::status_t status = sz::status_t::success_k;
    float kernel_milliseconds = 0.0f;
};

/**
 *  @brief Calls @p invocable and decomposes its returned status through opaque memory, in one
 *      non-inlined frame. Every timed engine call in the suite goes through here.
 *
 *  Both halves must stay in this frame. Compilers miscompile the engines' return-by-value inside
 *  these large translation units at @b -O2 : NVCC 12.x corrupts the two leading enum fields of a
 *  @c cuda_status_t while its `float elapsed_milliseconds` survives, and g++ trunk corrupts the
 *  status when it folds the return into an inlined caller. The libraries are correct: a separate-TU
 *  call returns @c success_k, and sharing one `[[gnu::noinline]]` frame reproduces that code path.
 */
template <typename invocable_type_>
STRINGZILLA_NOINLINE_ engine_timing_t invoke_engine_(invocable_type_ &&invocable) noexcept {
    using status_t = std::invoke_result_t<invocable_type_>;
    status_t engine_result = invocable();
    do_not_optimize(engine_result);

    status_t materialized;
    std::memcpy((void *)&materialized, (void const *)&engine_result, sizeof(status_t));

    engine_timing_t timing;
    timing.status = static_cast<sz::status_t>(materialized);
    timing.kernel_milliseconds = engine_elapsed_milliseconds(materialized);
    return timing;
}

/**
 *  @brief Rounds the number @b down to the preceding power of two.
 *  @see Equivalent to @c std::bit_floor: https://en.cppreference.com/w/cpp/numeric/bit_floor
 */
inline std::size_t bit_floor(std::size_t n) {
    if (n == 0) return 0;
    std::size_t most_significant_bit_position = 0;
    while (n > 1) n >>= 1, most_significant_bit_position++;
    return static_cast<std::size_t>(1) << most_significant_bit_position;
}

/** Parses a human byte size like @c 64mb or @c 1gb into bytes; @c kb, @c mb and @c gb are powers of
 *  1024, matching the @c parse_size of StringWars. A bare number is bytes, and an empty or zero
 *  value is the whole file. */
inline std::size_t parse_size(std::string const &text) {
    std::size_t cursor = 0;
    while (cursor < text.size() && (std::isdigit((unsigned char)text[cursor]) || text[cursor] == '.')) ++cursor;
    double const number = cursor == 0 ? 0.0 : std::stod(text.substr(0, cursor));
    std::string unit;
    for (; cursor < text.size(); ++cursor)
        if (!std::isspace((unsigned char)text[cursor])) unit.push_back((char)std::tolower((unsigned char)text[cursor]));
    std::size_t multiplier = 1;
    if (unit.empty() || unit == "b") multiplier = 1;
    else if (unit == "kb") multiplier = 1024ull;
    else if (unit == "mb") multiplier = 1024ull * 1024ull;
    else if (unit == "gb") multiplier = 1024ull * 1024ull * 1024ull;
    else throw std::invalid_argument("Invalid STRINGWARS_DATASET_LIMIT unit: " + unit);
    return static_cast<std::size_t>(number * (double)multiplier);
}

/** The smallest read that still exercises the control-flow paths of a compute-bound bench on the
 *  multilingual corpus. Memory-bound benches ignore it and read the whole file. */
static constexpr std::size_t compute_bound_slice_bytes_k = 64ull * 1024ull * 1024ull;

#if !STRINGZILLA_TARGET_CUDA
using dataset_t = std::string;
using token_view_t = std::string_view;
using tokens_t = std::vector<token_view_t>;
#else
using dataset_t = std::basic_string<char, std::char_traits<char>, unified_alloc<char>>;
using token_view_t = stringzilla::span<char const>;
using tokens_t = std::vector<token_view_t, unified_alloc<token_view_t>>;
#endif

/**
 *  @brief Tokenizes a string with the given separator predicate.
 *  @see For faster ways to tokenize a string with STL: https://ashvardanian.com/posts/splitting-strings-cpp/
 */
template <typename is_separator_callback_type_>
tokens_t tokenize(std::string_view str, is_separator_callback_type_ &&is_separator) {

    std::size_t separator_count = 0;
    for (std::size_t i = 0; i < str.length(); ++i)
        if (is_separator(str[i])) separator_count++;

    std::size_t const token_upper_bound = separator_count + 1;
    tokens_t tokens(token_upper_bound);

    // Empty tokens are dropped, so the count is an upper bound until the trailing resize.
    std::size_t tokens_found = 0;
    for (std::size_t start = 0, end = 0; end <= str.length(); ++end)
        if (end == str.length() || is_separator(str[end])) {
            if (start < end) tokens[tokens_found] = {&str[start], end - start}, ++tokens_found;
            start = end + 1;
        }

    tokens.resize(tokens_found);
    return tokens;
}

/**
 *  @brief Tokenizes a string around the given separator @p byteset in one lazy SIMD pass.
 *
 *  Each step of the underlying @c split issues one @c sz_find_byteset scan. The whole corpus is
 *  already bounded by the dataset read, so the walk runs to the end without a cap of its own.
 */
inline tokens_t tokenize(std::string_view str, sz::byteset_t separators) {
    tokens_t tokens;
    for (auto token : sz::string_view_t {str.data(), str.size()}.split(separators)) {
        if (token.size() == 0) continue; // ? Runs of separators yield empty segments
        tokens.push_back({token.data(), token.size()});
    }
    return tokens;
}

/** Splits a string into words around newlines, tabs, and other ASCII whitespaces. */
inline tokens_t tokenize(std::string_view str) { return tokenize(str, sz::whitespaces_set()); }

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
 *  @brief Environment for the benchmarking scripts pulled from the CLI arguments.
 *
 *  The original CLI arguments include the @c path to the dataset file and the number of seconds per
 *  benchmark, the Regex @c filter to select only the backends that match the given pattern, as well
 *  as the @c tokenization mode to convert the loaded textual @c dataset to a @c tokens array.
 *
 *  In the RELEASE mode, the tokens will be shuffled to avoid any bias in the benchmarking process.
 *  The @c seed is used to guarantee reproducibility of the results between different runs.
 */
struct environment_t {
    enum tokenization_t : unsigned char {
        file_k = 255,
        lines_k = 254,
        words_k = 253,
    };

    /** How @c fmt spells a tokenization mode: @c file, @c line, @c word, or @c N-grams. */
    friend std::string format_as(tokenization_t mode) {
        switch (mode) {
        case file_k: return "file";
        case lines_k: return "line";
        case words_k: return "word";
        default: return fmt::format("{}-grams", static_cast<std::size_t>(mode));
        }
    }

    /** Absolute path of the textual input file on disk. */
    std::string path;

    /** Stress-testing results directory. */
    std::string stress_dir;

    /** Tokenization mode to convert the @c dataset to @c tokens. */
    tokenization_t tokenization = tokenization_t::words_k;

    /** Regular expression to filter the backends. */
    std::string filter;

    /** The @c filter compiled once, so @c allow never rebuilds it per benchmark. */
    std::regex filter_pattern;

    /** Whether to stress-test the backends. */
    bool stress = true;

    /** Upper time bound on a duration of the stress-test for a single callable. */
    std::size_t stress_seconds = STRINGZILLA_DEBUG ? 1 : 10;

    /** Upper time bound on a duration of a single callable. */
    std::size_t benchmark_seconds = STRINGZILLA_DEBUG ? 1 : 10;

    /** Seed for the random number generator. */
    std::uint64_t seed = 0;

    /** Upper bound on the number of stress test failures on a callable. */
    std::size_t stress_limit = 1;

    /** Whether to deduplicate tokens before benchmarking. */
    bool unique = false;

    /** Dataset bytes to read at most, 0 meaning the whole file; @c STRINGWARS_DATASET_LIMIT. */
    std::size_t dataset_limit_bytes = 0;

    /** Per-benchmark batch sizes from @c STRINGWARS_BATCH, empty means the backend default. */
    std::vector<std::size_t> batch_sizes_override;

    /** Textual content of the dataset file, fully loaded into memory. */
    dataset_t dataset;

    /** Array of tokens extracted from the @c dataset. */
    tokens_t tokens;

    /** This machine's cache geometry, which cache-resident benchmark shapes are sized from. */
    sz::cpu_specs_t specs;

    bool allow(std::string_view benchmark_name) const {
        return filter.empty() || std::regex_search(benchmark_name.begin(), benchmark_name.end(), filter_pattern);
    }

    std::string_view operator[](std::size_t i) const {
        if (i >= tokens.size()) throw std::out_of_range("Index out of range");
        return {tokens[i].data(), tokens[i].size()};
    }
};

/** Whether @c build_environment stress-tests the backends absent @c STRINGWARS_STRESS. */
enum class stress_default_t : bool { quick_k, stress_k };

/** The run's @b STRINGWARS_SEED: 0 when unset, which keeps the tokens in order. */
inline std::size_t bench_seed() noexcept {
    std::size_t const seed = env_variable("STRINGWARS_SEED", std::size_t {0});
    if (seed == 0 && *env_variable("STRINGWARS_SEED", "")) {
        fmt::println(stderr, "STRINGWARS_SEED must be a positive integer");
        std::abort();
    }
    return seed;
}

/** Prints the run's seed below the lines of @c log_environment. */
inline void print_bench_environment() noexcept { fmt::println("- Seed: {}", bench_seed()); }

/**
 *  @brief Prepares the environment for benchmarking based on environment variables and default
 *      settings. Different workloads may use different default datasets and tokenization modes, but
 *      time limits and seeds are usually consistent across all benchmarks.
 *
 *  @param[in] argc Number of command-line string arguments. Not used in reality.
 *  @param[in] argv Array of command-line string arguments. Not used in reality.
 *
 *  @param[in] default_dataset Default dataset file path, if @b STRINGWARS_DATASET is not set.
 *  @param[in] default_tokens Tokenization mode, if @b STRINGWARS_TOKENS is not set.
 *  @param[in] default_duration Time limit per benchmark, if @b STRINGWARS_MAX_SECONDS is not set.
 *
 *  @param[in] default_stress Whether to stress-test backends, if @b STRINGWARS_STRESS is not set.
 *  @param[in] default_stress_dir Stress-test log directory, if @b STRINGWARS_STRESS_DIR is unset.
 *  @param[in] default_stress_limit Failures to tolerate, if @b STRINGWARS_STRESS_LIMIT is not set.
 *  @param[in] default_stress_duration Stress-test time, if @b STRINGWARS_STRESS_DURATION is unset.
 *
 *  @param[in] default_filter Regex to filter the backends, if @b STRINGWARS_FILTER is not set.
 */
inline environment_t build_environment(                                        //
    int argc, char const *argv[],                                              //< Ignored
    std::string default_dataset, environment_t::tokenization_t default_tokens, //< Mandatory
    std::size_t default_dataset_limit_bytes = 0,                               //< Optional, 0 = whole file
    std::size_t default_duration = STRINGZILLA_DEBUG ? 1 : 10,                 //< Optional
    stress_default_t default_stress = stress_default_t::stress_k,              //
    std::string default_stress_dir = ".tmp",                                   //
    std::size_t default_stress_limit = 1,                                      //
    std::size_t default_stress_duration = STRINGZILLA_DEBUG ? 1 : 10,          //
    std::string default_filter = ""                                            //
    ) noexcept(false) {

    sz_unused_(argc && argv); // Unused in this context
    environment_t env;
    env.dataset_limit_bytes = default_dataset_limit_bytes;

    env.path = env_variable("STRINGWARS_DATASET", default_dataset.c_str());
    env.filter = env_variable("STRINGWARS_FILTER", default_filter.c_str());
    if (!env.filter.empty()) env.filter_pattern = std::regex(env.filter);

    env.benchmark_seconds = env_variable("STRINGWARS_MAX_SECONDS", default_duration);
    if (env.benchmark_seconds == 0) throw std::invalid_argument("The time limit must be greater than 0.");
    env.seed = bench_seed();

    char const *const tokenization = env_variable("STRINGWARS_TOKENS", "");
    if (!*tokenization) env.tokenization = default_tokens;
    else if (std::strcmp(tokenization, "file") == 0) env.tokenization = environment_t::file_k;
    else if (std::strcmp(tokenization, "lines") == 0) env.tokenization = environment_t::lines_k;
    else if (std::strcmp(tokenization, "words") == 0) env.tokenization = environment_t::words_k;
    else {
        // Anything else names an N-gram length, which `env_variable` parses strictly.
        env.tokenization = static_cast<environment_t::tokenization_t>(
            env_variable("STRINGWARS_TOKENS", std::size_t {0}));
        if (env.tokenization == 0)
            throw std::invalid_argument(
                "The tokenization mode must be 'file', 'lines', 'words', or a positive integer.");
    }

    env.stress = env_variable("STRINGWARS_STRESS", default_stress == stress_default_t::stress_k);
    env.stress_seconds = env_variable("STRINGWARS_STRESS_DURATION", default_stress_duration);
    if (env.stress_seconds == 0) throw std::invalid_argument("The stress-testing time limit must be greater than 0.");
    env.stress_dir = env_variable("STRINGWARS_STRESS_DIR", default_stress_dir.c_str());
    env.stress_limit = env_variable("STRINGWARS_STRESS_LIMIT", default_stress_limit);
    if (env.stress_limit == 0) throw std::invalid_argument("The stress-testing limit must be greater than 0.");

    // @sa `STRINGWARS_UNIQUE=1` sorts the tokenized set and drops duplicates before benchmarking.
    env.unique = env_variable("STRINGWARS_UNIQUE", false);

    // Use `STRINGWARS_DATASET_LIMIT` to bound the dataset read, so the file tail is never touched.
    if (char const *const limit = env_variable("STRINGWARS_DATASET_LIMIT", ""); *limit)
        env.dataset_limit_bytes = parse_size(limit);

    // Use `STRINGWARS_BATCH` to override the per-benchmark batch sizes with a comma-separated list,
    // e.g. `STRINGWARS_BATCH=1024` to run a single batch and skip the slow/largest default sweep entries.
    // @sa `STRINGWARS_BATCH=1024,4096` replaces the default batch-size sweep with exactly these sizes.
    std::string const batch_argument = env_variable("STRINGWARS_BATCH", "");
    for (std::size_t start = 0; start < batch_argument.size();) {
        std::size_t const comma = batch_argument.find(',', start);
        std::size_t const end = comma == std::string::npos ? batch_argument.size() : comma;
        if (end > start) env.batch_sizes_override.push_back(std::stoull(batch_argument.substr(start, end - start)));
        start = end + 1;
    }

    env.dataset = read_file(env.path, env.dataset_limit_bytes); // A non-zero limit stops the read early
    env.dataset.resize(bit_floor(env.dataset.size()));          // Shrink to the nearest power of two

    // Tokenize the dataset according to the tokenization mode. The corpus is already bounded by the read,
    // so each mode walks it to the end.
    if (env.tokenization == environment_t::file_k) { env.tokens.push_back({env.dataset.data(), env.dataset.size()}); }
    else if (env.tokenization == environment_t::lines_k) { env.tokens = tokenize(env.dataset, sz::byteset_t {'\n'}); }
    else if (env.tokenization == environment_t::words_k) { env.tokens = tokenize(env.dataset); }
    else {
        std::size_t n = static_cast<std::size_t>(env.tokenization);
        env.tokens = filter_by_length<token_view_t>(tokenize(env.dataset), n, std::equal_to<std::size_t>());
    }

    // Deduplicate tokens if requested
    if (env.unique) {
        std::sort(env.tokens.begin(), env.tokens.end());
        auto last = std::unique(env.tokens.begin(), env.tokens.end());
        env.tokens.erase(last, env.tokens.end());
    }

    env.tokens.resize(bit_floor(env.tokens.size())); // Shrink to the nearest power of two

    // In "RELEASE" mode, shuffle tokens to avoid bias.
    if (env.seed != 0) {
        std::mt19937_64 generator(static_cast<unsigned long>(env.seed));
        std::shuffle(env.tokens.begin(), env.tokens.end(), generator);
    }

    auto const mean_token_length = std::accumulate(env.tokens.begin(), env.tokens.end(), (std::size_t)0u,
                                                   [](std::size_t sum, token_view_t token) -> std::size_t {
                                                       return sum + token.size();
                                                   }) *
                                   1.0 / env.tokens.size();

    // Group integer decimal separators by 3
    // https://www.ibm.com/docs/en/i/7.4?topic=categories-lc-numeric-category
    std::setlocale(LC_NUMERIC, "en_US.UTF-8");
    // Only the core count is portable to query, so every cache level keeps the conservative default a shape sized
    // from it is nominal against.
    if (unsigned const cores = std::thread::hardware_concurrency()) env.specs.cores_per_socket = cores;

    fmt::print(R"(Environment built with the following settings:
- Dataset path: {path}
- Time limit: {benchmark_seconds} seconds per benchmark ({stress_seconds} per stress-test)
- Algorithm filter: {filter}
- Tokenization mode: {tokenization}
- Stress-testing: {stress}
- Unique tokens: {unique}
- Loaded dataset size: {dataset_bytes} bytes
- Dataset limit: {dataset_limit}
- Number of tokens: {tokens}
- Mean token length: {mean_token_length:.2f} bytes
- Caches: {l1_bytes} B first-level (assumed), {l3_bytes} B confined to a compute domain
)",
               fmt::arg("path", env.path), fmt::arg("benchmark_seconds", env.benchmark_seconds),
               fmt::arg("stress_seconds", env.stress_seconds),
               fmt::arg("filter", env.filter.empty() ? std::string("none") : env.filter),
               fmt::arg("tokenization", env.tokenization), fmt::arg("stress", env.stress ? "yes" : "no"),
               fmt::arg("unique", env.unique ? "yes" : "no"), fmt::arg("dataset_bytes", env.dataset.size()),
               fmt::arg("dataset_limit", env.dataset_limit_bytes ? fmt::format("{} bytes", env.dataset_limit_bytes)
                                                                 : std::string("whole file")),
               fmt::arg("tokens", env.tokens.size()), fmt::arg("mean_token_length", mean_token_length),
               fmt::arg("l1_bytes", env.specs.l1_bytes), fmt::arg("l3_bytes", env.specs.l3_bytes));

    return env;
}

/** The slice's median token length in bytes: the short query, and the typical candidate. */
inline std::size_t median_token_bytes(environment_t const &env) {
    std::vector<std::size_t> lengths(env.tokens.size());
    std::transform(env.tokens.begin(), env.tokens.end(), lengths.begin(),
                   [](token_view_t token) { return token.size(); });
    std::nth_element(lengths.begin(), lengths.begin() + lengths.size() / 2, lengths.end());
    return lengths[lengths.size() / 2];
}

/** Candidates one call scores: the first @c STRINGWARS_BATCH entry if set, else as many median
 *  tokens as fill the first-level cache. */
inline std::size_t candidates_per_call(environment_t const &env) {
    if (!env.batch_sizes_override.empty()) return env.batch_sizes_override.front();
    return std::max<std::size_t>(1, env.specs.l1_bytes / median_token_bytes(env));
}

#if STRINGZILLA_TARGET_CUDA

/** Candidates one device call scores: the first @c STRINGWARS_BATCH entry if set, else one per
 *  resident thread of the bound device. */
inline std::size_t resident_candidates_per_call(environment_t const &env) {
    if (!env.batch_sizes_override.empty()) return env.batch_sizes_override.front();
    int device = 0;
    cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("The device would not report its geometry.");
    return (std::size_t)properties.multiProcessorCount * (std::size_t)properties.maxThreadsPerMultiProcessor;
}
#endif

/** Uses C-style file IO to save information about the most recent stress test failure. Files are
 *  written to @c STRINGWARS_STRESS_DIR as `failed_<time>_<name>.txt`. */
inline void log_failure(                                              //
    environment_t const &env, std::string_view name,                  //
    std::size_t expected_check_value, std::size_t actual_check_value, //
    std::optional<std::size_t> token_index) noexcept(false) {

    std::string timestamp = std::to_string(std::time(nullptr));
    // Benchmark names embed shape labels like `...:q4xc4:allpairs`; `:` and `/` are invalid in path
    // components on common filesystems, so map every non-portable character to `_` before composing the path.
    std::string safe_name {name};
    for (char &c : safe_name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '.' || c == '_')) c = '_';
    std::string file_name = "failed_" + timestamp + "_" + safe_name + "_" + ".txt";
    std::string file_path = env.stress_dir + "/" + file_name;
    std::FILE *file = std::fopen(file_path.c_str(), "w");
    if (!file) throw std::runtime_error("Failed to open file for writing: " + file_name);

    fmt::println(file, "Dataset path: {}\nTokenization mode: {}\nSeed: {}", env.path, env.tokenization,
                 static_cast<std::size_t>(env.seed));
    if (token_index)
        fmt::println(file, "Token index: {}\nToken Hex: {:02X}", *token_index,
                     fmt::join(std::as_bytes(std::span(env[*token_index])), " "));
    fmt::println(file, "Expected: {}\nActual: {}", expected_check_value, actual_check_value);
    std::fclose(file);
}

/**
 *  @brief Light-weight structure to construct a histogram of function call durations for a very
 *      wide range of floating point values using logarithmic binning, queryable by percentile.
 *
 *  Every call site below feeds it raw CPU-cycle deltas rather than wall-clock seconds, so the
 *  bounds are sized for cycle counts: from a single cycle up to a call spanning hours at multi-GHz
 *  clocks, not for literal seconds.
 */
template <std::size_t slots_ = 128>
struct duration_histogram {
    using count_t = std::uint32_t;
    std::array<count_t, slots_> bins = {};
    static constexpr double max_cpu_cycles_k = 1e14; // ~9 hours at 3 GHz — hard to imagine a slower single call
    static constexpr double min_cpu_cycles_k = 1;    // A call can't take fewer than a single cycle

    inline count_t &operator[](double cpu_cycles) {
        auto bin_float = std::log(cpu_cycles / min_cpu_cycles_k) / std::log(max_cpu_cycles_k / min_cpu_cycles_k) *
                         bins.size();
        // A call quicker than `min_cpu_cycles_k` makes the logarithm negative, and casting that to an unsigned
        // type wraps to something enormous, so the low end is clamped before the cast rather than after.
        // The high end clamps to the last slot: `bins.size()` is one past it.
        std::size_t bin = bin_float <= 0 ? 0 : std::min(bins.size() - 1, static_cast<std::size_t>(bin_float));
        return bins[bin];
    }

    /**
     *  @brief Upper edge of the bin holding the nearest-rank percentile, e.g. @c 0.99 for the p99.
     *  @return The upper edge, or @b 0 when the histogram is empty and there is nothing to report.
     *
     *  Nearest rank is the `ceil(fraction * count)`-th sample, so the p99 of a hundred samples is
     *  the 99th rather than the largest. The bin's upper edge is reported because a bin only knows
     *  its own bounds, which rounds every reading up by at most one bin width.
     */
    inline double percentile(double fraction) const noexcept {
        std::size_t total_count = 0;
        for (count_t bin_count : bins) total_count += bin_count;
        if (total_count == 0) return 0;

        std::size_t target_rank = static_cast<std::size_t>(std::ceil(fraction * total_count));
        if (target_rank == 0) target_rank = 1; // ? A `fraction` of zero must not select an empty leading bin
        std::size_t cumulative_count = 0;
        for (std::size_t bin_index = 0; bin_index < bins.size(); ++bin_index) {
            cumulative_count += bins[bin_index];
            if (cumulative_count >= target_rank)
                return min_cpu_cycles_k *
                       std::pow(max_cpu_cycles_k / min_cpu_cycles_k, (bin_index + 1.0) / bins.size());
        }
        return max_cpu_cycles_k;
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

    /** Cheapest single call observed, in CPU cycles; a less noisy cost estimator than the mean. */
    std::uint64_t profiled_cpu_cycles_min = std::numeric_limits<std::uint64_t>::max();

    std::size_t bytes_passed = 0; //< Pulled from the `call_result_t`
    std::size_t operations = 0;   //< Pulled from the `call_result_t`
    std::size_t errors = 0;       //< Pulled from the `call_result_t`

    /**
     *  @brief Logs the benchmark results to the console, including the throughput and latency,
     *      comparing against one or more baselines.
     *
     *  The output reads like this:
     *
     *  @code{.unparsed}
     *  Benchmarking `sz_find_skylake`:
     *  > Throughput: 0.00 TB/s @ 0.00 ns/call
     *  > Latency: min 0.00 ns/call, p99 0.00 ns/call
     *  > Efficiency: 0.00 TOps/s @ 0.00 ops/cycle
     *  > Errors: 0 in 10 calls
     *  > + 3.5 x against `sz_find_serial`
     *  > + 70 % against `memmem`
     *  @endcode
     *
     *  When running on Linux, additional hardware counters can be sampled using @c perf:
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
     *  After a section of benchmarks is completed, you can use other functionality to visualize the
     *  results in a more structured way, like a table or a graph or a set of progress bars.
     */
    template <typename... baselines_types_>
    bench_result_t const &log(baselines_types_ const &...bases) const {
        if (skipped) return *this;
        fmt::println("\nBenchmarking `{}`:", fmt::styled(name, fmt::emphasis::bold));

        // Print the number of errors, if any
        if (errors) fmt::println("> Errors: {} in {} calls", errors, stress_calls);

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
        fmt::println("> Throughput: {:.2f} {} @ {:.2f} {}/call", //
                     bytes_printable, bytes_printable_unit,      //
                     seconds_printable, seconds_printable_unit);

        // Scheduler interference only slows a call down, so the minimum is a less noisy estimator than the
        // mean above, and the 99th percentile shows whether outliers drag that mean up. Both come from the
        // CPU-cycle histogram, rescaled by this run's average seconds-per-cycle ratio.
        if (profiled_cpu_cycles > 0 && profiled_cpu_cycles_min != std::numeric_limits<std::uint64_t>::max()) {
            double const seconds_per_cycle = profiled_seconds / profiled_cpu_cycles;

            auto minimum_printable = profiled_cpu_cycles_min * seconds_per_cycle * 1e9;
            char const *minimum_printable_unit = "ns";
            if (minimum_printable > 1e3) minimum_printable /= 1e3, minimum_printable_unit = "us";
            if (minimum_printable > 1e3) minimum_printable /= 1e3, minimum_printable_unit = "ms";
            if (minimum_printable > 1e3) minimum_printable /= 1e3, minimum_printable_unit = "s";

            auto p99_printable = cpu_cycles_histogram.percentile(0.99) * seconds_per_cycle * 1e9;
            char const *p99_printable_unit = "ns";
            if (p99_printable > 1e3) p99_printable /= 1e3, p99_printable_unit = "us";
            if (p99_printable > 1e3) p99_printable /= 1e3, p99_printable_unit = "ms";
            if (p99_printable > 1e3) p99_printable /= 1e3, p99_printable_unit = "s";

            fmt::println("> Latency: min {:.2f} {}/call, p99 {:.2f} {}/call", //
                         minimum_printable, minimum_printable_unit,           //
                         p99_printable, p99_printable_unit);
        }

        // Print the number of operations, if there was a separate tracking mechanism for those.
        if (operations) {
            auto ops_printable = operations * 1.0 / profiled_seconds;
            auto ops_per_cycle = operations * 1.0 / profiled_cpu_cycles;
            char const *ops_printable_unit = (operations ? "Ops/s" : "B/s");
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "KOps/s";
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "MOps/s";
            if (ops_printable > 1e3) ops_printable /= 1e3, ops_printable_unit = "GOps/s";
            fmt::println("> Efficiency: {:.2f} {} @ {:.2f} ops/cycle", ops_printable, ops_printable_unit,
                         ops_per_cycle);
        }

        // Define a helper lambda to log relative performance with folding expressions.
        auto log_relative = [this](bench_result_t const &base) {
            if (skipped || base.skipped) return;
            auto relative_throughput = (bytes_passed / profiled_seconds) / (base.bytes_passed / base.profiled_seconds);
            if (operations)
                relative_throughput = (operations / profiled_seconds) / (base.operations / base.profiled_seconds);

            // Format relative improvements: green and a plus for improvements, red and a minus for regressions.
            auto const relative_color = fmt::fg(relative_throughput > 1 ? fmt::terminal_color::green
                                                                        : fmt::terminal_color::red);
            char const *relative_sign = (relative_throughput > 1) ? "+" : "-";
            char const *relative_unit = (relative_throughput > 2) ? "x" : "%";
            if (relative_throughput < 0.5) relative_throughput = 1 / relative_throughput, relative_unit = "x";
            if (std::strcmp(relative_unit, "%") == 0) relative_throughput = (relative_throughput - 1) * 100;
            fmt::println("> {} {:.1f} {} against `{}`", fmt::styled(relative_sign, relative_color),
                         fmt::styled(std::abs(relative_throughput), relative_color),
                         fmt::styled(relative_unit, relative_color), base.name);
        };

        // Expand over all provided baselines.
        [[maybe_unused]] std::initializer_list<int> const expanded {(log_relative(bases), 0)...};
        sz_unused_(log_relative); // In case no `bases` were provided

        return *this;
    }
};

/**
 *  @brief Repeatedly calls and profiles a given @b nullary function, comparing it to a baseline.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] baseline Optional serial analog to stress-test the accelerated function against.
 *  @param[in] callable Nullary function taking no arguments and returning a @b call_result_t.
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
    std::string_view name,     //
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
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value) {
        repeat_up_to_t stress = repeat_up_to(env.stress ? env.stress_seconds : 0.0);
        for (auto call_index : stress) {
            sz_unused_(call_index);
            call_result_t const accelerated_result = callable();
            call_result_t const baseline_result = baseline();
            ++result.stress_calls;
            result.stress_inputs += accelerated_result.inputs_processed;
            if (check_validator(accelerated_result.check_value, baseline_result.check_value)) continue; // No failures

            // If we got here, the error needs to be reported and investigated.
            ++result.errors;
            if (result.errors > env.stress_limit) {
                fmt::println("Too many errors in {} after {:.3f} seconds. Stopping the test.", name, stress.seconds());
                std::terminate();
            }
            log_failure(env, name, baseline_result.check_value, accelerated_result.check_value, {});
        }
    }

    // Repeat the benchmark of the unary function. Assume most of them are applied to the entire
    // dataset and take a lot of time, so we don't unroll much, unlike `bench_unary`.
    repeat_up_to_t repeat = repeat_up_to(env.benchmark_seconds);
    for (auto call_index : repeat) {
        sz_unused_(call_index);
        std::uint64_t cpu_cycles_at_start = cpu_cycle_counter();
        call_result_t call_result = callable();
        std::uint64_t cpu_cycles_at_end = cpu_cycle_counter();
        std::uint64_t const cpu_cycles_spent = cpu_cycles_at_end - cpu_cycles_at_start;

        // Aggregate:
        result.operations += call_result.operations;
        result.bytes_passed += call_result.bytes_passed;
        result.profiled_inputs += call_result.inputs_processed;
        result.profiled_cpu_cycles += cpu_cycles_spent;
        result.profiled_cpu_cycles_min = std::min(result.profiled_cpu_cycles_min, cpu_cycles_spent);
        result.cpu_cycles_histogram[static_cast<double>(cpu_cycles_spent)] += 1;
    }
    result.profiled_calls += repeat.count();
    result.profiled_seconds = repeat.seconds();

    return result;
}

/**
 *  @brief Loops over all tokens of the environment in loop-unrolled batches, applying the given
 *      @b unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] baseline Optional serial analog to stress-test the accelerated function against.
 *  @param[in] callable Unary function from a @c std::size_t token index to a @b call_result_t.
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
    std::string_view name,     //
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
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value) {
        repeat_up_to_t stress = repeat_up_to(env.stress ? env.stress_seconds : 0.0);
        for (auto call_index : stress) {
            std::size_t const token_index = call_index & lookup_mask;
            call_result_t const accelerated_result = callable(token_index);
            call_result_t const baseline_result = baseline(token_index);
            ++result.stress_calls;
            result.stress_inputs += accelerated_result.inputs_processed;
            if (check_validator(accelerated_result.check_value, baseline_result.check_value)) continue; // No failures

            // If we got here, the error needs to be reported and investigated.
            ++result.errors;
            if (result.errors > env.stress_limit) {
                fmt::println("Too many errors in {} after {:.3f} seconds. Stopping the test.", name, stress.seconds());
                std::terminate();
            }
            log_failure(env, name, baseline_result.check_value, accelerated_result.check_value, token_index);
        }
    }

    // Run once to estimate the per-call duration and size the unrolled loop below. This call pays cold-cache
    // and branch-predictor costs a steady-state call does not, so it stays out of the reported statistics.
    call_result_t warm_up_result;
    std::uint64_t warm_up_cpu_cycles = 0;
    auto const first_call_duration = seconds_per_call([&] {
        std::uint64_t cpu_cycles_at_start = cpu_cycle_counter();
        warm_up_result = callable((std::size_t)0); //? Use the first token
        std::uint64_t cpu_cycles_at_end = cpu_cycle_counter();
        warm_up_cpu_cycles = cpu_cycles_at_end - cpu_cycles_at_start;
    });
    if (first_call_duration >= env.benchmark_seconds) {
        // No budget remains for a second, uncontaminated sample, so the cold warm-up call is reported as-is.
        result.operations += warm_up_result.operations;
        result.bytes_passed += warm_up_result.bytes_passed;
        result.profiled_inputs += warm_up_result.inputs_processed;
        result.profiled_calls += 1;
        result.profiled_cpu_cycles += warm_up_cpu_cycles;
        result.profiled_cpu_cycles_min = warm_up_cpu_cycles;
        result.cpu_cycles_histogram[static_cast<double>(warm_up_cpu_cycles)] += 1;
        result.profiled_seconds = first_call_duration;
        return result;
    }

    // Repeat the benchmarks in unrolled batches of `unroll_factor` until the time limit is reached.
    constexpr std::size_t unroll_factor = 8;
    repeat_up_to_t repeat = repeat_up_to(env.benchmark_seconds - first_call_duration);
    for (auto batch_index : repeat) {
        std::size_t const batch_start = batch_index * unroll_factor;
        std::uint64_t t0 = cpu_cycle_counter();
        call_result_t r0 = callable((batch_start + 0) & lookup_mask);
        call_result_t r1 = callable((batch_start + 1) & lookup_mask);
        call_result_t r2 = callable((batch_start + 2) & lookup_mask);
        call_result_t r3 = callable((batch_start + 3) & lookup_mask);
        call_result_t r4 = callable((batch_start + 4) & lookup_mask);
        call_result_t r5 = callable((batch_start + 5) & lookup_mask);
        call_result_t r6 = callable((batch_start + 6) & lookup_mask);
        call_result_t r7 = callable((batch_start + 7) & lookup_mask);
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

        std::uint64_t const batch_cpu_cycles = t7 - t0;
        result.profiled_cpu_cycles += batch_cpu_cycles;
        result.profiled_cpu_cycles_min = std::min(result.profiled_cpu_cycles_min, batch_cpu_cycles / unroll_factor);
        result.cpu_cycles_histogram[static_cast<double>(batch_cpu_cycles)] += unroll_factor;
    }
    result.profiled_calls += repeat.count() * unroll_factor;
    result.profiled_seconds = repeat.seconds();
    return result;
}

/**
 *  @brief Loops over all tokens of the environment in loop-unrolled batches, applying the given
 *      @b nullary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] callable Nullary function taking no arguments and returning a @b call_result_t.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_>
bench_result_t bench_nullary(environment_t const &env, std::string_view name, callable_type_ &&callable) {
    return bench_nullary(env, name, callable_no_op_t {}, callable);
}

/**
 *  @brief Loops over all tokens of the environment in loop-unrolled batches, applying the given
 *      @b unary function.
 *  @param[in] env Environment with the dataset and tokens.
 *  @param[in] name Name of the benchmark, used for logging.
 *  @param[in] callable Unary function from a @c std::size_t token index to a @b call_result_t.
 *  @return Profiling results, including the number of cycles, bytes processed, and error counts.
 */
template <typename callable_type_>
bench_result_t bench_unary(environment_t const &env, std::string_view name, callable_type_ &&callable) {
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
                fmt::println("Mismatch at index {}", i);
                return false;
            }
        return true;
    }
};

} // namespace ashvardanian::stringzilla::bench
