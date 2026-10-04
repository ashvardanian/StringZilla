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
 *  3. Integrate with Linux @b perf and other tools for more detailed analysis.
 *
 *  We can isolate the relevant pieces of code, excluding the preprocessing costs from the actual
 *  workload. We can also track individual hardware counters, including platform-specific
 *  @c PERF_TYPE_RAW ones, that are not handled by most tools.
 *
 *  4. Visualize results differently, with compact output for generic workloads and special cases.
 *
 *  Environment variables, read once by @c read_settings. The @b STRINGWARS_* ones mean the same in
 *  StringWars, and the @b STRINGZILLA_STRESS* ones only StringZilla has:
 *
 *  @verbatim
 *  Variable                       Default      Meaning
 *  STRINGWARS_DATASET             per corpus   Corpus file, instead of "leipzig1M.txt" or "xlsum.csv"
 *  STRINGWARS_TOKENS              per corpus   "file", "lines", "words", or an N-gram length like 64
 *  STRINGWARS_BYTES               whole file   Bytes read from the corpus, like 4096, 64KB or 1GB; UTF-8 families read 64MB
 *  STRINGWARS_UNIQUE              0            Sorts the tokens and drops duplicates first
 *  STRINGWARS_FILTER              none         ECMAScript regex over benchmark names, or a substring when it does not compile
 *  STRINGWARS_SEED                42           Seed for every random generator, or "random" to draw one, which a run prints
 *  STRINGWARS_WARMUP              1s           Untimed run before each benchmark, like 200ms or 1s
 *  STRINGWARS_TIME_LIMIT          10s          Timed run of each benchmark, like 200ms or 10s; 1s in debug builds
 *  STRINGWARS_BATCH_PER_CORE      per backend  Candidates per call per CPU core, CUDA or ROCm multiprocessor, or Metal GPU core
 *  STRINGZILLA_STRESS             1            Checks each backend against its baseline before timing
 *  STRINGZILLA_STRESS_TIME_LIMIT  time limit   Time of each stress check, like 200ms or 10s
 *  STRINGZILLA_STRESS_DIR         .tmp         Directory for stress-check failure logs
 *  STRINGZILLA_STRESS_LIMIT       2            The stress-check failure that stops the run, so 1 stops at the first
 *  @endverbatim
 */
#pragma once
#include <cctype>  // `std::isalnum`
#include <clocale> // `std::setlocale`
#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`
#include <cstdio>  // `std::fopen`, `std::fclose`, `std::FILE`
#include <cstdint> // `std::uint32_t`
#include <cstdlib> // `std::exit`, `std::getenv`
#include <cstring> // `std::memcpy`, `std::strcmp`, `std::strlen`

#include <algorithm>
#include <array>        // `std::array`
#include <bit>          // `std::bit_floor`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <concepts>     // `std::predicate`
#include <exception>    // `std::invalid_argument`
#include <filesystem>   // `std::filesystem::create_directories`
#include <functional>   // `std::equal_to`
#include <limits>       // `std::numeric_limits`
#include <map>          // `std::map`
#include <new>          // `std::bad_alloc`
#include <numeric>      // `std::accumulate`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`, `std::mt19937`
#include <regex>        // `std::regex`, `std::regex_search`
#include <span>         // `std::span`, `std::as_bytes`
#include <string>       // `std::hash`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <thread>       // `std::this_thread::sleep_for`, `std::thread::hardware_concurrency`
#include <tuple>        // `std::tuple`, for `call_best`
#include <type_traits>  // `std::remove_reference`
#include <utility>      // `std::index_sequence`
#include <vector>       // `std::vector`

#if defined(_MSC_VER)
#include <intrin.h> // `_ReadWriteBarrier`
#endif
#if defined(_WIN32)
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h> // `std::byte`

#include "stringzilla/metal.h" // The Metal layer the Metal benchmarks drive
#include "stringzilla/stringzilla.h"
#include "stringzilla/types.hpp"
#if !STRINGZILLA_HEADER_ONLY
#include "stringzilla/stringzilla.hpp" // `sz::string_view_t::split`, which `tokenize` scans with
#endif

namespace sz = ashvardanian::stringzilla;

namespace ashvardanian::stringzilla::bench {

/** A 32-bit generator seed, kept apart from counts so neither passes for the other. */
struct seed_t {
    std::uint32_t value = 0;
};

/** A positive thread count: "0" in the environment resolves to every core this process may use. */
struct threads_t {
    std::size_t count = 0;
};

/** A size in bytes, kept apart from element counts. */
struct bytes_t {
    std::size_t value = 0;
};

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> env_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Parses the environment variable @p name with @p parse, or returns @p fallback when it is unset
 *  or empty. Text that does not parse prints `NAME="text" does not parse, expected <expected>` and
 *  exits with status 1, which leaves crash handlers quiet. */
template <typename value_type_, typename parse_type_>
[[nodiscard]] value_type_ env_parsed(char const *name, value_type_ fallback, parse_type_ &&parse,
                                     char const *expected) noexcept {
    std::optional<std::string_view> const text = env_text(name);
    if (!text) return fallback;
    if (std::optional<value_type_> value = parse(*text)) return *std::move(value);
    fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", name, *text, expected);
    std::exit(1);
}

/** A positive whole number, like "64". */
inline std::optional<std::size_t> parse_count(std::string_view text) noexcept {
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || end != text.data() + text.size() || !count) return std::nullopt;
    return count;
}

/** A positive duration in whole milliseconds or seconds, like "200ms" or "10s". */
inline std::optional<std::chrono::milliseconds> parse_duration(std::string_view text) noexcept {
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || !count) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    if (unit == "ms") return std::chrono::milliseconds(count);
    if (unit == "s") return std::chrono::seconds(count);
    return std::nullopt;
}

/** A positive size in whole bytes or binary units of any case, like "4096", "64KB" or "1GB". */
inline std::optional<bytes_t> parse_size(std::string_view text) noexcept {
    std::size_t bytes = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
    if (error != std::errc {} || !bytes) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    auto const same_letter = [](char given, char lower) noexcept {
        return std::tolower(static_cast<unsigned char>(given)) == lower;
    };
    for (std::string_view const known : std::array<std::string_view, 5> {"", "kb", "mb", "gb", "tb"}) {
        if (std::equal(unit.begin(), unit.end(), known.begin(), known.end(), same_letter)) return bytes_t {bytes};
        if (bytes > std::numeric_limits<std::size_t>::max() / 1024) return std::nullopt;
        bytes *= 1024;
    }
    return std::nullopt;
}

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<seed_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return seed_t {static_cast<std::uint32_t>(std::random_device {}())};
    std::uint32_t seed = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed_t {seed};
}

inline std::size_t env_count(char const *name, std::size_t fallback) noexcept {
    return env_parsed(name, fallback, parse_count, "a positive count");
}

inline std::chrono::milliseconds env_duration(char const *name, std::chrono::milliseconds fallback) noexcept {
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s");
}

inline bytes_t env_size(char const *name, bytes_t fallback) noexcept {
    return env_parsed(name, fallback, parse_size, "a size like 4096, 64KB or 1GB");
}

inline bool env_flag(char const *name, bool fallback) noexcept {
    auto const parse_flag = [](std::string_view text) noexcept -> std::optional<bool> {
        if (text == "1" || text == "true") return true;
        if (text == "0" || text == "false") return false;
        return std::nullopt;
    };
    return env_parsed(name, fallback, parse_flag, "0, 1, true or false");
}

inline seed_t env_seed(char const *name, seed_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
}

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** Spells @p size as a user types it, in the largest binary unit dividing it, like "256MB". */
inline std::string spell_size(bytes_t size) {
    constexpr std::array<char const *, 5> units {"", "KB", "MB", "GB", "TB"};
    std::size_t bytes = size.value, power = 0;
    while (power + 1 != units.size() && bytes && bytes % 1024 == 0) bytes /= 1024, ++power;
    return std::to_string(bytes) + units[power];
}

/** Calls the dispatch point @p best_ over @p capabilities with the arguments of its capability
 *  kernels, whose last one, the stream, follows the mask. */
template <auto best_, typename... arguments_types_>
sz_status_t call_best(sz_capability_t capabilities, arguments_types_... arguments) noexcept {
    std::tuple<arguments_types_...> const tuple {arguments...};
    return [&]<std::size_t... indices_>(std::index_sequence<indices_...>) {
        return best_(std::get<indices_>(tuple)..., capabilities, std::get<sizeof...(indices_)>(tuple));
    }(std::make_index_sequence<sizeof...(arguments_types_) - 1> {});
}

/** The dispatch point @p best_ in the shape of its capability kernels, over the CPU capabilities
 *  this process enables: callable like them, and convertible to their function pointers. */
template <auto best_>
inline constexpr auto cpu_best =
    [](auto... arguments) noexcept { return call_best<best_>(default_capabilities(), arguments...); };

/*  The one place a benchmark picks a GPU vendor's runtime, by its compiler: the baseline and the
 *  helpers the rows reach past the library. */
#if STRINGZILLA_ARCH_ROCM_
inline constexpr sz_capability_t gpu_baseline_k = sz_cap_rocm_k;
inline constexpr auto gpu_multiprocessors = &sz_device_multiprocessors_rocm_;
inline constexpr auto gpu_threads_per_multiprocessor = &sz_device_threads_per_multiprocessor_rocm_;
inline constexpr auto gpu_free_bytes = &sz_device_free_bytes_rocm_;
inline constexpr auto gpu_copy = &sz_copy_rocm_;
#elif STRINGZILLA_ARCH_CUDA_
inline constexpr sz_capability_t gpu_baseline_k = sz_cap_cuda_k;
inline constexpr auto gpu_multiprocessors = &sz_device_multiprocessors_cuda_;
inline constexpr auto gpu_threads_per_multiprocessor = &sz_device_threads_per_multiprocessor_cuda_;
inline constexpr auto gpu_free_bytes = &sz_device_free_bytes_cuda_;
inline constexpr auto gpu_copy = &sz_copy_cuda_;
#endif

#if !STRINGZILLA_ARCH_CUDA_ && !STRINGZILLA_ARCH_ROCM_
template <typename value_type_>
using unified_vector = std::vector<value_type_, std::allocator<value_type_>>;
#else
template <typename value_type_>
using unified_vector = std::vector<value_type_, unified_alloc<value_type_, gpu_baseline_k>>;

/**
 *  @brief Allocator over plain @b device memory, which no host code may dereference.
 *
 *  For scratch that only a kernel ever reads or writes, where unified memory would pay page
 *  migration on every access from the wrong side. @ref vector is the only container that grows
 *  it, through @c resize_uninitialized, because moving elements on the host is exactly what
 *  @c host_accessible_k forbids.
 */
template <typename value_type_>
struct device_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;

    /** Plain device memory: a container must not move elements through it on the host to grow. */
    static constexpr bool host_accessible_k = false;

    template <typename other_value_type_>
    struct rebind {
        using other = device_alloc<other_value_type_>;
    };

    constexpr device_alloc() noexcept = default;
    template <typename other_value_type_>
    constexpr device_alloc(device_alloc<other_value_type_> const &) noexcept {}

    value_type *allocate(size_type count) const noexcept {
        if (count > (std::numeric_limits<size_type>::max)() / sizeof(value_type)) return nullptr;
        sz_allocator_t allocator;
        if (sz_allocator_init_device_best(&allocator, gpu_baseline_k) != sz_success_k) return nullptr;
        pointer result = static_cast<pointer>(
            allocator.allocate(count * sizeof(value_type), allocator.handle, nullptr));
        if (!result || reinterpret_cast<sz_size_t>(result) % alignof(value_type) == 0) return result;
        allocator.free(result, count * sizeof(value_type), allocator.handle, nullptr);
        return nullptr;
    }
    void deallocate(pointer start, size_type count) const noexcept {
        sz_allocator_t allocator;
        if (sz_allocator_init_device_best(&allocator, gpu_baseline_k) != sz_success_k) return;
        allocator.free(start, count * sizeof(value_type), allocator.handle, nullptr);
    }
    template <typename other_type_>
    bool operator==(device_alloc<other_type_> const &) const noexcept {
        return true;
    }
    template <typename other_type_>
    bool operator!=(device_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

/** Page-locked host storage for transfers to and from the device. */
template <typename value_type_>
struct pinned_alloc {
    using value_type = value_type_;
    using pointer = value_type *;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;

    template <typename other_value_type_>
    struct rebind {
        using other = pinned_alloc<other_value_type_>;
    };

    constexpr pinned_alloc() noexcept = default;
    template <typename other_value_type_>
    constexpr pinned_alloc(pinned_alloc<other_value_type_> const &) noexcept {}

    value_type *allocate(size_type count) const {
        sz_allocator_t allocator;
        pointer result = nullptr;
        count = count ? count : 1;
        if (count > (std::numeric_limits<size_type>::max)() / sizeof(value_type)) goto failed;
        if (sz_allocator_init_pinned_best(&allocator, gpu_baseline_k) != sz_success_k) goto failed;
        result = static_cast<pointer>(allocator.allocate(count * sizeof(value_type), allocator.handle, nullptr));
        if (!result) goto failed;
        if (reinterpret_cast<sz_size_t>(result) % alignof(value_type) == 0) return result;
        allocator.free(result, count * sizeof(value_type), allocator.handle, nullptr);
    failed:
        throw std::bad_alloc();
    }
    void deallocate(pointer start, size_type count) const noexcept {
        sz_allocator_t allocator;
        if (sz_allocator_init_pinned_best(&allocator, gpu_baseline_k) != sz_success_k) return;
        allocator.free(start, (count ? count : 1) * sizeof(value_type), allocator.handle, nullptr);
    }
    template <typename other_type_>
    bool operator==(pinned_alloc<other_type_> const &) const noexcept {
        return true;
    }
    template <typename other_type_>
    bool operator!=(pinned_alloc<other_type_> const &) const noexcept {
        return false;
    }
};

/** Page-locked host memory, which the driver reports as host and every engine refuses. */
template <typename value_type_>
using pinned_vector = std::vector<value_type_, pinned_alloc<value_type_>>;

/**
 *  @brief Plain device memory a kernel can write and the host cannot touch.
 *
 *  @c vector is what the engines already store device-resident scratch in, and its
 *  @c try_resize_uninitialized is the only growth a non-host-accessible allocator admits.
 */
template <typename value_type_>
using device_vector = vector<value_type_, device_alloc<value_type_>>;

/**
 *  @brief Queues a copy of a device-resident buffer into @p destination on the default stream,
 *      which the caller joins.
 *  @param[out] destination At least as many elements as @p source holds; only that prefix is set.
 */
template <typename value_type_>
inline sz_status_t copy_device_to_host(device_vector<value_type_> const &source, std::span<value_type_> destination) {
    if (source.size() == 0) return sz_success_k;
    if (destination.size() < source.size()) return sz_unexpected_dimensions_k;
    return gpu_copy(destination.data(), source.data(), source.size() * sizeof(value_type_), nullptr);
}
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

/** Reads a file into a string via LibC @c <cstdio>. A @p read_limit stops the read after that many
 *  bytes, so the file tail is never touched. */
inline std::string read_file(std::string const &path, std::optional<bytes_t> read_limit) noexcept(false) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Failed to open file: " + path);
    std::size_t capacity = read_limit ? read_limit->value : 0;
    if (!read_limit) {
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
    return n == 0 ? multiple_ : sz::round_up_to_multiple(n, multiple_);
}

using check_value_t = std::uint64_t;

struct call_result_t {

    /** Number of input bytes processed. */
    std::size_t bytes_passed = 0;

    /** Some value used to compare execution result between the baseline and accelerated backend. */
    check_value_t check_value = 0;

    /** For operations with non-linear complexity, the throughput should be measured differently. */
    std::size_t operations_count = 0;

    call_result_t() = default;
    call_result_t(std::size_t bytes_passed, check_value_t check_value = 0, std::size_t operations_count = 0)
        : bytes_passed(bytes_passed), check_value(check_value), operations_count(operations_count) {}
};

struct callable_no_op_t {
    call_result_t operator()(std::size_t) const { return {}; }
};

/** Keeps @p value, and every write before it, from being optimized away. */
template <typename value_type_>
inline void do_not_optimize(value_type_ &&value) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    [[maybe_unused]] auto volatile *pointer = &value;
    _ReadWriteBarrier();
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** How @c print shows a counter: as is, or per second in decimal units or in binary bytes. */
enum class counter_kind_t : unsigned char { value_k, decimal_rate_k, byte_rate_k };

/** A named amount per call, shown as its @c kind says. */
struct counter_t {
    char const *name = nullptr;
    double value = 0;
    counter_kind_t kind = counter_kind_t::value_k;
};

/** The width every row pads its name to, so the counters line up in one column. */
inline constexpr std::size_t row_name_width_k = 56;

/** A finished benchmark: its name, calls per second and counters, or why it was skipped. The name
 *  is owned, since @c environment_t::serial_rows keeps rows past the call. */
struct row_t {
    std::string name;
    double calls_per_second = 0;
    std::array<counter_t, 4> counters {};
    std::optional<std::string_view> skip_reason;
};

/** Prints @p row on one line: byte rates in binary units like "GB/s", other rates like "47.6/s" or
 *  "1.234 M/s", then the speed-up over @p baseline_calls_per_second when there is one. */
inline void print(row_t const &row, std::optional<double> baseline_calls_per_second = std::nullopt) {
    if (row.skip_reason) return fmt::println("{:<{}} skipped: {}", row.name, row_name_width_k, *row.skip_reason);
    auto const print_rate = [](char const *name, double per_second, counter_kind_t kind) {
        bool const binary = kind == counter_kind_t::byte_rate_k;
        double const step = binary ? 1024 : 1000;
        char const *prefix = "";
        for (char const *next : binary ? std::array {"K", "M", "G", "T", "P"} : std::array {"k", "M", "G", "T", "P"})
            if (per_second >= step) per_second /= step, prefix = next;
        // Significant digits rather than decimals, so a slow rate never rounds to zero
        char const *const separator = binary || *prefix ? " " : "";
        fmt::print("  {} {:.4g}{}{}{}/s", name, per_second, separator, prefix, binary ? "B" : "");
    };
    fmt::print("{:<{}}", row.name, row_name_width_k);
    print_rate("calls", row.calls_per_second, counter_kind_t::decimal_rate_k);
    for (counter_t const &counter : row.counters)
        if (counter.kind != counter_kind_t::value_k)
            print_rate(counter.name, counter.value * row.calls_per_second, counter.kind);
        else if (counter.name) fmt::print("  {} {:.4g}", counter.name, counter.value);
    if (baseline_calls_per_second) fmt::print("  {:.2f}x", row.calls_per_second / *baseline_calls_per_second);
    fmt::print("\n");
}

/** Prints @p row when the filter selected it. */
inline void print(std::optional<row_t> const &row, std::optional<double> baseline_calls_per_second = std::nullopt) {
    if (row) print(*row, baseline_calls_per_second);
}

/** The calls per second of @p row to print others against, unless filtered out or skipped. */
inline std::optional<double> baseline_of(std::optional<row_t> const &row) noexcept {
    if (!row || row->skip_reason) return std::nullopt;
    return row->calls_per_second;
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed, and the clock starts at @c begin. The loop runs untimed for
 *  the warm-up, then counts calls until the time limit. It reads the clock again only once the
 *  calls so far have grown by a 64th, so a short call doesn't time the clock itself.
 */
class loop_t {
    using steady_clock_t = std::chrono::steady_clock;
    using time_point_t = steady_clock_t::time_point;
    using seconds_t = std::chrono::duration<double>;

    /** Where the loop is: untimed, timed, or finished by the time limit or a skip. */
    enum class phase_t : unsigned char {

        /** Calls run untimed until the warm-up passes. */
        warming_up_k,

        /** Calls count until the time limit passes. */
        timed_k,

        /** No call runs again. */
        done_k,
    };

    /** The calls so far, divided by this, run between two reads of the clock. */
    static constexpr std::size_t clock_stride_k = 64;

    std::chrono::milliseconds warmup_, time_limit_;
    time_point_t start_ {};
    steady_clock_t::duration elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0;
    phase_t phase_ = phase_t::warming_up_k;
    std::optional<std::string_view> skip_reason_;
    std::array<counter_t, 4> counters_ {};

    bool keep_running() noexcept {
        if (phase_ == phase_t::done_k) return false;
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (phase_ == phase_t::warming_up_k && elapsed_ >= warmup_)
            phase_ = phase_t::timed_k, start_ = steady_clock_t::now(), elapsed_ = {}, calls_ = 0;
        else if (phase_ == phase_t::timed_k && elapsed_ >= time_limit_) return phase_ = phase_t::done_k, false;
        next_check_ = calls_ + calls_ / clock_stride_k + 1;
        return true;
    }

    void add(counter_t counter) noexcept {
        for (counter_t &slot : counters_)
            if (!slot.name) return void(slot = counter);
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t *loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop and reports @p reason instead of results. */
    void skip(std::string_view reason) noexcept { skip_reason_ = reason, phase_ = phase_t::done_k; }

    /** Reports @p per_call of @p name per second, in decimal units. */
    void rate(char const *name, double per_call) noexcept { add({name, per_call, counter_kind_t::decimal_rate_k}); }

    /** Reports @p bytes_per_call per second, in binary units. */
    void byte_rate(double bytes_per_call) noexcept { add({"bytes", bytes_per_call, counter_kind_t::byte_rate_k}); }

    /** The finished benchmark under @p name. */
    row_t row(std::string_view name) const {
        if (skip_reason_) return {std::string(name), 0, {}, skip_reason_};
        return {std::string(name), calls_ / seconds_t(elapsed_).count(), counters_, std::nullopt};
    }
};

/** The smallest read that still exercises the control-flow paths of a compute-bound bench on the
 *  multilingual corpus. Memory-bound benches ignore it and read the whole file. */
inline constexpr bytes_t compute_bound_slice_k {64ull << 20};

#if !STRINGZILLA_ARCH_CUDA_ && !STRINGZILLA_ARCH_ROCM_
using dataset_t = std::string;
using token_view_t = std::string_view;
using tokens_t = std::vector<token_view_t>;
#else
using dataset_t = std::basic_string<char, std::char_traits<char>, unified_alloc<char, gpu_baseline_k>>;
using token_view_t = std::string_view;
using tokens_t = std::vector<token_view_t, unified_alloc<token_view_t, gpu_baseline_k>>;
#endif

/**
 *  @brief Tokenizes a string with the given separator predicate.
 *  @see For faster ways to tokenize a string with STL: https://ashvardanian.com/posts/splitting-strings-cpp/
 */
template <std::predicate<char> is_separator_callback_type_>
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
 *  @brief Tokenizes a string around any of the @p separators bytes in one lazy SIMD pass.
 *
 *  Each step of the underlying @c split issues one @c sz_find_byteset_best scan. The whole corpus
 *  is already bounded by the dataset read, so the walk runs to the end without a cap of its own.
 *  Header-only builds have no dispatch point to split with, so they test every byte instead.
 */
inline tokens_t tokenize(std::string_view str, std::string_view separators) {
#if STRINGZILLA_HEADER_ONLY
    return tokenize(str, [&](char c) { return separators.find(c) != std::string_view::npos; });
#else
    tokens_t tokens;
    sz::byteset_t const separators_set(separators.data(), separators.size());
    for (auto token : sz::string_view_t {str.data(), str.size()}.split(separators_set)) {
        if (token.size() == 0) continue; // ? Runs of separators yield empty segments
        tokens.push_back({token.data(), token.size()});
    }
    return tokens;
#endif
}

/** Splits a string into words around newlines, tabs, and other ASCII whitespaces. */
inline tokens_t tokenize(std::string_view str) { return tokenize(str, " \t\n\r\f\v"); }

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

/** How a corpus splits its dataset into tokens. */
enum class tokenization_t : unsigned char { file_k, lines_k, words_k, ngrams_k };

/** How @c fmt spells a tokenization: @c file, @c lines, @c words, or @c ngrams. */
inline std::string_view format_as(tokenization_t tokenization) noexcept {
    std::string_view constexpr names[] = {"file", "lines", "words", "ngrams"};
    return names[static_cast<unsigned char>(tokenization)];
}

/** The tokenization @c format_as spells as @p text. */
inline std::optional<tokenization_t> parse_tokenization(std::string_view text) noexcept {
    if (text == "file") return tokenization_t::file_k;
    if (text == "lines") return tokenization_t::lines_k;
    if (text == "words") return tokenization_t::words_k;
    if (text == "ngrams") return tokenization_t::ngrams_k;
    return std::nullopt;
}

/**
 *  @brief One corpus the benchmarks run on: the dataset file, loaded and split into @c tokens.
 *
 *  The tokens keep their file order, so every language's harness sees the same inputs; the seed
 *  only seeds the random generators that sample from them.
 */
struct corpus_t {

    /** Absolute path of the textual input file on disk. */
    std::string path;

    /** Tokenization mode to convert the @c dataset to @c tokens. */
    tokenization_t tokenization = tokenization_t::words_k;

    /** Token length in bytes that @c ngrams_k keeps. */
    std::size_t ngram_bytes = 0;

    /** Textual content of the dataset file, fully loaded into memory. */
    dataset_t dataset;

    /** Array of tokens extracted from the @c dataset. */
    tokens_t tokens;

    std::string_view operator[](std::size_t i) const {
        if (i >= tokens.size()) throw std::out_of_range("Index out of range");
        return {tokens[i].data(), tokens[i].size()};
    }
};

/** Whether a corpus keeps its duplicate tokens or sorts them out. */
enum class duplicates_t : bool { keep_k, drop_k };

/** How @c fmt spells a duplicates policy: @c keep or @c drop. */
inline std::string_view format_as(duplicates_t duplicates) noexcept {
    return duplicates == duplicates_t::drop_k ? "drop" : "keep";
}

/** The duplicates policy @c format_as spells as @p text. */
inline std::optional<duplicates_t> parse_duplicates(std::string_view text) noexcept {
    if (text == "keep") return duplicates_t::keep_k;
    if (text == "drop") return duplicates_t::drop_k;
    return std::nullopt;
}

/** Whether backends are checked against baselines before timing. */
enum class stress_t : bool { skip_k, check_k };

/** How @c fmt spells a stress policy: @c skip or @c check. */
inline std::string_view format_as(stress_t stress) noexcept { return stress == stress_t::check_k ? "check" : "skip"; }

/** The stress policy @c format_as spells as @p text. */
inline std::optional<stress_t> parse_stress(std::string_view text) noexcept {
    if (text == "skip") return stress_t::skip_k;
    if (text == "check") return stress_t::check_k;
    return std::nullopt;
}

/** Every benchmark setting, its default as the initializer, filled once by @c read_settings. An
 *  unset optional leaves the choice to each corpus or backend. */
struct settings_t {

    /** Corpus file, instead of each corpus' own. */
    std::optional<std::string_view> dataset;

    /** Tokenization, instead of each corpus' own. */
    std::optional<tokenization_t> tokenization;

    /** Token length in bytes when @c tokenization is @c ngrams_k. */
    std::size_t ngram_bytes = 0;

    /** Bytes read from the corpus file, else each corpus' own. */
    std::optional<bytes_t> read_limit;

    /** Whether duplicate tokens are kept or sorted out. */
    duplicates_t duplicates = duplicates_t::keep_k;

    /** Benchmarks to run, by ECMAScript regex or else substring. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** Seed for every random generator. */
    seed_t seed {42};

    /** Untimed run ahead of each benchmark. */
    std::chrono::milliseconds warmup = std::chrono::seconds(1);

    /** Timed run of each benchmark. */
    std::chrono::milliseconds time_limit = std::chrono::seconds(STRINGZILLA_DEBUG ? 1 : 10);

    /** Candidates per call per core, else each backend's own. */
    std::optional<std::size_t> candidates_per_core;

    /** Whether backends are checked against baselines before timing. */
    stress_t stress = stress_t::check_k;

    /** Time of each stress check. */
    std::chrono::milliseconds stress_time_limit = time_limit;

    /** Directory for stress-check failure logs. */
    std::string_view stress_log_dir = ".tmp";

    /** The stress-check failure that stops the run: 1 stops at the first. */
    std::size_t stress_failures_to_stop = 2;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Reads every @c settings_t variable; a stress time limit left unset follows the time limit. */
inline settings_t read_settings() noexcept {
    settings_t settings;
    auto const parse_tokens = [&](std::string_view text) noexcept -> std::optional<tokenization_t> {
        if (std::optional<std::size_t> const length = parse_count(text))
            return settings.ngram_bytes = *length, tokenization_t::ngrams_k;
        std::optional<tokenization_t> const tokenization = parse_tokenization(text);
        if (tokenization == tokenization_t::ngrams_k) return std::nullopt;
        return tokenization;
    };
    settings.dataset = env_text("STRINGWARS_DATASET");
    if (env_text("STRINGWARS_TOKENS"))
        settings.tokenization = env_parsed("STRINGWARS_TOKENS", tokenization_t::words_k, parse_tokens,
                                           "file, lines, words or an N-gram length");
    if (env_text("STRINGWARS_BYTES")) settings.read_limit = env_size("STRINGWARS_BYTES", {});
    settings.duplicates = duplicates_t(env_flag("STRINGWARS_UNIQUE", settings.duplicates == duplicates_t::drop_k));
    settings.filter = env_text("STRINGWARS_FILTER").value_or("");
    if (!settings.filter.empty()) {
#if defined(__cpp_exceptions) && __cpp_exceptions
        try {
            settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
        }
        catch (std::regex_error const &) {
        }
#else
        settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
#endif
    }
    settings.seed = env_seed("STRINGWARS_SEED", settings.seed);
    settings.warmup = env_duration("STRINGWARS_WARMUP", settings.warmup);
    settings.time_limit = env_duration("STRINGWARS_TIME_LIMIT", settings.time_limit);
    if (env_text("STRINGWARS_BATCH_PER_CORE")) settings.candidates_per_core = env_count("STRINGWARS_BATCH_PER_CORE", 1);
    settings.stress = stress_t(env_flag("STRINGZILLA_STRESS", settings.stress == stress_t::check_k));
    settings.stress_time_limit = env_duration("STRINGZILLA_STRESS_TIME_LIMIT", settings.time_limit);
    settings.stress_log_dir = env_text("STRINGZILLA_STRESS_DIR").value_or(settings.stress_log_dir);
    settings.stress_failures_to_stop = env_count("STRINGZILLA_STRESS_LIMIT", settings.stress_failures_to_stop);
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const &settings) {
    fmt::println("- Seed: {}", settings.seed.value);
    fmt::println("- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    fmt::println("- Warm-up: {}", spell_duration(settings.warmup));
    fmt::println("- Time limit: {}", spell_duration(settings.time_limit));
    fmt::println("- Bytes: {}", settings.read_limit ? spell_size(*settings.read_limit) : std::string("per corpus"));
    fmt::println("- Batch per core: {}", settings.candidates_per_core ? std::to_string(*settings.candidates_per_core)
                                                                      : std::string("per backend"));
    fmt::println("- Dataset: {}", settings.dataset.value_or("per corpus"));
    fmt::println("- Tokens: {}", !settings.tokenization ? std::string("per corpus")
                                 : *settings.tokenization == tokenization_t::ngrams_k
                                     ? std::to_string(settings.ngram_bytes)
                                     : fmt::format("{}", *settings.tokenization));
    fmt::println("- Unique: {}", settings.duplicates == duplicates_t::drop_k);
    fmt::println("- Stress: {}", settings.stress == stress_t::check_k);
    fmt::println("- Stress time limit: {}", spell_duration(settings.stress_time_limit));
    fmt::println("- Stress dir: {}", settings.stress_log_dir);
    fmt::println("- Stress limit: {}", settings.stress_failures_to_stop);
}

/** The facts this binary and this machine report: the library version, the capabilities compiled
 *  in and detected, the cache geometry that cache-resident benchmark shapes are sized from, and in
 *  GPU builds the first visible device. */
struct machine_t {
    std::array<unsigned, 3> version {STRINGZILLA_H_VERSION_MAJOR, STRINGZILLA_H_VERSION_MINOR,
                                     STRINGZILLA_H_VERSION_PATCH};
    sz_capability_t compiled = 0;
    sz_capability_t detected = 0;
    sz::cpu_specs_t specs;
#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_ || STRINGZILLA_WITH_METAL

    /** The first visible device, with the architecture CUDA and ROCm name, like @c sm_90 or
     *  @c gfx942, or empty. */
    std::string device_name;
#endif
};

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    sz_capabilities_compiled_cpu(&machine.compiled), sz_capabilities_detected_cpu(&machine.detected);
#if STRINGZILLA_ARCH_ROCM_
    int device_count = 0;
    hipDeviceProp_t properties;
    if (hipGetDeviceCount(&device_count) == hipSuccess && device_count != 0 &&
        hipGetDeviceProperties(&properties, 0) == hipSuccess)
        machine.device_name = fmt::format("{} {}", properties.name, properties.gcnArchName);
#elif STRINGZILLA_ARCH_CUDA_
    int device_count = 0;
    cudaDeviceProp properties;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count != 0 &&
        cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
        machine.device_name = fmt::format("{} sm_{}{}", properties.name, properties.major, properties.minor);
#elif STRINGZILLA_WITH_METAL
    sz_stream_t queue = nullptr;
    if (sz_stream_init_metal(0, &queue) == sz_success_k) {
        void *(*const message)(void *, SEL) = reinterpret_cast<void *(*)(void *, SEL)>(objc_msgSend);
        void *const name = message(message(queue, sel_registerName("device")), sel_registerName("name"));
        machine.device_name = static_cast<char const *>(message(name, sel_registerName("UTF8String")));
        sz_stream_free_metal(queue);
    }
#endif
    return machine;
}

/** Prints the version line, then "- Compiled for:", "- This machine:", "- Caches:", and in GPU
 *  builds "- CUDA:", "- ROCm:" or "- Metal:". */
inline void print(machine_t const &machine) {
    char compiled[STRINGZILLA_CAPABILITIES_NAME_CAPACITY], detected[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
    sz_capabilities_name(machine.compiled, compiled, sizeof(compiled));
    sz_capabilities_name(machine.detected, detected, sizeof(detected));
    fmt::println("StringZilla {}.{}.{}", machine.version[0], machine.version[1], machine.version[2]);
    fmt::println("- Compiled for: {}", compiled);
    fmt::println("- This machine: {}", detected);
    fmt::println("- Caches: {} first-level, assumed, and {} confined to a compute domain",
                 spell_size({machine.specs.l1_bytes}), spell_size({machine.specs.l3_bytes}));
#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_ || STRINGZILLA_WITH_METAL
    fmt::println("- {}: {}", STRINGZILLA_WITH_METAL ? "Metal" : (STRINGZILLA_ARCH_ROCM_ ? "ROCm" : "CUDA"),
                 machine.device_name.empty() ? std::string_view("no device") : machine.device_name);
#endif
}

/** Loads one corpus: @p default_dataset split by @p default_tokenization and cut at
 *  @p default_read_limit, or read whole without one, unless the @p settings override them. */
inline corpus_t build_corpus(settings_t const &settings, char const *default_dataset,
                             tokenization_t default_tokenization,
                             std::optional<bytes_t> default_read_limit) noexcept(false) {
    corpus_t corpus {std::string(settings.dataset.value_or(default_dataset)),
                     settings.tokenization.value_or(default_tokenization),
                     settings.ngram_bytes,
                     {},
                     {}};
    std::optional<bytes_t> const read_limit = settings.read_limit ? settings.read_limit : default_read_limit;

    corpus.dataset = read_file(corpus.path, read_limit);
    corpus.dataset.resize(std::bit_floor(corpus.dataset.size())); // Shrink to the nearest power of two

    // Tokenize the dataset according to the tokenization mode. The corpus is already bounded by the read,
    // so each mode walks it to the end.
    if (corpus.tokenization == tokenization_t::file_k) {
        corpus.tokens.push_back({corpus.dataset.data(), corpus.dataset.size()});
    }
    else if (corpus.tokenization == tokenization_t::lines_k) { corpus.tokens = tokenize(corpus.dataset, "\n"); }
    else if (corpus.tokenization == tokenization_t::words_k) { corpus.tokens = tokenize(corpus.dataset); }
    else {
        corpus.tokens = filter_by_length<token_view_t>(tokenize(corpus.dataset), corpus.ngram_bytes,
                                                       std::equal_to<std::size_t>());
    }

    // Deduplicate tokens if requested
    if (settings.duplicates == duplicates_t::drop_k) {
        std::sort(corpus.tokens.begin(), corpus.tokens.end());
        auto last = std::unique(corpus.tokens.begin(), corpus.tokens.end());
        corpus.tokens.erase(last, corpus.tokens.end());
    }

    corpus.tokens.resize(std::bit_floor(corpus.tokens.size())); // Shrink to the nearest power of two

    auto const mean_token_length = std::accumulate(corpus.tokens.begin(), corpus.tokens.end(), (std::size_t)0u,
                                                   [](std::size_t sum, token_view_t token) -> std::size_t {
                                                       return sum + token.size();
                                                   }) *
                                   1.0 / corpus.tokens.size();

    fmt::print(
        R"(Corpus loaded:
- Dataset path: {path}
- Tokenization mode: {tokenization}
- Loaded dataset size: {dataset_bytes} bytes
- Dataset limit: {dataset_limit}
- Number of tokens: {tokens}
- Mean token length: {mean_token_length:.2f} bytes
)",
        fmt::arg("path", corpus.path), fmt::arg("tokenization", corpus.tokenization),
        fmt::arg("dataset_bytes", corpus.dataset.size()),
        fmt::arg("dataset_limit", read_limit ? spell_size(*read_limit) : std::string("whole file")),
        fmt::arg("tokens", corpus.tokens.size()), fmt::arg("mean_token_length", mean_token_length));

    return corpus;
}

/**
 *  @brief The corpora the families run on, each built by @c build_corpus on first use and kept for
 *      the rest of the run, so a family's kernels see the same tokens as its dispatch points.
 *
 *  The @b STRINGWARS_* variables override every corpus alike, so a run over one dataset loads it
 *  once per tokenization and limit rather than once per family.
 */
class corpora_t {
    settings_t const &settings_;
    std::optional<corpus_t> words_, lines_, multilingual_lines_, multilingual_slice_;

    corpus_t const &build_(std::optional<corpus_t> &corpus, char const *dataset, tokenization_t tokenization,
                           std::optional<bytes_t> read_limit) {
        if (!corpus) corpus.emplace(build_corpus(settings_, dataset, tokenization, read_limit));
        return *corpus;
    }

  public:
    explicit corpora_t(settings_t const &settings) noexcept : settings_(settings) {}

    /** English words of `leipzig1M.txt`: short tokens for search, sorting, and containers. */
    corpus_t const &words() { return build_(words_, "leipzig1M.txt", tokenization_t::words_k, std::nullopt); }

    /** English lines of `leipzig1M.txt`: longer tokens for hashing, memory, and ciphers. */
    corpus_t const &lines() { return build_(lines_, "leipzig1M.txt", tokenization_t::lines_k, std::nullopt); }

    /** Multilingual lines of the whole `xlsum.csv`: the engines' queries and candidates. */
    corpus_t const &multilingual_lines() {
        return build_(multilingual_lines_, "xlsum.csv", tokenization_t::lines_k, std::nullopt);
    }

    /** Multilingual lines of a compute-bound `xlsum.csv` slice: the UTF-8 families' text. */
    corpus_t const &multilingual_slice() {
        return build_(multilingual_slice_, "xlsum.csv", tokenization_t::lines_k, compute_bound_slice_k);
    }
};

/** Everything a benchmark reads, built once in @c main and passed down by reference. Not copyable,
 *  since @c corpora reads the @c settings beside it. */
struct environment_t {
    settings_t settings;
    machine_t machine;
    corpora_t corpora {settings};

    /** The serial kernels' rows by name, which the other capabilities' kernels print against. */
    std::map<std::string, row_t, std::less<>> serial_rows {};

    environment_t(settings_t settings, machine_t machine) noexcept
        : settings(std::move(settings)), machine(std::move(machine)) {}
    environment_t(environment_t const &) = delete;
    environment_t &operator=(environment_t const &) = delete;
};

/** Prints @p title, and returns whether this machine runs @p capabilities; otherwise prints why
 *  it skips. */
inline bool section(environment_t const &env, std::string_view title, sz_capability_t capabilities) noexcept {
    sz_capability_t const missing = capabilities & ~env.machine.detected;
    if (!missing) {
        fmt::println("\n{}:", title);
        return true;
    }
    char missing_names[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
    sz_capabilities_name(missing, missing_names, sizeof(missing_names));
    fmt::println("\n{}: skipped, {} not detected", title, missing_names);
    return false;
}

/** The median token length of @p corpus in bytes: the short query, and the typical candidate. */
inline std::size_t median_token_bytes(corpus_t const &corpus) {
    std::vector<std::size_t> lengths(corpus.tokens.size());
    std::transform(corpus.tokens.begin(), corpus.tokens.end(), lengths.begin(),
                   [](token_view_t token) { return token.size(); });
    std::nth_element(lengths.begin(), lengths.begin() + lengths.size() / 2, lengths.end());
    return lengths[lengths.size() / 2];
}

/** Candidates one call scores on its one core: @c STRINGWARS_BATCH_PER_CORE if set, else as many
 *  median tokens of @p corpus as fill the first-level cache. */
inline std::size_t candidates_per_call(environment_t const &env, corpus_t const &corpus) {
    if (env.settings.candidates_per_core) return *env.settings.candidates_per_core;
    return std::max<std::size_t>(1, env.machine.specs.l1_bytes / median_token_bytes(corpus));
}

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

/** Candidates one device call scores: @c STRINGWARS_BATCH_PER_CORE per multiprocessor if set, else
 *  one per resident thread of the bound device. */
inline std::size_t resident_candidates_per_call(environment_t const &env) {
    std::size_t const multiprocessors = gpu_multiprocessors();
    if (multiprocessors == 0) throw std::runtime_error("The device would not report its geometry.");
    std::size_t const per_core = env.settings.candidates_per_core.value_or(gpu_threads_per_multiprocessor());
    return per_core * multiprocessors;
}
#endif

/** Uses C-style file IO to save information about the most recent stress test failure. Files are
 *  written to @c STRINGZILLA_STRESS_DIR as `failed_<time>_<name>.txt`. */
inline void log_failure(                                              //
    settings_t const &settings, corpus_t const &corpus,               //
    std::string_view name,                                            //
    std::size_t expected_check_value, std::size_t actual_check_value, //
    std::optional<std::size_t> token_index) noexcept(false) {

    std::string timestamp = std::to_string(std::time(nullptr));
    // Benchmark names embed shape labels like `...:q4xc4:allpairs`; `:` and `/` are invalid in path
    // components on common filesystems, so map every non-portable character to `_` before composing the path.
    std::string safe_name {name};
    for (char &c : safe_name)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '.' || c == '_')) c = '_';
    std::string file_name = "failed_" + timestamp + "_" + safe_name + "_" + ".txt";
    std::string file_path = std::string(settings.stress_log_dir) + "/" + file_name;
    std::error_code ignored;
    std::filesystem::create_directories(settings.stress_log_dir, ignored);
    std::FILE *file = std::fopen(file_path.c_str(), "w");
    if (!file) throw std::runtime_error("Failed to open file for writing: " + file_path);

    fmt::println(file, "Dataset path: {}\nTokenization mode: {}\nSeed: {}", corpus.path, corpus.tokenization,
                 settings.seed.value);
    if (token_index)
        fmt::println(file, "Token index: {}\nToken Hex: {:02X}", *token_index,
                     fmt::join(std::as_bytes(std::span(corpus[*token_index])), " "));
    fmt::println(file, "Expected: {}\nActual: {}", expected_check_value, actual_check_value);
    std::fclose(file);
}

/** Checks @p callable against @p baseline for the stress time limit, untimed, logging each
 *  mismatch and ending the run with status 1 at the failure that stops it. A unary pair is called
 *  with the token index. */
template <typename callable_type_, typename baseline_type_, typename check_validator_type_>
void stress_check(environment_t const &env, corpus_t const &corpus, std::string_view name, callable_type_ &callable,
                  baseline_type_ &baseline, check_validator_type_ &check_validator) {
    constexpr bool unary_k = std::is_invocable_v<callable_type_ &, std::size_t>;
    std::size_t const lookup_mask = std::bit_floor(corpus.tokens.size()) - 1;
    std::size_t errors = 0, calls = 0;
    loop_t stress({}, env.settings.stress_time_limit);
    for (std::size_t call : stress) {
        std::size_t const token_index = call & lookup_mask;
        call_result_t accelerated_result, baseline_result;
        if constexpr (unary_k) accelerated_result = callable(token_index), baseline_result = baseline(token_index);
        else accelerated_result = callable(), baseline_result = baseline();
        ++calls;
        if (check_validator(accelerated_result.check_value, baseline_result.check_value)) continue;
        log_failure(env.settings, corpus, name, baseline_result.check_value, accelerated_result.check_value,
                    unary_k ? std::optional<std::size_t>(token_index) : std::nullopt);
        if (++errors >= env.settings.stress_failures_to_stop) {
            fmt::println(stderr, "{}: {} stress calls disagreed with the baseline after {} calls", name, errors, calls);
            std::exit(1);
        }
    }
    if (errors) fmt::println("{}: {} of {} stress calls disagreed with the baseline", name, errors, calls);
}

/**
 *  @brief Times a @b nullary function over the whole dataset, checked against a baseline first.
 *  @param[in] env The settings, the machine, and the corpora.
 *  @param[in] corpus The dataset and tokens the stress failures are logged with.
 *  @param[in] name Name of the benchmark, which the filter matches.
 *  @param[in] baseline Optional serial analog to stress-test the accelerated function against.
 *  @param[in] callable Nullary function taking no arguments and returning a @b call_result_t.
 *  @param[in] check_validator Optional function to validate the results of the benchmark.
 */
template <                                                        //
    typename callable_type_,                                      //
    typename baseline_type_ = callable_no_op_t,                   //
    typename preprocessing_type_ = callable_no_op_t,              //
    typename check_validator_type_ = std::equal_to<check_value_t> //
    >
std::optional<row_t> bench_nullary(                   //
    environment_t const &env, corpus_t const &corpus, //
    std::string_view name,                            //
    baseline_type_ &&baseline,                        //
    callable_type_ &&callable,                        //
    preprocessing_type_ &&preprocessing = preprocessing_type_ {},
    check_validator_type_ &&check_validator = check_validator_type_ {}) {

    if (!env.settings.selects(name)) return std::nullopt;
    if constexpr (!is_same_type<preprocessing_type_, callable_no_op_t>::value) preprocessing();
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value)
        if (env.settings.stress == stress_t::check_k)
            stress_check(env, corpus, name, callable, baseline, check_validator);

    loop_t loop(env.settings.warmup, env.settings.time_limit);
    std::size_t calls = 0, bytes = 0, operations = 0;
    for ([[maybe_unused]] std::size_t call : loop) {
        call_result_t const result = callable();
        ++calls, bytes += result.bytes_passed, operations += result.operations_count;
    }
    loop.byte_rate(double(bytes) / calls);
    if (operations) loop.rate("ops", double(operations) / calls);
    return loop.row(name);
}

/**
 *  @brief Times a @b unary function over the tokens of @p corpus, checked against a baseline first.
 *  @param[in] env The settings, the machine, and the corpora.
 *  @param[in] corpus The dataset and tokens the calls rotate over.
 *  @param[in] name Name of the benchmark, which the filter matches.
 *  @param[in] baseline Optional serial analog to stress-test the accelerated function against.
 *  @param[in] callable Unary function from a @c std::size_t token index to a @b call_result_t.
 *  @param[in] preprocessing Optional function to pre-process the data before the calls.
 *  @param[in] check_validator Optional function to validate the results of the benchmark.
 *
 *  The token index is the call index masked to the largest power of two of tokens.
 */
template <                                                        //
    typename callable_type_,                                      //
    typename baseline_type_ = callable_no_op_t,                   //
    typename preprocessing_type_ = callable_no_op_t,              //
    typename check_validator_type_ = std::equal_to<check_value_t> //
    >
std::optional<row_t> bench_unary(                     //
    environment_t const &env, corpus_t const &corpus, //
    std::string_view name,                            //
    baseline_type_ &&baseline,                        //
    callable_type_ &&callable,                        //
    preprocessing_type_ &&preprocessing = preprocessing_type_ {},
    check_validator_type_ &&check_validator = check_validator_type_ {}) {

    if (!env.settings.selects(name)) return std::nullopt;
    if constexpr (!is_same_type<preprocessing_type_, callable_no_op_t>::value) preprocessing();
    if constexpr (!is_same_type<baseline_type_, callable_no_op_t>::value)
        if (env.settings.stress == stress_t::check_k)
            stress_check(env, corpus, name, callable, baseline, check_validator);

    std::size_t const lookup_mask = std::bit_floor(corpus.tokens.size()) - 1;
    loop_t loop(env.settings.warmup, env.settings.time_limit);
    std::size_t calls = 0, bytes = 0, operations = 0;
    for (std::size_t call : loop) {
        call_result_t const result = callable(call & lookup_mask);
        ++calls, bytes += result.bytes_passed, operations += result.operations_count;
    }
    loop.byte_rate(double(bytes) / calls);
    if (operations) loop.rate("ops", double(operations) / calls);
    return loop.row(name);
}

/** Times a @b nullary function over the whole dataset, with no baseline to check it against. */
template <typename callable_type_>
std::optional<row_t> bench_nullary(environment_t const &env, corpus_t const &corpus, std::string_view name,
                                   callable_type_ &&callable) {
    return bench_nullary(env, corpus, name, callable_no_op_t {}, callable);
}

/** Times a @b unary function over the tokens of @p corpus, with no baseline to check it against. */
template <typename callable_type_>
std::optional<row_t> bench_unary(environment_t const &env, corpus_t const &corpus, std::string_view name,
                                 callable_type_ &&callable) {
    return bench_unary(env, corpus, name, callable_no_op_t {}, callable);
}

/**
 *  @brief Times the serial baseline @p name, printed against @p against when the filter keeps it.
 *  @return Its calls per second, whenever the filter keeps it or any of @p arms printed against it.
 */
template <typename callable_type_>
std::optional<double> bench_baseline(environment_t const &env, corpus_t const &corpus, std::string const &name,
                                     std::initializer_list<std::string_view> arms, callable_type_ &callable,
                                     std::optional<double> against = std::nullopt) {
    bool const printed = env.settings.selects(name);
    std::string_view timed_as = name;
    if (!printed)
        for (std::string_view const arm : arms)
            if (env.settings.selects(arm)) timed_as = arm;
    // ? Timed under a kept arm's name, so a filter naming only the arms still times their baseline
    std::optional<row_t> row = bench_unary(env, corpus, timed_as, callable);
    if (!row) return std::nullopt;
    row->name = name;
    if (printed) print(*row, against);
    return baseline_of(row);
}

/*  The families, one file each, time the dispatch points against the standard baselines. */
void bench_find(environment_t &env);
void bench_token(environment_t &env);
void bench_sequence(environment_t &env);
void bench_memory(environment_t &env);
void bench_cipher(environment_t &env);
void bench_container(environment_t &env);
void bench_levenshtein(environment_t &env);
void bench_overlap(environment_t &env);
void bench_substrings(environment_t &env);
void bench_utf8_traverse(environment_t &env);
void bench_utf8_scan(environment_t &env);
void bench_utf8_segment(environment_t &env);
void bench_utf8_norm(environment_t &env);
void bench_utf8_uncased(environment_t &env);

/*  The kernels by name, one file per architecture, each against its operation's serial kernel. */
void bench_cross_serial(environment_t &env);
void bench_cross_x8664(environment_t &env);
void bench_cross_arm64(environment_t &env);
void bench_cross_riscv64(environment_t &env);
void bench_cross_loongarch64(environment_t &env);
void bench_cross_ppc64(environment_t &env);
void bench_cross_wasm(environment_t &env);

} // namespace ashvardanian::stringzilla::bench
