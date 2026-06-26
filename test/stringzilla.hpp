/**
 *  @brief  Helper structures and functions for C++ unit- and stress-tests.
 *  @file   scripts/test_stringzilla.hpp
 *  @author Ash Vardanian
 *  @date June 16, 2026
 *
 *  @section Environment Variables
 *
 *  The test infrastructure supports the following environment variables for reproducible
 *  stress testing and fuzzing:
 *
 *  - `SZ_TESTS_SEED` : Seed for the random number generator. If not set, a random seed is
 *    generated using `std::random_device`. The actual seed used is always
 *    printed at startup for reproducibility.
 *  - `SZ_TESTS_MULTIPLIER` : Multiplier for stress-test iteration counts. Defaults to 1.0.
 *    Each test has its own baseline iteration count tuned for its
 *    operation complexity. This multiplier scales all baselines
 *    proportionally (e.g., 0.1 for quick smoke tests, 10 for
 *    thorough CI fuzzing).
 *  - `SZ_TESTS_FILTER` : ECMAScript regex matched against test names; only matching tests run
 *    (e.g. `SZ_TESTS_FILTER=utf8`). Unset or empty runs everything. Honored by `run_test`.
 *
 *  @section Example Usage
 *
 *  @code{.sh}
 *  # Run with a specific seed for reproducibility
 *  SZ_TESTS_SEED=42 ./build_release/stringzilla_test_cpp20
 *
 *  # Quick smoke test (10% of normal iterations)
 *  SZ_TESTS_MULTIPLIER=0.1 ./build_release/stringzilla_test_cpp20
 *
 *  # Fast inner loop: only the UTF-8 tests, at 10% iterations
 *  SZ_TESTS_FILTER=utf8 SZ_TESTS_MULTIPLIER=0.1 ./build_release/stringzilla_test_cpp20
 *
 *  # Thorough CI stress test (10x normal iterations)
 *  SZ_TESTS_MULTIPLIER=10 ./build_release/stringzilla_test_cpp20
 *
 *  # Combine both for CI fuzzing
 *  SZ_TESTS_SEED=12345 SZ_TESTS_MULTIPLIER=5 ./build_release/stringzilla_test_cpp20
 *  @endcode
 */
#pragma once
#include <cassert> // `assert` used directly by `with_guarded_buffer_`
#include <csignal> // `std::signal`, `SIGSEGV`, `SIGABRT`
#include <cstdint> // `std::uintptr_t` for cache-line alignment
#include <cstdio>  // `std::printf`, `std::fflush`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::strcmp`

#include <algorithm> // `std::copy`, `std::generate`
#include <chrono>    // `std::chrono::steady_clock` for per-test timing
#include <exception> // `std::exception`
#include <fstream>   // `std::ifstream`
#include <random>    // `std::random_device`
#include <regex>     // `std::regex_search` for `SZ_TESTS_FILTER`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#if defined(__linux__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#include <unistd.h>   // `STDERR_FILENO`
#endif

#include "stringzilla/types.hpp"
#if SZ_USE_CUDA
#include "stringzillas/types.cuh"
#endif

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

using arrow_strings_view_t = arrow_strings_view<char, sz_size_t>;

#if !SZ_USE_CUDA
using arrow_strings_tape_t = arrow_strings_tape<char, sz_size_t, std::allocator<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, std::allocator<value_type_>>;
#else
using arrow_strings_tape_t = arrow_strings_tape<char, sz_size_t, stringzillas::unified_alloc<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, stringzillas::unified_alloc<value_type_>>;
#endif

inline std::string read_file(std::string path) noexcept(false) {
    std::ifstream stream(path);
    if (!stream.is_open()) throw std::runtime_error("Failed to open file: " + path);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

inline void write_file(std::string path, std::string content) noexcept(false) {
    std::ofstream stream(path);
    if (!stream.is_open()) throw std::runtime_error("Failed to open file: " + path);
    stream << content;
    stream.close();
}

/**
 *  @brief Returns the seed used for the global random number generator.
 *
 *  If `SZ_TESTS_SEED` is set, returns its value. Otherwise, generates a random seed
 *  using `std::random_device`. The seed is cached after the first call.
 */
inline std::mt19937::result_type global_random_seed() noexcept {
    static std::mt19937::result_type seed = []() {
        char const *seed_env = std::getenv("SZ_TESTS_SEED");
        if (seed_env && seed_env[0] != '\0')
            return static_cast<std::mt19937::result_type>(std::strtoul(seed_env, nullptr, 10));
        std::random_device seed_source;
        return static_cast<std::mt19937::result_type>(seed_source());
    }();
    return seed;
}

/** @brief Returns true if the seed was set via environment variable. */
inline bool global_random_seed_from_env() noexcept {
    char const *seed_env = std::getenv("SZ_TESTS_SEED");
    return seed_env && seed_env[0] != '\0';
}

/**
 *  @brief Returns a reference to the global random number generator.
 *
 *  The generator is seeded once using `global_random_seed()`, which respects the
 *  `SZ_TESTS_SEED` environment variable for reproducible testing.
 */
inline std::mt19937 &global_random_generator() noexcept {
    static std::mt19937 generator(global_random_seed());
    return generator;
}

/**
 *  @brief Returns the multiplier for stress-test iteration counts.
 *
 *  Reads from the `SZ_TESTS_MULTIPLIER` environment variable. Defaults to 1.0.
 *  Use values < 1.0 for quick smoke tests, > 1.0 for thorough stress testing in CI.
 */
inline double get_iterations_multiplier() noexcept {
    static double multiplier = []() {
        char const *env = std::getenv("SZ_TESTS_MULTIPLIER");
        if (env && env[0] != '\0') {
            double parsed = std::strtod(env, nullptr);
            if (parsed > 0.0) return parsed;
        }
        return 1.0;
    }();
    return multiplier;
}

/**
 *  @brief Scales a baseline iteration count by the global multiplier.
 *
 *  Use this to wrap hardcoded iteration counts in stress tests, e.g.:
 *  @code{.cpp}
 *  for (std::size_t i = 0; i < scale_iterations(1000); ++i) { ... }
 *  @endcode
 *
 *  @param baseline The default number of iterations for this test.
 *  @return The scaled iteration count, guaranteed to be at least 1.
 */
inline std::size_t scale_iterations(std::size_t baseline) noexcept {
    double scaled = baseline * get_iterations_multiplier();
    return scaled < 1.0 ? 1 : static_cast<std::size_t>(scaled);
}

template <typename string_type_, typename other_string_type_>
inline string_type_ to_str(other_string_type_ const &other) noexcept {
    return string_type_(other.data(), other.size());
}

/**
 *  @brief A uniform distribution of characters, with a given alphabet size.
 *         The alphabet size is the number of distinct characters in the distribution.
 *
 *  We can't use `std::uniform_int_distribution<char>` because `char` overload is not supported by some platforms.
 *  MSVC, for example, requires one of `short`, `int`, `long`, `long long`, `unsigned short`, `unsigned int`,
 *  `unsigned long`, or `unsigned long long`.
 */
struct uniform_u8_distribution_t {
    std::uniform_int_distribution<std::uint32_t> distribution;

    inline uniform_u8_distribution_t(std::size_t alphabet_size = 255)
        : distribution(1, static_cast<std::uint32_t>(alphabet_size)) {}
    inline uniform_u8_distribution_t(char from, char to)
        : distribution(static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(to)) {}

    template <typename generator_type_>
    std::uint8_t operator()(generator_type_ &&generator) noexcept {
        return static_cast<std::uint8_t>(distribution(generator));
    }
};

inline void randomize_string(char *string, std::size_t length, char const *alphabet, std::size_t cardinality) noexcept {
    uniform_u8_distribution_t distribution(0, static_cast<char>(cardinality - 1));
    std::generate(string, string + length, [&]() -> char { return alphabet[distribution(global_random_generator())]; });
}

inline void randomize_string(char *string, std::size_t length) noexcept {
    uniform_u8_distribution_t distribution;
    std::generate(string, string + length, [&]() -> char { return distribution(global_random_generator()); });
}

inline std::string random_string(std::size_t length, char const *alphabet, std::size_t cardinality) noexcept(false) {
    std::string result(length, '\0');
    randomize_string(&result[0], length, alphabet, cardinality);
    return result;
}

inline std::string repeat(std::string const &patten, std::size_t count) noexcept(false) {
    std::string result(patten.size() * count, '\0');
    for (std::size_t i = 0; i < count; ++i) std::copy(patten.begin(), patten.end(), result.begin() + i * patten.size());
    return result;
}

/**
 *  @brief Randomly slices a string into consecutive parts and passes those to @p slice_callback.
 *  @warning Is @b single-threaded in nature, as it depends on the `global_random_generator`.
 */
template <typename slice_callback_type_>
inline void iterate_in_random_slices(std::string const &text, slice_callback_type_ &&slice_callback) noexcept {
    std::size_t remaining = text.size();
    while (remaining > 0) {
        std::uniform_int_distribution<std::size_t> slice_length_distribution(1, remaining);
        std::size_t slice_length = slice_length_distribution(global_random_generator());
        slice_callback({text.data() + text.size() - remaining, slice_length});
        remaining -= slice_length;
    }
}

/**
 *  @brief Invokes @p body with a writable buffer placed at each of a representative spread of
 *         sub-cache-line byte offsets, so SIMD kernels are exercised at every alignment.
 *  @param usable_length Minimum number of writable bytes guaranteed past the passed pointer.
 *  @param body Callable as `body(sz_ptr_t pointer, std::size_t offset)`; the buffer is zero-filled per offset.
 */
template <typename body_type_>
inline void for_each_cacheline_offset_(std::size_t usable_length, body_type_ &&body) noexcept {
    static constexpr std::size_t offsets[] = {0, 1, 7, 8, 15, 16, 31, 32, 33, 48, 63};
    for (std::size_t offset : offsets) {
        std::vector<char> storage(usable_length + 2 * SZ_CACHE_LINE_WIDTH + 1, '\0');
        char *pointer = storage.data();
        while (reinterpret_cast<std::uintptr_t>(pointer) % SZ_CACHE_LINE_WIDTH != offset) ++pointer;
        body(reinterpret_cast<sz_ptr_t>(pointer), offset);
    }
}

/**
 *  @brief Runs @p body on a @p length -byte writable buffer flanked by canary bytes on both sides, then
 *         asserts the guards are intact - catching out-of-bounds writes from a kernel under adversarial input.
 *  @param length Number of usable bytes handed to @p body.
 *  @param body Callable as `body(sz_ptr_t pointer, std::size_t length)`; the buffer is canary-filled per call.
 */
template <typename body_type_>
inline void with_guarded_buffer_(std::size_t length, body_type_ &&body) noexcept {
    static constexpr std::size_t guard_width = 64;
    static constexpr unsigned char canary_value = 0xA5;
    std::vector<unsigned char> storage(length + 2 * guard_width, canary_value);
    unsigned char *usable = storage.data() + guard_width;
    body(reinterpret_cast<sz_ptr_t>(usable), length);
    for (std::size_t index = 0; index != guard_width; ++index) {
        assert(storage[index] == canary_value && "front canary overwritten");
        assert(storage[guard_width + length + index] == canary_value && "back canary overwritten");
    }
}

struct fuzzy_config_t {
    std::string alphabet = "ABC";
    std::size_t batch_size = 16;
    std::size_t min_string_length = 1;
    std::size_t max_string_length = 200;
};

inline void randomize_strings(fuzzy_config_t config, std::vector<std::string> &array, bool unique = false) {
    array.resize(config.batch_size);

    std::uniform_int_distribution<std::size_t> length_distribution(config.min_string_length, config.max_string_length);
    for (std::size_t i = 0; i != config.batch_size; ++i) {
        std::size_t length = length_distribution(global_random_generator());
        array[i] = random_string(length, config.alphabet.data(), config.alphabet.size());
    }

    if (unique) {
        std::sort(array.begin(), array.end());
        auto last = std::unique(array.begin(), array.end());
        array.erase(last, array.end());
    }
}

inline void randomize_strings(fuzzy_config_t config, std::vector<std::string> &array, arrow_strings_tape_t &tape,
                              bool unique = false) {

    randomize_strings(config, array, unique);

    // Convert to a GPU-friendly layout
    status_t status = tape.try_assign(array.data(), array.data() + array.size());
    sz_assert_(status == status_t::success_k);
}

inline char const *status_name(status_t s) noexcept {
    switch (s) {
    case status_t::success_k: return "success";
    case status_t::bad_alloc_k: return "bad_alloc";
    case status_t::invalid_utf8_k: return "invalid_utf8";
    case status_t::contains_duplicates_k: return "contains_duplicates";
    case status_t::overflow_risk_k: return "overflow_risk";
    case status_t::unexpected_dimensions_k: return "unexpected_dimensions";
    case status_t::missing_gpu_k: return "missing_gpu";
    case status_t::device_code_mismatch_k: return "device_code_mismatch";
    case status_t::device_memory_mismatch_k: return "device_memory_mismatch";
    case status_t::unknown_k: return "unknown";
    default: return "unrecognized";
    }
}

inline int log_environment() {
    std::printf("- Uses Westmere: %s \n", SZ_USE_WESTMERE ? "yes" : "no");
    std::printf("- Uses Goldmont: %s \n", SZ_USE_GOLDMONT ? "yes" : "no");
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICELAKE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses NEON AES: %s \n", SZ_USE_NEONAES ? "yes" : "no");
    std::printf("- Uses NEON SHA: %s \n", SZ_USE_NEONSHA ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses SVE2 AES: %s \n", SZ_USE_SVE2AES ? "yes" : "no");
    std::printf("- Uses WASM SIMD128: %s \n", SZ_USE_V128 ? "yes" : "no");
    std::printf("- Uses WASM relaxed SIMD: %s \n", SZ_USE_V128RELAXED ? "yes" : "no");
    std::printf("- Uses RISC-V RVV: %s \n", SZ_USE_RVV ? "yes" : "no");
    std::printf("- Uses LoongArch LASX: %s \n", SZ_USE_LASX ? "yes" : "no");
    std::printf("- Uses Power VSX: %s \n", SZ_USE_POWERVSX ? "yes" : "no");
    std::printf("- Uses CUDA: %s \n", SZ_USE_CUDA ? "yes" : "no");
    std::printf("- Uses Kepler CUDA: %s \n", SZ_USE_KEPLER ? "yes" : "no");
    std::printf("- Uses Hopper CUDA: %s \n", SZ_USE_HOPPER ? "yes" : "no");

#if SZ_USE_CUDA
    cudaError_t cuda_error = cudaFree(0); // Force context initialization
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA initialization error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    int device_count = 0;
    cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
        return 1;
    }
    std::printf("CUDA device count: %d\n", device_count);
    if (device_count == 0) {
        std::printf("No CUDA devices found.\n");
        return 1;
    }
    std::printf("- CUDA devices:\n");
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cuda_error = cudaGetDeviceProperties(&prop, i);
        if (cuda_error != cudaSuccess) {
            std::printf("Error retrieving properties for device %d: %s\n", i, cudaGetErrorString(cuda_error));
            continue;
        }
        int count = 1;
        for (int j = i + 1; j < device_count; ++j) {
            cudaDeviceProp next;
            if (cudaGetDeviceProperties(&next, j) == cudaSuccess && std::strcmp(next.name, prop.name) == 0) { ++count; }
            else { break; }
        }
        int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        int shared_memory_per_warp = (warps_per_sm > 0) ? (prop.sharedMemPerMultiprocessor / warps_per_sm) : 0;
        std::printf("  - %d x %s\n", count, prop.name);
        std::printf("    Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        std::printf("    Maximum Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        std::printf("    Warp Size: %d threads\n", prop.warpSize);
        std::printf("    Max Warps per SM: %d warps\n", warps_per_sm);
        std::printf("    Shared Memory per Warp: %d bytes\n", shared_memory_per_warp);
        std::printf("    Managed memory: %s\n", prop.managedMemory ? "yes" : "no");
        std::printf("    Unified addressing: %s\n", prop.unifiedAddressing ? "yes" : "no");
        i += count - 1;
    }
#endif
    return 0;
}

/**
 *  @brief Prints test environment configuration (seed and multiplier).
 *
 *  Call this at the start of main() to display test configuration alongside
 *  other environment info. Format matches capability flags style.
 */
inline void print_test_environment() noexcept {
    auto seed = global_random_seed();
    bool from_env = global_random_seed_from_env();
    std::printf("- Test seed: %u%s\n", static_cast<unsigned>(seed), from_env ? " (from SZ_TESTS_SEED)" : "");
    double multiplier = get_iterations_multiplier();
    if (multiplier != 1.0) std::printf("- Iterations multiplier: %.2fx\n", multiplier);
    std::fflush(stdout); // Ensure output is visible even on crash
}

#pragma region - Test Runner

/**
 *  @brief Prints a backtrace on a fatal signal, so a crashing/aborting kernel self-localizes
 *         instead of dying silently - especially under output redirection in CI.
 */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    std::fprintf(stderr, "\n*** Fatal signal %d - backtrace follows ***\n", signal_number);
#if defined(__linux__)
    void *frames[64];
    int const frames_count = backtrace(frames, sizeof(frames) / sizeof(frames[0]));
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

/**
 *  @brief Installs SIGSEGV/SIGABRT backtrace handlers and line-buffers stdout.
 *         Shared by the serial (`.cpp`) and CUDA (`.cu`) test entry points; call once from `main`.
 */
inline void install_test_signal_handlers() noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // Line-buffer, so progress survives a crash under output redirection.
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
}

/**
 *  @brief Returns true if a test named @p name should run, honoring the `SZ_TESTS_FILTER` regex.
 *
 *  `SZ_TESTS_FILTER` is an ECMAScript regular expression matched against the test name (e.g.
 *  `SZ_TESTS_FILTER=fingerprint` runs only the rolling-hasher tests, skipping the slower similarity
 *  suite). An empty or unset filter runs everything; an invalid pattern runs everything rather than
 *  silently skipping the whole suite.
 */
inline bool test_should_run(char const *name) noexcept {
    static std::string const filter = []() -> std::string {
        char const *env = std::getenv("SZ_TESTS_FILTER");
        return env && env[0] != '\0' ? std::string(env) : std::string();
    }();
    if (filter.empty()) return true;
    try {
        return std::regex_search(name, std::regex(filter));
    }
    catch (std::regex_error const &) {
        return true;
    }
}

/**
 *  @brief Runs one named test: honors `SZ_TESTS_FILTER`, wall-clock times it, and reports the outcome.
 *  @return The number of failures (0 on success or when skipped, 1 on a thrown exception).
 *
 *  Hard failures via `sz_assert_` abort the process (and self-localize through the installed signal
 *  handler); this wrapper additionally turns thrown exceptions into a localized, named failure instead
 *  of a bare `what()` at the top of `main`, and surfaces per-test durations so slow tests are obvious.
 */
template <typename function_type_>
inline int run_test(char const *name, function_type_ &&test_function) noexcept {
    if (!test_should_run(name)) {
        std::printf("- %s ... skipped (SZ_TESTS_FILTER)\n", name);
        std::fflush(stdout);
        return 0;
    }
    std::printf("- %s ...\n", name);
    std::fflush(stdout);
    auto const start = std::chrono::steady_clock::now();
    try {
        test_function();
    }
    catch (std::exception const &error) {
        std::fprintf(stderr, "- %s ... FAILED: %s\n", name, error.what());
        std::fflush(stderr);
        return 1;
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("- %s ... ok (%.2f s)\n", name, seconds);
    std::fflush(stdout);
    return 0;
}

#pragma endregion - Test Runner

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian

#pragma region Cross-Translation-Unit Test Declarations

/*  Fused from the former `test_stringzilla_decls.hpp`. These declarations live at global scope to
 *  match the TU definitions; the using-declaration makes `scale_iterations` visible for the default
 *  arguments below. */
using ashvardanian::stringzilla::scripts::scale_iterations;

#pragma region Assertion Helpers

#define scope_assert(init, operation, condition) \
    do {                                         \
        init;                                    \
        operation;                               \
        assert(condition);                       \
    } while (0)

#define let_assert(init, condition) \
    do {                            \
        init;                       \
        assert(condition);          \
    } while (0)

#define assert_throws(expression, exception_type) \
    do {                                          \
        bool threw = false;                       \
        try {                                     \
            sz_unused_(expression);               \
        }                                         \
        catch (exception_type const &) {          \
            threw = true;                         \
        }                                         \
        assert(threw);                            \
    } while (0)

#pragma endregion // Assertion Helpers

#pragma region Basic Utilities

void test_arithmetic_unit();
void test_sequence_unit();
void test_allocator_unit();
void test_byteset_unit();

#pragma endregion // Basic Utilities

#pragma region Hashing

void test_hash_unit();
void test_hash_all();
void test_hash_multiseed_all();

#pragma endregion // Hashing

#pragma region UTF-8

void test_utf8_runes_unit();
void test_utf8_runes_safety();
void test_utf8_runes_all();
void test_utf8_tokens_unit();
void test_utf8_tokens_safety();
void test_utf8_tokens_all();
void test_utf8_words_unit();
void test_utf8_words_rules();
void test_utf8_words_safety();
void test_utf8_words_all();
void test_utf8_graphemes_unit();
void test_utf8_graphemes_rules();
void test_utf8_graphemes_safety();
void test_utf8_graphemes_all();
void test_utf8_sentences_unit();
void test_utf8_sentences_rules();
void test_utf8_sentences_safety();
void test_utf8_sentences_all();
void test_utf8_linewraps_unit();
void test_utf8_linewraps_rules();
void test_utf8_linewraps_safety();
void test_utf8_linewraps_all();
void test_utf8_norm_unit();
void test_utf8_norm_safety();
void test_utf8_norm_all();
void test_utf8_delimiters_unit();
void test_utf8_delimiters_safety();
void test_utf8_delimiters_all();

#pragma endregion // UTF-8

#pragma region Uncased UTF-8

void test_uncased_unit();
void test_uncased_all();
void test_uncased_safety();

#pragma endregion // Uncased UTF-8

#pragma region String Class and STL Compatibility

template <typename string_type>
void test_ascii_unit();

void test_memory_unit(std::size_t max_l2_size = 1024ull * 1024ull);
void test_memory_large_unit();
void test_memory_all();
void test_memory_safety();

template <typename string_type>
void test_stl_reads_unit();

template <typename string_type>
void test_stl_updates_unit();

void test_stl_conversions_unit();
void test_stl_containers_unit();

template <typename string_type>
void test_extensions_reads_unit();

void test_extensions_updates_unit();
void test_string_constructors_unit();
void test_memory_stability_unit(std::size_t length = 1ull << 10, std::size_t iterations = scale_iterations(100));
void test_string_updates_unit(std::size_t repetitions = 1024);

#pragma endregion // String Class and STL Compatibility

#pragma region Search and Comparison

void test_compare_unit();
void test_find_unit();
void test_find_all();
void test_find_misaligned_fuzz();
void test_lookup_fuzz(std::size_t lookup_tables_to_try = 32, std::size_t slices_per_table = 16);

#pragma endregion // Search and Comparison

#pragma region Sequence Algorithms

void test_sort_all();
void test_sort_unit();
void test_intersect_unit();

#pragma endregion // Sequence Algorithms

#pragma endregion // Cross-Translation-Unit Test Declarations
