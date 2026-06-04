/**
 *  @brief  Helper structures and functions for C++ unit- and stress-tests.
 *  @file   test_stringzilla.hpp
 *  @author Ash Vardanian
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
 *  # Thorough CI stress test (10x normal iterations)
 *  SZ_TESTS_MULTIPLIER=10 ./build_release/stringzilla_test_cpp20
 *
 *  # Combine both for CI fuzzing
 *  SZ_TESTS_SEED=12345 SZ_TESTS_MULTIPLIER=5 ./build_release/stringzilla_test_cpp20
 *  @endcode
 */
#pragma once
#include <cstdio>  // `std::printf`, `std::fflush`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::strcmp`

#include <fstream>   // `std::ifstream`
#include <random>    // `std::random_device`
#include <string>    // `std::string`
#include <vector>    // `std::vector`
#include <algorithm> // `std::copy`, `std::generate`

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
 *  @brief  Returns the seed used for the global random number generator.
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
 *  @brief  Returns a reference to the global random number generator.
 *
 *  The generator is seeded once using `global_random_seed()`, which respects the
 *  `SZ_TESTS_SEED` environment variable for reproducible testing.
 */
inline std::mt19937 &global_random_generator() noexcept {
    static std::mt19937 generator(global_random_seed());
    return generator;
}

/**
 *  @brief  Returns the multiplier for stress-test iteration counts.
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
 *  @brief  Scales a baseline iteration count by the global multiplier.
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
 *  @brief  A uniform distribution of characters, with a given alphabet size.
 *          The alphabet size is the number of distinct characters in the distribution.
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
    std::printf("- Uses Haswell: %s \n", SZ_USE_HASWELL ? "yes" : "no");
    std::printf("- Uses Skylake: %s \n", SZ_USE_SKYLAKE ? "yes" : "no");
    std::printf("- Uses Ice Lake: %s \n", SZ_USE_ICE ? "yes" : "no");
    std::printf("- Uses NEON: %s \n", SZ_USE_NEON ? "yes" : "no");
    std::printf("- Uses NEON AES: %s \n", SZ_USE_NEON_AES ? "yes" : "no");
    std::printf("- Uses NEON SHA: %s \n", SZ_USE_NEON_SHA ? "yes" : "no");
    std::printf("- Uses SVE: %s \n", SZ_USE_SVE ? "yes" : "no");
    std::printf("- Uses SVE2: %s \n", SZ_USE_SVE2 ? "yes" : "no");
    std::printf("- Uses SVE2 AES: %s \n", SZ_USE_SVE2_AES ? "yes" : "no");
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
 *  @brief  Prints test environment configuration (seed and multiplier).
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

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
