/**
 *  @brief  Helper structures and functions for C++ unit- and stress-tests.
 *  @file   test/stringzilla.hpp
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
 *  @section Driver Tiers
 *
 *  A driver's suffix states what it costs and what it may assume, so the name answers both without
 *  reading the body. A family names its drivers `test_<family>_<tier>`, or `test_<family>_<operation>_<tier>`
 *  where one family covers several operations - `substrings` counts, finds, rewrites and scores, and each
 *  wants its own tiers. Helpers that are not drivers take a `check_` prefix and a trailing underscore, and
 *  are never registered in a `main`.
 *
 *  - `_unit`        Known-answer vectors against an external ground truth. Fixed cost: it must run
 *                   identically at every `SZ_TESTS_MULTIPLIER`, so no randomness and no sweeps.
 *  - `_equivalence` A reference against a candidate over generated corpora - serial against each compiled
 *                   backend, or the library against `std::`. This tier owns randomness.
 *  - `_safety`      Malformed, adversarial and boundary inputs. Asserts survival, bounds and stated
 *                   refusals - never answers, since a wrong answer is not what is under test here.
 *                   Scales with `SZ_TESTS_MULTIPLIER` alongside `_equivalence`; only `_unit` is pinned.
 *  - `_all`         Walks the family's backend table and drives the tiers above. Holds no assertions
 *                   of its own; a literal here belongs in `_unit`.
 *  - `_rules`       Annex rule coverage, where a family transcribes a published spec (UAX-29, UAX-14).
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
#include <csignal> // `std::signal`, `SIGSEGV`, `SIGABRT`
#include <cstdint> // `std::uintptr_t` for cache-line alignment
#include <cstdio>  // `std::printf`, `std::fflush`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::strcmp`

#include <algorithm> // `std::copy`, `std::generate`
#include <chrono>    // `std::chrono::steady_clock` for per-test timing
#include <exception> // `std::exception`
#include <random>    // `std::random_device`
#include <regex>     // `std::regex_search` for `SZ_TESTS_FILTER`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#include <unistd.h>   // `STDERR_FILENO`
#endif

#include "stringzilla/types.hpp"
#if SZ_USE_CUDA
#include "stringzillas/types.cuh"
#endif

#pragma region Assertion Helpers

/**
 *  @brief Test-suite verification - always active, regardless of `NDEBUG` or `SZ_DEBUG`. Unlike `sz_assert_`,
 *         which is a debug-only invariant check for the library, a test's oracle must never be a no-op.
 */
#define verify(condition)                                                                                  \
    do {                                                                                                   \
        if (!(condition)) {                                                                                \
            std::fprintf(stderr, "Test verification failed: %s, %s:%d\n", #condition, __FILE__, __LINE__); \
            std::abort();                                                                                  \
        }                                                                                                  \
    } while (0)

/**
 *  @brief One case whose subject has to be named before it can be asserted on, scoped to the case.
 *
 *  Prefer it wherever a bare `verify` would need a preceding declaration that outlives its one use:
 *  a run of these reads as a table of cases, where the same run written longhand reads as prose.
 */
#define let_verify(init, condition) \
    do {                            \
        init;                       \
        verify(condition);          \
    } while (0)

/** @brief As `let_verify`, when the subject must also be acted on before the assertion holds - a mutation
 *         whose result is the subject itself, so there is nothing for the condition to bind. */
#define scope_verify(init, operation, condition) \
    do {                                         \
        init;                                    \
        operation;                               \
        verify(condition);                       \
    } while (0)

/** @brief That @p expression throws @p exception_type. The only assertion whose subject is the failure,
 *         so a passing call - or one that throws something else - is the defect it reports. */
#define throws_verify(expression, exception_type) \
    do {                                          \
        bool threw = false;                       \
        try {                                     \
            sz_unused_(expression);               \
        }                                         \
        catch (exception_type const &) {          \
            threw = true;                         \
        }                                         \
        verify(threw);                            \
    } while (0)

#pragma endregion // Assertion Helpers

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

/**
 *  @brief Copies @p texts into unified memory a CUDA kernel can reach, as one span per string.
 *
 *  Owns the bytes the spans point into, so it has to outlive every call that reads `view()`.
 */
struct unified_texts_t {
    std::vector<unified_vector<char>> storage;
    unified_vector<span<char const>> spans;

    explicit unified_texts_t(std::vector<std::string> const &texts) : storage(texts.size()), spans(texts.size()) {
        for (std::size_t index = 0; index != texts.size(); ++index) {
            storage[index].assign(texts[index].begin(), texts[index].end());
            spans[index] = {storage[index].data(), storage[index].size()};
        }
    }

    span<span<char const> const> view() const noexcept { return {spans.data(), spans.size()}; }
};

/**
 *  @brief Reads a file into a string via LibC `<cstdio>`. A non-zero @p max_bytes stops the read after
 *         that many bytes, so the file tail is never touched.
 */
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
    std::size_t const read_bytes = std::fread(content.data(), 1, capacity, file);
    std::fclose(file);
    content.resize(read_bytes);
    return content;
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

/**
 *  @brief Baseline for a loop whose work grows with the square of its trip count - a sweep over lengths that
 *         re-scans a growing buffer, say - so that doubling the multiplier doubles the work, not quadruples it.
 */
inline std::size_t scale_iterations_quadratic(std::size_t baseline) noexcept {
    std::size_t const work = scale_iterations(baseline * baseline);
    std::size_t bound = 1;
    while (bound * bound < work) ++bound;
    return bound;
}

/**
 *  @brief The @p step -th value of a rotation over @p count items that also advances a phase each full turn, so
 *         crossing it with another rotation of the same length still reaches every pair.
 */
inline std::size_t rotating_index(std::size_t step, std::size_t count) noexcept {
    return count ? (step + step / count) % count : 0;
}

/** @brief Views a C array as a `sz::span`, so tables pass as one argument and keep their length attached. */
template <typename value_type_, std::size_t count_>
constexpr span<value_type_ const> span_over(value_type_ const (&array)[count_]) noexcept {
    return span<value_type_ const>(array, count_);
}

/**
 *  @brief Step for walking an exhaustive space, so `SZ_TESTS_MULTIPLIER` dials sweeps as well as loops.
 *
 *  Striding rather than truncating keeps the far end of the space - where the window-edge cases live -
 *  reachable at a low multiplier. The sweep becomes complete at the `10x` stress point rather than at the
 *  default, so a default run samples every space and a stress run covers them.
 */
inline std::size_t sweep_stride(std::size_t complete) noexcept {
    double const coverage = get_iterations_multiplier() / 10.0;
    if (coverage >= 1.0 || complete == 0) return 1;
    std::size_t const wanted = static_cast<std::size_t>(complete * coverage);
    return wanted < 1 ? complete : complete / wanted;
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
    std::vector<char> storage(usable_length + 2 * SZ_CACHE_LINE_WIDTH + 1, '\0');
    for (std::size_t offset : offsets) {
        std::fill(storage.begin(), storage.end(), '\0');
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
        verify(storage[index] == canary_value && "front canary overwritten");
        verify(storage[guard_width + length + index] == canary_value && "back canary overwritten");
    }
}

/**
 *  @brief Splits @p alphabet into its UTF-8 characters, so a multi-byte alphabet still generates valid text.
 *
 *  A character runs from a lead byte to the last continuation byte after it, which needs no decoder and leaves
 *  an ASCII alphabet one character per byte.
 */
inline std::vector<std::string> alphabet_characters(std::string const &alphabet) noexcept(false) {
    auto const continues_character = [&](std::size_t offset) {
        return (static_cast<unsigned char>(alphabet[offset]) & 0xC0) == 0x80;
    };
    std::vector<std::string> characters;
    for (std::size_t start = 0; start < alphabet.size();) {
        std::size_t end = start + 1;
        while (end < alphabet.size() && continues_character(end)) ++end;
        characters.push_back(alphabet.substr(start, end - start));
        start = end;
    }
    return characters;
}

/** @brief Concatenates @p length characters drawn uniformly from @p characters. */
inline std::string random_string(std::size_t length, std::vector<std::string> const &characters) noexcept(false) {
    std::uniform_int_distribution<std::size_t> distribution(0, characters.size() - 1);
    std::string result;
    while (length--) result += characters[distribution(global_random_generator())];
    return result;
}

/** @brief Reads the start of a member string, for `sz_sequence_t` views over a `std::vector<std::string>`. */
inline sz_cptr_t sequence_get_start_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].data();
}

/** @brief Reads the length of a member string, for `sz_sequence_t` views over a `std::vector<std::string>`. */
inline sz_size_t sequence_get_length_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].size();
}

/** @brief Fills an `sz_sequence_t` view over a `std::vector<std::string>` via the shared accessor helpers. */
inline sz_sequence_t sequence_from_(std::vector<std::string> const &strings) {
    sz_sequence_t sequence;
    sequence.handle = &strings;
    sequence.count = (sz_size_t)strings.size();
    sequence.get_start = sequence_get_start_;
    sequence.get_length = sequence_get_length_;
    return sequence;
}


struct fuzzy_config_t {
    std::string alphabet = "ABC"; // ? Drawn one UTF-8 character at a time, so `"αβγ"` yields valid multi-byte text.
    std::size_t batch_size = 16;
    std::size_t min_string_length = 1; // ? In characters, which equals bytes only for an ASCII alphabet.
    std::size_t max_string_length = 200;
};

inline void randomize_strings(fuzzy_config_t config, std::vector<std::string> &array) {
    array.resize(config.batch_size);

    std::vector<std::string> const characters = alphabet_characters(config.alphabet);
    std::uniform_int_distribution<std::size_t> length_distribution(config.min_string_length, config.max_string_length);
    for (std::size_t index = 0; index != config.batch_size; ++index)
        array[index] = random_string(length_distribution(global_random_generator()), characters);
}

inline void randomize_strings(fuzzy_config_t config, std::vector<std::string> &array, arrow_strings_tape_t &tape) {

    randomize_strings(config, array);

    // Convert to a GPU-friendly layout
    status_t const status = tape.try_assign(array.data(), array.data() + array.size());
    verify(status == status_t::success_k);
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
        std::size_t count = 1;
        for (int j = i + 1; j < device_count; ++j) {
            cudaDeviceProp next;
            if (cudaGetDeviceProperties(&next, j) == cudaSuccess && std::strcmp(next.name, prop.name) == 0) { ++count; }
            else { break; }
        }
        int warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
        std::size_t shared_memory_per_warp =
            (warps_per_sm > 0) ? (prop.sharedMemPerMultiprocessor / static_cast<std::size_t>(warps_per_sm)) : 0;
        std::printf("  - %zu x %s\n", count, prop.name);
        std::printf("    Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        std::printf("    Maximum Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        std::printf("    Warp Size: %d threads\n", prop.warpSize);
        std::printf("    Max Warps per SM: %d warps\n", warps_per_sm);
        std::printf("    Shared Memory per Warp: %zu bytes\n", shared_memory_per_warp);
        std::printf("    Managed memory: %s\n", prop.managedMemory ? "yes" : "no");
        std::printf("    Unified addressing: %s\n", prop.unifiedAddressing ? "yes" : "no");
        i += static_cast<int>(count) - 1;
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
#if defined(__linux__) && defined(__GLIBC__)
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
    // Line-buffer, so progress survives a crash under output redirection.
    // Size must be nonzero: Windows ucrt fast-fails on a zero-sized buffering mode.
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
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
 *  @brief Reseeds the global generator from the global seed and a test name.
 *
 *  Mixing happens in `std::seed_seq`, whose output the standard specifies exactly, rather than in `sz_hash` or
 *  `std::hash`. The harness must not draw its inputs through the kernels it validates, and it must land on the
 *  same stream everywhere, or `SZ_TESTS_SEED=7` stops meaning the same bytes on Arm as it does on x86.
 */
inline void seed_generator_for_test(char const *name) noexcept {
    std::vector<std::uint32_t> entropy {static_cast<std::uint32_t>(global_random_seed())};
    for (char const *character = name; *character; ++character)
        entropy.push_back(static_cast<std::uint32_t>(static_cast<unsigned char>(*character)));
    std::seed_seq sequence(entropy.begin(), entropy.end());
    global_random_generator().seed(sequence);
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
inline std::size_t run_test(char const *name, function_type_ &&test_function) noexcept {
    if (!test_should_run(name)) {
        std::printf("- %s ... skipped (SZ_TESTS_FILTER)\n", name);
        std::fflush(stdout);
        return 0;
    }
    std::printf("- %s ...\n", name);
    std::fflush(stdout);
    // Reseed per test so inputs don't depend on which tests ran first, and `SZ_TESTS_FILTER` reproduces faithfully.
    seed_generator_for_test(name);
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

/*  Cross-translation-unit test declarations. These live at global scope to match the TU definitions;
 *  the using-declaration makes `scale_iterations` visible for the default arguments below. */
using ashvardanian::stringzilla::scripts::scale_iterations;

#pragma region Basic Utilities

void test_arithmetic_unit();
void test_sequence_unit();
void test_allocator_unit();
void test_byteset_unit();

#pragma endregion // Basic Utilities

#pragma region Hashing

void test_hash_unit();
void test_hash_safety();
void test_hash_all();
void test_hash_multiseed_all();

#pragma endregion // Hashing

#pragma region Ciphers

void test_cipher_unit();
void test_cipher_safety();
void test_cipher_all();

#pragma endregion // Ciphers

#pragma region UTF-8

void test_utf8_runes_unit();
void test_utf8_runes_scripts_unit();
void test_utf8_runes_safety();
void test_utf8_runes_all();
void test_utf8_tokens_unit();
void test_utf8_tokens_scripts_unit();
void test_utf8_tokens_safety();
void test_utf8_tokens_all();
void test_utf8_wordbreaks_unit();
void test_utf8_wordbreaks_rules();
void test_utf8_wordbreaks_safety();
void test_utf8_wordbreaks_all();
void test_utf8_graphemes_unit();
void test_utf8_graphemes_rules();
void test_utf8_graphemes_safety();
void test_utf8_graphemes_all();
void test_utf8_sentences_unit();
void test_utf8_sentences_rules();
void test_utf8_sentences_safety();
void test_utf8_sentences_all();
void test_utf8_linebreaks_unit();
void test_utf8_linebreaks_rules();
void test_utf8_linebreaks_safety();
void test_utf8_linebreaks_all();
void test_utf8_norm_unit();
void test_utf8_norm_safety();
void test_utf8_norm_all();
void test_utf8_delimiters_unit();
void test_utf8_delimiters_safety();
void test_utf8_delimiters_all();

#pragma endregion // UTF-8

#pragma region Uncased UTF-8

void test_uncased_unit();
void test_uncased_scripts_unit();
void test_uncased_regressions_unit();
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
void test_memory_stability_equivalence(std::size_t length = 1ull << 10,
                                       std::size_t iterations = scale_iterations(100));
void test_string_updates_equivalence(std::size_t repetitions = 1024);

#pragma endregion // String Class and STL Compatibility

#pragma region Search and Comparison

void test_compare_unit();
void test_extensions_ranges_unit();
void test_find_unit();
void test_find_safety();
void test_find_all();
void test_find_misaligned_equivalence();
void test_lookup_equivalence(std::size_t lookup_tables_to_try = 32, std::size_t slices_per_table = 16);

#pragma endregion // Search and Comparison

#pragma region Sequence Algorithms

void test_sort_all();
void test_sort_unit();
void test_sort_safety();
void test_sort_reference_equivalence();
void test_intersect_unit();
void test_intersect_equivalence();

#pragma endregion // Sequence Algorithms
