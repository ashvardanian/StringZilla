/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2024
 *  @brief Helper structures and functions for C++ unit- and stress-tests.
 *
 *  @section test_environment_variables Environment Variables
 *
 *  The test infrastructure supports the following environment variables for reproducible stress
 *  testing and fuzzing:
 *
 *  - @c STRINGZILLA_SEED : Seed for the random number generator, 42 by default; @c random draws
 *    one from @c std::random_device. The seed is printed at startup beside a rerun template, and
 *    every failure @c run_test catches prints a @c rerun line reproducing it.
 *  - @c STRINGZILLA_SCALE : Multiplier for stress-test iteration counts, 1.0 by default. It
 *    scales each test's own baseline, e.g. 0.1 for quick smoke tests, 10 for thorough CI fuzzing.
 *  - @c STRINGZILLA_FILTER : ECMAScript regex matched against test names; only matching tests run,
 *    e.g. `STRINGZILLA_FILTER=utf8`. Unset or empty runs everything, and a pattern that does not
 *    compile matches as a plain substring. Honored by @c run_test.
 *
 *  @section test_driver_tiers Driver Tiers
 *
 *  A driver's suffix states what it costs and what it may assume, so the name answers both without
 *  reading the body. A family names its drivers `test_<family>_<tier>`, or
 *  `test_<family>_<operation>_<tier>` where one family covers several operations - @c substrings
 *  counts, finds, rewrites and scores, and each wants its own tiers. Helpers that are not drivers
 *  take a @c check_ prefix and a trailing underscore, and are never registered in a @c main.
 *
 *  - @c _unit : Known-answer vectors against an external ground truth. Fixed cost: it must run
 *    identically at every @c STRINGZILLA_SCALE, so no randomness and no sweeps.
 *  - @c _equivalence : A reference against a candidate over generated corpora - serial against
 *    each compiled backend, or the library against `std::`. This tier owns randomness.
 *  - @c _safety : Malformed, adversarial and boundary inputs. Asserts survival, bounds and stated
 *    refusals - never answers, since a wrong answer is not what is under test here. Scales with
 *    @c STRINGZILLA_SCALE alongside @c _equivalence; only @c _unit is pinned.
 *  - @c _all : Walks the family's backend table and drives the tiers above. Holds no assertions
 *    of its own; a literal here belongs in @c _unit.
 *  - @c _rules : Annex rule coverage, where a family transcribes a published spec: UAX-29, UAX-14.
 *
 *  @section test_example_usage Example Usage
 *
 *  @code{.sh}
 *  # Draw a fresh seed instead of the default 42
 *  STRINGZILLA_SEED=random ./build_release/stringzilla_test_cpp20
 *
 *  # Quick smoke test (10% of normal iterations)
 *  STRINGZILLA_SCALE=0.1 ./build_release/stringzilla_test_cpp20
 *
 *  # Fast inner loop: only the UTF-8 tests, at 10% iterations
 *  STRINGZILLA_FILTER=utf8 STRINGZILLA_SCALE=0.1 ./build_release/stringzilla_test_cpp20
 *
 *  # Thorough CI stress test (10x normal iterations)
 *  STRINGZILLA_SCALE=10 ./build_release/stringzilla_test_cpp20
 *
 *  # Combine both for CI fuzzing
 *  STRINGZILLA_SEED=12345 STRINGZILLA_SCALE=5 ./build_release/stringzilla_test_cpp20
 *  @endcode
 */
#pragma once
#include <csignal> // `std::signal`, `SIGSEGV`, `SIGABRT`
#include <cstdint> // `std::uintptr_t` for cache-line alignment
#include <cstdio>  // `std::FILE`, `std::fopen`, `stderr`
#include <cstdlib> // `std::getenv`, `std::strtod`, `std::abort`, `std::malloc`, `std::free`
#include <cstring> // `std::strcmp`, `std::strlen`

#include <algorithm>    // `std::copy`, `std::generate`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock` for per-test timing
#include <exception>    // `std::exception`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex_search` for `STRINGZILLA_FILTER`
#include <span>         // `std::span`, `std::as_bytes`
#include <string>       // `std::string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <type_traits>  // `std::is_enum_v`
#include <vector>       // `std::vector`

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
#include <fmt/std.h>

#include "stringzilla/types.hpp"

#pragma region Assertion Helpers

namespace ashvardanian::stringzilla::test {

/** One operand of a failed comparison, cut at a few hundred bytes so a megabyte string cannot drown
 *  the report printed on failure. */
template <typename value_type_>
std::string render_operand(value_type_ const &value) {
    std::string rendered;
    if constexpr (std::is_null_pointer_v<value_type_>) rendered = "nullptr";
    else if constexpr (std::is_pointer_v<value_type_>) rendered = fmt::format("{}", fmt::ptr(value));
    else if constexpr (std::is_array_v<value_type_> &&
                       std::is_same_v<std::remove_cv_t<std::remove_extent_t<value_type_>>, char>)
        rendered = fmt::format("{:?}", std::string_view(value, std::find(value, std::end(value), '\0')));
    else if constexpr (std::is_same_v<value_type_, char>) rendered = fmt::format("{:?}", value);
    else if constexpr (std::is_convertible_v<value_type_ const &, std::string_view>)
        rendered = fmt::format("{:?}", std::string_view(value));
    else if constexpr (std::is_enum_v<value_type_>) rendered = fmt::format("{}", fmt::underlying(value));
    else if constexpr (fmt::is_formattable<value_type_>::value) rendered = fmt::format("{}", value);
    else rendered = "{?}";
    std::size_t constexpr limit_bytes = 256;
    if (rendered.size() > limit_bytes)
        rendered = fmt::format("{}... {} more bytes", rendered.substr(0, limit_bytes), rendered.size() - limit_bytes);
    return rendered;
}

/** The bytes of @p text, space-separated, printing a hex dump under @c {:02X}. */
inline auto hex_bytes(std::string_view text) noexcept { return fmt::join(std::as_bytes(std::span(text)), " "); }

/** What a failed @c verify throws once its diagnostic is printed, for @c run_test to report. */
struct test_failure_t : std::exception {
    char const *what() const noexcept override { return "verification failed"; }
};

/** The state of one @c verify: its leftmost comparison's operands, rendered only if it fails. */
struct assertion_t {
    std::string expansion;

    /** Throws outside the checking function, so a @c verify in a @c noexcept one terminates without
     *  tripping GCC's @c -Wterminate. */
    [[noreturn]] void fail(char const *expression, char const *file, int line) const {
        fmt::println(stderr, "Test verification failed: {}, {}:{}", expression, file, line);
        if (!expansion.empty()) fmt::println(stderr, "  with expansion: {}", expansion);
        throw test_failure_t {};
    }
};

/** Left operand of a @c verify, awaiting the comparison that decides whether to render it. */
template <typename left_type_>
struct left_operand {
    assertion_t &assertion;
    left_type_ const &value;

    explicit operator bool() const { return static_cast<bool>(value); }

    template <typename right_type_>
    bool record(bool holds, char const *operator_name, right_type_ const &right) const {
        if (!holds)
            assertion.expansion = fmt::format("{} {} {}", render_operand(value), operator_name, render_operand(right));
        return holds;
    }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
    template <typename right_type_>
    friend bool operator==(left_operand &&left, right_type_ const &right) {
        return left.record(left.value == right, "==", right);
    }
    template <typename right_type_>
    friend bool operator!=(left_operand &&left, right_type_ const &right) {
        return left.record(left.value != right, "!=", right);
    }
    template <typename right_type_>
    friend bool operator<(left_operand &&left, right_type_ const &right) {
        return left.record(left.value < right, "<", right);
    }
    template <typename right_type_>
    friend bool operator<=(left_operand &&left, right_type_ const &right) {
        return left.record(left.value <= right, "<=", right);
    }
    template <typename right_type_>
    friend bool operator>(left_operand &&left, right_type_ const &right) {
        return left.record(left.value > right, ">", right);
    }
    template <typename right_type_>
    friend bool operator>=(left_operand &&left, right_type_ const &right) {
        return left.record(left.value >= right, ">=", right);
    }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
};

/** Binds tighter than any comparison without being one, so `assertion <=> a == b` captures @c a
 *  before @c == runs, and GCC's @c -Wparentheses sees no comparison nested in another. */
template <typename left_type_>
left_operand<left_type_> operator<=>(assertion_t &assertion, left_type_ const &value) noexcept {
    return {assertion, value};
}

} // namespace ashvardanian::stringzilla::test

/** Test-suite verification - always active, regardless of @c NDEBUG or @c STRINGZILLA_DEBUG. Unlike
 *  @c sz_assert_, a debug-only invariant check for the library, a test's oracle must never be a
 *  no-op. A failing comparison prints both of its operands. */
#define verify(condition)                                                                       \
    do {                                                                                        \
        ::ashvardanian::stringzilla::test::assertion_t sz_assertion_;                           \
        if (!(sz_assertion_ <=> condition)) sz_assertion_.fail(#condition, __FILE__, __LINE__); \
    } while (0)

/**
 *  @brief One case whose subject has to be named before it can be asserted on, scoped to the case.
 *
 *  Prefer it wherever a bare @c verify would need a preceding declaration that outlives its one
 *  use: a run of these reads as a table of cases, the same run written longhand as prose.
 */
#define let_verify(init, condition) \
    do {                            \
        init;                       \
        verify(condition);          \
    } while (0)

/** As @c let_verify, when the subject must also be acted on before the assertion holds - a mutation
 *  whose result is the subject itself, so there is nothing for the condition to bind. */
#define scope_verify(init, operation, condition) \
    do {                                         \
        init;                                    \
        operation;                               \
        verify(condition);                       \
    } while (0)

/** That @p expression throws @p exception_type. The only assertion whose subject is the failure,
 *  so a passing call - or one that throws something else - is the defect it reports. */
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

#pragma endregion Assertion Helpers

#pragma region Backend Tables

/**
 *  @brief Reports which backend broke a check, then fails the test through the suite's oracle.
 *  @param[in] name The row's spelling in its family's backend table.
 *  @param[in] what What the row disagreed with: its oracle, known answer, or reference backend.
 */
inline void fail_backend_(char const *name, char const *what) {
    fmt::println(stderr, "Backend {} failed: {}", name, what);
    verify(false && "A backend disagreed with its oracle, its known answer, or the reference");
}

/** The row named @p name, so a reordering cannot silently hand the differential a new reference. */
template <typename backend_type_, std::size_t count_>
backend_type_ const &backend_named_(backend_type_ const (&backends)[count_], char const *name) {
    for (std::size_t index = 0; index != count_; ++index)
        if (std::strcmp(backends[index].name, name) == 0) return backends[index];
    verify(false && "The backend table must carry the named reference");
    return backends[0];
}

#pragma endregion Backend Tables

namespace ashvardanian::stringzilla::test {

using arrow_strings_view_t = arrow_strings_view<char, sz_size_t>;

#if !STRINGZILLA_TARGET_CUDA
using arrow_strings_tape_t = arrow_strings_tape<char, sz_size_t, std::allocator<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, std::allocator<value_type_>>;
#else
using arrow_strings_tape_t = arrow_strings_tape<char, sz_size_t, unified_alloc<char>>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, unified_alloc<value_type_>>;
#endif

#if STRINGZILLA_TARGET_CUDA

/**
 *  @brief Page-locked host memory, which the driver reports as host and every engine refuses.
 *
 *  A third memory kind beside unified and device, and the one a caller most expects to work.
 */
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

/**
 *  @brief Copies @p texts into unified memory a CUDA kernel can reach, as one span per string.
 *
 *  Owns the bytes the spans point into, so it has to outlive every call that reads @c view().
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

/** The run's knobs, read once by @c main and passed to every @c run_test. */
struct test_environment_t {
    std::uint32_t seed = 42;
    double scale = 1.0;
    char const *filter = nullptr;      // ? Points into `environ`, alive for the whole run
    std::optional<std::regex> pattern; // ? The compiled @c filter, empty when it does not compile
    char const *program = "";          // ? `argv[0]`, which closes every rerun line

    /** Whether @c filter selects @p name: as a regex, or as a substring if it does not compile. */
    bool selects(std::string_view name) const {
        if (!filter) return true;
        if (pattern) return std::regex_search(name.begin(), name.end(), *pattern);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Reads @c STRINGZILLA_SEED, @c STRINGZILLA_SCALE and @c STRINGZILLA_FILTER, drawing a seed for
 *  @c random; @p program is `argv[0]`. */
inline test_environment_t read_test_environment(char const *program) {
    test_environment_t environment;
    environment.program = program;
    bool const random_seed = std::strcmp(env_variable("STRINGZILLA_SEED", ""), "random") == 0;
    environment.seed = random_seed ? std::random_device {}() : env_variable("STRINGZILLA_SEED", environment.seed);
    environment.scale = env_variable("STRINGZILLA_SCALE", environment.scale);
    environment.filter = env_variable("STRINGZILLA_FILTER", environment.filter);
    try {
        if (environment.filter) environment.pattern.emplace(environment.filter);
    }
    catch (std::regex_error const &) {
        // ? `selects` then matches the pattern as a plain substring
    }
    return environment;
}

/**
 *  @brief The seed of the test named @p name in a run seeded with @p seed.
 *
 *  FNV-1a over @c std::uint32_t is exact on every platform, like @c std::seed_seq, and never
 *  touches the kernels under test, unlike @c sz_hash. The harness must not draw its inputs through
 *  the kernels it validates, and it must land on the same stream everywhere, or
 *  `STRINGZILLA_SEED=7` stops meaning the same bytes on Arm as it does on x86.
 */
constexpr std::uint32_t mix_seed(std::uint32_t seed, std::string_view name) noexcept {
    std::uint32_t mixed = seed ^ 2166136261u;
    for (char const character : name) mixed = (mixed ^ static_cast<unsigned char>(character)) * 16777619u;
    return mixed;
}

/** What one test draws its inputs from and sizes its loops by, built for it by @c run_test. */
struct test_context_t {
    std::mt19937 generator;
    double scale = 1.0;

    /** Scales a @p baseline iteration count by @c STRINGZILLA_SCALE, never below 1. */
    std::size_t iterations(std::size_t baseline) const noexcept {
        double const scaled = baseline * scale;
        return scaled < 1.0 ? 1 : static_cast<std::size_t>(scaled);
    }

    /** Baseline for a loop whose work grows with the square of its trip count, like a sweep over
     *  lengths that re-scans a growing buffer, so doubling the multiplier only doubles the work. */
    std::size_t iterations_quadratic(std::size_t baseline) const noexcept {
        std::size_t const work = iterations(baseline * baseline);
        std::size_t bound = 1;
        while (bound * bound < work) ++bound;
        return bound;
    }

    /**
     *  @brief Step for walking an exhaustive space, so the multiplier dials sweeps, not just loops.
     *
     *  Striding rather than truncating keeps the far end of the space - where the window-edge cases
     *  live - reachable at a low multiplier. The sweep completes at the 10× stress point rather
     *  than at the default, so a default run samples every space and a stress run covers them.
     */
    std::size_t sweep_stride(std::size_t complete) const noexcept {
        double const coverage = scale / 10.0;
        if (coverage >= 1.0 || complete == 0) return 1;
        std::size_t const wanted = static_cast<std::size_t>(complete * coverage);
        return wanted < 1 ? complete : complete / wanted;
    }
};

/** The @p step -th value of a rotation over @p count items that also advances a phase each full
 *  turn, so crossing it with another rotation of the same length still reaches every pair. */
inline std::size_t rotating_index(std::size_t step, std::size_t count) noexcept {
    return count ? (step + step / count) % count : 0;
}

/** Views a C array as a @c sz::span, so tables pass as one argument with their length. */
template <typename value_type_, std::size_t count_>
constexpr span<value_type_ const> span_over(value_type_ const (&array)[count_]) noexcept {
    return span<value_type_ const>(array, count_);
}

template <typename string_type_, typename other_string_type_>
inline string_type_ to_str(other_string_type_ const &other) noexcept {
    return string_type_(other.data(), other.size());
}

/**
 *  @brief A uniform distribution of characters, with a given alphabet size: the number of distinct
 *      characters in the distribution.
 *
 *  We can't use `std::uniform_int_distribution<char>` because the @c char overload is not supported
 *  by some platforms. MSVC, for example, requires one of @c short, @c int, @c long, `long long`,
 *  `unsigned short`, `unsigned int`, `unsigned long`, or `unsigned long long`.
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

/** Fills @p text with bytes drawn uniformly from @p alphabet, or from 1 to 255 when it is empty. */
inline void randomize_string(std::mt19937 &generator, std::span<char> text, std::string_view alphabet = {}) noexcept {
    if (alphabet.empty()) {
        uniform_u8_distribution_t distribution;
        std::generate(text.begin(), text.end(), [&]() -> char { return distribution(generator); });
        return;
    }
    uniform_u8_distribution_t distribution(0, static_cast<char>(alphabet.size() - 1));
    std::generate(text.begin(), text.end(), [&]() -> char { return alphabet[distribution(generator)]; });
}

inline std::string random_string(std::mt19937 &generator, std::size_t length,
                                 std::string_view alphabet) noexcept(false) {
    std::string result(length, '\0');
    randomize_string(generator, result, alphabet);
    return result;
}

inline std::string repeat(std::string const &patten, std::size_t count) noexcept(false) {
    std::string result(patten.size() * count, '\0');
    for (std::size_t i = 0; i < count; ++i) std::copy(patten.begin(), patten.end(), result.begin() + i * patten.size());
    return result;
}

/** Randomly slices a string into consecutive parts and passes those to @p slice_callback. */
template <typename slice_callback_type_>
inline void iterate_in_random_slices(std::mt19937 &generator, std::string const &text,
                                     slice_callback_type_ &&slice_callback) {
    std::size_t remaining = text.size();
    while (remaining > 0) {
        std::uniform_int_distribution<std::size_t> slice_length_distribution(1, remaining);
        std::size_t slice_length = slice_length_distribution(generator);
        slice_callback({text.data() + text.size() - remaining, slice_length});
        remaining -= slice_length;
    }
}

/**
 *  @brief Invokes @p body with a writable buffer placed at each of a representative spread of
 *      sub-cache-line byte offsets, so SIMD kernels are exercised at every alignment.
 *  @param[in] usable_length Minimum number of writable bytes guaranteed past the passed pointer.
 *  @param[in] body Callable as `body(sz_ptr_t pointer, std::size_t offset)`; the buffer is
 *      zero-filled per offset.
 */
template <typename body_type_>
inline void for_each_cacheline_offset_(std::size_t usable_length, body_type_ &&body) {
    static constexpr std::size_t offsets[] = {0, 1, 7, 8, 15, 16, 31, 32, 33, 48, 63};
    std::vector<char> storage(usable_length + 2 * STRINGZILLA_CACHE_LINE_BYTES + 1, '\0');
    for (std::size_t offset : offsets) {
        std::fill(storage.begin(), storage.end(), '\0');
        char *pointer = storage.data();
        while (reinterpret_cast<std::uintptr_t>(pointer) % STRINGZILLA_CACHE_LINE_BYTES != offset) ++pointer;
        body(reinterpret_cast<sz_ptr_t>(pointer), offset);
    }
}

/**
 *  @brief Runs @p body on a @p length -byte buffer flanked by canary bytes on both sides, then
 *      asserts the guards are intact, catching out-of-bounds writes on adversarial inputs.
 *  @param[in] length Number of usable bytes handed to @p body.
 *  @param[in] body Callable as `body(sz_ptr_t pointer, std::size_t length)`; the buffer is
 *      canary-filled per call.
 */
template <typename body_type_>
inline void with_guarded_buffer_(std::size_t length, body_type_ &&body) {
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

/** An allocator refusing every request, for asserting a kernel reports @c sz_bad_alloc_k and leaves
 *  its outputs alone. */
inline sz_memory_allocator_t refusing_allocator_() noexcept {
    sz_memory_allocator_t refusing;
    refusing.allocate = +[](sz_size_t, void *) -> void * { return nullptr; };
    refusing.free = +[](void *, sz_size_t, void *) {};
    refusing.handle = nullptr;
    return refusing;
}

/** A heap whose callbacks refuse every handle but its own, so a kernel passing the allocator where
 *  its @c handle belongs fails with @c sz_bad_alloc_k instead of reinterpreting the allocator. It
 *  counts the blocks it holds, so a test can assert that every one of them came back. */
struct handle_checked_heap_t {
    handle_checked_heap_t const *self = this;
    std::size_t live_allocations = 0;
    sz_memory_allocator_t allocator {};

    handle_checked_heap_t() noexcept {
        allocator.allocate = +[](sz_size_t length, void *handle) -> void * {
            handle_checked_heap_t &heap = *static_cast<handle_checked_heap_t *>(handle);
            if (heap.self != &heap) return nullptr;
            ++heap.live_allocations;
            return std::malloc(length);
        };
        allocator.free = +[](void *pointer, sz_size_t, void *handle) {
            --static_cast<handle_checked_heap_t *>(handle)->live_allocations;
            std::free(pointer);
        };
        allocator.handle = this;
    }
    handle_checked_heap_t(handle_checked_heap_t const &) = delete;
    handle_checked_heap_t &operator=(handle_checked_heap_t const &) = delete;
};

/**
 *  @brief Splits @p alphabet into its UTF-8 characters, so a multi-byte alphabet yields valid text.
 *
 *  A character runs from a lead byte to the last continuation byte after it, which needs no decoder
 *  and leaves an ASCII alphabet one character per byte.
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

/** Concatenates @p length characters drawn uniformly from @p characters. */
inline std::string random_string(std::mt19937 &generator, std::size_t length,
                                 std::vector<std::string> const &characters) noexcept(false) {
    std::uniform_int_distribution<std::size_t> distribution(0, characters.size() - 1);
    std::string result;
    while (length--) result += characters[distribution(generator)];
    return result;
}

/** Reads a member string start, for @c sz_sequence_t views over a `std::vector<std::string>`. */
inline sz_cptr_t sequence_get_start_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].data();
}

/** Reads a member string length, for @c sz_sequence_t views over a `std::vector<std::string>`. */
inline sz_size_t sequence_get_length_(void const *handle, sz_sorted_idx_t index) {
    return (*reinterpret_cast<std::vector<std::string> const *>(handle))[index].size();
}

/** Fills an @c sz_sequence_t view over a `std::vector<std::string>` via the shared accessors. */
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

    fuzzy_config_t() = default;
    fuzzy_config_t(char const *alphabet, std::size_t batch_size, std::size_t min_string_length,
                   std::size_t max_string_length)
        : alphabet(alphabet), batch_size(batch_size), min_string_length(min_string_length),
          max_string_length(max_string_length) {}
};

inline void randomize_strings(std::mt19937 &generator, fuzzy_config_t config, std::vector<std::string> &array) {
    array.resize(config.batch_size);

    std::vector<std::string> const characters = alphabet_characters(config.alphabet);
    std::uniform_int_distribution<std::size_t> length_distribution(config.min_string_length, config.max_string_length);
    for (std::size_t index = 0; index != config.batch_size; ++index)
        array[index] = random_string(generator, length_distribution(generator), characters);
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

/** Prints the lines every test and benchmark opens with: the version and both capability lists. */
inline void log_environment() {
    fmt::println("StringZilla {}.{}.{}", STRINGZILLA_H_VERSION_MAJOR, STRINGZILLA_H_VERSION_MINOR,
                 STRINGZILLA_H_VERSION_PATCH);
    // The library already answers both questions; a wall of `STRINGZILLA_TARGET_*` echoes only repeats the first one.
    fmt::println("- Compiled for: {}", sz_capabilities_to_string(sz_capabilities_comptime()));
    fmt::println("- This machine: {}", sz_capabilities_to_string(sz_capabilities_runtime()));
}

#if STRINGZILLA_TARGET_CUDA

/**
 *  @brief Prints `- CUDA: <name> sm_<major><minor>` for the first visible device, or
 *      `- CUDA: no device`.
 *  @return Whether a device is visible; without one, the GPU tests and benchmarks skip.
 */
inline bool log_cuda_device() {
    // The device is asked directly rather than through `sz_capabilities`: that verb answers for the library this
    // binary links, and `define_stringzilla_library` compiles the core without CUDA, so it reports none.
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

/** Prints the run's seed, how to rerun one test under it, and its scale unless 1, below the lines
 *  of @c log_environment. */
inline void print_test_environment(test_environment_t const &environment) noexcept {
    fmt::println("- Seed: {}", environment.seed);
    fmt::println("- Rerun one test: STRINGZILLA_SEED={} STRINGZILLA_FILTER='^<name>$' {}", environment.seed,
                 environment.program);
    if (environment.scale != 1.0) fmt::println("- Scale: {}", environment.scale);
    std::fflush(stdout); // Ensure output is visible even on crash
}

#pragma region Test Runner

/** Prints a backtrace on a fatal signal, so a crashing/aborting kernel self-localizes instead of
 *  dying silently - especially under output redirection in CI. Writes raw, since the crashing
 *  thread may already hold the stdio lock that @c fmt::println takes. */
inline void test_fatal_signal_handler(int signal_number) noexcept {
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

/** Installs the fatal-signal backtrace handler and line-buffers stdout. Shared by the serial
 *  @c .cpp and CUDA @c .cu test entry points; call once from @c main. */
inline void install_test_signal_handlers() noexcept {
    // Line-buffer, so progress survives a crash under output redirection.
    // Size must be nonzero: Windows ucrt fast-fails on a zero-sized buffering mode.
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    for (int signal_number : {SIGSEGV, SIGABRT, SIGILL, SIGFPE}) std::signal(signal_number, test_fatal_signal_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
}

/**
 *  @brief Runs one named test: honors @c STRINGZILLA_FILTER, times it, and reports the outcome.
 *  @return The number of failures: 0 on success or when skipped, 1 on a failed check or a thrown
 *      exception, which are reported with the line that reruns the test alone.
 *
 *  A test taking a @c test_context_t draws from a generator seeded by @c mix_seed, so its inputs
 *  never depend on which tests ran first. Library asserts via @c sz_assert_ still abort the
 *  process, self-localizing through the installed signal handler.
 */
template <typename function_type_>
inline std::size_t run_test(test_environment_t const &environment, std::string_view name,
                            function_type_ &&test_function) noexcept {
    if (!environment.selects(name)) {
        fmt::println("- {} ... skipped (STRINGZILLA_FILTER)", name);
        std::fflush(stdout);
        return 0;
    }
    fmt::println("- {} ...", name);
    std::fflush(stdout);
    test_context_t context {std::mt19937(mix_seed(environment.seed, name)), environment.scale};
    auto const start = std::chrono::steady_clock::now();
    try {
        if constexpr (std::is_invocable_v<function_type_ &, test_context_t &>) test_function(context);
        else test_function();
    }
    catch (std::exception const &error) {
        fmt::println(stderr, "- {} ... FAILED: {}", name, error.what());
        fmt::println(stderr, "  rerun: STRINGZILLA_SEED={} STRINGZILLA_FILTER='^{}$' {}", environment.seed, name,
                     environment.program);
        return 1;
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::println("- {} ... ok ({:.2f} s)", name, seconds);
    std::fflush(stdout);
    return 0;
}

#pragma endregion Test Runner

} // namespace ashvardanian::stringzilla::test

/*  Cross-translation-unit test declarations. These live at global scope to match the TU
 *  definitions; the using-declaration names the context the drawing tests take. */
using ashvardanian::stringzilla::test::test_context_t;

#pragma region Basic Utilities

void test_arithmetic_unit();
void test_sequence_unit();
void test_strings_tape_assign_unit();
void test_strings_tape_overflow_unit();
void test_allocator_unit();
void test_byteset_unit();

#pragma endregion Basic Utilities

#pragma region Hashing

void test_hash_unit();
void test_hash_safety();
void test_hash_all(test_context_t &context);
void test_hash_multiseed_all(test_context_t &context);

#pragma endregion Hashing

#pragma region Ciphers

void test_cipher_unit();
void test_cipher_safety(test_context_t &context);
void test_cipher_all(test_context_t &context);

#pragma endregion Ciphers

#pragma region UTF8

void test_utf8_runes_unit();
void test_utf8_runes_scripts_unit();
void test_utf8_runes_safety(test_context_t &context);
void test_utf8_runes_all(test_context_t &context);
void test_utf8_tokens_unit();
void test_utf8_tokens_scripts_unit();
void test_utf8_tokens_safety(test_context_t &context);
void test_utf8_tokens_all(test_context_t &context);
void test_utf8_wordbreaks_unit();
void test_utf8_wordbreaks_rules();
void test_utf8_wordbreaks_safety(test_context_t &context);
void test_utf8_wordbreaks_all(test_context_t &context);
void test_utf8_graphemes_unit();
void test_utf8_graphemes_rules();
void test_utf8_graphemes_safety(test_context_t &context);
void test_utf8_graphemes_all(test_context_t &context);
void test_utf8_sentences_unit();
void test_utf8_sentences_rules();
void test_utf8_sentences_safety(test_context_t &context);
void test_utf8_sentences_all(test_context_t &context);
void test_utf8_linebreaks_unit();
void test_utf8_linebreaks_rules();
void test_utf8_linebreaks_safety(test_context_t &context);
void test_utf8_linebreaks_all(test_context_t &context);
void test_utf8_norm_unit();
void test_utf8_norm_safety(test_context_t &context);
void test_utf8_norm_all(test_context_t &context);
void test_utf8_delimiters_unit();
void test_utf8_delimiters_safety(test_context_t &context);
void test_utf8_delimiters_all(test_context_t &context);

#pragma endregion UTF8

#pragma region Uncased UTF8

void test_uncased_unit();
void test_uncased_scripts_unit();
void test_uncased_regressions_unit();
void test_uncased_all(test_context_t &context);
void test_uncased_safety(test_context_t &context);

#pragma endregion Uncased UTF8

#pragma region String Class and STL Compatibility

template <typename string_type>
void test_ascii_unit();

void test_memory_unit(std::size_t max_l2_size = 1024ull * 1024ull);
void test_memory_large_unit();
void test_memory_all(test_context_t &context);
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
void test_string_reserve_unit();
void test_memory_stability_equivalence(test_context_t &context, std::size_t length);
void test_string_updates_equivalence(test_context_t &context, std::size_t repetitions = 1024);

#pragma endregion String Class and STL Compatibility

#pragma region Search and Comparison

void test_compare_unit();
void test_extensions_ranges_unit();
void test_find_unit();
void test_find_safety();
void test_find_all(test_context_t &context);
void test_find_misaligned_equivalence(test_context_t &context);
void test_lookup_equivalence(test_context_t &context, std::size_t lookup_tables_to_try = 32,
                             std::size_t slices_per_table = 16);

#pragma endregion Search and Comparison

#pragma region Sequence Algorithms

void test_sort_all(test_context_t &context);
void test_sort_unit();
void test_sort_safety();
void test_sort_reference_equivalence(test_context_t &context);
void test_intersect_unit();
void test_intersect_equivalence(test_context_t &context);
void test_levenshtein_unit();
void test_levenshtein_all(test_context_t &context);
void test_levenshtein_safety(test_context_t &context);
void test_overlap_unit();
void test_overlap_all(test_context_t &context);
void test_overlap_safety(test_context_t &context);
void test_substrings_unit();
void test_substrings_all(test_context_t &context);
void test_substrings_safety(test_context_t &context);

#pragma endregion Sequence Algorithms
