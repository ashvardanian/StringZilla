/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2024
 *  @brief Helper structures and functions for C++ unit- and stress-tests.
 *
 *  @section test_environment_variables Environment Variables
 *
 *  Read once by @c read_settings, with a value that does not parse ending the run with status 1:
 *
 *  @verbatim
 *  Variable            Default  Meaning
 *  STRINGZILLA_SEED    42       Seed for every random generator, or "random" to draw one, which a run prints
 *  STRINGZILLA_SCALE   1        Multiplier for every baseline iteration count, like 0.1 or 10
 *  STRINGZILLA_FILTER  none     ECMAScript regex over test names, or a substring when it does not compile
 *  @endverbatim
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
 *    a dispatch point or a capability's kernel, or the library against `std::`. This tier owns
 *    randomness.
 *  - @c _safety : Malformed, adversarial and boundary inputs. Asserts survival, bounds and stated
 *    refusals - never answers, since a wrong answer is not what is under test here. Scales with
 *    @c STRINGZILLA_SCALE alongside @c _equivalence; only @c _unit is pinned.
 *  - @c _all : Drives the tiers above over the family's dispatch points. Holds no assertions of
 *    its own; a literal here belongs in @c _unit.
 *  - @c _rules : Annex rule coverage, where a family transcribes a published spec: UAX-29, UAX-14.
 *
 *  The family files test the dispatch points and `stringzilla.hpp`. The kernel-level checks live in
 *  `cross.hpp`, and each `cross_<arch>.cpp` runs them over that architecture's kernels by name, one
 *  @c cross_section_t section per capability, as `test_<family>_<tier>_<capability>`.
 *
 *  @section test_example_usage Example Usage
 *
 *  @code{.sh}
 *  # Draw a fresh seed instead of the default 42
 *  STRINGZILLA_SEED=random ./build_release/stringzilla_test
 *
 *  # Quick smoke test (10% of normal iterations)
 *  STRINGZILLA_SCALE=0.1 ./build_release/stringzilla_test
 *
 *  # Fast inner loop: only the UTF-8 tests, at 10% iterations
 *  STRINGZILLA_FILTER=utf8 STRINGZILLA_SCALE=0.1 ./build_release/stringzilla_test
 *
 *  # Thorough CI stress test (10x normal iterations)
 *  STRINGZILLA_SCALE=10 ./build_release/stringzilla_test
 *
 *  # Combine both for CI fuzzing
 *  STRINGZILLA_SEED=12345 STRINGZILLA_SCALE=5 ./build_release/stringzilla_test
 *  @endcode
 */
#pragma once
#include <climits> // `INT_MAX`
#include <cmath>   // `std::isfinite`
#include <csignal> // `std::signal`, `SIGSEGV`, `SIGABRT`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uintptr_t` for cache-line alignment
#include <cstdio>  // `std::setvbuf`, `stderr`
#include <cstdlib> // `std::getenv`, `std::strtod`, `std::exit`, `std::malloc`, `std::free`
#include <cstring> // `std::strcmp`, `std::strlen`

#include <algorithm>    // `std::copy`, `std::generate`
#include <array>        // `std::array`
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
#include <tuple>        // `std::tuple`, for `call_best`
#include <type_traits>  // `std::is_enum_v`
#include <utility>      // `std::index_sequence`
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

#include "stringzilla/metal.h" // The Metal layer the Metal suites drive
#include "stringzilla/types.hpp"

namespace sz = ashvardanian::stringzilla;

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
        try {                                     \
            sz_unused_(expression);               \
        }                                         \
        catch (exception_type const &) {          \
            break;                                \
        }                                         \
        verify(false);                            \
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

/** Calls @p kernel, a CPU capability kernel or a @c cpu_best, whose result is the out-parameter
 *  just before the stream; verifies that it succeeded and returns that result. */
template <typename result_type_, typename kernel_type_, typename... arguments_types_>
result_type_ kernel_result(kernel_type_ const &kernel, arguments_types_... arguments) {
    result_type_ result {};
    verify(kernel(arguments..., &result, nullptr) == sz_success_k);
    return result;
}

using tape_view_t = tape_view<char, sz_size_t>;

struct device_backend_t {
    sz::device_t selected = sz::device_t::cpu();
    sz_capability_t capabilities = 0;
    sz_stream_t stream = nullptr;
    sz_allocator_t unified {};
    sz_status_t (*init)(sz_size_t, sz_stream_t *) = nullptr;
    sz_status_t (*free)(sz_stream_t) = nullptr;
};

using tape_t = tape<char, sz_u64_t, unified_alloc<char>, tape_termination_t::packed_k>;
template <typename value_type_>
using unified_vector = std::vector<value_type_, unified_alloc<value_type_>>;

/** Owns one native stream, released after its dependent allocations and engines. */
struct stream_t {
    sz_stream_t handle = nullptr;
    sz_status_t (*release)(sz_stream_t);

    stream_t(sz_size_t ordinal, sz_status_t (*init)(sz_size_t, sz_stream_t *), sz_status_t (*release)(sz_stream_t))
        : release(release) {
        verify(init(ordinal, &handle) == sz_success_k);
    }
    stream_t(stream_t const &) = delete;
    stream_t &operator=(stream_t const &) = delete;
    ~stream_t() noexcept { release(handle); }
};

/** A 32-bit generator seed, kept apart from counts so neither passes for the other. */
struct seed_t {
    std::uint32_t value = 0;
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

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<seed_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return seed_t {static_cast<std::uint32_t>(std::random_device {}())};
    std::uint32_t seed = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed_t {seed};
}

inline seed_t env_seed(char const *name, seed_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
}

/** A requested GPU, before checking whether its runtime and ordinal are available. */
struct device_selection_t {
    sz::device_kind_t backend;
    std::size_t ordinal;
};

inline std::string_view device_name(sz::device_kind_t kind) noexcept {
    switch (kind) {
    case sz::device_kind_t::cpu_k: return "cpu";
    case sz::device_kind_t::cuda_k: return "cuda";
    case sz::device_kind_t::rocm_k: return "rocm";
    case sz::device_kind_t::metal_k: return "metal";
    }
    return "unrecognized";
}

inline std::optional<std::vector<device_selection_t>> parse_devices(std::string_view text) {
    std::vector<device_selection_t> devices;
    do {
        std::size_t const comma = text.find(',');
        std::string_view const entry = text.substr(0, comma);
        std::size_t const colon = entry.find(':');
        if (colon == std::string_view::npos) return std::nullopt;
        std::string_view const vendor = entry.substr(0, colon);
        sz::device_kind_t backend;
        if (vendor == "cuda") backend = sz::device_kind_t::cuda_k;
        else if (vendor == "rocm") backend = sz::device_kind_t::rocm_k;
        else if (vendor == "metal") backend = sz::device_kind_t::metal_k;
        else return std::nullopt;
        std::string_view const number = entry.substr(colon + 1);
        std::size_t ordinal = 0;
        auto const [end, error] = std::from_chars(number.data(), number.data() + number.size(), ordinal);
        if (error != std::errc {} || end != number.data() + number.size()) return std::nullopt;
        devices.push_back({backend, ordinal});
        if (comma == std::string_view::npos) return devices;
        text.remove_prefix(comma + 1);
    } while (!text.empty());
    return std::nullopt;
}

/** Every test setting, with its default as the initializer, filled once by @c read_settings. */
struct settings_t {
    std::optional<std::vector<device_selection_t>> devices;

    /** Seed every test's generator is mixed from. */
    seed_t seed {42};

    /** Multiplier for each test's own iteration counts. */
    double iterations_scale = 1.0;

    /** Tests to run, by ECMAScript regex or else substring; points into @c environ. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** The binary's @c argv[0], which closes every rerun line. */
    std::string_view program;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Reads every @c settings_t variable for the binary named @p program. */
inline settings_t read_settings(char const *program) noexcept {
    settings_t settings;
    if (env_text("STRINGZILLA_DEVICES"))
        settings.devices = env_parsed("STRINGZILLA_DEVICES", std::vector<device_selection_t> {}, parse_devices,
                                      "comma-separated devices such as cuda:0,rocm:1");
    settings.program = program;
    auto const parse_scale = [](std::string_view text) -> std::optional<double> {
        std::string const terminated(text);
        char *end = nullptr;
        double const value = std::strtod(terminated.c_str(), &end);
        if (end != terminated.c_str() + terminated.size() || !(value > 0) || !std::isfinite(value)) return std::nullopt;
        return value;
    };
    settings.filter = env_text("STRINGZILLA_FILTER").value_or("");
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
    settings.seed = env_seed("STRINGZILLA_SEED", settings.seed);
    settings.iterations_scale = env_parsed("STRINGZILLA_SCALE", settings.iterations_scale, parse_scale,
                                           "a positive number like 0.1 or 10");
    return settings;
}

/**
 *  @brief The seed of the test named @p name in a run seeded with @p seed.
 *
 *  FNV-1a over @c std::uint32_t is exact on every platform, like @c std::seed_seq, and never
 *  touches the kernels under test, unlike @c sz_hash_best. The harness must not draw its inputs
 *  through the kernels it validates, and it must land on the same stream everywhere, or
 *  `STRINGZILLA_SEED=7` stops meaning the same bytes on Arm as it does on x86.
 */
constexpr std::uint32_t mix_seed(seed_t seed, std::string_view name) noexcept {
    std::uint32_t mixed = seed.value ^ 2166136261u;
    for (char const character : name) mixed = (mixed ^ static_cast<unsigned char>(character)) * 16777619u;
    return mixed;
}

/** What one test draws its inputs from and sizes its loops by, built for it by @c run_test. */
struct test_context_t {
    std::mt19937 generator;
    double iterations_scale = 1.0;

    /** Scales a @p baseline iteration count by @c STRINGZILLA_SCALE, never below 1. */
    std::size_t iterations(std::size_t baseline) const noexcept {
        double const scaled = baseline * iterations_scale;
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
        double const coverage = iterations_scale / 10.0;
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

/** Views a C array as a @c std::span, so tables pass as one argument with their length. */
template <typename value_type_, std::size_t count_>
constexpr std::span<value_type_ const> span_over(value_type_ const (&array)[count_]) noexcept {
    return std::span<value_type_ const>(array, count_);
}

/** The byte offset within @p text of each slice @p range yields, to assert where slices land. */
template <typename text_type_, typename range_type_>
std::vector<std::ptrdiff_t> offsets_within(text_type_ const &text, range_type_ const &range) {
    std::vector<std::ptrdiff_t> offsets;
    for (auto const slice : range) offsets.push_back(slice.data() - text.data());
    return offsets;
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
inline sz_allocator_t refusing_allocator_() noexcept {
    sz_allocator_t refusing;
    refusing.allocate = +[](sz_size_t, void *, sz_stream_t) -> void * { return nullptr; };
    refusing.free = +[](void *, sz_size_t, void *, sz_stream_t) {};
    refusing.handle = nullptr;
    return refusing;
}

/** A heap whose callbacks refuse every handle but its own, so a kernel passing the allocator where
 *  its @c handle belongs fails with @c sz_bad_alloc_k instead of reinterpreting the allocator. It
 *  counts the blocks it holds, so a test can assert that every one of them came back. */
struct handle_checked_heap_t {
    handle_checked_heap_t const *self = this;
    std::size_t live_allocations = 0;
    sz_allocator_t allocator {};

    handle_checked_heap_t() noexcept {
        allocator.allocate = +[](sz_size_t length, void *handle, sz_stream_t) -> void * {
            handle_checked_heap_t &heap = *static_cast<handle_checked_heap_t *>(handle);
            if (heap.self != &heap) return nullptr;
            ++heap.live_allocations;
            return std::malloc(length);
        };
        allocator.free = +[](void *pointer, sz_size_t, void *handle, sz_stream_t) {
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

inline sz_sequence_t sequence_from_(std::span<sz_string_view_t const> views) {
    sz_sequence_t sequence;
    sz_sequence_from_string_views(views.data(), views.size(), &sequence);
    return sequence;
}

/** Borrows a sequence view of the supplied strings. */
inline sz_sequence_t sequence_from_(std::vector<std::string> const &strings) {
    sz_sequence_t sequence;
    sequence.handle = &strings;
    sequence.count = (sz_size_t)strings.size();
    sequence.get_start = [](void const *handle, sz_size_t index) -> sz_cptr_t {
        auto const &strings = *static_cast<std::vector<std::string> const *>(handle);
        return strings[index].data();
    };
    sequence.get_length = [](void const *handle, sz_size_t index) -> sz_size_t {
        auto const &strings = *static_cast<std::vector<std::string> const *>(handle);
        return strings[index].size();
    };
    return sequence;
}

struct fuzzy_config_t {

    /** Drawn one UTF-8 character at a time, so `"αβγ"` yields valid multi-byte text. */
    std::string alphabet = "ABC";
    std::size_t batch_size = 16;

    /** In characters, which equals bytes only for an ASCII alphabet. */
    std::size_t min_string_length = 1;
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

inline void print(sz::device_t const &device) {
    fmt::println("Device: {}:{}", device_name(device.kind()), device.ordinal());
}

/** Prints each setting as "- Name: value", in the grammar it parses from, then a rerun template. */
inline void print(settings_t const &settings) {
    if (settings.devices) {
        for (device_selection_t const &device : *settings.devices)
            fmt::println("- Device: {}:{}", device_name(device.backend), device.ordinal);
    }
    else fmt::println("- Devices: auto");
    fmt::println("- Seed: {}", settings.seed.value);
    fmt::println("- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    fmt::println("- Scale: {}", settings.iterations_scale);
    fmt::println("- Rerun one test: STRINGZILLA_SEED={} STRINGZILLA_FILTER='^<name>$' {}", settings.seed.value,
                 settings.program);
}

/** The facts this binary and this machine report: the library version, the capabilities compiled
 *  in and detected. */
struct machine_t {
    std::array<unsigned, 3> version {STRINGZILLA_H_VERSION_MAJOR, STRINGZILLA_H_VERSION_MINOR,
                                     STRINGZILLA_H_VERSION_PATCH};
    sz_capability_t compiled = 0;
    sz_capability_t detected = 0;
};

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    sz_capabilities_compiled_cpu(&machine.compiled), sz_capabilities_detected_cpu(&machine.detected);
    return machine;
}

/** Prints the version line, the capabilities as "- Compiled for:" and "- This machine:", and in
 *  GPU builds "- CUDA:", "- ROCm:" or "- Metal:". */
inline void print(machine_t const &machine) {
    char compiled[STRINGZILLA_CAPABILITIES_NAME_CAPACITY], detected[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
    sz_capabilities_name(machine.compiled, compiled, sizeof(compiled));
    sz_capabilities_name(machine.detected, detected, sizeof(detected));
    fmt::println("StringZilla {}.{}.{}", machine.version[0], machine.version[1], machine.version[2]);
    fmt::println("- Compiled for: {}", compiled);
    fmt::println("- This machine: {}", detected);
}

/** Everything a test reads, built once in @c main and passed down by reference. */
struct environment_t {
    settings_t settings;
    machine_t machine;
};

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
inline std::size_t run_test(settings_t const &settings, std::string_view name,
                            function_type_ &&test_function) noexcept {
    if (!settings.selects(name)) {
        fmt::println("- {} ... skipped (STRINGZILLA_FILTER)", name);
        return 0;
    }
    fmt::println("- {} ...", name);
    test_context_t context {std::mt19937(mix_seed(settings.seed, name)), settings.iterations_scale};
    auto const start = std::chrono::steady_clock::now();
    try {
        if constexpr (std::is_invocable_v<function_type_ &, test_context_t &>) test_function(context);
        else test_function();
    }
    catch (std::exception const &error) {
        fmt::println(stderr, "- {} ... FAILED: {}", name, error.what());
        fmt::println(stderr, "  rerun: STRINGZILLA_SEED={} STRINGZILLA_FILTER='^{}$' {}", settings.seed.value, name,
                     settings.program);
        return 1;
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::println("- {} ... ok ({:.2f} s)", name, seconds);
    return 0;
}

/**
 *  @brief Runs an architecture's kernel cross-checks via @c run_test, a section per capability.
 *
 *  `#if STRINGZILLA_TARGET_<KIT>` says a capability's kernels are built, and @c section says
 *  whether this CPU runs them, announcing once a section it cannot run and skipping its checks.
 */
struct cross_section_t {
    environment_t const &env;
    sz_capability_t detected = env.machine.detected;
    bool runnable = true;
    std::size_t failures = 0;

    explicit cross_section_t(environment_t const &env) noexcept : env(env) {}

    /** Opens the section of the kernels that need @p capability. */
    void section(std::string_view title, sz_capability_t capability) noexcept {
        runnable = (detected & capability) != 0;
        if (runnable) return fmt::println("\n{}:", title);
        char missing_names[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        sz_capabilities_name(capability & ~detected, missing_names, sizeof(missing_names));
        fmt::println("\n{}: skipped, {} not detected", title, missing_names);
    }

    /** Runs @p test as @c run_test does, unless this CPU lacks the section's capability. */
    template <typename function_type_>
    void operator()(std::string_view name, function_type_ &&test) {
        if (runnable) failures += run_test(env.settings, name, std::forward<function_type_>(test));
    }
};

#pragma endregion Test Runner

#pragma region Kernel Cross Checks

std::size_t test_cross_serial(environment_t const &env);
std::size_t test_cross_x8664(environment_t const &env);
std::size_t test_cross_arm64(environment_t const &env);
std::size_t test_cross_riscv64(environment_t const &env);
std::size_t test_cross_loongarch64(environment_t const &env);
std::size_t test_cross_ppc64(environment_t const &env);
std::size_t test_cross_wasm(environment_t const &env);
std::size_t test_cross_cuda(environment_t const &env, std::size_t ordinal);
std::size_t test_cross_rocm(environment_t const &env, std::size_t ordinal);
std::size_t test_cross_metal(environment_t const &env, std::size_t ordinal);

#pragma endregion Kernel Cross Checks

#pragma region Basic Utilities

void test_arithmetic_unit();
void test_sequence_unit();
void test_tape_assign_unit();
void test_tape_overflow_unit();
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

void test_utf8_uncased_unit();
void test_utf8_uncased_scripts_unit();
void test_utf8_uncased_regressions_unit();
void test_utf8_uncased_all(test_context_t &context);
void test_utf8_uncased_safety(test_context_t &context);

#pragma endregion Uncased UTF8

#pragma region String Class and STL Compatibility

template <typename string_type>
void test_ascii_unit();

void test_memory_unit();
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

} // namespace ashvardanian::stringzilla::test
