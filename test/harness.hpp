/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2024
 *  @brief Helper structures and functions for C++ unit- and stress-tests.
 *
 *  Every setting is a @c STRINGZILLA_<NAME> variable that a @c --name flag overrides;
 *  CONTRIBUTING.md lists them. A filter that selects no case fails the run rather than passing
 *  an empty suite.
 *
 *  @section test_failure_model Failure Model
 *
 *  A failed check prints where it failed, counts into @c test_run, and lets the case go on, so one
 *  run lists every mismatch it reaches. @c run_test_case diffs that count around each case and
 *  prints a rerun line under a failing one; @c finish_test_run makes the exit status. Where the
 *  step after a failed check would index or dereference past what it checked, the case returns on
 *  the check's answer instead, as in <tt>if (!expect_true(found)) return;</tt>.
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
 *  An exhaustive space strides by <tt>sweep_stride(test_run.settings.scale / 10)</tt>, so a default run
 *  samples a tenth of it and the 10x stress run covers it, reaching the far end at every scale.
 *
 *  A case's label names the code under test: a driver's own function name, as @c test_hash_unit,
 *  or a kernel cross-check as a call, as @c check_find_unit_(find_haswell). The family files test
 *  the dispatch points and @c stringzilla.hpp. The kernel-level checks live in @c cross.hpp, and
 *  each @c cross_<arch>.cpp runs them over that architecture's kernels by name, behind one
 *  @c capability_gate_t per capability.
 *
 *  @section test_example_usage Example Usage
 *
 *  @code{.sh}
 *  # Draw a fresh seed instead of the default 42
 *  STRINGZILLA_SEED=random ./build_release/stringzilla_test
 *
 *  # Quick smoke test (10% of normal iterations)
 *  ./build_release/stringzilla_test --scale 0.1
 *
 *  # Fast inner loop: only the UTF-8 families, at 10% iterations
 *  STRINGZILLA_FILTER='^test_utf8_' STRINGZILLA_SCALE=0.1 ./build_release/stringzilla_test
 *
 *  # Thorough CI stress test (10x normal iterations)
 *  STRINGZILLA_SCALE=10 ./build_release/stringzilla_test
 *
 *  # Combine both for CI fuzzing
 *  STRINGZILLA_SEED=12345 STRINGZILLA_SCALE=5 ./build_release/stringzilla_test
 *  @endcode
 */
#pragma once
#include <cctype>    // `std::tolower`
#include <cinttypes> // `PRIu64`, `PRIuLEAST32`
#include <cmath>     // `std::isfinite`, `std::llround`
#include <csignal>   // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`, `SIGTRAP`
#include <cstddef>   // `std::ptrdiff_t`
#include <cstdint>   // `std::uintptr_t` for cache-line alignment, `std::uint_least32_t`
#include <cstdio>    // `std::setvbuf`, `std::fprintf`, `stderr`
#include <cstdlib>   // `std::getenv`, `std::strtod`, `std::exit`, `std::malloc`, `std::free`
#include <cstring>   // `std::strcmp`, `std::strncmp`, `std::strlen`

#include <algorithm>       // `std::copy`, `std::generate`, `std::max`
#include <array>           // `std::array`
#include <atomic>          // `std::atomic`, for `test_run`
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::steady_clock` for per-case timing
#include <functional>      // `std::invoke`
#include <memory>          // `std::unique_ptr`, which owns a backend's arena block
#include <optional>        // `std::optional`
#include <random>          // `std::random_device`, `std::uniform_int_distribution`
#include <regex>           // `std::regex_search` for `STRINGZILLA_FILTER`
#include <source_location> // `std::source_location`, the line a failed check names
#include <span>            // `std::span`, `std::as_bytes`
#include <string>          // `std::string`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <thread>          // `std::this_thread::yield`
#include <tuple>           // `std::tuple`, for `call_best`
#include <type_traits>     // `std::is_enum_v`, `std::invoke_result_t`
#include <typeinfo>        // `typeid`, naming the exception `expect_throws` wanted
#include <utility>         // `std::index_sequence`
#include <vector>          // `std::vector`

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include "stringzilla/metal.h" // The Metal layer the Metal suites drive
#include "stringzilla/types.hpp"

#if STRINGZILLA_OS_WINDOWS_
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
/*  Only glibc and Darwin ship `backtrace` in libc itself; FreeBSD and musl need another library. */
#if (defined(__GLIBC__) || STRINGZILLA_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

namespace sz = ashvardanian::stringzilla;

namespace ashvardanian::stringzilla::test {

/** Stops in an attached debugger, or dumps core without one; SIGTRAP stays outside the crash
 *  handlers. */
inline void trap_into_debugger() noexcept {
#if STRINGZILLA_OS_WINDOWS_
    __debugbreak();
#else
    std::raise(SIGTRAP);
#endif
}

/** Prints the case in progress and a backtrace on a fatal signal; defined beside the run. */
inline void install_test_signal_handlers() noexcept;

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    std::fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const *setting_text(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const *text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char **argv, char const *variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const *off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                double fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char **argv, char const *variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (error == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string pattern;
    std::optional<std::regex> regex;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(label.begin(), label.end(), *regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(filter.pattern);
    }
    catch (std::regex_error const &) {
    }
#else
    filter.regex.emplace(filter.pattern);
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
    char const *program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit state, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t state = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : state(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(state += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

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

/** What a run reads: the common settings, how hard to push and how to treat failures and skips,
 *  then the GPUs. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;

    /** The GPUs @c STRINGZILLA_DEVICES names, or every one this machine has when unset. */
    std::optional<std::vector<device_selection_t>> devices;
};

/** Reads every @c test_settings_t field; defined once the run's own reader exists. */
inline test_settings_t read_test_settings(int argc, char **argv) noexcept;

/** The facts this binary and this machine report: the library version, the capabilities compiled
 *  in and detected. */
struct machine_t {
    std::array<unsigned, 3> version {STRINGZILLA_VERSION_MAJOR, STRINGZILLA_VERSION_MINOR, STRINGZILLA_VERSION_PATCH};
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
inline void print(machine_t const &machine, std::FILE *stream = stdout) {
    char compiled[STRINGZILLA_CAPABILITIES_NAME_CAPACITY], detected[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
    sz_capabilities_name(machine.compiled, compiled, sizeof(compiled));
    sz_capabilities_name(machine.detected, detected, sizeof(detected));
    fmt::println(stream, "StringZilla {}.{}.{}", machine.version[0], machine.version[1], machine.version[2]);
    fmt::println(stream, "- Compiled for: {}", compiled);
    fmt::println(stream, "- This machine: {}", detected);
}

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const *skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const *reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const *current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const *> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_ &&predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = static_cast<double>(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char **argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    std::printf("- Seed: %" PRIu64 ", scale: %g, filter: %s\n", test_run.settings.seed, test_run.settings.scale,
                test_run.settings.filter.pattern.empty() ? "none" : test_run.settings.filter.pattern.c_str());
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    std::fprintf(stderr, "  check failed at %s:%" PRIuLEAST32 "\n", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_met does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_met, std::source_location where = std::source_location::current()) noexcept {
    if (!is_met) record_failed_check(where);
    return is_met;
}

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const *label, std::size_t failed_checks, char const *skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        std::printf("- %s ... skipped: %s\n", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        std::printf("- %s ... ok (%.2f s)\n", label, seconds);
        return;
    }
    std::fflush(stdout);
    std::fprintf(stderr, "- %s ... FAILED: %zu failed check(s)\n", label, failed_checks);
    std::fprintf(stderr, "  rerun: %s --seed %" PRIu64 " --filter '^", test_run.settings.program,
                 test_run.settings.seed);
    for (char const *character = label; *character; ++character) {
        if (std::strchr("\\^$.|?*+()[]{}", *character)) std::fputc('\\', stderr); // labels are code, the filter a regex
        std::fputc(*character, stderr);
    }
    std::fputs("$'\n", stderr);
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const *label, case_function_type_ &&case_function, argument_types_ &&...arguments) noexcept {
    if (!test_run.settings.filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    std::printf("%zu passed, %zu failed, %zu skipped, seed %" PRIu64 "\n", test_run.passed_cases, test_run.failed_cases,
                test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

inline test_settings_t read_test_settings(int argc, char **argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "STRINGZILLA_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "STRINGZILLA_FILTER", "--filter");
    settings.program = argc != 0 ? argv[0] : "";
    settings.scale = read_real_setting(argc, argv, "STRINGZILLA_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "STRINGZILLA_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "STRINGZILLA_NO_SKIPS", "--no-skips");
    if (char const *text = setting_text(argc, argv, "STRINGZILLA_DEVICES", "--devices")) {
        settings.devices = parse_devices(text);
        if (!settings.devices)
            exit_unparsed("STRINGZILLA_DEVICES", text, "comma-separated devices such as cuda:0,rocm:1");
    }
    return settings;
}

#pragma region Crash Localization

/**
 *  @brief Names the signal and the case in progress, prints a backtrace, then re-raises the signal,
 *      so a crashing or aborting kernel self-localizes instead of dying silently under CI's output
 *      redirection. Library asserts via @c sz_assert_ abort the process and land here too.
 *
 *  Writes through a raw @c write, since the crashing thread may already hold the stdio lock that
 *  @c fmt::println takes.
 */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    auto const write_text = [](std::string_view text) noexcept {
#if STRINGZILLA_OS_WINDOWS_
        [[maybe_unused]] auto const written = _write(2, text.data(), static_cast<unsigned>(text.size()));
#else
        [[maybe_unused]] auto const written = ::write(STDERR_FILENO, text.data(), text.size());
#endif
    };
    std::string_view signal_name = "SIGSEGV";
    if (signal_number == SIGABRT) signal_name = "SIGABRT";
    else if (signal_number == SIGILL) signal_name = "SIGILL";
    else if (signal_number == SIGFPE) signal_name = "SIGFPE";
#if defined(SIGBUS)
    else if (signal_number == SIGBUS) signal_name = "SIGBUS";
#endif
    write_text("\n*** ");
    write_text(signal_name);
    write_text(" in case '");
    write_text(test_run.current_case ? test_run.current_case : "none");
    write_text("' - backtrace follows ***\n");
#if (defined(__GLIBC__) || STRINGZILLA_OS_APPLE_) && __has_include(<execinfo.h>)
    void *frames[64];
    int const frames_count = backtrace(frames, sizeof(frames) / sizeof(frames[0]));
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    // Restore and re-raise, so the shell still sees the real signal and any core dump is produced.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

inline void install_test_signal_handlers() noexcept {
    for (int signal_number : {SIGSEGV, SIGABRT, SIGILL, SIGFPE}) std::signal(signal_number, test_fatal_signal_handler);
#if defined(SIGBUS)
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
}

#pragma endregion Crash Localization

/** Baseline for a loop whose work grows with the square of its trip count, like a sweep over
 *  lengths that re-scans a growing buffer, so doubling the scale only doubles the work. */
inline std::size_t iterations_quadratic(std::size_t baseline) noexcept {
    std::size_t const work = scaled_iterations(baseline * baseline);
    std::size_t bound = 1;
    while (bound * bound < work) ++bound;
    return bound;
}

#pragma region Assertion Helpers

/** One operand of a failed comparison, cut at a few hundred bytes so a megabyte string cannot drown
 *  the report printed on failure. */
template <typename value_type_>
std::string render_operand(value_type_ const &value) {
    std::string rendered;
    if constexpr (std::is_null_pointer_v<value_type_>) rendered = "nullptr";
    else if constexpr (std::is_pointer_v<value_type_> || std::is_function_v<value_type_>)
        rendered = fmt::format("{}", fmt::ptr(value));
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

/** A format string bundled with the source location of the call that spelled it, so a check's
 *  context rides behind a variadic pack, where a defaulted trailing argument cannot go. */
struct format_at_t {
    char const *format = nullptr;
    std::source_location location {};

    format_at_t(char const *format, std::source_location location = std::source_location::current()) noexcept
        : format(format), location(location) {}
};

/** Prints the @p context of a failed check, formatted with @p args , under its location line. */
template <typename... args_types_>
void print_check_context_(format_at_t context, args_types_ const &...args) {
    fmt::println(stderr, "    {}", fmt::vformat(context.format, fmt::make_format_args(args...)));
}

/** Holds when @p held does; otherwise counts a failed check and prints the @p context under it. */
template <typename... args_types_>
bool expect_true(bool is_met, format_at_t context, args_types_ const &...args) {
    if (expect_true(is_met, context.location)) return true;
    print_check_context_(context, args...);
    return false;
}

/** Holds when @p actual equals @p wanted; otherwise counts a failed check at the caller's line and
 *  prints both under it, cut by @ref render_operand . */
template <typename actual_type_, typename wanted_type_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                  std::source_location where = std::source_location::current()) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
    bool const held = actual == wanted;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (!held) {
        record_failed_check(where);
        fmt::println(stderr, "    expected {}, got {}", render_operand(wanted), render_operand(actual));
    }
    return held;
}

/** As @ref expect_equal , printing the @p context under the failed check. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted, format_at_t context,
                  args_types_ const &...args) {
    if (expect_equal(actual, wanted, context.location)) return true;
    print_check_context_(context, args...);
    return false;
}

/** Holds when @p actual lies within @p tolerance of @p wanted, scaled by the larger of one and
 *  either magnitude, and never for NaN; otherwise counts a failed check at the caller's line
 *  and prints both. */
template <typename actual_type_, typename wanted_type_, typename tolerance_type_>
bool expect_near(actual_type_ const &actual, wanted_type_ const &wanted, tolerance_type_ const &tolerance,
                 std::source_location where = std::source_location::current()) {
    double const actual_value = static_cast<double>(actual), wanted_value = static_cast<double>(wanted);
    double const magnitude = max_of(max_of(1.0, std::fabs(actual_value)), std::fabs(wanted_value));
    bool const held = std::fabs(actual_value - wanted_value) <= static_cast<double>(tolerance) * magnitude;
    if (!held) {
        record_failed_check(where);
        fmt::println(stderr, "    expected {} within {}, got {}", wanted, tolerance, actual);
    }
    return held;
}

/** As @ref expect_near , printing the @p context under the failed check. */
template <typename actual_type_, typename wanted_type_, typename tolerance_type_, typename... args_types_>
bool expect_near(actual_type_ const &actual, wanted_type_ const &wanted, tolerance_type_ const &tolerance,
                 format_at_t context, args_types_ const &...args) {
    if (expect_near(actual, wanted, tolerance, context.location)) return true;
    print_check_context_(context, args...);
    return false;
}

/** Holds when @p callable throws @p exception_type_ , the one check whose subject is the failure,
 *  so a call that returns, or throws something else, is the defect it counts. */
template <typename exception_type_, typename callable_type_>
bool expect_throws(callable_type_ &&callable, std::source_location where = std::source_location::current()) {
    try {
        callable();
    }
    catch (exception_type_ const &) {
        return true;
    }
    catch (...) {
    }
    record_failed_check(where);
    fmt::println(stderr, "    expected {} to be thrown", typeid(exception_type_).name());
    return false;
}

#pragma endregion Assertion Helpers

#pragma region Backend Tables

/**
 *  @brief Reports the backend that broke a check, counted as a failed check at the caller's line.
 *  @param[in] name The row's spelling in its family's backend table.
 *  @param[in] what What the row disagreed with: its oracle, known answer, or reference backend.
 */
inline void fail_backend_(char const *name, char const *what,
                          std::source_location where = std::source_location::current()) {
    record_failed_check(where);
    fmt::println(stderr, "    backend {} failed: {}", name, what);
}

/** The row named @p name, so a reordering cannot silently hand the differential a new reference. */
template <typename backend_type_, std::size_t count_>
backend_type_ const &backend_named_(backend_type_ const (&backends)[count_], char const *name) {
    for (std::size_t index = 0; index != count_; ++index)
        if (std::strcmp(backends[index].name, name) == 0) return backends[index];
    expect_true(false, "The backend table must carry the named reference");
    return backends[0];
}

#pragma endregion Backend Tables

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
 *  just before the stream; checks that it succeeded and returns that result. */
template <typename result_type_, typename kernel_type_, typename... arguments_types_>
result_type_ kernel_result(kernel_type_ const &kernel, arguments_types_... arguments) {
    result_type_ result {};
    expect_equal(kernel(arguments..., &result, nullptr), sz_success_k);
    return result;
}

/** A bump arena over one device block whose bookkeeping the host keeps, so carving a tape reads
 *  nothing back from the device and a rewind returns every tape at once. */
struct device_arena_t {
    sz_ptr_t block = nullptr;
    sz_size_t capacity = 0, consumed = 0;
};

/** Carves @p bytes from the arena at @p handle on a 256-byte boundary, or null once it is full. */
inline void *device_arena_allocate_(sz_size_t bytes, void *handle, sz_stream_t) {
    device_arena_t &arena = *static_cast<device_arena_t *>(handle);
    sz_size_t const start = (arena.consumed + 255) & ~(sz_size_t)255;
    if (start > arena.capacity || bytes > arena.capacity - start) return nullptr;
    arena.consumed = start + bytes;
    return arena.block + start;
}

/** Returns nothing, as an arena's tapes go back together when it rewinds. */
inline void device_arena_free_(void *, sz_size_t, void *, sz_stream_t) {}

/** The allocator carving from @p arena, which has no @c reallocate. */
inline sz_allocator_t device_arena_policy_(device_arena_t &arena) {
    sz_allocator_t policy {};
    policy.allocate = &device_arena_allocate_;
    policy.free = &device_arena_free_;
    policy.handle = &arena;
    return policy;
}

struct device_backend_t {
    sz::device_t selected = sz::device_t::cpu();
    sz_capability_t capabilities = 0;
    sz_stream_t stream = nullptr;

    /** Memory the host builds and reads, which the device addresses over the bus. */
    sz_allocator_t pinned {};

    /** Memory only the device touches, which holds the arena's block. */
    sz_allocator_t device {};

    /** The arena the device gathers its tapes into, owned by the vendor's driver function. */
    device_arena_t *arena = nullptr;

    sz_status_t (*init)(sz_size_t, sz_stream_t *) = nullptr;
    sz_status_t (*free)(sz_stream_t) = nullptr;
};

/** The adapter of a backend's pinned policy, riding its stream. */
template <typename value_type_>
unified_alloc<value_type_> pinned_alloc_(device_backend_t const &runtime) {
    return {runtime.pinned, runtime.stream};
}

/** Hands an arena's block back to the policy it came from, as a @c std::unique_ptr deleter. */
struct device_block_release_t {
    unified_alloc<char> allocator;
    std::size_t bytes = 0;
    void operator()(char *block) const noexcept { allocator.deallocate(block, bytes); }
};

using pinned_strings_t = strings<unified_alloc<char>>;
template <typename value_type_>
using pinned_vector = std::vector<value_type_, unified_alloc<value_type_>>;

/** Owns one native stream, released after its dependent allocations and engines. */
struct stream_t {
    sz_stream_t handle = nullptr;
    sz_status_t (*release)(sz_stream_t);

    stream_t(sz_size_t ordinal, sz_status_t (*init)(sz_size_t, sz_stream_t *), sz_status_t (*release)(sz_stream_t))
        : release(release) {
        expect_equal(init(ordinal, &handle), sz_success_k);
    }
    stream_t(stream_t const &) = delete;
    stream_t &operator=(stream_t const &) = delete;
    ~stream_t() noexcept { release(handle); }
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
inline void randomize_string(split_mix_generator_t &generator, std::span<char> text,
                             std::string_view alphabet = {}) noexcept {
    if (alphabet.empty()) {
        uniform_u8_distribution_t distribution;
        std::generate(text.begin(), text.end(), [&]() -> char { return distribution(generator); });
        return;
    }
    uniform_u8_distribution_t distribution(0, static_cast<char>(alphabet.size() - 1));
    std::generate(text.begin(), text.end(), [&]() -> char { return alphabet[distribution(generator)]; });
}

inline std::string random_string(split_mix_generator_t &generator, std::size_t length,
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
inline void iterate_in_random_slices(split_mix_generator_t &generator, std::string const &text,
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
    std::vector<char> storage(usable_length + 2 * sz_default_alignment_k + 1, '\0');
    for (std::size_t offset : offsets) {
        std::fill(storage.begin(), storage.end(), '\0');
        char *pointer = storage.data();
        while (reinterpret_cast<std::uintptr_t>(pointer) % sz_default_alignment_k != offset) ++pointer;
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
        expect_equal(storage[index], canary_value, "front canary overwritten");
        expect_equal(storage[guard_width + length + index], canary_value, "back canary overwritten");
    }
}

/** Runs @p body on a span of @p count strings over @p payload_bytes, borrowed from blocks of
 *  exactly those sizes between canaries, its offsets zeroed but the last, the payload size. */
template <typename body_type_>
inline void with_guarded_span_(std::size_t count, std::size_t payload_bytes, body_type_ &&body) {
    with_guarded_buffer_(payload_bytes, [&](sz_ptr_t payload, std::size_t) {
        with_guarded_buffer_((count + 1) * sizeof(sz_u32_t), [&](sz_ptr_t offsets_bytes, std::size_t) {
            sz_u32_t *const offsets = reinterpret_cast<sz_u32_t *>(offsets_bytes);
            std::memset(offsets, 0, count * sizeof(sz_u32_t));
            offsets[count] = static_cast<sz_u32_t>(payload_bytes);
            sz_strings_span_t borrowed {};
            borrowed.layout = sz_strings_u32_tape_k, borrowed.count = count;
            borrowed.data.u32_tape.data = payload, borrowed.data.u32_tape.offsets = offsets;
            body(borrowed);
        });
    });
}

/** An allocator refusing every request, for asserting a kernel reports @c sz_bad_alloc_k and leaves
 *  its outputs alone. */
inline sz_allocator_t refusing_c_allocator() noexcept {
    sz_allocator_t refusing;
    refusing.allocate = +[](sz_size_t, void *, sz_stream_t) -> void * { return nullptr; };
    refusing.reallocate = +[](void *, sz_size_t, sz_size_t, void *, sz_stream_t) -> void * { return nullptr; };
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
        allocator.reallocate = +[](void *pointer, sz_size_t, sz_size_t new_length, void *handle,
                                   sz_stream_t) -> void * {
            handle_checked_heap_t &heap = *static_cast<handle_checked_heap_t *>(handle);
            if (heap.self != &heap) return nullptr;
            heap.live_allocations += !pointer;
            if (new_length) return std::realloc(pointer, new_length);
            --heap.live_allocations;
            std::free(pointer);
            return nullptr;
        };
        allocator.free = +[](void *pointer, sz_size_t, void *handle, sz_stream_t) {
            if (!pointer) return;
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
inline std::string random_string(split_mix_generator_t &generator, std::size_t length,
                                 std::vector<std::string> const &characters) noexcept(false) {
    std::uniform_int_distribution<std::size_t> distribution(0, characters.size() - 1);
    std::string result;
    while (length--) result += characters[distribution(generator)];
    return result;
}

/** Borrows an accessor-backed collection of the supplied views. */
inline sz_strings_view_t strings_from_(std::span<sz_string_view_t const> views) {
    sz_strings_view_t strings;
    sz_strings_from_string_views(views.data(), views.size(), &strings);
    return strings;
}

/** Borrows an accessor-backed collection of the supplied strings. */
inline sz_strings_view_t strings_from_(std::vector<std::string> const &texts) {
    sz_strings_view_t strings;
    strings.layout = sz_strings_accessor_k, strings.count = (sz_size_t)texts.size();
    strings.data.accessor.handle = &texts;
    strings.data.accessor.get = [](void const *handle, sz_size_t const *indices, sz_size_t count,
                                   sz_string_view_t *views) {
        auto const &texts = *static_cast<std::vector<std::string> const *>(handle);
        for (sz_size_t position = 0; position != count; ++position) {
            std::string const &text = texts[indices[position]];
            views[position].start = text.data(), views[position].length = text.size();
        }
    };
    return strings;
}

/** Views the 32-bit tape over @p data that @p offsets describes, one more of them than strings. */
inline sz_strings_view_t strings_tape_over_(sz_cptr_t data, std::span<sz_u32_t const> offsets) {
    sz_strings_view_t strings;
    strings.layout = sz_strings_u32_tape_k, strings.count = (sz_size_t)offsets.size() - 1;
    strings.data.u32_tape.data = data, strings.data.u32_tape.offsets = offsets.data();
    return strings;
}

/** Views the 64-bit tape over @p data that @p offsets describes, one more of them than strings. */
inline sz_strings_view_t strings_tape_over_(sz_cptr_t data, std::span<sz_u64_t const> offsets) {
    sz_strings_view_t strings;
    strings.layout = sz_strings_u64_tape_k, strings.count = (sz_size_t)offsets.size() - 1;
    strings.data.u64_tape.data = data, strings.data.u64_tape.offsets = offsets.data();
    return strings;
}

/** Packs @p texts into a 32-bit tape over @p payload and @p offsets, which must outlive it. */
inline sz_strings_view_t strings_tape_from_(std::vector<std::string> const &texts, std::string &payload,
                                            std::vector<sz_u32_t> &offsets) {
    payload.clear(), offsets.assign(1, 0);
    for (std::string const &text : texts) payload += text, offsets.push_back((sz_u32_t)payload.size());
    sz_strings_view_t strings;
    strings.layout = sz_strings_u32_tape_k, strings.count = (sz_size_t)texts.size();
    strings.data.u32_tape.data = payload.data(), strings.data.u32_tape.offsets = offsets.data();
    return strings;
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

inline void randomize_strings(split_mix_generator_t &generator, fuzzy_config_t config,
                              std::vector<std::string> &array) {
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

#pragma region Test Runner

/**
 *  @brief The case adapter of the kernels that need @c needed : each case runs where @c detected
 *      has it, and otherwise skips as <tt>needs sz_cap_<capability>_k</tt>.
 *
 *  <tt>#if STRINGZILLA_TARGET_<CAPABILITY></tt> says a capability's kernels are built, and the
 *  gate says whether this machine runs them, as in
 *  <tt>run_test_case("check_find_unit_(find_haswell)", haswell, body)</tt>. A @p body taking a
 *  generator gets one keyed to its case.
 */
struct capability_gate_t {
    sz_capability_t needed = 0;
    sz_capability_t detected = 0;
    char skip_reason[STRINGZILLA_CAPABILITIES_NAME_CAPACITY + 16] = "needs sz_cap_";

    explicit capability_gate_t(sz_capability_t needed, sz_capability_t detected = probe_machine().detected) noexcept
        : needed(needed), detected(detected) {
        constexpr std::size_t prefix = sizeof("needs sz_cap_") - 1;
        sz_size_t length = sz_capabilities_name(needed, skip_reason + prefix, sizeof(skip_reason) - prefix - 3);
        std::memcpy(skip_reason + prefix + length, "_k", 3);
    }

    template <typename body_type_>
    case_result_t operator()(body_type_ &&body) const {
        if ((detected & needed) == 0) return case_result_t::skipped(skip_reason);
        if constexpr (std::is_invocable_v<body_type_ &, split_mix_generator_t &>) {
            split_mix_generator_t generator {case_stream_key()};
            body(generator);
        }
        else body();
        return {};
    }
};

#pragma endregion Test Runner

#pragma region Kernel Cross Checks

void test_cross_serial();
void test_cross_x8664();
void test_cross_arm64();
void test_cross_riscv64();
void test_cross_loongarch64();
void test_cross_ppc64();
void test_cross_wasm();
void test_cross_cuda(std::size_t ordinal);
void test_cross_rocm(std::size_t ordinal);
void test_cross_metal(std::size_t ordinal);

#pragma endregion Kernel Cross Checks

#pragma region Basic Utilities

void test_arithmetic_unit();
void test_strings_view_unit();
void test_strings_columns_unit();
void test_allocator_unit();
void test_byteset_unit();

#pragma endregion Basic Utilities

#pragma region Hashing

void test_hash_unit();
void test_hash_safety();
void test_hash_all();
void test_hash_multiseed_all();

#pragma endregion Hashing

#pragma region Ciphers

void test_cipher_unit();
void test_cipher_safety();
void test_cipher_all();

#pragma endregion Ciphers

#pragma region UTF8

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

#pragma endregion UTF8

#pragma region Uncased UTF8

void test_utf8_uncased_unit();
void test_utf8_uncased_scripts_unit();
void test_utf8_uncased_regressions_unit();
void test_utf8_uncased_all();
void test_utf8_uncased_safety();

#pragma endregion Uncased UTF8

#pragma region String Class and STL Compatibility

template <typename string_type>
void test_ascii_unit();

void test_strings_unit();
void test_strings_all();
void test_strings_safety();

template <typename string_type>
void test_stl_reads_unit();

void test_stl_updates_unit();

void test_stl_conversions_unit();
void test_stl_containers_unit();

template <typename string_type>
void test_extensions_reads_unit();

void test_extensions_updates_unit();
void test_string_constructors_unit();
void test_string_reserve_unit();
void test_memory_stability_equivalence(std::size_t length);
void test_string_updates_equivalence(std::size_t repetitions);

#pragma endregion String Class and STL Compatibility

#pragma region Search and Comparison

void test_compare_unit();
void test_extensions_ranges_unit();
void test_find_unit();
void test_find_safety();
void test_find_all();
void test_find_misaligned_equivalence();
void test_lookup_equivalence(std::size_t lookup_tables_to_try, std::size_t slices_per_table);

#pragma endregion Search and Comparison

#pragma region Sequence Algorithms

void test_sort_all();
void test_sort_unit();
void test_sort_safety();
void test_sort_reference_equivalence();
void test_intersect_unit();
void test_intersect_equivalence();
void test_levenshtein_unit();
void test_levenshtein_all();
void test_levenshtein_safety();
void test_overlap_unit();
void test_overlap_all();
void test_overlap_safety();
void test_substrings_unit();
void test_substrings_all();
void test_substrings_safety();
void test_regex_unit();
void test_regex_all();
void test_regex_safety();
void test_subdocuments_unit();
void test_subdocuments_all();
void test_subdocuments_safety();

#pragma endregion Sequence Algorithms

} // namespace ashvardanian::stringzilla::test
