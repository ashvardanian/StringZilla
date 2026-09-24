/**
 *  @file test/stringzilla.cu
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Extensive @b stress-testing suite for the StringZilla families that reach a GPU.
 *
 *  One launcher per family's GPU cases, the way @c test/stringzilla.cpp aggregates the CPU ones.
 *  Each family's cases live beside its CPU test - @c test/overlap.cu next to @c test/overlap.cpp -
 *  and are registered here.
 *
 *  @sa test/stringzilla.cpp for the CPU backends of the same families.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstdio> // `stdout`

#include <fmt/format.h>

#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `log_environment`, `run_test`

using namespace ashvardanian::stringzilla::test;

int main(int argc, char const **argv) {
    sz_unused_(argc && argv);
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    fmt::println("Hi, dear tester! You look nice today!");
    if (auto code = log_environment(); code != 0) return code;
    print_test_environment();

    std::size_t failures = 0;

    failures += run_test("test_levenshtein_all", test_levenshtein_all);
    failures += run_test("test_levenshtein_safety", test_levenshtein_safety);
    failures += run_test("test_overlap_all", test_overlap_all);
    failures += run_test("test_overlap_safety", test_overlap_safety);
    failures += run_test("test_substrings_unit", test_substrings_unit);
    failures += run_test("test_substrings_all", test_substrings_all);
    failures += run_test("test_substrings_safety", test_substrings_safety);

    if (failures) {
        fmt::println("Fail! {} tests failed.", failures);
        return 1;
    }
    fmt::println("All tests passed... Unbelievable!");
    return 0;
}
