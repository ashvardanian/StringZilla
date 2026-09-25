/**
 *  @file test/main.cu
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Extensive @b stress-testing suite for the StringZilla families that reach a GPU.
 *
 *  One launcher per family's GPU cases, the way @c test/main.cpp aggregates the CPU ones.
 *  Each family's cases live beside its CPU test - @c test/overlap.cu next to @c test/overlap.cpp -
 *  and are registered here.
 *
 *  @sa test/main.cpp for the CPU backends of the same families.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstdio> // `stdout`

#include <fmt/format.h>

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp" // `log_environment`, `run_test`

using namespace ashvardanian::stringzilla::test;

int main(int, char const **argv) {
    test_environment_t const environment = read_test_environment(argv[0]);
    install_test_signal_handlers(); // Backtrace on fatal signals + line-buffered stdout for crash localization.
    log_environment();
    print_test_environment(environment);
    if (!log_cuda_device()) return 0; // ? A build machine or CI runner need not have a device, so none is a skip

    std::size_t failures = 0;

    failures += run_test(environment, "test_levenshtein_all", test_levenshtein_all);
    failures += run_test(environment, "test_levenshtein_safety", test_levenshtein_safety);
    failures += run_test(environment, "test_overlap_all", test_overlap_all);
    failures += run_test(environment, "test_overlap_safety", test_overlap_safety);
    failures += run_test(environment, "test_substrings_unit", test_substrings_unit);
    failures += run_test(environment, "test_substrings_all", test_substrings_all);
    failures += run_test(environment, "test_substrings_safety", test_substrings_safety);

    if (failures != 0) {
        fmt::println(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    fmt::println("\nAll tests passed!");
    return 0;
}
