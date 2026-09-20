/**
 *  @brief  Extensive @b stress-testing suite for the StringZilla families that reach a GPU, written in CUDA C++.
 *  @see    The CPU backends of the same families are driven by @c test/stringzilla.cpp.
 *
 *  @file   test/stringzilla.cu
 *  @author Ash Vardanian
 *  @date   September 15, 2026
 *
 *  One launcher per family's GPU cases, the way @c test/stringzilla.cpp aggregates the CPU ones. Each family's
 *  cases live beside its CPU test - @c test/overlap.cu next to @c test/overlap.cpp - and are registered here.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include <cstdio> // `std::printf`

#include <stringzilla/stringzilla.h> // Primary C API

#include "stringzilla.hpp" // `log_environment`, `run_test`

using namespace ashvardanian::stringzilla::test;

int main(int argc, char const **argv) {
    sz_unused_(argc && argv);
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Hi, dear tester! You look nice today!\n");
    if (auto code = log_environment(); code != 0) return code;
    print_test_environment();

    std::size_t failures = 0;

    failures += run_test("test_levenshtein_all", test_levenshtein_all);
    failures += run_test("test_levenshtein_safety", test_levenshtein_safety);
    failures += run_test("test_overlap_all", test_overlap_all);
    failures += run_test("test_overlap_safety", test_overlap_safety);

    if (failures) {
        std::printf("Fail! %zu tests failed.\n", failures);
        return 1;
    }
    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
