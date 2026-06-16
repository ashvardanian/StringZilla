/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_stringzillas.cpp
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

#define SZ_USE_NEON 0
#define SZ_USE_SVE 0
#define SZ_USE_WESTMERE 0
#define SZ_USE_HASWELL 0
#define SZ_USE_SKYLAKE 0
#define SZ_USE_ICELAKE 0
#define SZ_USE_CUDA 0
#define SZ_USE_KEPLER 0
#define SZ_USE_HOPPER 0
*/
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // Enforce aggressive logging for this unit.

#include "test_stringzilla.hpp"

#include "test_fingerprints.cuh"
#include "test_similarities.cuh"

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;

int main(int argc, char const **argv) {
    sz_unused_(argc && argv);
    install_test_signal_handlers();
    std::printf("Hi, dear tester! You look nice today!\n");
    if (auto code = log_environment(); code != 0) return code;
    print_test_environment();

    int failures = 0;

    std::printf("\n=== Fingerprints ===\n");
    failures += run_test("test_fingerprints_unit", test_fingerprints_unit);
    failures += run_test("test_fingerprints_equivalence", test_fingerprints_equivalence);
    failures += run_test("test_fingerprints_safety", test_fingerprints_safety);

    std::printf("\n=== Similarities ===\n");
    failures += run_test("test_similarities_unit", test_similarities_unit);
    failures += run_test("test_similarities_equivalence", test_similarities_equivalence);
    failures += run_test("test_similarities_safety", test_similarities_safety);
    failures += run_test("test_similarities_memory_usage", test_similarities_memory_usage);

    if (failures != 0) {
        std::fprintf(stderr, "\n%d test(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nAll tests passed!\n");
    return 0;
}
