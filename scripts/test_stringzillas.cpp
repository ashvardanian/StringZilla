/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file scripts/test_stringzillas.cpp
 *  @author Ash Vardanian
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
    failures += run_test("test_rolling_hashers_equivalence", test_rolling_hashers_equivalence);
    failures += run_test("test_rolling_hasher", test_rolling_hasher);
    failures += run_test("test_similarity_scores_equivalence", test_similarity_scores_equivalence);
    failures += run_test("test_similarity_scores_memory_usage", test_similarity_scores_memory_usage);

    if (failures != 0) {
        std::fprintf(stderr, "\n%d test(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nAll tests passed!\n");
    return 0;
}
