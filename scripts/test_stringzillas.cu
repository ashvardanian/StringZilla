/**
 *  @brief   Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test.cu
 *  @author  Ash Vardanian
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
#define SZ_USE_ICE 0
#define SZ_USE_CUDA 1
#define SZ_USE_KEPLER 1
#define SZ_USE_HOPPER 1
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
    std::printf("Hi, dear tester! You look nice today!\n");
    if (auto code = log_environment(); code != 0) return code;
    print_test_environment();

    try {
        test_rolling_hashers_equivalence();
        test_rolling_hasher();
        test_similarity_scores_equivalence();
        test_similarity_scores_memory_usage();
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
