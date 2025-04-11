/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
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

#define SZ_USE_HASWELL 0
#define SZ_USE_SKYLAKE 0
#define SZ_USE_ICE 0
#define SZ_USE_NEON 0
#define SZ_USE_SVE 0
*/
#define SZ_USE_OPENMP 1
#define SZ_USE_CUDA 0
#define SZ_USE_KEPLER 0
#define SZ_USE_HOPPER 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // Enforce aggressive logging for this unit.

#include "test_stringcuzilla.cuh"

namespace sz = ashvardanian::stringzilla;

int main(int argc, char const **argv) {
    sz_unused(argc && argv);
    std::printf("Hi, dear tester! You look nice today!\n");
    if (auto code = sz::scripts::log_environment(); code != 0) return code;

    try {
        sz::scripts::test_similarity_scores_equivalence();
        sz::scripts::test_similarity_scores_memory_usage();
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
