/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file test/stringzillas.cpp
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

#include "stringzilla.hpp"

#include "substrings.cuh"
#include "fingerprints.cuh"
#include "similarities.cuh"

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;

int main(int argc, char const **argv) {
    sz_unused_(argc && argv);
    install_test_signal_handlers();
    std::printf("Hi, dear tester! You look nice today!\n");
    if (auto code = log_environment(); code != 0) return code;
    print_test_environment();

    std::size_t failures = 0;

    failures += run_test("test_fingerprints_unit", test_fingerprints_unit);
    failures += run_test("test_fingerprints_equivalence", test_fingerprints_equivalence);
    failures += run_test("test_fingerprints_safety", test_fingerprints_safety);

    failures += run_test("test_similarities_unit", test_similarities_unit);
    failures += run_test("test_similarities_equivalence", test_similarities_equivalence);
    failures += run_test("test_similarities_cross_product", test_similarities_cross_product);
    failures += run_test("test_similarities_safety", test_similarities_safety);
    failures += run_test("test_similarities_memory_usage", test_similarities_memory_usage);

    failures += run_test("test_substrings_unit", test_substrings_unit);
    failures += run_test("test_substrings_uncased", test_substrings_uncased);
    failures += run_test("test_substrings_agreement", test_substrings_agreement);
    failures += run_test("test_substrings_construction", test_substrings_construction);
    failures += run_test("test_substrings_adversarial", test_substrings_adversarial);
    failures += run_test("test_substrings_large_haystacks", test_substrings_large_haystacks);
    failures += run_test("test_substrings_cover", test_substrings_cover);
    failures += run_test("test_substrings_rewriting", test_substrings_rewriting);
    failures += run_test("test_substrings_scoring", test_substrings_scoring);
    failures += run_test("test_substrings_safety", test_substrings_safety);
    failures += run_test("test_substrings_buffer_contracts", test_substrings_buffer_contracts);

    if (failures != 0) {
        std::fprintf(stderr, "\n%zu test(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nAll tests passed!\n");
    return 0;
}
