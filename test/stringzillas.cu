/**
 *  @brief Extensive @b stress-testing suite for StringZillas parallel operations, written in CUDA C++.
 *  @see Stress-tests on real-world and synthetic data are integrated into the benchmarks under @b `bench/`.
 *
 *  @file test/stringzillas.cu
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
#define SZ_USE_CUDA 1
#define SZ_USE_KEPLER 1
#define SZ_USE_HOPPER 1
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

    std::printf("\nTesting fingerprints\n");
    failures += run_test("test_fingerprints_unit", test_fingerprints_unit);
    failures += run_test("test_fingerprints_equivalence", test_fingerprints_equivalence);
    failures += run_test("test_fingerprints_safety", test_fingerprints_safety);
    failures += run_test("test_fingerprints_cuda_memory_safety", test_fingerprints_cuda_memory_safety);

    std::printf("\nTesting similarities\n");
    failures += run_test("test_similarities_unit", test_similarities_unit);
    failures += run_test("test_similarities_equivalence", test_similarities_equivalence);
    failures += run_test("test_similarities_cross_product_equivalence", test_similarities_cross_product_equivalence);
    failures += run_test("test_similarities_safety", test_similarities_safety);
    failures += run_test("test_similarities_cuda_memory_safety", test_similarities_cuda_memory_safety);
    failures += run_test("test_similarities_memory_usage_equivalence", test_similarities_memory_usage_equivalence);

    std::printf("\nTesting substrings\n");
    failures += run_test("test_substrings_unit", test_substrings_unit);
    failures += run_test("test_substrings_uncased_unit", test_substrings_uncased_unit);
    failures += run_test("test_substrings_uncased_equivalence", test_substrings_uncased_equivalence);
    failures += run_test("test_substrings_construction_equivalence", test_substrings_construction_equivalence);
    failures += run_test("test_substrings_adversarial_equivalence", test_substrings_adversarial_equivalence);
    failures += run_test("test_substrings_large_haystacks_equivalence", test_substrings_large_haystacks_equivalence);
    failures += run_test("test_substrings_cover_equivalence", test_substrings_cover_equivalence);
    failures += run_test("test_substrings_rewriting_equivalence", test_substrings_rewriting_equivalence);
    failures += run_test("test_substrings_scoring_unit", test_substrings_scoring_unit);
    failures += run_test("test_substrings_scoring_wide_equivalence", test_substrings_scoring_wide_equivalence);
    failures += run_test("test_substrings_cuda_memory_safety", test_substrings_cuda_memory_safety);
    failures += run_test("test_substrings_cuda_equivalence", test_substrings_cuda_equivalence);
    failures += run_test("test_substrings_safety", test_substrings_safety);
    failures += run_test("test_substrings_buffer_safety", test_substrings_buffer_safety);

    if (failures != 0) {
        std::fprintf(stderr, "\n%zu test(s) failed.\n", failures);
        return 1;
    }
    std::printf("All tests passed... Unbelievable!\n");
    return 0;
}
