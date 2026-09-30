/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the CUDA kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include "cross_simt.cuh" // `simt_backend_t`, `check_simt_backend_`

using namespace ashvardanian::stringzilla::test;

std::size_t test_cross_cuda(test_environment_t const &environment) {
    simt_backend_t const cuda {
        "cuda",
        sz_levenshtein_distances_cuda,
        sz_levenshtein_distance_tiled_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
    };
    cross_section_t check(environment);
    check.detected = gpu_capabilities();
    check.section("Cross CUDA", sz_cap_cuda_k);
    check_simt_backend_(check, cuda);
    return check.failures;
}
