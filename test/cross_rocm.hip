/**
 *  @file test/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the ROCm kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include "cross_simt.cuh" // `simt_backend_t`, `check_simt_backend_`

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_rocm(environment_t const &env) {
    simt_backend_t const rocm {
        "rocm",
        sz_levenshtein_distances_rocm,
        sz_levenshtein_distance_tiled_rocm,
        sz_overlap_scores_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
    };
    cross_section_t check(env);
    check.detected = gpu_capabilities();
    check.section("Cross ROCm", sz_cap_rocm_k);
    check_simt_backend_(check, rocm);
    return check.failures;
}

} // namespace ashvardanian::stringzilla::test
