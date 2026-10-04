/**
 *  @file test/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the ROCm kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include "harness.hpp"

#if STRINGZILLA_ARCH_ROCM_
#include "stringzilla/rocm.cuh"

#include "cross_simt.cuh" // `simt_backend`, `check_simt_backend_`
#endif

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_rocm(environment_t const &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    std::size_t const capability_failures = run_test(env.settings, "test_rocm_capabilities_unit", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        verify(sz_device_count_rocm(&devices) == sz_success_k && devices != 0);
        verify(sz_capabilities_detected_rocm(ordinal, &reported) == sz_success_k && reported == sz_cap_rocm_k);
        verify(sz_capabilities_detected_rocm(devices, &reported) == sz_missing_gpu_k);
    });
    auto const selected = sz::device_t::make(sz::device_kind_t::rocm_k, ordinal);
    verify(selected.status == status_t::success_k);
    stream_t stream(ordinal, sz_stream_init_rocm, sz_stream_free_rocm);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    verify(sz_capabilities_detected_rocm(ordinal, &runtime.capabilities) == sz_success_k);
    verify(sz_allocator_init_unified_rocm(&runtime.unified) == sz_success_k);
    verify(sz_allocator_init_device_rocm(&runtime.device) == sz_success_k);
    runtime.query = sz_stream_query_rocm_;
    runtime.reachable = sz_memory_accessible_rocm_;
    runtime.fill = sz_fill_rocm_;
    runtime.init = sz_stream_init_rocm;
    runtime.free = sz_stream_free_rocm;
    int caller = 0;
    verify(sz_device_enter_rocm_(runtime.stream, &caller) == sz_success_k);
    runtime.multiprocessors = sz_device_multiprocessors_rocm_();
    runtime.threads_per_multiprocessor = sz_device_threads_per_multiprocessor_rocm_();
    sz_device_leave_rocm_(caller);
    simt_backend const rocm {
        "rocm",
        runtime,
        sz_levenshtein_distances_rocm,
        sz_levenshtein_distance_tiled_rocm,
        sz_overlap_scores_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
        sz_utf8_uncased_fold_rocm,
        sz_utf8_norm_rocm,
    };
    cross_section_t check(env);
    check.detected = runtime.capabilities;
    check.section("Cross ROCm", sz_cap_rocm_k);
    check_simt_backend_(check, rocm);
    auto const failures = capability_failures + check.failures + test_cross_dispatch_simt(env, runtime);
    return failures;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::test
