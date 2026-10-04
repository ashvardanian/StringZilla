/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the CUDA kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include "harness.hpp"

#if STRINGZILLA_ARCH_CUDA_
#include "cross_device.hpp" // `device_kernels`, `check_device_kernels_`
#endif

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_cuda(environment_t const &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    std::size_t const capability_failures = run_test(env.settings, "test_cuda_capabilities_unit", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        verify(sz_device_count_cuda(&devices) == sz_success_k && devices != 0);
        verify(sz_capabilities_detected_cuda(ordinal, &reported) == sz_success_k && reported == sz_cap_cuda_k);
        verify(sz_capabilities_detected_cuda(devices, &reported) == sz_missing_gpu_k);
    });
    auto const selected = sz::device_t::make(sz::device_kind_t::cuda_k, ordinal);
    verify(selected.status == status_t::success_k);
    stream_t stream(ordinal, sz_stream_init_cuda, sz_stream_free_cuda);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    verify(sz_capabilities_detected_cuda(ordinal, &runtime.capabilities) == sz_success_k);
    verify(sz_allocator_init_unified_cuda(&runtime.unified) == sz_success_k);
    runtime.init = sz_stream_init_cuda;
    runtime.free = sz_stream_free_cuda;
    device_kernels const cuda {
        "cuda",
        runtime,
        sz_levenshtein_distances_cuda,
        sz_levenshtein_distance_tiled_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
    };
    cross_section_t check(env);
    check.detected = runtime.capabilities;
    check.section("Cross CUDA", sz_cap_cuda_k);
    check_device_kernels_(check, cuda);
    auto const failures = capability_failures + check.failures + test_cross_dispatch_device(env, runtime);
    return failures;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::test
