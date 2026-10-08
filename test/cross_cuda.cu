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
        int major = 0;
        verify(sz_device_count_cuda(&devices) == sz_success_k && devices != 0);
        verify(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)ordinal) == cudaSuccess);
        sz_capability_t const tiers = sz_cap_cuda_k | (major >= 9 ? sz_cap_hopper_k : 0) |
                                      (major >= 10 ? sz_cap_blackwell_k : 0);
        verify(sz_capabilities_detected_cuda(ordinal, &reported) == sz_success_k && reported == tiers);
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
    // Each tier's section narrows the engines its checks build to that tier, as a verb runs the
    // kernel of the capability its engine records.
    cross_section_t check(env);
    check.detected = runtime.capabilities;
#if STRINGZILLA_TARGET_CUDA
    device_backend_t cuda_runtime = runtime;
    cuda_runtime.capabilities &= sz_cap_cuda_k;
    device_kernels const cuda {
        "cuda",
        cuda_runtime,
        sz_levenshtein_distances_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
    };
    check.section("Cross CUDA", sz_cap_cuda_k);
    check_device_kernels_(check, cuda);
#endif
#if STRINGZILLA_TARGET_HOPPER
    device_backend_t hopper_runtime = runtime;
    hopper_runtime.capabilities &= sz_cap_cuda_k | sz_cap_hopper_k;
    device_kernels const hopper {
        "hopper",
        hopper_runtime,
        sz_levenshtein_distances_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_counts_hopper,
        sz_substrings_find_hopper,
        sz_substrings_replace_hopper,
        sz_substrings_bm25_scores_hopper,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
    };
    check.section("Cross Hopper", sz_cap_hopper_k);
    check_device_substrings_(check, hopper);
#endif
#if STRINGZILLA_TARGET_BLACKWELL
    device_backend_t blackwell_runtime = runtime;
    blackwell_runtime.capabilities &= sz_cap_cuda_k | sz_cap_blackwell_k;
    device_kernels const blackwell {
        "blackwell",
        blackwell_runtime,
        sz_levenshtein_distances_blackwell,
        sz_overlap_scores_blackwell,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
    };
    check.section("Cross Blackwell", sz_cap_blackwell_k);
    check_device_levenshtein_(check, blackwell);
    check_device_overlap_(check, blackwell);
#endif
    auto const failures = capability_failures + check.failures + test_cross_dispatch_device(env, runtime);
    return failures;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::test
