/**
 *  @file bench/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the ROCm kernels against the CPU tiers this build carries.
 */
#include "harness.hpp"

#if STRINGZILLA_ARCH_ROCM_
#include "stringzilla/rocm.cuh"
#include "cross_simt.cuh"
#endif

namespace ashvardanian::stringzilla::bench {
int bench_cross_rocm(environment_t &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    auto const selected = sz::device_t::make(sz::device_kind_t::rocm_k, ordinal);
    if (selected.status != status_t::success_k) throw std::runtime_error("The requested device is unavailable.");
    stream_t stream(ordinal, sz_stream_init_rocm, sz_stream_free_rocm);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    if (sz_capabilities_detected_rocm(ordinal, &runtime.capabilities) != sz_success_k)
        throw std::runtime_error("Device capabilities could not be queried.");
    if (sz_allocator_init_unified_rocm(&runtime.unified) != sz_success_k)
        throw std::runtime_error("The unified allocator could not be initialized.");
    if (sz_allocator_init_device_rocm(&runtime.device) != sz_success_k)
        throw std::runtime_error("The device allocator could not be initialized.");
    if (sz_allocator_init_pinned_rocm(&runtime.pinned) != sz_success_k)
        throw std::runtime_error("The pinned allocator could not be initialized.");
    runtime.copy = sz_copy_rocm_;
    runtime.free_bytes = [](sz_stream_t stream) -> sz_size_t {
        int caller = 0;
        if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return 0;
        sz_size_t const bytes = sz_device_free_bytes_rocm_();
        sz_device_leave_rocm_(caller);
        return bytes;
    };
    int caller = 0;
    if (sz_device_enter_rocm_(runtime.stream, &caller) != sz_success_k)
        throw std::runtime_error("The stream device could not be selected.");
    runtime.multiprocessors = sz_device_multiprocessors_rocm_();
    runtime.threads_per_multiprocessor = sz_device_threads_per_multiprocessor_rocm_();
    sz_device_leave_rocm_(caller);
    fmt::println("Device: rocm:{} ({} multiprocessors)", ordinal, runtime.multiprocessors.value_or(0));
    simt_backend_t const backend {"rocm",
                                  runtime,
                                  sz_levenshtein_engine_init_rocm,
                                  sz_levenshtein_distances_rocm,
                                  sz_overlap_engine_init_rocm,
                                  sz_overlap_scores_rocm,
                                  sz_substrings_engine_init_rocm,
                                  sz_substrings_counts_rocm,
                                  sz_substrings_find_rocm,
                                  sz_substrings_replace_rocm,
                                  sz_substrings_bm25_scores_rocm};
    int const result = bench_cross_simt(env, backend);
    return result;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}
} // namespace ashvardanian::stringzilla::bench
