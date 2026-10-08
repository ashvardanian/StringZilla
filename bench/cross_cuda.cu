/**
 *  @file bench/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the CUDA kernels against the CPU tiers this build carries.
 */
#include "harness.hpp"

#if STRINGZILLA_ARCH_CUDA_
#include "stringzilla/cuda.cuh"
#include "cross_simt.cuh"
#endif

namespace ashvardanian::stringzilla::bench {

int bench_cross_cuda(environment_t &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    auto const selected = sz::device_t::make(sz::device_kind_t::cuda_k, ordinal);
    if (selected.status != status_t::success_k) throw std::runtime_error("The requested device is unavailable.");
    stream_t stream(ordinal, sz_stream_init_cuda, sz_stream_free_cuda);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    if (sz_capabilities_detected_cuda(ordinal, &runtime.capabilities) != sz_success_k)
        throw std::runtime_error("Device capabilities could not be queried.");
    if (sz_allocator_init_unified_cuda(&runtime.unified) != sz_success_k)
        throw std::runtime_error("The unified allocator could not be initialized.");
    if (sz_allocator_init_device_cuda(&runtime.device) != sz_success_k)
        throw std::runtime_error("The device allocator could not be initialized.");
    if (sz_allocator_init_pinned_cuda(&runtime.pinned) != sz_success_k)
        throw std::runtime_error("The pinned allocator could not be initialized.");
    runtime.copy = sz_copy_cuda_;
    runtime.free_bytes = [](sz_stream_t stream) -> sz_size_t {
        int caller = 0;
        if (sz_device_enter_cuda_(stream, &caller) != sz_success_k) return 0;
        sz_size_t const bytes = sz_device_free_bytes_cuda_();
        sz_device_leave_cuda_(caller);
        return bytes;
    };
    int caller = 0;
    if (sz_device_enter_cuda_(runtime.stream, &caller) != sz_success_k)
        throw std::runtime_error("The stream device could not be selected.");
    runtime.multiprocessors = sz_device_multiprocessors_cuda_();
    runtime.threads_per_multiprocessor = sz_device_threads_per_multiprocessor_cuda_();
    sz_device_leave_cuda_(caller);
    fmt::println("Device: cuda:{} ({} multiprocessors)", ordinal, runtime.multiprocessors.value_or(0));
    simt_backend_t const backend {"cuda",
                                  runtime,
                                  sz_levenshtein_engine_init_cuda,
                                  sz_levenshtein_distances_cuda,
                                  sz_overlap_engine_init_cuda,
                                  sz_overlap_scores_cuda,
                                  sz_substrings_engine_init_cuda,
                                  sz_substrings_counts_cuda,
                                  sz_substrings_find_cuda,
                                  sz_substrings_replace_cuda,
                                  sz_substrings_bm25_scores_cuda,
                                  sz_utf8_uncased_fold_cuda,
                                  sz_utf8_norm_cuda};
    int result = bench_cross_simt(env, backend);
#if STRINGZILLA_TARGET_HOPPER
    if (runtime.capabilities & sz_cap_hopper_k) {
        simt_backend_t const hopper {"hopper",
                                     runtime,
                                     sz_levenshtein_engine_init_cuda,
                                     sz_levenshtein_distances_cuda,
                                     sz_overlap_engine_init_cuda,
                                     sz_overlap_scores_cuda,
                                     sz_substrings_engine_init_hopper,
                                     sz_substrings_counts_hopper,
                                     sz_substrings_find_hopper,
                                     sz_substrings_replace_hopper,
                                     sz_substrings_bm25_scores_hopper,
                                     sz_utf8_uncased_fold_cuda,
                                     sz_utf8_norm_cuda};
        result += bench_cross_simt_substrings(env, hopper);
    }
#endif
#if STRINGZILLA_TARGET_BLACKWELL
    if (runtime.capabilities & sz_cap_blackwell_k) {
        simt_backend_t const blackwell {"blackwell",
                                        runtime,
                                        sz_levenshtein_engine_init_blackwell,
                                        sz_levenshtein_distances_blackwell,
                                        sz_overlap_engine_init_blackwell,
                                        sz_overlap_scores_blackwell,
                                        sz_substrings_engine_init_cuda,
                                        sz_substrings_counts_cuda,
                                        sz_substrings_find_cuda,
                                        sz_substrings_replace_cuda,
                                        sz_substrings_bm25_scores_cuda,
                                        sz_utf8_uncased_fold_cuda,
                                        sz_utf8_norm_cuda};
        result += bench_cross_simt_levenshtein_overlap(env, blackwell);
    }
#endif
    return result;
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}
} // namespace ashvardanian::stringzilla::bench
