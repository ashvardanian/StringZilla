/**
 *  @file test/cross_cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the CUDA kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if STRINGZILLA_ARCH_CUDA_
#include "cross_device.hpp" // `device_kernels`, `check_device_kernels_`
#endif

namespace ashvardanian::stringzilla::test {

void test_cross_cuda(std::size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    run_test_case("sz_capabilities_detected_cuda", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        int major = 0;
        expect_true(sz_count_devices_cuda(&devices) == sz_success_k && devices != 0);
        expect_equal(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)ordinal), cudaSuccess);
        sz_capability_t const tiers = sz_cap_cuda_k | (major >= 8 ? sz_cap_ampere_k : 0) |
                                      (major >= 9 ? sz_cap_hopper_k : 0) | (major >= 10 ? sz_cap_blackwell_k : 0);
        expect_true(sz_capabilities_detected_cuda(ordinal, &reported) == sz_success_k && reported == tiers);
        expect_equal(sz_capabilities_detected_cuda(devices, &reported), sz_missing_gpu_k);
        // The runtime loads a unit's probe kernel where one of that unit's codes runs here.
        sz_capability_t const held = sz_cap_cuda_k | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
                                     sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER |
                                     sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL;
        expect_true(sz_capabilities_compiled_cuda(ordinal, &reported) == sz_success_k && reported == (tiers & held));
        expect_equal(sz_capabilities_compiled_cuda(devices, &reported), sz_missing_gpu_k);
    });
    auto const selected = sz::device_t::make(sz::device_kind_t::cuda_k, ordinal);
    if (!expect_equal(selected.status, status_t::success_k)) return;
    stream_t stream(ordinal, sz_stream_init_cuda, sz_stream_free_cuda);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    if (!expect_equal(sz_capabilities_detected_cuda(ordinal, &runtime.capabilities), sz_success_k)) return;
    if (!expect_equal(sz_allocator_init_pinned_cuda(&runtime.pinned), sz_success_k)) return;
    if (!expect_equal(sz_allocator_init_device_cuda(&runtime.device), sz_success_k)) return;
    // Every device tape a case gathers is carved from one block, which the case rewinds when done.
    std::size_t const arena_bytes = (std::size_t)512 << 20;
    unified_alloc<char> const arena_allocator(runtime.device, runtime.stream);
    std::unique_ptr<char, device_block_release_t> const arena_block(arena_allocator.allocate(arena_bytes),
                                                                    {arena_allocator, arena_bytes});
    if (!expect_true(arena_block != nullptr)) return;
    device_arena_t arena {arena_block.get(), arena_bytes, 0};
    runtime.arena = &arena;
    runtime.init = sz_stream_init_cuda;
    runtime.free = sz_stream_free_cuda;
    // Each tier's gate narrows the engines its checks build to that tier, as a verb runs the
    // kernel of the capability its engine records.
#if STRINGZILLA_TARGET_CUDA
    device_backend_t cuda_runtime = runtime;
    cuda_runtime.capabilities &= sz_cap_cuda_k;
    device_kernels const cuda {
        "cuda",
        cuda_runtime,
        sz_levenshtein_distances_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_contains_cuda,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
        sz_regex_contains_cuda,
        sz_regex_counts_cuda,
        sz_regex_find_cuda,
        sz_regex_replace_cuda,
        sz_regex_extract_cuda,
        sz_subdocuments_contains_cuda,
        sz_subdocuments_counts_cuda,
        sz_subdocuments_find_cuda,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
        sz_sha256_multistate_update_cuda,
        sz_sha256_multistate_digest_cuda,
    };
    capability_gate_t const cuda_gate(sz_cap_cuda_k, runtime.capabilities);
    check_device_kernels_(cuda_gate, cuda);
    strings_backend_t const strings_cuda {"cuda",
                                          sz_strings_to_tape_cuda,
                                          sz_strings_to_rows_cuda,
                                          sz_strings_escape_cuda,
                                          sz_strings_unescape_cuda,
                                          sz_strings_lookup_cuda,
                                          sz_strings_normalize_cuda};
    check_device_strings_(cuda_gate, cuda_runtime, strings_cuda);
#endif
#if STRINGZILLA_TARGET_AMPERE
    device_backend_t ampere_runtime = runtime;
    ampere_runtime.capabilities &= sz_cap_cuda_k | sz_cap_ampere_k;
    device_kernels const ampere {
        "ampere",
        ampere_runtime,
        sz_levenshtein_distances_ampere,
        sz_overlap_scores_ampere,
        sz_substrings_contains_ampere,
        sz_substrings_counts_ampere,
        sz_substrings_find_ampere,
        sz_substrings_replace_ampere,
        sz_substrings_bm25_scores_ampere,
        sz_regex_contains_ampere,
        sz_regex_counts_ampere,
        sz_regex_find_ampere,
        sz_regex_replace_ampere,
        sz_regex_extract_ampere,
        sz_subdocuments_contains_ampere,
        sz_subdocuments_counts_ampere,
        sz_subdocuments_find_ampere,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
        sz_sha256_multistate_update_cuda,
        sz_sha256_multistate_digest_cuda,
    };
    capability_gate_t const ampere_gate(sz_cap_ampere_k, runtime.capabilities);
    check_device_levenshtein_(ampere_gate, ampere);
    check_device_overlap_(ampere_gate, ampere);
    check_device_substrings_(ampere_gate, ampere);
    check_device_regex_(ampere_gate, ampere);
    check_device_subdocuments_(ampere_gate, ampere);
#endif
#if STRINGZILLA_TARGET_HOPPER
    device_backend_t hopper_runtime = runtime;
    hopper_runtime.capabilities &= sz_cap_cuda_k | sz_cap_hopper_k;
    device_kernels const hopper {
        "hopper",
        hopper_runtime,
        sz_levenshtein_distances_cuda,
        sz_overlap_scores_cuda,
        sz_substrings_contains_hopper,
        sz_substrings_counts_hopper,
        sz_substrings_find_hopper,
        sz_substrings_replace_hopper,
        sz_substrings_bm25_scores_hopper,
        sz_regex_contains_hopper,
        sz_regex_counts_hopper,
        sz_regex_find_hopper,
        sz_regex_replace_hopper,
        sz_regex_extract_hopper,
        sz_subdocuments_contains_hopper,
        sz_subdocuments_counts_hopper,
        sz_subdocuments_find_hopper,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
        sz_sha256_multistate_update_cuda,
        sz_sha256_multistate_digest_cuda,
    };
    capability_gate_t const hopper_gate(sz_cap_hopper_k, runtime.capabilities);
    check_device_substrings_(hopper_gate, hopper);
    check_device_regex_(hopper_gate, hopper);
    check_device_subdocuments_(hopper_gate, hopper);
#endif
#if STRINGZILLA_TARGET_BLACKWELL
    device_backend_t blackwell_runtime = runtime;
    blackwell_runtime.capabilities &= sz_cap_cuda_k | sz_cap_blackwell_k;
    device_kernels const blackwell {
        "blackwell",
        blackwell_runtime,
        sz_levenshtein_distances_blackwell,
        sz_overlap_scores_blackwell,
        sz_substrings_contains_cuda,
        sz_substrings_counts_cuda,
        sz_substrings_find_cuda,
        sz_substrings_replace_cuda,
        sz_substrings_bm25_scores_cuda,
        sz_regex_contains_cuda,
        sz_regex_counts_cuda,
        sz_regex_find_cuda,
        sz_regex_replace_cuda,
        sz_regex_extract_cuda,
        sz_subdocuments_contains_cuda,
        sz_subdocuments_counts_cuda,
        sz_subdocuments_find_cuda,
        sz_utf8_uncased_fold_cuda,
        sz_utf8_norm_cuda,
        sz_sha256_multistate_update_cuda,
        sz_sha256_multistate_digest_cuda,
    };
    capability_gate_t const blackwell_gate(sz_cap_blackwell_k, runtime.capabilities);
    check_device_levenshtein_(blackwell_gate, blackwell);
    check_device_overlap_(blackwell_gate, blackwell);
#endif
    test_cross_dispatch_device(runtime, gpu_best<sz_sha256_multistate_update_best>(runtime.capabilities),
                               gpu_best<sz_sha256_multistate_digest_best>(runtime.capabilities));
#else
    sz_unused_(ordinal);
#endif
}

} // namespace ashvardanian::stringzilla::test
