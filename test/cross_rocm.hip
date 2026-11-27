/**
 *  @file test/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine checks - the ROCm kernels by name, and the dispatching entry points.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if STRINGZILLA_ARCH_ROCM_
#include "cross_device.hpp" // `device_kernels`, `check_device_kernels_`
#endif

namespace ashvardanian::stringzilla::test {

void test_cross_rocm(std::size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    run_test_case("sz_capabilities_detected_rocm", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        hipDeviceProp_t properties;
        expect_true(sz_count_devices_rocm(&devices) == sz_success_k && devices != 0);
        expect_equal(hipGetDeviceProperties(&properties, (int)ordinal), hipSuccess);
        std::string_view const architecture = properties.gcnArchName;
        bool const runs_cdna3 = architecture.starts_with("gfx942");
        bool const runs_cdna4 = architecture.starts_with("gfx950");
        bool const runs_cdna5 = architecture.starts_with("gfx1250") || architecture.starts_with("gfx1251");
        sz_capability_t const tiers = sz_cap_rocm_k | (runs_cdna3 ? sz_cap_cdna3_k : 0) |
                                      (runs_cdna4 ? sz_cap_cdna4_k : 0) | (runs_cdna5 ? sz_cap_cdna5_k : 0);
        expect_true(sz_capabilities_detected_rocm(ordinal, &reported) == sz_success_k && reported == tiers);
        expect_equal(sz_capabilities_detected_rocm(devices, &reported), sz_missing_gpu_k);
    });
    device_selection_t const selected {sz::device_kind_t::rocm_k, ordinal};
    if (!expect_equal(check_device(selected), sz_success_k)) return;
    stream_t stream(ordinal, sz_stream_init_rocm, sz_stream_free_rocm);
    device_backend_t runtime;
    runtime.selected = selected;
    runtime.stream = stream.handle;
    if (!expect_equal(sz_capabilities_detected_rocm(ordinal, &runtime.capabilities), sz_success_k)) return;
    if (!expect_equal(sz_allocator_init_pinned_rocm(&runtime.pinned), sz_success_k)) return;
    if (!expect_equal(sz_allocator_init_device_rocm(&runtime.device), sz_success_k)) return;
    // Every device tape a case gathers is carved from one block, which the case rewinds when done.
    std::size_t const arena_bytes = (std::size_t)512 << 20;
    unified_alloc<char> const arena_allocator(runtime.device, runtime.stream);
    std::unique_ptr<char, device_block_release_t> const arena_block(arena_allocator.allocate(arena_bytes),
                                                                    {arena_allocator, arena_bytes});
    if (!expect_true(arena_block != nullptr)) return;
    device_arena_t arena {arena_block.get(), arena_bytes, 0};
    runtime.arena = &arena;
    runtime.init = sz_stream_init_rocm;
    runtime.free = sz_stream_free_rocm;
    // Each tier's gate narrows the engines its checks build to that tier, as a verb runs the
    // kernel of the capability its engine records.
#if STRINGZILLA_TARGET_ROCM
    device_backend_t rocm_runtime = runtime;
    rocm_runtime.capabilities &= sz_cap_rocm_k;
    device_kernels const rocm {
        "rocm",
        rocm_runtime,
        sz_levenshtein_distances_rocm,
        sz_ngrams_overlaps_rocm,
        sz_ngrams_sketches_rocm,
        sz_substrings_contains_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
        sz_regex_contains_rocm,
        sz_regex_counts_rocm,
        sz_regex_find_rocm,
        sz_regex_replace_rocm,
        sz_regex_extract_rocm,
        sz_subdocuments_contains_rocm,
        sz_subdocuments_counts_rocm,
        sz_subdocuments_find_rocm,
        sz_utf8_uncased_fold_rocm,
        sz_utf8_norm_rocm,
        sz_sha256_multistate_update_rocm,
        sz_sha256_multistate_digest_rocm,
    };
    capability_gate_t const rocm_gate(sz_cap_rocm_k, runtime.capabilities);
    check_device_kernels_(rocm_gate, rocm);
    strings_backend_t const strings_rocm {"rocm",
                                          sz_strings_to_tape_rocm,
                                          sz_strings_to_rows_rocm,
                                          sz_strings_escape_rocm,
                                          sz_strings_unescape_rocm,
                                          sz_strings_lookup_rocm,
                                          sz_strings_normalize_rocm};
    check_device_strings_(rocm_gate, rocm_runtime, strings_rocm);
#endif
#if STRINGZILLA_TARGET_CDNA3
    device_backend_t cdna3_runtime = runtime;
    cdna3_runtime.capabilities &= sz_cap_rocm_k | sz_cap_cdna3_k;
    device_kernels const cdna3 {
        "cdna3",
        cdna3_runtime,
        sz_levenshtein_distances_cdna3,
        sz_ngrams_overlaps_cdna3,
        sz_ngrams_sketches_cdna3,
        sz_substrings_contains_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
        sz_regex_contains_rocm,
        sz_regex_counts_rocm,
        sz_regex_find_rocm,
        sz_regex_replace_rocm,
        sz_regex_extract_rocm,
        sz_subdocuments_contains_rocm,
        sz_subdocuments_counts_rocm,
        sz_subdocuments_find_rocm,
        sz_utf8_uncased_fold_rocm,
        sz_utf8_norm_rocm,
        sz_sha256_multistate_update_rocm,
        sz_sha256_multistate_digest_rocm,
    };
    capability_gate_t const cdna3_gate(sz_cap_cdna3_k, runtime.capabilities);
    check_device_levenshtein_(cdna3_gate, cdna3);
    check_device_ngrams_(cdna3_gate, cdna3);
#endif
#if STRINGZILLA_TARGET_CDNA4
    device_backend_t cdna4_runtime = runtime;
    cdna4_runtime.capabilities &= sz_cap_rocm_k | sz_cap_cdna4_k;
    device_kernels const cdna4 {
        "cdna4",
        cdna4_runtime,
        sz_levenshtein_distances_cdna4,
        sz_ngrams_overlaps_cdna4,
        sz_ngrams_sketches_cdna4,
        sz_substrings_contains_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
        sz_regex_contains_rocm,
        sz_regex_counts_rocm,
        sz_regex_find_rocm,
        sz_regex_replace_rocm,
        sz_regex_extract_rocm,
        sz_subdocuments_contains_rocm,
        sz_subdocuments_counts_rocm,
        sz_subdocuments_find_rocm,
        sz_utf8_uncased_fold_rocm,
        sz_utf8_norm_rocm,
        sz_sha256_multistate_update_rocm,
        sz_sha256_multistate_digest_rocm,
    };
    capability_gate_t const cdna4_gate(sz_cap_cdna4_k, runtime.capabilities);
    check_device_levenshtein_(cdna4_gate, cdna4);
    check_device_ngrams_(cdna4_gate, cdna4);
#endif
#if STRINGZILLA_TARGET_CDNA5
    device_backend_t cdna5_runtime = runtime;
    cdna5_runtime.capabilities &= sz_cap_rocm_k | sz_cap_cdna5_k;
    device_kernels const cdna5 {
        "cdna5",
        cdna5_runtime,
        sz_levenshtein_distances_cdna5,
        sz_ngrams_overlaps_cdna5,
        sz_ngrams_sketches_cdna5,
        sz_substrings_contains_rocm,
        sz_substrings_counts_rocm,
        sz_substrings_find_rocm,
        sz_substrings_replace_rocm,
        sz_substrings_bm25_scores_rocm,
        sz_regex_contains_rocm,
        sz_regex_counts_rocm,
        sz_regex_find_rocm,
        sz_regex_replace_rocm,
        sz_regex_extract_rocm,
        sz_subdocuments_contains_rocm,
        sz_subdocuments_counts_rocm,
        sz_subdocuments_find_rocm,
        sz_utf8_uncased_fold_rocm,
        sz_utf8_norm_rocm,
        sz_sha256_multistate_update_rocm,
        sz_sha256_multistate_digest_rocm,
    };
    capability_gate_t const cdna5_gate(sz_cap_cdna5_k, runtime.capabilities);
    check_device_levenshtein_(cdna5_gate, cdna5);
    check_device_ngrams_(cdna5_gate, cdna5);
#endif
    test_cross_dispatch_device(runtime, gpu_best<sz_sha256_multistate_update_best>(runtime.capabilities),
                               gpu_best<sz_sha256_multistate_digest_best>(runtime.capabilities));
#else
    sz_unused_(ordinal);
#endif
}

} // namespace ashvardanian::stringzilla::test
