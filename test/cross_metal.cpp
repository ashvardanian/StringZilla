/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal kernels and dispatch points through the shared device suites.
 */
#undef NDEBUG // ! Enable all assertions for testing

#include "harness.hpp"

#if STRINGZILLA_ARCH_METAL_
#include "cross_device.hpp"
#endif

namespace ashvardanian::stringzilla::test {

std::size_t test_cross_metal(environment_t const &env, std::size_t ordinal) {
#if STRINGZILLA_ARCH_METAL_
    std::size_t const capability_failures = run_test(env.settings, "test_metal_capabilities_unit", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        verify(sz_device_count_metal(&devices) == sz_success_k && devices != 0);
        verify(sz_capabilities_detected_metal(ordinal, &reported) == sz_success_k);
        // A listed Metal device may fall below the Apple7 baseline and report no supported kernels.
        verify((reported & ~sz_cap_metal_k) == 0);
        verify(sz_capabilities_detected_metal(devices, &reported) == sz_missing_gpu_k);
    });
    auto const selected = sz::device_t::make(sz::device_kind_t::metal_k, ordinal);
    verify(selected.status == status_t::success_k);
    stream_t stream(ordinal, sz_stream_init_metal, sz_stream_free_metal);
    device_backend_t runtime;
    runtime.selected = selected.value;
    runtime.stream = stream.handle;
    verify(sz_capabilities_detected_metal(ordinal, &runtime.capabilities) == sz_success_k);
    verify(sz_allocator_init_unified_metal(&runtime.unified) == sz_success_k);
    runtime.init = sz_stream_init_metal;
    runtime.free = sz_stream_free_metal;
    cross_section_t check(env);
    check.detected = runtime.capabilities;
    check.section("Cross Metal", sz_cap_metal_k);
    check("test_allocator_metal", [&] {
        sz_allocator_t device {};
        verify(sz_allocator_init_device_best(&device, sz_cap_metal_k) == sz_success_k);
        verify(device.allocate == runtime.unified.allocate && device.free == runtime.unified.free &&
               device.handle == runtime.unified.handle);
        verify(sz_allocator_init_pinned_best(&device, sz_cap_metal_k) == sz_missing_kernel_k);
        verify(device.allocate == runtime.unified.allocate && device.free == runtime.unified.free &&
               device.handle == runtime.unified.handle);
    });
#if STRINGZILLA_TARGET_METAL
    device_kernels const metal {
        "metal",
        runtime,
        sz_levenshtein_distances_metal,
        sz_overlap_scores_metal,
        sz_substrings_counts_metal,
        sz_substrings_find_metal,
        sz_substrings_replace_metal,
        sz_substrings_bm25_scores_metal,
        sz_regex_counts_metal,
        sz_regex_find_metal,
        sz_regex_replace_metal,
        sz_regex_extract_metal,
        sz_subdocuments_counts_metal,
        sz_subdocuments_find_metal,
        sz_utf8_uncased_fold_metal,
        sz_utf8_norm_metal,
        nullptr,
        nullptr,
    };
    check_device_kernels_(check, metal);
    strings_backend_t const strings_metal {"metal",
                                           sz_strings_to_tape_metal,
                                           sz_strings_to_rows_metal,
                                           sz_strings_escape_metal,
                                           sz_strings_unescape_metal,
                                           sz_strings_lookup_metal,
                                           sz_strings_normalize_metal};
    check_device_strings_(check, runtime, strings_metal);
#endif
    return capability_failures + check.failures + test_cross_dispatch_device(env, runtime, nullptr, nullptr);
#else
    sz_unused_(env);
    sz_unused_(ordinal);
    return 0;
#endif
}

} // namespace ashvardanian::stringzilla::test
