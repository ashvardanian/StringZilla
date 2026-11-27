/**
 *  @file test/cross_metal.cpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal kernels and dispatch points through the shared device suites.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include "harness.hpp"

#if STRINGZILLA_ARCH_METAL_
#include "cross_device.hpp"
#endif

namespace ashvardanian::stringzilla::test {

void test_cross_metal(std::size_t ordinal) {
#if STRINGZILLA_ARCH_METAL_
    run_test_case("sz_capabilities_detected_metal", [&] {
        sz_size_t devices = 0;
        sz_capability_t reported = 0;
        expect_true(sz_count_devices_metal(&devices) == sz_success_k && devices != 0);
        expect_true(sz_capabilities_detected_metal(ordinal, &reported) == sz_success_k && reported == sz_cap_metal_k);
        expect_equal(sz_capabilities_detected_metal(devices, &reported), sz_missing_gpu_k);
    });
    device_selection_t const selected {sz::device_kind_t::metal_k, ordinal};
    if (!expect_equal(check_device(selected), sz_success_k)) return;
    stream_t stream(ordinal, sz_stream_init_metal, sz_stream_free_metal);
    device_backend_t runtime;
    runtime.selected = selected;
    runtime.stream = stream.handle;
    if (!expect_equal(sz_capabilities_detected_metal(ordinal, &runtime.capabilities), sz_success_k)) return;
    if (!expect_equal(sz_allocator_init_unified_metal(&runtime.pinned), sz_success_k)) return;
    runtime.device = runtime.pinned;
    // Every device tape a case gathers is carved from one block, which the case rewinds when done.
    std::size_t const arena_bytes = (std::size_t)512 << 20;
    unified_alloc<char> const arena_allocator(runtime.device, runtime.stream);
    std::unique_ptr<char, device_block_release_t> const arena_block(arena_allocator.allocate(arena_bytes),
                                                                    {arena_allocator, arena_bytes});
    if (!expect_true(arena_block != nullptr)) return;
    device_arena_t arena {arena_block.get(), arena_bytes, 0};
    runtime.arena = &arena;
    runtime.init = sz_stream_init_metal;
    runtime.free = sz_stream_free_metal;
    capability_gate_t const metal_gate(sz_cap_metal_k, runtime.capabilities);
    run_test_case("sz_allocator_init_device_best(sz_cap_metal_k)", metal_gate, [&] {
        sz_allocator_t device {};
        expect_equal(sz_allocator_init_device_best(&device, sz_cap_metal_k), sz_success_k);
        expect_true(device.allocate == runtime.pinned.allocate && device.free == runtime.pinned.free &&
                    device.handle == runtime.pinned.handle);
        expect_equal(sz_allocator_init_pinned_best(&device, sz_cap_metal_k), sz_missing_kernel_k);
        expect_true(device.allocate == runtime.pinned.allocate && device.free == runtime.pinned.free &&
                    device.handle == runtime.pinned.handle);
    });
#if STRINGZILLA_TARGET_METAL
    device_kernels const metal {
        "metal",
        runtime,
        sz_levenshtein_distances_metal,
        sz_ngrams_overlaps_metal,
        sz_ngrams_sketches_metal,
        sz_substrings_contains_metal,
        sz_substrings_counts_metal,
        sz_substrings_find_metal,
        sz_substrings_replace_metal,
        sz_substrings_bm25_scores_metal,
        sz_regex_contains_metal,
        sz_regex_counts_metal,
        sz_regex_find_metal,
        sz_regex_replace_metal,
        sz_regex_extract_metal,
        sz_subdocuments_contains_metal,
        sz_subdocuments_counts_metal,
        sz_subdocuments_find_metal,
        sz_utf8_uncased_fold_metal,
        sz_utf8_norm_metal,
        nullptr,
        nullptr,
    };
    check_device_kernels_(metal_gate, metal);
    strings_backend_t const strings_metal {"metal",
                                           sz_strings_to_tape_metal,
                                           sz_strings_to_rows_metal,
                                           sz_strings_escape_metal,
                                           sz_strings_unescape_metal,
                                           sz_strings_lookup_metal,
                                           sz_strings_normalize_metal};
    check_device_strings_(metal_gate, runtime, strings_metal);
#endif
    test_cross_dispatch_device(runtime, nullptr, nullptr);
#else
    sz_unused_(ordinal);
#endif
}

} // namespace ashvardanian::stringzilla::test
