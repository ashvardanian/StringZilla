/**
 *  @file c/stringzillas/fingerprints.cuh
 *  @brief Parallel Min-Hash fingerprints & UTF-8 fingerprints shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_SZS_FINGERPRINTS_CUH_
#define STRINGZILLAS_SZS_FINGERPRINTS_CUH_
#include "stringzillas.cuh"

/**
 *  @brief Allocates a `fingerprints_backends_t` holding the already-built @p variant, records @p dimensions, publishes
 *         the opaque handle, and folds the bad-alloc / success status reporting that every capability arm repeats.
 *  @tparam variant_t The concrete backend variant alternative to emplace (e.g. `vec<hasher_t>` or a fallback variant).
 */
template <typename variant_type_>
sz_status_t emplace_fingerprints_engine(szs_fingerprints_t *engine_punned, char const **error_message,
                                        sz_size_t dimensions, variant_type_ variant) noexcept {
    auto engine = new (std::nothrow) fingerprints_backends_t(std::in_place_type_t<variant_type_>(), std::move(variant));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Fingerprints engine");
    engine->dimensions = dimensions;
    *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

extern "C" {

#pragma region Fingerprints

SZ_DYNAMIC sz_status_t szs_fingerprints_init(                                     //
    sz_size_t dimensions, sz_size_t alphabet_size,                                //
    sz_size_t const *window_widths, sz_size_t window_widths_count, sz_u64_t seed, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,             //
    szs_fingerprints_t *engine_punned, char const **error_message) {

    sz_unused_(alloc);        // Custom allocator not yet implemented, using default
    sz_unused_(capabilities); // Optional backends may be compiled out
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // Use some default window widths if none are provided
    sz_size_t const default_window_widths[] = {3, 4, 5, 7, 9, 11, 15, 31};
    if (!window_widths || window_widths_count == 0) {
        window_widths = default_window_widths;
        window_widths_count = sizeof(default_window_widths) / sizeof(sz_size_t);
    }

    // For optimal performance the number of dimensions per window width must be divisible by the fingerprint
    // slice, so each width maps onto a whole `floating_rolling_hashers` block. When `dimensions` is not a clean
    // `window_widths_count * fingerprint_slice_k` multiple we fall back to the per-dimension `basic_rolling_hashers`
    // variant, which now uses the dimension-tiled, spill-free CUDA kernel and masks any unused lanes via its
    // `hashers_count` bound - so arbitrary dimension counts are supported, just without the sliced fast path.
    auto const dimensions_per_window_width_min = dimensions / window_widths_count;
    auto const dimensions_per_window_width_max = sz::divide_round_up(dimensions, window_widths_count);
    auto const can_use_sliced_sketchers = (dimensions_per_window_width_min == dimensions_per_window_width_max) &&
                                          (dimensions_per_window_width_min % fingerprint_slice_k == 0);
    using fallback_variant_cpus_t = typename fingerprints_backends_t::fallback_variant_cpus_t;

#if SZ_USE_HASWELL
    bool const can_use_haswell = (capabilities & sz_cap_haswell_k) == sz_cap_haswell_k;
    if (can_use_haswell && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_haswell_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset, seed);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        return emplace_fingerprints_engine<vec<hasher_t>>(engine_punned, error_message, dimensions, std::move(hashers));
    }
#endif // SZ_USE_HASWELL

#if SZ_USE_SKYLAKE
    bool const can_use_skylake = (capabilities & sz_cap_skylake_k) == sz_cap_skylake_k;
    if (can_use_skylake && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_skylake_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset, seed);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        return emplace_fingerprints_engine<vec<hasher_t>>(engine_punned, error_message, dimensions, std::move(hashers));
    }
#endif // SZ_USE_SKYLAKE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_cuda_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset, seed);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        return emplace_fingerprints_engine<vec<hasher_t>>(engine_punned, error_message, dimensions, std::move(hashers));
    }
    else if (can_use_cuda) {
        using fallback_variant_cuda_t = typename fingerprints_backends_t::fallback_variant_cuda_t;
        auto variant = fallback_variant_cuda_t();
        for (size_t dimension = 0; dimension < dimensions; ++dimension) {
            auto const window_width = window_widths[dimension % window_widths_count];
            auto const extend_status = variant.try_extend(window_width, 1, alphabet_size, seed);
            if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
        }

        auto engine = new (std::nothrow)
            fingerprints_backends_t(std::in_place_type_t<fallback_variant_cuda_t>(), std::move(variant));
        if (!engine) return sz_bad_alloc_k;

        engine->dimensions = dimensions;
        *engine_punned = reinterpret_cast<szs_fingerprints_t>(engine);
        return sz_success_k;
    }
#endif // SZ_USE_CUDA

    // Build the vectorized, but serial backend
    if (can_use_sliced_sketchers) {
        auto const count_hashers = dimensions / fingerprint_slice_k;
        using hasher_t = szs::floating_rolling_hashers<sz_cap_serial_k, fingerprint_slice_k>;
        vec<hasher_t> hashers;
        if (hashers.try_resize(count_hashers) != sz::status_t::success_k) return sz_bad_alloc_k;

        // Populate the hashers with the given window widths
        for (size_t i = 0; i < count_hashers; ++i) {
            auto const window_width = window_widths[i % window_widths_count];
            auto const first_dimension_offset = i * fingerprint_slice_k;
            auto const seed_status = hashers[i].try_seed(window_width, alphabet_size, first_dimension_offset, seed);
            if (seed_status != sz::status_t::success_k) return static_cast<sz_status_t>(seed_status);
        }

        return emplace_fingerprints_engine<vec<hasher_t>>(engine_punned, error_message, dimensions, std::move(hashers));
    }

    // Build the fallback variant with interleaving width dimensions
    auto variant = fallback_variant_cpus_t();
    for (size_t dimension = 0; dimension < dimensions; ++dimension) {
        auto const window_width = window_widths[dimension % window_widths_count];
        auto const extend_status = variant.try_extend(window_width, 1, alphabet_size, seed);
        if (extend_status != sz::status_t::success_k) return static_cast<sz_status_t>(extend_status);
    }

    return emplace_fingerprints_engine<fallback_variant_cpus_t>(engine_punned, error_message, dimensions,
                                                                std::move(variant));
}

SZ_DYNAMIC sz_status_t szs_fingerprints_sequence(                       //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *texts,                                         //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u32tape(                        //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *texts,                                 //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u32tape_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u64tape(                        //
    szs_fingerprints_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *texts,                                 //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    sz_assert_(texts != nullptr && "Input texts cannot be null");
    auto texts_container = sz_sequence_u64tape_as_cpp_container_t {texts};
    return szs_fingerprints_for_(                      //
        engine_punned, device_punned, texts_container, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC void szs_fingerprints_free(szs_fingerprints_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<fingerprints_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Fingerprints

#pragma region Fingerprints UTF8

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_init(                    //
    sz_size_t dimensions, sz_size_t alphabet_size,                    //
    sz_size_t const *window_widths, sz_size_t window_widths_count,    //
    sz_u64_t seed,                                                    //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities, //
    szs_fingerprints_utf8_t *engine_punned, char const **error_message) {

    return szs_fingerprints_init(                                            //
        dimensions, alphabet_size, window_widths, window_widths_count, seed, //
        alloc, capabilities, engine_punned, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_sequence(                       //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *texts,                                              //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_sequence(        //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_u32tape(                        //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *texts,                                      //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_u32tape(         //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_utf8_u64tape(                        //
    szs_fingerprints_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *texts,                                      //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {

    return szs_fingerprints_u64tape(         //
        engine_punned, device_punned, texts, //
        min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC void szs_fingerprints_utf8_free(szs_fingerprints_utf8_t engine_punned) {
    return szs_fingerprints_free(engine_punned);
}

#pragma endregion Fingerprints UTF8
}

#endif // STRINGZILLAS_SZS_FINGERPRINTS_CUH_
