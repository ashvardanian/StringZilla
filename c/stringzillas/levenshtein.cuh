/**
 *  @file c/stringzillas/levenshtein.cuh
 *  @brief Parallel Levenshtein & UTF-8 Levenshtein distances shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
#define STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
#include "stringzillas.cuh"

extern "C" {

#pragma region Levenshtein Distances

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_init(                                             //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_t *engine_punned, char const **error_message) {

    sz_unused_(alloc);        // Custom allocator not yet implemented, using default
    sz_unused_(capabilities); // Optional backends may be compiled out
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = szs::uniform_substitution_costs_t {match, mismatch};
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICELAKE
    bool const can_use_icelake = (capabilities & sz_cap_icelake_k) == sz_cap_icelake_k;
    if (can_use_icelake && can_use_linear_costs) {
        auto variant = szs::levenshtein_icelake_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_icelake_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_icelake) {
        auto variant = szs::affine_levenshtein_icelake_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_icelake_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICELAKE

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_linear_costs) {
        auto variant = szs::levenshtein_cuda_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_cuda) {
        auto variant = szs::affine_levenshtein_cuda_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_cuda_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_CUDA

#if SZ_USE_KEPLER
    bool const can_use_kepler = (capabilities & sz_caps_ck_k) == sz_caps_ck_k;
    if (can_use_kepler && can_use_linear_costs) {
        auto variant = szs::levenshtein_kepler_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_kepler_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_kepler) {
        auto variant = szs::affine_levenshtein_kepler_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_kepler_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_KEPLER

#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs) {
        auto variant = szs::levenshtein_hopper_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else if (can_use_hopper) {
        auto variant = szs::affine_levenshtein_hopper_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_hopper_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_HOPPER

    if (can_use_linear_costs) {
        auto variant = szs::levenshtein_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::levenshtein_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_levenshtein_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow)
            levenshtein_backends_t(std::in_place_type_t<szs::affine_levenshtein_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_sequence(                       //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u32tape(                        //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u64tape(                        //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,              //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_free(szs_levenshtein_distances_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein Distances

#pragma region Levenshtein UTF8 Distances

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_init(                                        //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_utf8_t *engine_punned, char const **error_message) {

    sz_unused_(alloc); // Custom allocator not yet implemented, using default
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const substitution_costs = szs::uniform_substitution_costs_t {match, mismatch};
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};

#if SZ_USE_ICELAKE
    bool const can_use_icelake = (capabilities & sz_cap_icelake_k) != 0;
    if (can_use_icelake && can_use_linear_costs) {
        auto variant = szs::levenshtein_utf8_icelake_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::levenshtein_utf8_icelake_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif // SZ_USE_ICELAKE

    bool const can_use_serial = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_serial && can_use_linear_costs) {
        auto variant = szs::levenshtein_utf8_serial_t(substitution_costs, linear_costs);
        auto engine = new (std::nothrow)
            levenshtein_utf8_backends_t(std::in_place_type_t<szs::levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }
    else {
        auto variant = szs::affine_levenshtein_utf8_serial_t(substitution_costs, affine_costs);
        auto engine = new (std::nothrow) levenshtein_utf8_backends_t(
            std::in_place_type_t<szs::affine_levenshtein_utf8_serial_t>(), std::move(variant));
        if (!engine)
            return propagate_error(sz::status_t::bad_alloc_k, error_message,
                                   "Failed to allocate UTF-8 Levenshtein engine");

        *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
        return propagate_error(sz::status_t::success_k, error_message);
    }

    return propagate_error(sz::status_t::unknown_k, error_message, "No supported UTF-8 Levenshtein backends available");
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_sequence(                       //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u32tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u64tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,                   //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_levenshtein_distances_utf8_for_(                 //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_utf8_free(szs_levenshtein_distances_utf8_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein UTF8 Distances
}

#endif // STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
