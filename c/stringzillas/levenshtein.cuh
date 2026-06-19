/**
 *  @file c/stringzillas/levenshtein.cuh
 *  @brief Parallel Levenshtein & UTF-8 Levenshtein distances shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
#define STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
#include "stringzillas.cuh"

/**
 *  @brief Allocates a `levenshtein_backends_t` holding the `engine_type_` arm built from @p ctor_args, publishes the
 *         opaque handle, and folds the bad-alloc / success status reporting that every capability arm repeats.
 *  @tparam engine_type_ The concrete backend variant alternative to emplace (the only argument that varies per arm).
 */
template <typename engine_type_, typename... ctor_args_types_>
sz_status_t emplace_levenshtein_engine(szs_levenshtein_distances_t *engine_punned, char const **error_message,
                                       ctor_args_types_ &&...ctor_args) noexcept {
    auto engine = new (std::nothrow) levenshtein_backends_t(std::in_place_type_t<engine_type_>(),
                                                            engine_type_(std::forward<ctor_args_types_>(ctor_args)...));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Levenshtein engine");
    *engine_punned = reinterpret_cast<szs_levenshtein_distances_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

/** @brief UTF-8 sibling of `emplace_levenshtein_engine` targeting `levenshtein_utf8_backends_t`. */
template <typename engine_type_, typename... ctor_args_types_>
sz_status_t emplace_levenshtein_utf8_engine(szs_levenshtein_distances_utf8_t *engine_punned, char const **error_message,
                                            ctor_args_types_ &&...ctor_args) noexcept {
    auto engine = new (std::nothrow) levenshtein_utf8_backends_t(
        std::in_place_type_t<engine_type_>(), engine_type_(std::forward<ctor_args_types_>(ctor_args)...));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate UTF-8 Levenshtein engine");
    *engine_punned = reinterpret_cast<szs_levenshtein_distances_utf8_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

/**
 *  @brief Cross-product (or symmetric self-similarity) dispatch shared by the byte-level and UTF-8 Levenshtein shims.
 *
 *  Builds an `szs::strided_rows<sz_size_t>` view over the caller's @p results matrix and invokes the migrated C++
 *  engine: the two-set overload `engine(queries, candidates, matrix, ...)` when @p candidates_container is non-null,
 *  or the symmetric overload `engine(queries, matrix, ...)` when it is null. The unsigned distance value type is
 *  `sz_size_t`, and @p results_row_stride counts elements (not bytes) between consecutive query rows.
 */
template <typename backends_type_, typename queries_type_, typename candidates_type_>
sz_status_t szs_levenshtein_cross_(                                                       //
    backends_type_ *engine, szs_device_scope_t device_punned,                             //
    queries_type_ const &queries_container, candidates_type_ const *candidates_container, //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(results != nullptr && "Results must not be null");

    auto *device = reinterpret_cast<device_scope_t *>(device_punned);
    auto const queries_count = queries_container.size();
    auto const candidates_count = candidates_container != nullptr ? candidates_container->size() : queries_count;
    auto results_matrix = szs::strided_rows<sz_size_t> {results, queries_count, candidates_count, results_row_stride};

    sz_status_t result = sz_success_k;
    auto variant_logic = [&](auto &engine_variant) {
        using engine_variant_t = std::decay_t<decltype(engine_variant)>;
        constexpr sz_capability_t engine_capability_k = engine_variant_t::capability_k;

        // GPU backends are only compatible with GPU scopes
        if constexpr (is_gpu_capability(engine_capability_k)) {
#if SZ_USE_CUDA
            if (std::holds_alternative<gpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<gpu_scope_t>(device->variants);
                szs::cuda_status_t status = candidates_container != nullptr
                                                ? engine_variant(queries_container, *candidates_container,
                                                                 results_matrix, get_executor(device_scope),
                                                                 get_specs(device_scope))
                                                : engine_variant(queries_container, results_matrix, //
                                                                 get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            // Try ephemeral GPU on default scope (device 0)
            else if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &ctx = default_gpu_context();
                szs::cuda_status_t status = ctx.status != sz::status_t::success_k ? ctx.status
                                            : candidates_container != nullptr
                                                ? engine_variant(queries_container, *candidates_container,
                                                                 results_matrix, ctx.executor, ctx.specs)
                                                : engine_variant(queries_container, results_matrix, ctx.executor,
                                                                 ctx.specs);
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::device_code_mismatch_k, error_message); }
#else
            result = propagate_error(sz::status_t::missing_gpu_k, error_message);
#endif // SZ_USE_CUDA
        }
        // CPU backends are only compatible with CPU scopes
        else {
            if (std::holds_alternative<default_scope_t>(device->variants)) {
                auto &device_scope = std::get<default_scope_t>(device->variants);
                sz::status_t status = candidates_container != nullptr
                                          ? engine_variant(queries_container, *candidates_container, results_matrix,
                                                           get_executor(device_scope), get_specs(device_scope))
                                          : engine_variant(queries_container, results_matrix, //
                                                           get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else if (std::holds_alternative<cpu_scope_t>(device->variants)) {
                auto &device_scope = std::get<cpu_scope_t>(device->variants);
                sz::status_t status = candidates_container != nullptr
                                          ? engine_variant(queries_container, *candidates_container, results_matrix,
                                                           get_executor(device_scope), get_specs(device_scope))
                                          : engine_variant(queries_container, results_matrix, //
                                                           get_executor(device_scope), get_specs(device_scope));
                result = propagate_error(status, error_message);
            }
            else { result = propagate_error(sz::status_t::device_code_mismatch_k, error_message); }
        }
    };

    std::visit(variant_logic, engine->variants);
    return result;
}

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
    if (can_use_icelake && can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_icelake_t>(engine_punned, error_message, substitution_costs,
                                                                      linear_costs);
    else if (can_use_icelake)
        return emplace_levenshtein_engine<szs::affine_levenshtein_icelake_t>(engine_punned, error_message,
                                                                             substitution_costs, affine_costs);
#endif // SZ_USE_ICELAKE

#if SZ_USE_NEON
    bool const can_use_neon = (capabilities & sz_cap_neon_k) == sz_cap_neon_k;
    if (can_use_neon && can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_neon_t>(engine_punned, error_message, substitution_costs,
                                                                   linear_costs);
    else if (can_use_neon)
        return emplace_levenshtein_engine<szs::affine_levenshtein_neon_t>(engine_punned, error_message,
                                                                          substitution_costs, affine_costs);
#endif // SZ_USE_NEON

    // GPU tiers are tested most-specific-first: a Hopper device reports the Kepler & base-CUDA bits too, so checking
    // base CUDA first would shadow the Hopper/Kepler engines. Hopper → Kepler → CUDA keeps each device on its best tier.
#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_hopper_t>(engine_punned, error_message, substitution_costs,
                                                                     linear_costs);
    else if (can_use_hopper)
        return emplace_levenshtein_engine<szs::affine_levenshtein_hopper_t>(engine_punned, error_message,
                                                                            substitution_costs, affine_costs);
#endif // SZ_USE_HOPPER

#if SZ_USE_KEPLER
    bool const can_use_kepler = (capabilities & sz_caps_ck_k) == sz_caps_ck_k;
    if (can_use_kepler && can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_kepler_t>(engine_punned, error_message, substitution_costs,
                                                                     linear_costs);
    else if (can_use_kepler)
        return emplace_levenshtein_engine<szs::affine_levenshtein_kepler_t>(engine_punned, error_message,
                                                                            substitution_costs, affine_costs);
#endif // SZ_USE_KEPLER

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_cuda_t>(engine_punned, error_message, substitution_costs,
                                                                   linear_costs);
    else if (can_use_cuda)
        return emplace_levenshtein_engine<szs::affine_levenshtein_cuda_t>(engine_punned, error_message,
                                                                          substitution_costs, affine_costs);
#endif // SZ_USE_CUDA

    if (can_use_linear_costs)
        return emplace_levenshtein_engine<szs::levenshtein_serial_t>(engine_punned, error_message, substitution_costs,
                                                                     linear_costs);
    else
        return emplace_levenshtein_engine<szs::affine_levenshtein_serial_t>(engine_punned, error_message,
                                                                            substitution_costs, affine_costs);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances(                                //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates,               //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u32tape(                          //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned,   //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_u32tape_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_u32tape_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u64tape(                          //
    szs_levenshtein_distances_t engine_punned, szs_device_scope_t device_punned,   //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_u64tape_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_u64tape_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
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
    if (can_use_icelake && can_use_linear_costs)
        return emplace_levenshtein_utf8_engine<szs::levenshtein_utf8_icelake_t>(engine_punned, error_message,
                                                                                substitution_costs, linear_costs);
#endif // SZ_USE_ICELAKE

#if SZ_USE_NEON
    bool const can_use_neon = (capabilities & sz_cap_neon_k) != 0;
    if (can_use_neon && can_use_linear_costs)
        return emplace_levenshtein_utf8_engine<szs::levenshtein_utf8_neon_t>(engine_punned, error_message,
                                                                             substitution_costs, linear_costs);
#endif // SZ_USE_NEON

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda && can_use_linear_costs)
        return emplace_levenshtein_utf8_engine<szs::levenshtein_utf8_cuda_t>(engine_punned, error_message,
                                                                             substitution_costs, linear_costs);
#endif // SZ_USE_CUDA

    bool const can_use_serial = (capabilities & sz_cap_serial_k) == sz_cap_serial_k;
    if (can_use_serial && can_use_linear_costs)
        return emplace_levenshtein_utf8_engine<szs::levenshtein_utf8_serial_t>(engine_punned, error_message,
                                                                               substitution_costs, linear_costs);
    else
        return emplace_levenshtein_utf8_engine<szs::affine_levenshtein_utf8_serial_t>(engine_punned, error_message,
                                                                                      substitution_costs, affine_costs);

    return propagate_error(sz::status_t::unknown_k, error_message, "No supported UTF-8 Levenshtein backends available");
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8(                                //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates,                    //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u32tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates,    //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_u32tape_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_u32tape_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u64tape(                        //
    szs_levenshtein_distances_utf8_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates,    //
    sz_size_t *results, sz_size_t results_row_stride, char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(queries != nullptr && "Query texts cannot be null");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    auto queries_container = sz_sequence_u64tape_as_cpp_container_t {queries};
    auto candidates_container = sz_sequence_u64tape_as_cpp_container_t {candidates};
    return szs_levenshtein_cross_(                                                                         //
        engine, device_punned, queries_container, candidates != nullptr ? &candidates_container : nullptr, //
        results, results_row_stride, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_utf8_free(szs_levenshtein_distances_utf8_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<levenshtein_utf8_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Levenshtein UTF8 Distances
}

#endif // STRINGZILLAS_SZS_LEVENSHTEIN_CUH_
