/**
 *  @file c/stringzillas/smith_waterman.cuh
 *  @brief Parallel Smith-Waterman local alignment scores shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_SZS_SMITH_WATERMAN_CUH_
#define STRINGZILLAS_SZS_SMITH_WATERMAN_CUH_
#include "stringzillas.cuh"

/**
 *  @brief Allocates a `smith_waterman_backends_t` holding the `engine_type_` arm built from @p ctor_args, publishes the
 *         opaque handle, and folds the bad-alloc / success status reporting that every capability arm repeats.
 *  @tparam engine_type_ The concrete backend variant alternative to emplace (the only argument that varies per arm).
 */
template <typename engine_type_, typename... ctor_args_types_>
inline sz_status_t emplace_smith_waterman_engine(szs_smith_waterman_scores_t *engine_punned, char const **error_message,
                                                 ctor_args_types_ &&...ctor_args) noexcept {
    auto engine = new (std::nothrow) smith_waterman_backends_t(
        std::in_place_type_t<engine_type_>(), engine_type_(std::forward<ctor_args_types_>(ctor_args)...));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Smith-Waterman engine");
    *engine_punned = reinterpret_cast<szs_smith_waterman_scores_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

extern "C" {

#pragma region Smith Waterman

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_init(                             //
    sz_u8_t const *byte_to_class, sz_error_cost_t const *class_substitution_costs, //
    sz_error_cost_t open, sz_error_cost_t extend,                                  //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,              //
    szs_smith_waterman_scores_t *engine_punned, char const **error_message) {

    sz_unused_(alloc);        // Custom allocator not yet implemented, using default
    sz_unused_(capabilities); // Optional backends may be compiled out
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

    // If the gap opening and extension costs are identical we can use less memory
    auto const can_use_linear_costs = open == extend;
    auto const linear_costs = szs::linear_gap_costs_t {open};
    auto const affine_costs = szs::affine_gap_costs_t {open, extend};
    auto substitution_costs = szs::error_costs_32x32_t {};
    std::memcpy((void *)substitution_costs.byte_to_class, (void const *)byte_to_class,
                sizeof(substitution_costs.byte_to_class));
    std::memcpy((void *)substitution_costs.class_substitution_costs, (void const *)class_substitution_costs,
                sizeof(substitution_costs.class_substitution_costs));

#if SZ_USE_ICELAKE
    bool const can_use_icelake = (capabilities & sz_cap_icelake_k) == sz_cap_icelake_k;
    if (can_use_icelake && can_use_linear_costs)
        return emplace_smith_waterman_engine<szs::smith_waterman_icelake_t>(engine_punned, error_message,
                                                                            substitution_costs, linear_costs);
#endif // SZ_USE_ICELAKE

#if SZ_USE_HASWELL
    bool const can_use_haswell = (capabilities & sz_cap_haswell_k) == sz_cap_haswell_k;
    if (can_use_haswell && can_use_linear_costs)
        return emplace_smith_waterman_engine<szs::smith_waterman_haswell_t>(engine_punned, error_message,
                                                                            substitution_costs, linear_costs);
#endif // SZ_USE_HASWELL

    // Hopper reports the base-CUDA bit too, so the Hopper (DPX) tier must be tested before plain CUDA.
#if SZ_USE_HOPPER
    bool const can_use_hopper = (capabilities & sz_caps_ckh_k) == sz_caps_ckh_k;
    if (can_use_hopper && can_use_linear_costs)
        return emplace_smith_waterman_engine<szs::smith_waterman_hopper_t>(engine_punned, error_message,
                                                                           substitution_costs, linear_costs);
    else if (can_use_hopper)
        return emplace_smith_waterman_engine<szs::affine_smith_waterman_hopper_t>(engine_punned, error_message,
                                                                                  substitution_costs, affine_costs);
#endif // SZ_USE_HOPPER

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) != 0;
    if (can_use_cuda && can_use_linear_costs)
        return emplace_smith_waterman_engine<szs::smith_waterman_cuda_t>(engine_punned, error_message,
                                                                         substitution_costs, linear_costs);
    else if (can_use_cuda)
        return emplace_smith_waterman_engine<szs::affine_smith_waterman_cuda_t>(engine_punned, error_message,
                                                                                substitution_costs, affine_costs);
#endif // SZ_USE_CUDA

    if (can_use_linear_costs)
        return emplace_smith_waterman_engine<szs::smith_waterman_serial_t>(engine_punned, error_message,
                                                                           substitution_costs, linear_costs);
    else
        return emplace_smith_waterman_engine<szs::affine_smith_waterman_serial_t>(engine_punned, error_message,
                                                                                  substitution_costs, affine_costs);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_sequence(                       //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *a, sz_sequence_t const *b,                              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_as_cpp_container_t {a};
    auto b_container = sz_sequence_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u32tape(                        //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u32tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u32tape_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u64tape(                        //
    szs_smith_waterman_scores_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {

    sz_assert_(a != nullptr && b != nullptr && "Input texts cannot be null");
    auto a_container = sz_sequence_u64tape_as_cpp_container_t {a};
    auto b_container = sz_sequence_u64tape_as_cpp_container_t {b};
    return szs_smith_waterman_scores_for_(                      //
        engine_punned, device_punned, a_container, b_container, //
        results, results_stride, error_message);
}

SZ_DYNAMIC void szs_smith_waterman_scores_free(szs_smith_waterman_scores_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<smith_waterman_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Smith Waterman
}

#endif // STRINGZILLAS_SZS_SMITH_WATERMAN_CUH_
