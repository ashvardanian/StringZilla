/**
 *  @file c/stringzillas/substrings.cuh
 *  @brief Multi-pattern Aho-Corasick search shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date August 6, 2026
 */
#ifndef SZS_SUBSTRINGS_CUH_
#define SZS_SUBSTRINGS_CUH_
#include "stringzillas.cuh"

#include <cstddef> // `offsetof`

#pragma region Dispatch

/**
 *  @brief Allocates a `substrings_backends_t` holding the already-built @p engine_variant and publishes the
 *         opaque handle. The variant arrives pre-built, as `try_build` is fallible.
 */
template <typename engine_type_>
sz_status_t emplace_substrings_engine(szs_substrings_t *engine_punned, char const **error_message,
                                      engine_type_ engine_variant) noexcept {
    auto engine = new (std::nothrow)
        substrings_backends_t(std::in_place_type_t<engine_type_>(), std::move(engine_variant));
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Substrings engine");
    *engine_punned = reinterpret_cast<szs_substrings_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

/**
 *  @brief Counts every needle's occurrences in every haystack, on whichever scope @p device_punned names.
 *  @sa `szs_levenshtein_cross_` for the same dispatch skeleton.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_count_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                         haystacks_type_ const &haystacks, sz_size_t *counts,
                                         char const **error_message) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(counts != nullptr && "Counts output cannot be null");

    auto &engine = *reinterpret_cast<substrings_backends_t *>(engine_punned);
    auto &device = *reinterpret_cast<device_scope_t *>(device_punned);
    sz::span<size_t> const counts_span {counts, haystacks.size()};
    size_t matches_total = 0; // ? `szs_substrings_count` publishes only the per-haystack breakdown

    sz::status_t const status = std::visit(
        [&](auto &engine_variant) -> sz::status_t {
            using engine_variant_t = std::decay_t<decltype(engine_variant)>;
            if constexpr (is_gpu_capability(engine_variant_t::capability_k)) {
#if SZ_USE_CUDA
                auto [gpu_scope, scope_status] = gpu_scope_for(device);
                if (scope_status.status != sz::status_t::success_k) return scope_status.status;
                return engine_variant.try_count(haystacks, counts_span, matches_total, get_executor(gpu_scope),
                                                get_specs(gpu_scope));
#else
                return sz::status_t::missing_gpu_k;
#endif
            }
            else
                return std::visit(
                    [&](auto &scope_variant) -> sz::status_t {
                        using scope_t = std::decay_t<decltype(scope_variant)>;
                        if constexpr (!is_cpu_scope<scope_t>()) return sz::status_t::device_code_mismatch_k;
                        else
                            return engine_variant.try_count(haystacks, counts_span, matches_total,
                                                            get_executor(scope_variant), get_specs(scope_variant));
                    },
                    device.variants);
        },
        engine.variants);
    return propagate_error(status, error_message);
}

/**
 *  @brief Locates every needle's occurrences in every haystack. @sa `szs_substrings_count_`.
 *
 *  The caller's array IS the output on every arm, viewed as the C++ match. Capacity checks and host-memory
 *  staging live in the engines, not here.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_find_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                        haystacks_type_ const &haystacks, szs_substrings_match_t *matches,
                                        sz_size_t matches_capacity, sz_size_t *matches_found_out,
                                        char const **error_message) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");
    sz_assert_(matches_found_out != nullptr && "Matches-found output cannot be null");

    static_assert(
        sizeof(szs::substrings_match_t) == sizeof(szs_substrings_match_t) &&
            offsetof(szs::substrings_match_t, haystack_index) == offsetof(szs_substrings_match_t, haystack_index) &&
            offsetof(szs::substrings_match_t, needle_index) == offsetof(szs_substrings_match_t, needle_index) &&
            offsetof(szs::substrings_match_t, byte_offset) == offsetof(szs_substrings_match_t, byte_offset) &&
            offsetof(szs::substrings_match_t, byte_length) == offsetof(szs_substrings_match_t, byte_length),
        "The C++ match must mirror the C ABI field for field, or this view would misreport matches");

    auto &engine = *reinterpret_cast<substrings_backends_t *>(engine_punned);
    auto &device = *reinterpret_cast<device_scope_t *>(device_punned);
    sz::span<szs::substrings_match_t> const matches_out {reinterpret_cast<szs::substrings_match_t *>(matches),
                                                         matches_capacity};
    *matches_found_out = 0;
    size_t matches_found = 0;

    sz::status_t const status = std::visit(
        [&](auto &engine_variant) -> sz::status_t {
            using engine_variant_t = std::decay_t<decltype(engine_variant)>;
            if constexpr (is_gpu_capability(engine_variant_t::capability_k)) {
#if SZ_USE_CUDA
                auto [gpu_scope, scope_status] = gpu_scope_for(device);
                if (scope_status.status != sz::status_t::success_k) return scope_status.status;
                return engine_variant.try_find(haystacks, matches_out, matches_found, get_executor(gpu_scope),
                                               get_specs(gpu_scope));
#else
                return sz::status_t::missing_gpu_k;
#endif
            }
            else
                return std::visit(
                    [&](auto &scope_variant) -> sz::status_t {
                        using scope_t = std::decay_t<decltype(scope_variant)>;
                        if constexpr (!is_cpu_scope<scope_t>()) return sz::status_t::device_code_mismatch_k;
                        else
                            return engine_variant.try_find(haystacks, matches_out, matches_found,
                                                           get_executor(scope_variant), get_specs(scope_variant));
                    },
                    device.variants);
        },
        engine.variants);

    sz_status_t const result = propagate_error(status, error_message);
    if (result != sz_success_k) return result;
    *matches_found_out = (sz_size_t)matches_found;
    return sz_success_k;
}

#pragma endregion Dispatch

extern "C" {

#pragma region Substrings

SZ_API_RUNTIME sz_status_t szs_substrings_init(                                       //
    sz_sequence_t const *needles, szs_substrings_case_sensitivity_t case_sensitivity, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                 //
    szs_substrings_t *engine_punned, char const **error_message) {

    sz_unused_(alloc); // Custom allocator not yet implemented, using default
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");
    sz_assert_(needles != nullptr && "Needle collection cannot be null");

    szs::substrings_case_sensitivity_t const engine_case_sensitivity = case_sensitivity == szs_substrings_uncased_k
                                                                           ? szs::substrings_uncased_k
                                                                           : szs::substrings_cased_k;
    auto const needles_container = sz_sequence_as_cpp_container_t {needles};

#if SZ_USE_CUDA
    bool const can_use_cuda = (capabilities & sz_cap_cuda_k) == sz_cap_cuda_k;
    if (can_use_cuda) {
        szs::substrings_u32_cuda_t engine_variant;
        szs::cuda_status_t const status = engine_variant.try_build(needles_container, engine_case_sensitivity);
        if (status.status != sz::status_t::success_k) return propagate_error(status, error_message);
        return emplace_substrings_engine<szs::substrings_u32_cuda_t>(engine_punned, error_message,
                                                                     std::move(engine_variant));
    }
#endif // SZ_USE_CUDA

    bool const can_use_parallel = (capabilities & sz_cap_parallel_k) == sz_cap_parallel_k;
    if (can_use_parallel) {
        szs::substrings_u32_parallel_t engine_variant;
        sz::status_t const status = engine_variant.try_build(needles_container, engine_case_sensitivity);
        if (status != sz::status_t::success_k) return propagate_error(status, error_message);
        return emplace_substrings_engine<szs::substrings_u32_parallel_t>(engine_punned, error_message,
                                                                         std::move(engine_variant));
    }

    szs::substrings_u32_serial_t engine_variant;
    sz::status_t const status = engine_variant.try_build(needles_container, engine_case_sensitivity);
    if (status != sz::status_t::success_k) return propagate_error(status, error_message);
    return emplace_substrings_engine<szs::substrings_u32_serial_t>(engine_punned, error_message,
                                                                   std::move(engine_variant));
}

SZ_API_RUNTIME sz_status_t szs_substrings_count(                      //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *haystacks, sz_size_t *counts,                //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_as_cpp_container_t {haystacks}, counts,
                                 error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find(                                            //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                      //
    sz_sequence_t const *haystacks,                                                        //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_as_cpp_container_t {haystacks}, matches,
                                matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_count_u32tape(              //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *haystacks, sz_size_t *counts,        //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                 counts, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_count_u64tape(              //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *haystacks, sz_size_t *counts,        //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                 counts, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find_u32tape(                                    //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                      //
    sz_sequence_u32tape_t const *haystacks,                                                //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                matches, matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find_u64tape(                                    //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                      //
    sz_sequence_u64tape_t const *haystacks,                                                //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                matches, matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME void szs_substrings_free(szs_substrings_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<substrings_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Substrings
}

#endif // SZS_SUBSTRINGS_CUH_
