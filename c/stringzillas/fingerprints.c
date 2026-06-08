/**
 *  @file c/stringzillas/fingerprints.c
 *  @brief Public `szs_fingerprints_*` wrappers + dispatch updater for the fingerprint engine.
 *  @author Ash Vardanian
 *
 *  Mirrors `c/stringzilla/find.c`: fills the `fingerprints` slot of `szs_dispatch_table` via
 *  `szs_dispatch_fingerprints_update_` (skylake > haswell > serial cascade, plus the per-tier PTX
 *  module under SZ_USE_CUDA) and defines the `SZ_DYNAMIC` engine lifecycle + per-input-format
 *  wrappers. The engine POD holds the dimensions, window widths, and alphabet; the per-text sweep is
 *  parallelized inside the entry via fork_union when the device scope carries a pool.
 */
#include "dispatch.h" // `szs_dispatch_table`, updaters, sets `SZ_DYNAMIC_DISPATCH 1`

#include "stringzillas/fingerprints/serial.h"  // `szs_fingerprints_serial` + shared body/ctx
#include "stringzillas/fingerprints/haswell.h" // `szs_fingerprints_haswell` (no-op when SZ_USE_HASWELL=0)
#include "stringzillas/fingerprints/skylake.h" // `szs_fingerprints_skylake` (no-op when SZ_USE_SKYLAKE=0)

#if SZ_USE_CUDA
#include "stringzillas/fingerprints/cuda_host.cuh" // `szs_fingerprints_cuda` Driver-API launcher
#endif

#include <string.h> // `memset`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Dispatch Updater

SZ_DISPATCH_INTERNAL void szs_dispatch_fingerprints_update_(sz_capability_t caps) {
    szs_dispatch_table.fingerprints = &szs_fingerprints_serial;
#if SZ_USE_HASWELL
    if ((caps & sz_cap_haswell_k) == sz_cap_haswell_k) szs_dispatch_table.fingerprints = &szs_fingerprints_haswell;
#endif
#if SZ_USE_SKYLAKE
    if ((caps & sz_cap_skylake_k) == sz_cap_skylake_k) szs_dispatch_table.fingerprints = &szs_fingerprints_skylake;
#endif
    sz_unused_(caps);
}

#pragma endregion

#pragma region - Engine Lifecycle

SZ_DYNAMIC sz_status_t szs_fingerprints_init(                         //
    sz_size_t dimensions, sz_size_t alphabet_size,                    //
    sz_size_t const *window_widths, sz_size_t window_widths_count,    //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities, //
    szs_fingerprints_t *engine, char const **error_message) {

    if (!engine) {
        if (error_message) *error_message = "Engine handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }

    // Default window widths match the historical C++ defaults.
    sz_size_t const default_window_widths[] = {3, 4, 5, 7, 9, 11, 15, 31};
    if (!window_widths || window_widths_count == 0) {
        window_widths = default_window_widths;
        window_widths_count = sizeof(default_window_widths) / sizeof(default_window_widths[0]);
    }
    if (window_widths_count > SZS_FINGERPRINTS_MAX_WINDOWS) {
        if (error_message) *error_message = "Too many window widths";
        return sz_unexpected_dimensions_k;
    }
    if (dimensions == 0) {
        if (error_message) *error_message = "Dimensions must be positive";
        return sz_unexpected_dimensions_k;
    }

    szs_fingerprints_engine_t *eng = (szs_fingerprints_engine_t *)malloc(sizeof(szs_fingerprints_engine_t));
    if (!eng) {
        if (error_message) *error_message = "Failed to allocate Fingerprints engine";
        return sz_bad_alloc_k;
    }
    memset(eng, 0, sizeof(*eng));
    eng->dimensions = dimensions;
    eng->alphabet_size = alphabet_size;
    eng->window_widths_count = window_widths_count;
    for (sz_size_t i = 0; i < window_widths_count; ++i) eng->window_widths[i] = window_widths[i];
    eng->capabilities = capabilities;
    if (alloc) eng->alloc = *alloc;
    *engine = (szs_fingerprints_t)eng;
    return sz_success_k;
}

SZ_DYNAMIC void szs_fingerprints_free(szs_fingerprints_t engine) {
    if (engine) free(engine);
}

#pragma endregion

#pragma region - Per-Op Dispatch Helper

static int szs_fingerprints_scope_wants_gpu_(szs_device_scope_impl_t *scope) {
#if SZ_USE_CUDA
    if (!scope) return 0;
    if (scope->kind == szs_scope_gpu_k) return 1;
    if (scope->kind == szs_scope_default_k && (szs_dispatch_table.gpu_tier & sz_cap_cuda_k) == sz_cap_cuda_k) return 1;
    return 0;
#else
    sz_unused_(scope);
    return 0;
#endif
}

SZ_INTERNAL sz_status_t szs_fingerprints_run_(szs_fingerprints_engine_t const *engine, szs_device_scope_impl_t *device,
                                              szs_sequence_view_t inputs, sz_u32_t *min_hashes,
                                              sz_size_t min_hashes_stride, sz_u32_t *min_counts,
                                              sz_size_t min_counts_stride, char const **error) {
#if SZ_USE_CUDA
    if (szs_fingerprints_scope_wants_gpu_(device))
        return szs_fingerprints_cuda(engine, device, inputs, min_hashes, min_hashes_stride, min_counts,
                                     min_counts_stride, error);
#endif
    return szs_dispatch_table.fingerprints(engine, device, inputs, min_hashes, min_hashes_stride, min_counts,
                                           min_counts_stride, error);
}

#pragma endregion

#pragma region - Public Call Wrappers

SZ_DYNAMIC sz_status_t szs_fingerprints_sequence(         //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_t const *texts,                           //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_sequence_k, {.sequence = texts}, {.sequence = SZ_NULL}};
    return szs_fingerprints_run_((szs_fingerprints_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                 min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u32tape(          //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_u32tape_t const *texts,                   //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u32tape_k, {.u32tape = texts}, {.u32tape = SZ_NULL}};
    return szs_fingerprints_run_((szs_fingerprints_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                 min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_fingerprints_u64tape(          //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_u64tape_t const *texts,                   //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u64tape_k, {.u64tape = texts}, {.u64tape = SZ_NULL}};
    return szs_fingerprints_run_((szs_fingerprints_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                 min_hashes, min_hashes_stride, min_counts, min_counts_stride, error_message);
}

#pragma endregion

#ifdef __cplusplus
} // extern "C"
#endif
