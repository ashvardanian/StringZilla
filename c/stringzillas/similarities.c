/**
 *  @file c/stringzillas/similarities.c
 *  @brief Public `szs_*` wrappers + dispatch updaters for the similarity engines.
 *  @author Ash Vardanian
 *
 *  Mirrors `c/stringzilla/find.c`: this translation unit fills its slice of `szs_dispatch_table` for
 *  the four pairwise-alignment operations (Levenshtein, UTF-8 Levenshtein, Needleman-Wunsch,
 *  Smith-Waterman) via the per-op `szs_dispatch_<op>_update_` updaters, and defines the `SZ_DYNAMIC`
 *  public engine lifecycle + per-input-format wrappers. Each updater sets the CPU function pointer by
 *  the same priority cascade as the C core (icelake > serial) and, under SZ_USE_CUDA, records the GPU
 *  tier so the public wrappers can route to the GPU entries. Each public wrapper builds the typed
 *  `szs_sequence_view_t` once and routes through the op's single `szs_<op>_run_` dispatch helper.
 */
#include "dispatch.h" // `szs_dispatch_table`, updaters, sets `SZ_DYNAMIC_DISPATCH 1`

#include "stringzillas/similarities/serial.h"  // `szs_*_serial` entries + shared helpers
#include "stringzillas/similarities/icelake.h" // `szs_*_icelake` entries (no-op when SZ_USE_ICELAKE=0)

#if SZ_USE_CUDA
#include "stringzillas/similarities/cuda_host.cuh" // `szs_*_cuda/_kepler/_hopper` Driver-API launchers
#endif

#include <string.h> // `memset`, `memcpy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Dispatch Updaters

SZ_DISPATCH_INTERNAL void szs_dispatch_levenshtein_update_(sz_capability_t caps) {
    szs_dispatch_table.levenshtein = &szs_levenshtein_serial;
#if SZ_USE_ICELAKE
    if ((caps & sz_cap_icelake_k) == sz_cap_icelake_k) szs_dispatch_table.levenshtein = &szs_levenshtein_icelake;
#endif
#if SZ_USE_CUDA
    // Record the GPU tier mask so the public wrappers can route default/gpu scopes to the GPU entries.
    szs_dispatch_table.gpu_tier = (sz_capability_t)(caps & sz_caps_cuda_k);
#endif
    sz_unused_(caps);
}

SZ_DISPATCH_INTERNAL void szs_dispatch_levenshtein_utf8_update_(sz_capability_t caps) {
    szs_dispatch_table.levenshtein_utf8 = &szs_levenshtein_utf8_serial;
#if SZ_USE_ICELAKE
    if ((caps & sz_cap_icelake_k) == sz_cap_icelake_k)
        szs_dispatch_table.levenshtein_utf8 = &szs_levenshtein_utf8_icelake;
#endif
    sz_unused_(caps);
}

SZ_DISPATCH_INTERNAL void szs_dispatch_needleman_wunsch_update_(sz_capability_t caps) {
    szs_dispatch_table.needleman_wunsch = &szs_needleman_wunsch_serial;
#if SZ_USE_ICELAKE
    if ((caps & sz_cap_icelake_k) == sz_cap_icelake_k)
        szs_dispatch_table.needleman_wunsch = &szs_needleman_wunsch_icelake;
#endif
    sz_unused_(caps);
}

SZ_DISPATCH_INTERNAL void szs_dispatch_smith_waterman_update_(sz_capability_t caps) {
    szs_dispatch_table.smith_waterman = &szs_smith_waterman_serial;
#if SZ_USE_ICELAKE
    if ((caps & sz_cap_icelake_k) == sz_cap_icelake_k) szs_dispatch_table.smith_waterman = &szs_smith_waterman_icelake;
#endif
    sz_unused_(caps);
}

#pragma endregion

#pragma region - GPU Entry Selection

#if SZ_USE_CUDA
/** @brief Pick the highest-tier similarity GPU entry for an op given the system capabilities. */
static szs_levenshtein_fn_t szs_gpu_levenshtein_entry_(sz_capability_t caps) {
    if ((caps & sz_cap_hopper_k) == sz_cap_hopper_k) return &szs_levenshtein_hopper;
    if ((caps & sz_cap_kepler_k) == sz_cap_kepler_k) return &szs_levenshtein_kepler;
    if ((caps & sz_cap_cuda_k) == sz_cap_cuda_k) return &szs_levenshtein_cuda;
    return SZ_NULL;
}
static szs_needleman_wunsch_fn_t szs_gpu_needleman_wunsch_entry_(sz_capability_t caps) {
    if ((caps & sz_cap_hopper_k) == sz_cap_hopper_k) return &szs_needleman_wunsch_hopper;
    if ((caps & sz_cap_cuda_k) == sz_cap_cuda_k) return &szs_needleman_wunsch_cuda;
    return SZ_NULL;
}
static szs_smith_waterman_fn_t szs_gpu_smith_waterman_entry_(sz_capability_t caps) {
    if ((caps & sz_cap_hopper_k) == sz_cap_hopper_k) return &szs_smith_waterman_hopper;
    if ((caps & sz_cap_cuda_k) == sz_cap_cuda_k) return &szs_smith_waterman_cuda;
    return SZ_NULL;
}
#endif

#pragma endregion

#pragma region - Engine Lifecycle

/** @brief Allocate & populate a Levenshtein engine POD (shared by the byte & UTF-8 handles). */
static sz_status_t szs_levenshtein_init_(sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open,
                                         sz_error_cost_t extend, sz_memory_allocator_t const *alloc,
                                         sz_capability_t capabilities, void **engine_out, char const **error_message) {
    if (!engine_out) {
        if (error_message) *error_message = "Engine handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }
    szs_levenshtein_engine_t *eng = (szs_levenshtein_engine_t *)malloc(sizeof(szs_levenshtein_engine_t));
    if (!eng) {
        if (error_message) *error_message = "Failed to allocate Levenshtein engine";
        return sz_bad_alloc_k;
    }
    memset(eng, 0, sizeof(*eng));
    eng->substitution.match = match;
    eng->substitution.mismatch = mismatch;
    eng->gaps.open = open;
    eng->gaps.extend = extend;
    eng->gap_mode = (open == extend) ? sz_gaps_linear_k : sz_gaps_affine_k;
    eng->capabilities = capabilities;
    if (alloc) eng->alloc = *alloc;
    *engine_out = (void *)eng;
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_init(                                             //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_t *engine, char const **error_message) {
    return szs_levenshtein_init_(match, mismatch, open, extend, alloc, capabilities, (void **)engine, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_init(                                        //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_utf8_t *engine, char const **error_message) {
    return szs_levenshtein_init_(match, mismatch, open, extend, alloc, capabilities, (void **)engine, error_message);
}

SZ_DYNAMIC void szs_levenshtein_distances_free(szs_levenshtein_distances_t engine) {
    if (engine) free(engine);
}
SZ_DYNAMIC void szs_levenshtein_distances_utf8_free(szs_levenshtein_distances_utf8_t engine) {
    if (engine) free(engine);
}

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_init(                               //
    sz_substitution_costs_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                  //
    szs_needleman_wunsch_scores_t *engine, char const **error_message) {
    if (!engine || !subs) {
        if (error_message) *error_message = "Engine handle and substitution costs must not be NULL";
        return sz_unexpected_dimensions_k;
    }
    szs_needleman_wunsch_engine_t *eng = (szs_needleman_wunsch_engine_t *)malloc(sizeof(szs_needleman_wunsch_engine_t));
    if (!eng) {
        if (error_message) *error_message = "Failed to allocate Needleman-Wunsch engine";
        return sz_bad_alloc_k;
    }
    memset(eng, 0, sizeof(*eng));
    eng->substitution = *subs;
    eng->gaps.open = open;
    eng->gaps.extend = extend;
    eng->gap_mode = (open == extend) ? sz_gaps_linear_k : sz_gaps_affine_k;
    eng->capabilities = capabilities;
    if (alloc) eng->alloc = *alloc;
    *engine = (szs_needleman_wunsch_scores_t)eng;
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_init(                                 //
    sz_substitution_costs_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                  //
    szs_smith_waterman_scores_t *engine, char const **error_message) {
    if (!engine || !subs) {
        if (error_message) *error_message = "Engine handle and substitution costs must not be NULL";
        return sz_unexpected_dimensions_k;
    }
    szs_smith_waterman_engine_t *eng = (szs_smith_waterman_engine_t *)malloc(sizeof(szs_smith_waterman_engine_t));
    if (!eng) {
        if (error_message) *error_message = "Failed to allocate Smith-Waterman engine";
        return sz_bad_alloc_k;
    }
    memset(eng, 0, sizeof(*eng));
    eng->substitution = *subs;
    eng->gaps.open = open;
    eng->gaps.extend = extend;
    eng->gap_mode = (open == extend) ? sz_gaps_linear_k : sz_gaps_affine_k;
    eng->capabilities = capabilities;
    if (alloc) eng->alloc = *alloc;
    *engine = (szs_smith_waterman_scores_t)eng;
    return sz_success_k;
}

SZ_DYNAMIC void szs_needleman_wunsch_scores_free(szs_needleman_wunsch_scores_t engine) {
    if (engine) free(engine);
}
SZ_DYNAMIC void szs_smith_waterman_scores_free(szs_smith_waterman_scores_t engine) {
    if (engine) free(engine);
}

#pragma endregion

#pragma region - Per-Op Dispatch Helpers

/** @brief True if a device scope wants GPU execution (explicit gpu, or default with a usable GPU entry). */
static int szs_scope_wants_gpu_(szs_device_scope_impl_t *scope) {
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

SZ_INTERNAL sz_status_t szs_levenshtein_run_(szs_levenshtein_engine_t const *engine, szs_device_scope_impl_t *device,
                                             szs_sequence_view_t inputs, sz_size_t *results, sz_size_t stride,
                                             char const **error) {
#if SZ_USE_CUDA
    if (szs_scope_wants_gpu_(device)) {
        szs_levenshtein_fn_t gpu = szs_gpu_levenshtein_entry_(szs_dispatch_table.gpu_tier);
        if (gpu) return gpu(engine, device, inputs, results, stride, error);
    }
#endif
    return szs_dispatch_table.levenshtein(engine, device, inputs, results, stride, error);
}

SZ_INTERNAL sz_status_t szs_levenshtein_utf8_run_(szs_levenshtein_engine_t const *engine,
                                                  szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                  sz_size_t *results, sz_size_t stride, char const **error) {
    // UTF-8 Levenshtein has no GPU backend: always the CPU table entry.
    return szs_dispatch_table.levenshtein_utf8(engine, device, inputs, results, stride, error);
}

SZ_INTERNAL sz_status_t szs_needleman_wunsch_run_(szs_needleman_wunsch_engine_t const *engine,
                                                  szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                  sz_ssize_t *results, sz_size_t stride, char const **error) {
#if SZ_USE_CUDA
    if (szs_scope_wants_gpu_(device)) {
        szs_needleman_wunsch_fn_t gpu = szs_gpu_needleman_wunsch_entry_(szs_dispatch_table.gpu_tier);
        if (gpu) return gpu(engine, device, inputs, results, stride, error);
    }
#endif
    return szs_dispatch_table.needleman_wunsch(engine, device, inputs, results, stride, error);
}

SZ_INTERNAL sz_status_t szs_smith_waterman_run_(szs_smith_waterman_engine_t const *engine,
                                                szs_device_scope_impl_t *device, szs_sequence_view_t inputs,
                                                sz_ssize_t *results, sz_size_t stride, char const **error) {
#if SZ_USE_CUDA
    if (szs_scope_wants_gpu_(device)) {
        szs_smith_waterman_fn_t gpu = szs_gpu_smith_waterman_entry_(szs_dispatch_table.gpu_tier);
        if (gpu) return gpu(engine, device, inputs, results, stride, error);
    }
#endif
    return szs_dispatch_table.smith_waterman(engine, device, inputs, results, stride, error);
}

#pragma endregion

#pragma region - Public Call Wrappers

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_sequence(         //
    szs_levenshtein_distances_t engine, szs_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_sequence_k, {.sequence = a}, {.sequence = b}};
    return szs_levenshtein_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u32tape(           //
    szs_levenshtein_distances_t engine, szs_device_scope_t device,  //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b, //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u32tape_k, {.u32tape = a}, {.u32tape = b}};
    return szs_levenshtein_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_levenshtein_distances_u64tape(           //
    szs_levenshtein_distances_t engine, szs_device_scope_t device,  //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b, //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u64tape_k, {.u64tape = a}, {.u64tape = b}};
    return szs_levenshtein_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device, inputs,
                                results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_sequence(         //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                     //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_sequence_k, {.sequence = a}, {.sequence = b}};
    return szs_levenshtein_utf8_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u32tape(          //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,     //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u32tape_k, {.u32tape = a}, {.u32tape = b}};
    return szs_levenshtein_utf8_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_levenshtein_distances_utf8_u64tape(          //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,     //
    sz_size_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u64tape_k, {.u64tape = a}, {.u64tape = b}};
    return szs_levenshtein_utf8_run_((szs_levenshtein_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_sequence(         //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                  //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_sequence_k, {.sequence = a}, {.sequence = b}};
    return szs_needleman_wunsch_run_((szs_needleman_wunsch_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_u32tape(          //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device, //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b,  //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u32tape_k, {.u32tape = a}, {.u32tape = b}};
    return szs_needleman_wunsch_run_((szs_needleman_wunsch_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_needleman_wunsch_scores_u64tape(          //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device, //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b,  //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u64tape_k, {.u64tape = a}, {.u64tape = b}};
    return szs_needleman_wunsch_run_((szs_needleman_wunsch_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                     inputs, results, results_stride, error_message);
}

SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_sequence(         //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_sequence_k, {.sequence = a}, {.sequence = b}};
    return szs_smith_waterman_run_((szs_smith_waterman_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                   inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u32tape(           //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device,  //
    sz_sequence_u32tape_t const *a, sz_sequence_u32tape_t const *b, //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u32tape_k, {.u32tape = a}, {.u32tape = b}};
    return szs_smith_waterman_run_((szs_smith_waterman_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                   inputs, results, results_stride, error_message);
}
SZ_DYNAMIC sz_status_t szs_smith_waterman_scores_u64tape(           //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device,  //
    sz_sequence_u64tape_t const *a, sz_sequence_u64tape_t const *b, //
    sz_ssize_t *results, sz_size_t results_stride, char const **error_message) {
    szs_sequence_view_t inputs = {szs_inputs_u64tape_k, {.u64tape = a}, {.u64tape = b}};
    return szs_smith_waterman_run_((szs_smith_waterman_engine_t const *)engine, (szs_device_scope_impl_t *)device,
                                   inputs, results, results_stride, error_message);
}

#pragma endregion

#ifdef __cplusplus
} // extern "C"
#endif
