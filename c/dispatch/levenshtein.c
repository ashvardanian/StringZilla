/**
 *  @file c/dispatch/levenshtein.c
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief The Levenshtein engine's dispatch points, their
 *      lists and finder.
 */
#include <stringzilla/levenshtein.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_levenshtein_engine_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_levenshtein_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_levenshtein_distances_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_levenshtein_distances_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_levenshtein_distances_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_levenshtein_distances_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_levenshtein_distances_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_levenshtein_distances_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_levenshtein_distances_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_levenshtein_distances_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_levenshtein_distances_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                       sz_levenshtein_symbol_t symbol, sz_capability_t capabilities,
                                                       sz_allocator_t *allocator, sz_stream_t stream) {
    sz_kernel_levenshtein_engine_init_t const kernel = (sz_kernel_levenshtein_engine_init_t)sz_kernel_pick_(
        capabilities, sz_levenshtein_engine_init_capabilities());
    return kernel ? kernel(engine, queries, symbol, allocator, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine, sz_stream_t stream) {
    sz_levenshtein_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                                     sz_size_t *distances, sz_size_t distances_stride,
                                                     sz_stream_t stream) {
    sz_kernel_levenshtein_distances_t const kernel = (sz_kernel_levenshtein_distances_t)sz_kernel_pick_(
        engine->capability, sz_levenshtein_distances_capabilities());
    return kernel ? kernel(engine, candidates, distances, distances_stride, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_levenshtein_engine_init_k: lists = sz_levenshtein_engine_init_capabilities(); break;
    case sz_kernel_levenshtein_distances_k: lists = sz_levenshtein_distances_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
