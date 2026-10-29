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
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL
                                             : (sz_kernel_punned_t)&sz_levenshtein_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_engine_init_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_engine_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_levenshtein_engine_init_neon,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) *
                        !STRINGZILLA_MULTIVERSIONED) >>
                       48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_levenshtein_engine_init_cuda,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1)
            ? STRINGZILLA_NULL
            : (sz_kernel_punned_t)&sz_levenshtein_engine_init_blackwell,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_levenshtein_engine_init_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_levenshtein_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_levenshtein_distances_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_levenshtein_distances_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_distances_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_distances_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL
                                              : (sz_kernel_punned_t)&sz_levenshtein_distances_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_levenshtein_distances_neon,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) *
                        !STRINGZILLA_MULTIVERSIONED) >>
                       48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_levenshtein_distances_cuda,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_levenshtein_distances_blackwell,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_levenshtein_distances_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_levenshtein_distances_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine,
                                                       sz_strings_view_t const *queries, sz_levenshtein_symbol_t symbol,
                                                       sz_capability_t capabilities, sz_allocator_t *allocator,
                                                       sz_stream_t stream) {
    sz_kernel_levenshtein_engine_init_t const kernel = (sz_kernel_levenshtein_engine_init_t)sz_kernel_pick_(
        capabilities, sz_levenshtein_engine_init_capabilities());
    return kernel ? kernel(engine, queries, symbol, allocator, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine, sz_stream_t stream) {
    sz_levenshtein_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine,
                                                     sz_strings_view_t const *candidates,
                                                     sz_size_t const *symbol_counts, sz_size_t const *order,
                                                     sz_size_t bound, sz_size_t *distances, sz_size_t distances_stride,
                                                     sz_stream_t stream) {
    sz_kernel_levenshtein_distances_t const kernel = (sz_kernel_levenshtein_distances_t)sz_kernel_pick_(
        engine->capability, sz_levenshtein_distances_capabilities());
    return kernel ? kernel(engine, candidates, symbol_counts, order, bound, distances, distances_stride, stream)
                  : sz_missing_kernel_k;
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
