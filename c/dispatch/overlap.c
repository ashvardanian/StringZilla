/**
 *  @file c/dispatch/overlap.c
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief The window-overlap engine's dispatch points, their lists and finder.
 */
#include <stringzilla/overlap.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_overlap_engine_init_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_engine_init_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_engine_init_neon,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) &
                        STRINGZILLA_PROFILE_CAPABILITIES) >>
                       48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_overlap_engine_init_cuda,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_overlap_engine_init_blackwell,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_overlap_engine_init_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_overlap_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_overlap_scores_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_scores_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_scores_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_scores_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_overlap_scores_neon,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) &
                        STRINGZILLA_PROFILE_CAPABILITIES) >>
                       48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_overlap_scores_cuda,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_overlap_scores_blackwell,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_overlap_scores_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_overlap_scores_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                   sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                   sz_size_t candidates_budget, sz_capability_t capabilities,
                                                   sz_allocator_t *allocator, sz_stream_t stream) {
    sz_kernel_overlap_engine_init_t const kernel = (sz_kernel_overlap_engine_init_t)sz_kernel_pick_(
        capabilities, sz_overlap_engine_init_capabilities());
    return kernel ? kernel(engine, queries, window_widths, window_widths_count, candidates_budget, allocator, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_overlap_engine_free(sz_overlap_engine_t *engine, sz_stream_t stream) {
    sz_overlap_engine_close_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                              sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                              sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_kernel_overlap_scores_t const kernel = (sz_kernel_overlap_scores_t)sz_kernel_pick_(
        engine->capability, sz_overlap_scores_capabilities());
    return kernel ? kernel(engine, candidates, radius, matches, matches_query_stride, matches_candidate_stride, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_overlap_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_overlap_engine_init_k: lists = sz_overlap_engine_init_capabilities(); break;
    case sz_kernel_overlap_scores_k: lists = sz_overlap_scores_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
