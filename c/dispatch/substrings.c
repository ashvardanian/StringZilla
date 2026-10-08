/**
 *  @file c/dispatch/substrings.c
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief The multi-pattern search engine's dispatch points, their lists and finder.
 */
#include <stringzilla/substrings.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. Every list
 *  spans the same capabilities, so each verb has the kernel of the capability its init recorded. */

static sz_capability_kernels_t const *sz_substrings_engine_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_substrings_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_substrings_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_substrings_engine_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_substrings_engine_init_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_substrings_engine_init_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_substrings_engine_init_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_substrings_engine_init_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_substrings_engine_init_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_substrings_engine_init_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_substrings_engine_init_cuda,
#endif
#if STRINGZILLA_TARGET_HOPPER
        (sz_kernel_punned_t)&sz_substrings_engine_init_hopper,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_substrings_engine_init_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_substrings_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_substrings_counts_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_substrings_counts_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_substrings_counts_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_substrings_counts_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_substrings_counts_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_substrings_counts_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_substrings_counts_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_substrings_counts_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_substrings_counts_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_substrings_counts_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_substrings_counts_cuda,
#endif
#if STRINGZILLA_TARGET_HOPPER
        (sz_kernel_punned_t)&sz_substrings_counts_hopper,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_substrings_counts_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_substrings_counts_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_substrings_find_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_substrings_find_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_substrings_find_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_substrings_find_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_substrings_find_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_substrings_find_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_substrings_find_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_substrings_find_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_substrings_find_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_substrings_find_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_substrings_find_cuda,
#endif
#if STRINGZILLA_TARGET_HOPPER
        (sz_kernel_punned_t)&sz_substrings_find_hopper,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_substrings_find_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_substrings_find_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_substrings_replace_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_substrings_replace_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_substrings_replace_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_substrings_replace_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_substrings_replace_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_substrings_replace_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_substrings_replace_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_substrings_replace_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_substrings_replace_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_substrings_replace_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_substrings_replace_cuda,
#endif
#if STRINGZILLA_TARGET_HOPPER
        (sz_kernel_punned_t)&sz_substrings_replace_hopper,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_substrings_replace_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_substrings_replace_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_substrings_bm25_scores_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_cuda,
#endif
#if STRINGZILLA_TARGET_HOPPER
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_hopper,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_substrings_bm25_scores_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_substrings_engine_init(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                      sz_substrings_case_sensitivity_t case_sensitivity,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_size_t hot_states, sz_size_t matches_budget,
                                                      sz_size_t haystacks_budget, sz_capability_t capabilities,
                                                      sz_allocator_t *allocator, sz_stream_t stream) {
    sz_kernel_substrings_engine_init_t const kernel = (sz_kernel_substrings_engine_init_t)sz_kernel_pick_(
        capabilities, sz_substrings_engine_init_capabilities());
    if (!kernel) return sz_missing_kernel_k;
    sz_status_t const status = kernel(engine, needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                      haystacks_budget, allocator, stream);
    if (status != sz_success_k) return status;
    // Init kernels leave the serial copy, which a device mask, finding no copy kernel, never calls.
    sz_kernel_punned_t copy;
    sz_capability_t copy_capability;
    if (sz_find_kernel_punned(sz_kernel_copy_k, capabilities, &copy, &copy_capability) == sz_success_k)
        engine->copy = (sz_kernel_copy_t)copy;
    return sz_success_k;
}

STRINGZILLA_API void sz_substrings_engine_free(sz_substrings_engine_t *engine, sz_stream_t stream) {
    sz_substrings_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_kernel_substrings_counts_t const kernel = (sz_kernel_substrings_counts_t)sz_kernel_pick_(
        engine->capability, sz_substrings_counts_capabilities());
    return kernel ? kernel(engine, haystacks, counts, counts_stride, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                               sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                               sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_kernel_substrings_find_t const kernel = (sz_kernel_substrings_find_t)sz_kernel_pick_(
        engine->capability, sz_substrings_find_capabilities());
    return kernel ? kernel(engine, haystacks, matches, matches_capacity, matches_offsets, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                  sz_sequence_t const *replacements, sz_ptr_t target,
                                                  sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    sz_kernel_substrings_replace_t const kernel = (sz_kernel_substrings_replace_t)sz_kernel_pick_(
        engine->capability, sz_substrings_replace_capabilities());
    return kernel ? kernel(engine, haystacks, replacements, target, target_capacity, offsets, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_f32_t const *document_lengths,
                                                      sz_substrings_bm25_t const *parameters,
                                                      sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                      sz_size_t scores_stride, sz_stream_t stream) {
    sz_kernel_substrings_bm25_scores_t const kernel = (sz_kernel_substrings_bm25_scores_t)sz_kernel_pick_(
        engine->capability, sz_substrings_bm25_scores_capabilities());
    return kernel
               ? kernel(engine, haystacks, document_lengths, parameters, needle_weights, scores, scores_stride, stream)
               : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_substrings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_substrings_engine_init_k: lists = sz_substrings_engine_init_capabilities(); break;
    case sz_kernel_substrings_counts_k: lists = sz_substrings_counts_capabilities(); break;
    case sz_kernel_substrings_find_k: lists = sz_substrings_find_capabilities(); break;
    case sz_kernel_substrings_replace_k: lists = sz_substrings_replace_capabilities(); break;
    case sz_kernel_substrings_bm25_scores_k: lists = sz_substrings_bm25_scores_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
