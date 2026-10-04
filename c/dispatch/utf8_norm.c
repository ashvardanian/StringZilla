/**
 *  @file c/dispatch/utf8_norm.c
 *  @author Ash Vardanian
 *  @date June 14, 2026
 *  @brief The normalization capability lists, dispatch points and finder.
 *
 *  The normalizer and the violation finder share one streaming engine, parameterized by a
 *  force-inlined scan primitive; each SIMD backend overrides only that scan.
 */
#include <stringzilla/utf8_norm.h> // `sz_utf8_norm_*`, `sz_utf8_find_denormalized_*`

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_norm_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_norm_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_norm_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_utf8_norm_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_norm_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_norm_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_utf8_norm_sve,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_norm_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_norm_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_norm_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_utf8_norm_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_norm_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_norm_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_utf8_norm_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_utf8_norm_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_utf8_norm_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_sve_k * STRINGZILLA_TARGET_SVE |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_utf8_find_denormalized_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_sve,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_find_denormalized_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_sve_k * STRINGZILLA_TARGET_SVE |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_utf8_norm_best(                        //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, //
    sz_ptr_t target, sz_size_t *target_length,                        //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_norm_t const kernel = (sz_kernel_utf8_norm_t)sz_kernel_pick_(capabilities,
                                                                                sz_utf8_norm_capabilities());
    return kernel ? kernel(source, source_length, form, target, target_length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_denormalized_best(                             //
    sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form, sz_cptr_t *match, //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_find_denormalized_t const kernel = (sz_kernel_utf8_find_denormalized_t)sz_kernel_pick_(
        capabilities, sz_utf8_find_denormalized_capabilities());
    return kernel ? kernel(source, source_length, form, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_norm_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_norm_k: lists = sz_utf8_norm_capabilities(); break;
    case sz_kernel_utf8_find_denormalized_k: lists = sz_utf8_find_denormalized_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
