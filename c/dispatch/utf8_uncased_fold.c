/**
 *  @file c/dispatch/utf8_uncased_fold.c
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief The UTF-8 case folding capability list, dispatch point and finder.
 *
 *  Split from the uncased find and order unit, <tt>utf8_uncased.c</tt>, so the cheap folding kernels
 *  compile as their own translation unit, in parallel with the heavy AVX-512 find.
 */
#include <stringzilla/utf8_uncased_fold.h> // `sz_utf8_uncased_fold_*`

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_uncased_fold_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        above_k & ~(sz_cap_sve2_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_utf8_uncased_fold_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_uncased_fold_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_utf8_uncased_fold_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_utf8_uncased_fold_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) & STRINGZILLA_PROFILE_CAPABILITIES) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_utf8_uncased_fold_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_best( //
    sz_cptr_t source, sz_size_t source_length,         //
    sz_ptr_t target, sz_size_t *target_length,         //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_uncased_fold_t const kernel = (sz_kernel_utf8_uncased_fold_t)sz_kernel_pick_(
        capabilities, sz_utf8_uncased_fold_capabilities());
    return kernel ? kernel(source, source_length, target, target_length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                             sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_uncased_fold_k: lists = sz_utf8_uncased_fold_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
