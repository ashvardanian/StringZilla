/**
 *  @file c/dispatch/subdocuments.c
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The path query engine's dispatch points, their lists and finder.
 */
#include "stringzilla/subdocuments.h"

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. Every list
 *  spans the same capabilities, so each verb has the kernel of the capability its init recorded. */

static sz_capability_kernels_t const *sz_subdocuments_engine_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_subdocuments_engine_init_metal,
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

static sz_capability_kernels_t const *sz_subdocuments_contains_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_subdocuments_contains_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_subdocuments_contains_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_subdocuments_contains_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_subdocuments_contains_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_subdocuments_contains_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_subdocuments_contains_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_subdocuments_contains_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_subdocuments_contains_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_subdocuments_contains_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_subdocuments_contains_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_subdocuments_contains_rocm,
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
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_subdocuments_counts_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_subdocuments_counts_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_subdocuments_counts_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_subdocuments_counts_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_subdocuments_counts_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_subdocuments_counts_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_subdocuments_counts_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_subdocuments_counts_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_subdocuments_counts_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_subdocuments_counts_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_subdocuments_counts_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_subdocuments_counts_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_subdocuments_counts_metal,
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

static sz_capability_kernels_t const *sz_subdocuments_find_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_subdocuments_find_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_subdocuments_find_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_subdocuments_find_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_subdocuments_find_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_subdocuments_find_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_subdocuments_find_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_subdocuments_find_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_subdocuments_find_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_subdocuments_find_powervsx,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_subdocuments_find_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_subdocuments_find_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_subdocuments_find_metal,
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

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *paths, sz_subdocuments_format_t format,
                                                        sz_subdocuments_options_t options, sz_size_t *malformed_index,
                                                        sz_capability_t capabilities, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    sz_kernel_subdocuments_engine_init_t const kernel = (sz_kernel_subdocuments_engine_init_t)sz_kernel_pick_(
        capabilities, sz_subdocuments_engine_init_capabilities());
    return kernel ? kernel(engine, paths, format, options, malformed_index, allocator, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_subdocuments_engine_free(sz_subdocuments_engine_t *engine, sz_stream_t stream) {
    sz_subdocuments_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains(sz_subdocuments_engine_t *engine,
                                                     sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                     sz_size_t *malformed_index, sz_stream_t stream) {
    sz_kernel_subdocuments_contains_t const kernel = (sz_kernel_subdocuments_contains_t)sz_kernel_pick_(
        engine->capability, sz_subdocuments_contains_capabilities());
    return kernel ? kernel(engine, documents, contained, malformed_index, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                   sz_size_t *counts, sz_size_t counts_stride,
                                                   sz_size_t *malformed_index, sz_stream_t stream) {
    sz_kernel_subdocuments_counts_t const kernel = (sz_kernel_subdocuments_counts_t)sz_kernel_pick_(
        engine->capability, sz_subdocuments_counts_capabilities());
    return kernel ? kernel(engine, documents, counts, counts_stride, malformed_index, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find(sz_subdocuments_engine_t *engine, sz_strings_view_t const *documents,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_size_t *malformed_index,
                                                 sz_stream_t stream) {
    sz_kernel_subdocuments_find_t const kernel = (sz_kernel_subdocuments_find_t)sz_kernel_pick_(
        engine->capability, sz_subdocuments_find_capabilities());
    return kernel ? kernel(engine, documents, matches, matches_capacity, matches_offsets, malformed_index, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_subdocuments_engine_init_k: lists = sz_subdocuments_engine_init_capabilities(); break;
    case sz_kernel_subdocuments_contains_k: lists = sz_subdocuments_contains_capabilities(); break;
    case sz_kernel_subdocuments_counts_k: lists = sz_subdocuments_counts_capabilities(); break;
    case sz_kernel_subdocuments_find_k: lists = sz_subdocuments_find_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
