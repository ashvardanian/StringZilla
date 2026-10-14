/**
 *  @file c/dispatch/regex.c
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The regular expression engine's dispatch points, their lists and finder.
 */
#include "stringzilla/regex.h"

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. Every list
 *  spans the same capabilities, so each verb has the kernel of the capability its init recorded. */

static sz_capability_kernels_t const *sz_regex_engine_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_regex_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_regex_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_regex_engine_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_regex_engine_init_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_regex_engine_init_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_regex_engine_init_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_regex_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_regex_counts_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_regex_counts_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_regex_counts_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_regex_counts_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_regex_counts_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_regex_counts_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_regex_counts_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_regex_counts_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_regex_find_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_regex_find_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_regex_find_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_regex_find_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_regex_find_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_regex_find_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_regex_find_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_regex_find_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_regex_replace_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_regex_replace_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_regex_replace_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_regex_replace_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_regex_replace_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_regex_replace_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_regex_replace_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_regex_replace_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_regex_extract_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_regex_extract_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_regex_extract_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_regex_extract_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_regex_extract_neon,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_regex_extract_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_regex_extract_rocm,
#endif
    };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        (sz_kernel_punned_t)&sz_regex_extract_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_regex_engine_init(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                 sz_case_sensitivity_t case_sensitivity, sz_size_t *malformed_index,
                                                 sz_capability_t capabilities, sz_allocator_t *allocator,
                                                 sz_stream_t stream) {
    sz_kernel_regex_engine_init_t const kernel = (sz_kernel_regex_engine_init_t)sz_kernel_pick_(
        capabilities, sz_regex_engine_init_capabilities());
    return kernel ? kernel(engine, patterns, case_sensitivity, malformed_index, allocator, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_regex_engine_free(sz_regex_engine_t *engine, sz_stream_t stream) {
    sz_regex_engine_free_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_regex_counts(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                            sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_kernel_regex_counts_t const kernel = (sz_kernel_regex_counts_t)sz_kernel_pick_(engine->capability,
                                                                                      sz_regex_counts_capabilities());
    return kernel ? kernel(engine, haystacks, counts, counts_stride, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_regex_find(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                          sz_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                          sz_stream_t stream) {
    sz_kernel_regex_find_t const kernel = (sz_kernel_regex_find_t)sz_kernel_pick_(engine->capability,
                                                                                  sz_regex_find_capabilities());
    return kernel ? kernel(engine, haystacks, matches, matches_capacity, matches_offsets, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_regex_replace(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_strings_view_t const *replacements, sz_ptr_t target,
                                             sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    sz_kernel_regex_replace_t const kernel = (sz_kernel_regex_replace_t)sz_kernel_pick_(
        engine->capability, sz_regex_replace_capabilities());
    return kernel ? kernel(engine, haystacks, replacements, target, target_capacity, offsets, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_regex_extract(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                             sz_size_t pattern_index, sz_match_t *groups, sz_size_t groups_capacity,
                                             sz_size_t *groups_offsets, sz_stream_t stream) {
    sz_kernel_regex_extract_t const kernel = (sz_kernel_regex_extract_t)sz_kernel_pick_(
        engine->capability, sz_regex_extract_capabilities());
    return kernel ? kernel(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_regex_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                 sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_regex_engine_init_k: lists = sz_regex_engine_init_capabilities(); break;
    case sz_kernel_regex_counts_k: lists = sz_regex_counts_capabilities(); break;
    case sz_kernel_regex_find_k: lists = sz_regex_find_capabilities(); break;
    case sz_kernel_regex_replace_k: lists = sz_regex_replace_capabilities(); break;
    case sz_kernel_regex_extract_k: lists = sz_regex_extract_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
