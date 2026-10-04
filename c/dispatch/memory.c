/**
 *  @file c/dispatch/memory.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Raw memory dispatch points, @c sz_copy_best, @c sz_move_best, @c sz_fill_best, and
 *      @c sz_lookup_best, and their finder, then the unified memory ones each capability group
 *      answers once, @c sz_allocator_init_unified_best and @c sz_sequence_realloc_best.
 */
#include <stringzilla/memory.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_copy_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_copy_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_copy_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_copy_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_copy_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_copy_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_copy_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_copy_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_copy_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_copy_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_copy_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_move_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_move_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_move_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_move_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_move_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_move_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_move_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_move_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_move_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_move_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_move_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_fill_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_fill_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_fill_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_fill_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_fill_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_fill_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_fill_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_fill_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_fill_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_fill_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_fill_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_lookup_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_lookup_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_lookup_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_lookup_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_lookup_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_lookup_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_lookup_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_lookup_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_lookup_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_lookup_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_lookup_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_copy_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_copy_t const kernel = (sz_kernel_copy_t)sz_kernel_pick_(capabilities, sz_copy_capabilities());
    return kernel ? kernel(target, source, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_move_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                         sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_move_t const kernel = (sz_kernel_move_t)sz_kernel_pick_(capabilities, sz_move_capabilities());
    return kernel ? kernel(target, source, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_fill_best(sz_ptr_t target, sz_size_t length, sz_u8_t value, sz_capability_t capabilities,
                                         sz_stream_t stream) {
    sz_kernel_fill_t const kernel = (sz_kernel_fill_t)sz_kernel_pick_(capabilities, sz_fill_capabilities());
    return kernel ? kernel(target, length, value, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_lookup_best(sz_ptr_t target, sz_cptr_t source, sz_size_t length,
                                           char const lut[sz_at_least_(256)], sz_capability_t capabilities,
                                           sz_stream_t stream) {
    sz_kernel_lookup_t const kernel = (sz_kernel_lookup_t)sz_kernel_pick_(capabilities, sz_lookup_capabilities());
    return kernel ? kernel(target, source, length, lut, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_memory_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_copy_k: lists = sz_copy_capabilities(); break;
    case sz_kernel_move_k: lists = sz_move_capabilities(); break;
    case sz_kernel_fill_k: lists = sz_fill_capabilities(); break;
    case sz_kernel_lookup_k: lists = sz_lookup_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_allocator_init_unified_serial(allocator);
#if STRINGZILLA_TARGET_CUDA
    case sz_capability_group_cuda_k: return sz_allocator_init_unified_cuda(allocator);
#endif
#if STRINGZILLA_TARGET_ROCM
    case sz_capability_group_rocm_k: return sz_allocator_init_unified_rocm(allocator);
#endif
#if STRINGZILLA_TARGET_METAL
    case sz_capability_group_metal_k: return sz_allocator_init_unified_metal(allocator);
#endif
    default: return sz_missing_gpu_k;
    }
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator);
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_missing_kernel_k;
#if STRINGZILLA_TARGET_CUDA
    case sz_capability_group_cuda_k: return sz_allocator_init_device_cuda(allocator);
#endif
#if STRINGZILLA_TARGET_ROCM
    case sz_capability_group_rocm_k: return sz_allocator_init_device_rocm(allocator);
#endif
#if STRINGZILLA_TARGET_METAL
    case sz_capability_group_metal_k: return sz_allocator_init_unified_metal(allocator);
#endif
    default: return sz_missing_gpu_k;
    }
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator);
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k:
    case sz_capability_group_metal_k: return sz_missing_kernel_k;
#if STRINGZILLA_TARGET_CUDA
    case sz_capability_group_cuda_k: return sz_allocator_init_pinned_cuda(allocator);
#endif
#if STRINGZILLA_TARGET_ROCM
    case sz_capability_group_rocm_k: return sz_allocator_init_pinned_rocm(allocator);
#endif
    default: return sz_missing_gpu_k;
    }
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_best(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_capability_t capabilities, sz_stream_t stream) {
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k:
        return sz_sequence_realloc_serial(target, source, allocator, allocated_bytes, stream);
#if STRINGZILLA_TARGET_CUDA
    case sz_capability_group_cuda_k:
        return sz_sequence_realloc_cuda(target, source, allocator, allocated_bytes, stream);
#endif
#if STRINGZILLA_TARGET_ROCM
    case sz_capability_group_rocm_k:
        return sz_sequence_realloc_rocm(target, source, allocator, allocated_bytes, stream);
#endif
#if STRINGZILLA_TARGET_METAL
    case sz_capability_group_metal_k:
        return sz_sequence_realloc_metal(target, source, allocator, allocated_bytes, stream);
#endif
    default: return sz_missing_gpu_k;
    }
}
