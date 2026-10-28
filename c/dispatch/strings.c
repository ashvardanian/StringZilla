/**
 *  @file c/dispatch/strings.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Collection dispatch points, @c sz_strings_to_tape_best, @c sz_strings_to_rows_best,
 *      @c sz_strings_escape_best, @c sz_strings_unescape_best, @c sz_strings_lookup_best,
 *      @c sz_strings_normalize_best, @c sz_strings_split_best, @c sz_strings_split_lines_best and
 *      @c sz_strings_join_best, and their finder, then the allocators each capability group answers
 *      once, like @c sz_allocator_init_unified_best.
 */
#include "stringzilla/strings.h"

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_strings_to_tape_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_to_tape_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_to_tape_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_tape_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_to_tape_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_to_tape_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_to_tape_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_to_rows_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_to_rows_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_to_rows_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_to_rows_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_to_rows_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_to_rows_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_to_rows_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_escape_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_escape_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_escape_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_escape_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_escape_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_escape_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_escape_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_unescape_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_unescape_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_unescape_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_unescape_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_unescape_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_unescape_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_unescape_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_lookup_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_lookup_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_lookup_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_lookup_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_lookup_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_lookup_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_lookup_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_normalize_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_normalize_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL
                                                  : (sz_kernel_punned_t)&sz_strings_normalize_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_normalize_powervsx,
#endif
    };
    enum { cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_normalize_cuda,
#endif
    };
    enum { rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_strings_normalize_rocm,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_strings_normalize_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_split_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
                   sz_cap_neon_k * STRINGZILLA_TARGET_NEON) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_serial,
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_neon,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_split_lines_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
                   sz_cap_neon_k * STRINGZILLA_TARGET_NEON) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_lines_serial,
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_lines_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_split_lines_neon,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_strings_join_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
                   sz_cap_neon_k * STRINGZILLA_TARGET_NEON) *
                  !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_join_serial,
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_join_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_strings_join_neon,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_strings_to_tape_t const kernel = (sz_kernel_strings_to_tape_t)sz_kernel_pick_(
        capabilities, sz_strings_to_tape_capabilities());
    return kernel ? kernel(strings, order, count, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_strings_to_rows_t const kernel = (sz_kernel_strings_to_rows_t)sz_kernel_pick_(
        capabilities, sz_strings_to_rows_capabilities());
    return kernel ? kernel(strings, order, count, rows, longest, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_escape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_capability_t capabilities,
                                                   sz_stream_t stream) {
    sz_kernel_strings_escape_t const kernel = (sz_kernel_strings_escape_t)sz_kernel_pick_(
        capabilities, sz_strings_escape_capabilities());
    return kernel ? kernel(strings, order, count, kind, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_unescape_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_strings_unescape_t const kernel = (sz_kernel_strings_unescape_t)sz_kernel_pick_(
        capabilities, sz_strings_unescape_capabilities());
    return kernel ? kernel(strings, order, count, kind, allocator, tape, malformed_index, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_lookup_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_strings_lookup_t const kernel = (sz_kernel_strings_lookup_t)sz_kernel_pick_(
        capabilities, sz_strings_lookup_capabilities());
    return kernel ? kernel(strings, order, count, lut, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_normalize_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_capability_t capabilities,
                                                      sz_stream_t stream) {
    sz_kernel_strings_normalize_t const kernel = (sz_kernel_strings_normalize_t)sz_kernel_pick_(
        capabilities, sz_strings_normalize_capabilities());
    return kernel ? kernel(strings, order, count, form, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_best(sz_cptr_t text, sz_size_t length, sz_byteset_t const *delimiters,
                                                  sz_split_kind_t kind, sz_allocator_t *allocator,
                                                  sz_strings_span_t *tape, sz_capability_t capabilities,
                                                  sz_stream_t stream) {
    sz_kernel_strings_split_t const kernel = (sz_kernel_strings_split_t)sz_kernel_pick_(
        capabilities, sz_strings_split_capabilities());
    return kernel ? kernel(text, length, delimiters, kind, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_split_lines_best(sz_cptr_t text, sz_size_t length, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_capability_t capabilities,
                                                        sz_stream_t stream) {
    sz_kernel_strings_split_lines_t const kernel = (sz_kernel_strings_split_lines_t)sz_kernel_pick_(
        capabilities, sz_strings_split_lines_capabilities());
    return kernel ? kernel(text, length, allocator, tape, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_join_best(sz_strings_view_t const *columns, sz_size_t columns_count,
                                                 sz_size_t rows, sz_string_view_t const *frames,
                                                 sz_string_view_t between, sz_ptr_t output, sz_size_t *length,
                                                 sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_strings_join_t const kernel = (sz_kernel_strings_join_t)sz_kernel_pick_(capabilities,
                                                                                      sz_strings_join_capabilities());
    return kernel ? kernel(columns, columns_count, rows, frames, between, output, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_strings_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_strings_to_tape_k: lists = sz_strings_to_tape_capabilities(); break;
    case sz_kernel_strings_to_rows_k: lists = sz_strings_to_rows_capabilities(); break;
    case sz_kernel_strings_escape_k: lists = sz_strings_escape_capabilities(); break;
    case sz_kernel_strings_unescape_k: lists = sz_strings_unescape_capabilities(); break;
    case sz_kernel_strings_lookup_k: lists = sz_strings_lookup_capabilities(); break;
    case sz_kernel_strings_normalize_k: lists = sz_strings_normalize_capabilities(); break;
    case sz_kernel_strings_split_k: lists = sz_strings_split_capabilities(); break;
    case sz_kernel_strings_split_lines_k: lists = sz_strings_split_lines_capabilities(); break;
    case sz_kernel_strings_join_k: lists = sz_strings_join_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_allocator_init_unified_serial(allocator);
#if STRINGZILLA_ARCH_CUDA_
    case sz_capability_group_cuda_k: return sz_allocator_init_unified_cuda(allocator);
#endif
#if STRINGZILLA_ARCH_ROCM_
    case sz_capability_group_rocm_k: return sz_allocator_init_unified_rocm(allocator);
#endif
#if STRINGZILLA_ARCH_METAL_
    case sz_capability_group_metal_k: return sz_allocator_init_unified_metal(allocator);
#endif
    default: return sz_missing_gpu_k;
    }
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_best(sz_allocator_t *allocator, sz_capability_t capabilities) {
    sz_unused_(allocator);
    switch (sz_capability_group_of_(capabilities)) {
    case sz_capability_group_cpu_k: return sz_missing_kernel_k;
#if STRINGZILLA_ARCH_CUDA_
    case sz_capability_group_cuda_k: return sz_allocator_init_device_cuda(allocator);
#endif
#if STRINGZILLA_ARCH_ROCM_
    case sz_capability_group_rocm_k: return sz_allocator_init_device_rocm(allocator);
#endif
#if STRINGZILLA_ARCH_METAL_
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
#if STRINGZILLA_ARCH_CUDA_
    case sz_capability_group_cuda_k: return sz_allocator_init_pinned_cuda(allocator);
#endif
#if STRINGZILLA_ARCH_ROCM_
    case sz_capability_group_rocm_k: return sz_allocator_init_pinned_rocm(allocator);
#endif
    default: return sz_missing_gpu_k;
    }
}
