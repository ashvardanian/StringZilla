/**
 *  @file src/dispatch/ngrams.c
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief The ngrams engine's dispatch points, their lists and finder.
 */
#include "stringzilla/ngrams.h"
#include "stringzilla/ngrams/serial.h" // `sz_ngrams_engine_close_`

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_ngrams_engine_init_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
        !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_engine_init_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_engine_init_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_engine_init_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_engine_init_neon,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_engine_init_powervsx,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
                         sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_engine_init_cuda,
#endif
#if STRINGZILLA_TARGET_AMPERE
        cuda_above_k & ~((sz_cap_ampere_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_engine_init_ampere,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_ngrams_engine_init_blackwell,
#endif
    };
    enum {
        rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
                         sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_engine_init_rocm,
#endif
#if STRINGZILLA_TARGET_CDNA3
        rocm_above_k & ~((sz_cap_cdna3_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_engine_init_cdna3,
#endif
#if STRINGZILLA_TARGET_CDNA4
        rocm_above_k & ~((sz_cap_cdna4_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_engine_init_cdna4,
#endif
#if STRINGZILLA_TARGET_CDNA5
        rocm_above_k & ~((sz_cap_cdna5_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_engine_init_cdna5,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_engine_init_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
             sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL,
         cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
             sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5,
         rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_ngrams_overlaps_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
        !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_overlaps_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_overlaps_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_overlaps_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_overlaps_neon,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_overlaps_powervsx,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
                         sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_overlaps_cuda,
#endif
#if STRINGZILLA_TARGET_AMPERE
        cuda_above_k & ~((sz_cap_ampere_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_overlaps_ampere,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_ngrams_overlaps_blackwell,
#endif
    };
    enum {
        rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
                         sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_overlaps_rocm,
#endif
#if STRINGZILLA_TARGET_CDNA3
        rocm_above_k & ~((sz_cap_cdna3_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_overlaps_cdna3,
#endif
#if STRINGZILLA_TARGET_CDNA4
        rocm_above_k & ~((sz_cap_cdna4_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_overlaps_cdna4,
#endif
#if STRINGZILLA_TARGET_CDNA5
        rocm_above_k & ~((sz_cap_cdna5_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_overlaps_cdna5,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_overlaps_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
             sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL,
         cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
             sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5,
         rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_ngrams_sketches_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) *
        !STRINGZILLA_MULTIVERSIONED
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_sketches_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_sketches_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_sketches_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_sketches_neon,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_ngrams_sketches_powervsx,
#endif
    };
    enum {
        cuda_above_k = ((sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
                         sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        cuda_above_k & ~((sz_cap_cuda_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_sketches_cuda,
#endif
#if STRINGZILLA_TARGET_AMPERE
        cuda_above_k & ~((sz_cap_ampere_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_sketches_ampere,
#endif
#if STRINGZILLA_TARGET_BLACKWELL
        cuda_above_k & ~((sz_cap_blackwell_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                             : (sz_kernel_punned_t)&sz_ngrams_sketches_blackwell,
#endif
    };
    enum {
        rocm_above_k = ((sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
                         sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5) *
                        !STRINGZILLA_MULTIVERSIONED) >>
        48
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        rocm_above_k & ~((sz_cap_rocm_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                        : (sz_kernel_punned_t)&sz_ngrams_sketches_rocm,
#endif
#if STRINGZILLA_TARGET_CDNA3
        rocm_above_k & ~((sz_cap_cdna3_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_sketches_cdna3,
#endif
#if STRINGZILLA_TARGET_CDNA4
        rocm_above_k & ~((sz_cap_cdna4_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_sketches_cdna4,
#endif
#if STRINGZILLA_TARGET_CDNA5
        rocm_above_k & ~((sz_cap_cdna5_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                         : (sz_kernel_punned_t)&sz_ngrams_sketches_cdna5,
#endif
    };
    enum { metal_above_k = ((sz_cap_metal_k * STRINGZILLA_TARGET_METAL) * !STRINGZILLA_MULTIVERSIONED) >> 48 };
    static sz_kernel_punned_t const metal[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_METAL
        metal_above_k & ~((sz_cap_metal_k >> 48) * 2 - 1) ? STRINGZILLA_NULL
                                                          : (sz_kernel_punned_t)&sz_ngrams_sketches_metal,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_ampere_k * STRINGZILLA_TARGET_AMPERE |
             sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL,
         cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM | sz_cap_cdna3_k * STRINGZILLA_TARGET_CDNA3 |
             sz_cap_cdna4_k * STRINGZILLA_TARGET_CDNA4 | sz_cap_cdna5_k * STRINGZILLA_TARGET_CDNA5,
         rocm},
        {sz_cap_metal_k * STRINGZILLA_TARGET_METAL, metal},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                  sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                  sz_size_t candidates_budget, sz_capability_t capabilities,
                                                  sz_allocator_t *allocator, sz_stream_t stream) {
    sz_kernel_ngrams_engine_init_t const kernel = (sz_kernel_ngrams_engine_init_t)sz_kernel_pick_(
        capabilities, sz_ngrams_engine_init_capabilities());
    return kernel ? kernel(engine, queries, window_widths, window_widths_count, candidates_budget, allocator, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API void sz_ngrams_engine_free(sz_ngrams_engine_t *engine, sz_stream_t stream) {
    sz_ngrams_engine_close_(engine, stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                               sz_f32_t radius, sz_f32_t *shares, sz_size_t shares_query_stride,
                                               sz_size_t shares_candidate_stride, sz_stream_t stream) {
    sz_kernel_ngrams_overlaps_t const kernel = (sz_kernel_ngrams_overlaps_t)sz_kernel_pick_(
        engine->capability, sz_ngrams_overlaps_capabilities());
    return kernel ? kernel(engine, candidates, radius, shares, shares_query_stride, shares_candidate_stride, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                               sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                               sz_stream_t stream) {
    sz_kernel_ngrams_sketches_t const kernel = (sz_kernel_ngrams_sketches_t)sz_kernel_pick_(
        engine->capability, sz_ngrams_sketches_capabilities());
    return kernel ? kernel(engine, texts, slots, sketches, sketches_stride, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_ngrams_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_ngrams_engine_init_k: lists = sz_ngrams_engine_init_capabilities(); break;
    case sz_kernel_ngrams_overlaps_k: lists = sz_ngrams_overlaps_capabilities(); break;
    case sz_kernel_ngrams_sketches_k: lists = sz_ngrams_sketches_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
