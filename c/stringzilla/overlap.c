/**
 *  @file c/stringzilla/overlap.c
 *  @brief Per-domain dispatch shim for the window-overlap engine (`sz_overlap_engine_*`, `sz_overlap_scores`).
 *  @author Ash Vardanian
 *  @date January 27, 2024
 */
#include <stringzilla/overlap.h>
#include <stringzilla/stringzilla.h> // `sz_capabilities`

#include "dispatch.h"

SZ_DISPATCH_INTERNAL void sz_dispatch_overlap_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->overlap_scores = sz_overlap_scores_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) impl->overlap_scores = sz_overlap_scores_haswell;
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) impl->overlap_scores = sz_overlap_scores_skylake;
#endif
}

/*  Filled from the GPU runtime rather than from `sz_capabilities`, and left null where none is compiled in or no
 *  device answers. Assigning the slot is the whole body, so a second call answers the same thing.
 */
SZ_DISPATCH_INTERNAL void sz_dispatch_overlap_gpu_update_(void) {
#if SZ_USE_CUDA
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0)
        sz_dispatch_gpu_table.overlap_scores = sz_overlap_scores_cuda;
    else
        sz_dispatch_gpu_table.overlap_scores = (sz_overlap_scores_t)SZ_NULL;
#else
    sz_dispatch_gpu_table.overlap_scores = (sz_overlap_scores_t)SZ_NULL;
#endif
}

SZ_API_RUNTIME sz_status_t sz_overlap_engine_init_cpu(sz_sequence_t const *queries, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                      sz_overlap_engine_t *engine) {
    sz_capability_t const caps = sz_capabilities();
    sz_unused_(caps);
#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k)
        return sz_overlap_engine_init_skylake(queries, window_widths, window_widths_count, alloc, engine);
#endif
#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k)
        return sz_overlap_engine_init_haswell(queries, window_widths, window_widths_count, alloc, engine);
#endif
    return sz_overlap_engine_init_serial(queries, window_widths, window_widths_count, alloc, engine);
}

#if SZ_USE_CUDA
SZ_API_RUNTIME sz_status_t sz_overlap_engine_init_gpu(sz_sequence_t const *queries, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                      void *stream, sz_overlap_engine_t *engine) {
    // The only caller that needs a device table is the one building an engine for a device, so this is where it fills.
    sz_dispatch_gpu_table_init();
    return sz_overlap_engine_init_cuda(queries, window_widths, window_widths_count, alloc, stream, engine);
}
#else
SZ_API_RUNTIME sz_status_t sz_overlap_engine_init_gpu(sz_sequence_t const *queries, sz_size_t const *window_widths,
                                                      sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                      void *stream, sz_overlap_engine_t *engine) {
    sz_unused_(queries), sz_unused_(window_widths), sz_unused_(window_widths_count);
    sz_unused_(alloc), sz_unused_(stream), sz_unused_(engine);
    return sz_device_code_mismatch_k;
}
#endif

SZ_API_RUNTIME void sz_overlap_engine_free(sz_overlap_engine_t *engine) { sz_overlap_engine_close_(engine); }

SZ_API_RUNTIME sz_status_t sz_overlap_scores(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                             sz_f32_t *scores, sz_size_t scores_query_stride,
                                             sz_size_t scores_candidate_stride) {
    if (!(engine->capability & sz_caps_cuda_k))
        return sz_dispatch_cpu_table.overlap_scores(engine, candidates, scores, scores_query_stride,
                                                    scores_candidate_stride);
    // A device engine outlives the runtime that built it only if something unloaded the driver under it.
    if (!sz_dispatch_gpu_table.overlap_scores) return sz_device_code_mismatch_k;
    return sz_dispatch_gpu_table.overlap_scores(engine, candidates, scores, scores_query_stride,
                                                scores_candidate_stride);
}
