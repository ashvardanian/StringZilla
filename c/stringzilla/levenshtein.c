/**
 *  @file c/stringzilla/levenshtein.c
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Per-domain dispatch shim for the Levenshtein edit distances (`sz_levenshtein_*`).
 */

#include <stringzilla/levenshtein.h>
#include <stringzilla/stringzilla.h> // `sz_capabilities`

#include "dispatch.h"

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_levenshtein_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->levenshtein_distances = sz_levenshtein_distances_serial;
    impl->levenshtein_distances_utf8 = sz_levenshtein_distances_serial;

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->levenshtein_distances = sz_levenshtein_distances_haswell;
        impl->levenshtein_distances_utf8 = sz_levenshtein_distances_haswell;
    }
#endif

#if STRINGZILLA_TARGET_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->levenshtein_distances = sz_levenshtein_distances_skylake;
        impl->levenshtein_distances_utf8 = sz_levenshtein_distances_skylake;
    }
#endif

#if STRINGZILLA_TARGET_ICELAKE
    // Runes never reach the byte lanes, so the rune slot stays whatever the Skylake tier bound above.
    if (caps & sz_cap_icelake_k) { impl->levenshtein_distances = sz_levenshtein_distances_icelake; }
#endif
}

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_levenshtein_gpu_update_(void) {
#if STRINGZILLA_TARGET_CUDA
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return;
    sz_dispatch_gpu_table.levenshtein_distances = sz_levenshtein_distances_cuda;
    sz_dispatch_gpu_table.levenshtein_distances_utf8 = sz_levenshtein_distances_cuda;
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_levenshtein_engine_init_cpu(sz_sequence_t const *queries,
                                                                   sz_levenshtein_symbol_t symbol,
                                                                   sz_memory_allocator_t *alloc,
                                                                   sz_levenshtein_engine_t *engine) {
    return sz_levenshtein_engine_init_cpu_(queries, symbol, sz_capabilities(), alloc, engine);
}

#if STRINGZILLA_TARGET_CUDA
STRINGZILLA_API_RUNTIME sz_status_t sz_levenshtein_engine_init_gpu(sz_sequence_t const *queries,
                                                                   sz_levenshtein_symbol_t symbol,
                                                                   sz_memory_allocator_t *alloc, void *stream,
                                                                   sz_levenshtein_engine_t *engine) {
    // The only caller that needs the device table is the only one that has already chosen a device.
    sz_dispatch_gpu_table_init();
    return sz_levenshtein_engine_init_cuda(queries, symbol, alloc, stream, engine);
}
#else
STRINGZILLA_API_RUNTIME sz_status_t sz_levenshtein_engine_init_gpu(sz_sequence_t const *queries,
                                                                   sz_levenshtein_symbol_t symbol,
                                                                   sz_memory_allocator_t *alloc, void *stream,
                                                                   sz_levenshtein_engine_t *engine) {
    sz_unused_(queries), sz_unused_(symbol), sz_unused_(alloc), sz_unused_(stream), sz_unused_(engine);
    return sz_device_code_mismatch_k;
}
#endif

STRINGZILLA_API_RUNTIME void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine) {
    sz_levenshtein_engine_free_(engine);
}

STRINGZILLA_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride) {
    if (engine->capability & sz_caps_cuda_k)
        return engine->symbol == sz_levenshtein_runes_k
                   ? sz_dispatch_gpu_table.levenshtein_distances_utf8(engine, candidates, distances, distances_stride)
                   : sz_dispatch_gpu_table.levenshtein_distances(engine, candidates, distances, distances_stride);
    return engine->symbol == sz_levenshtein_runes_k
               ? sz_dispatch_cpu_table.levenshtein_distances_utf8(engine, candidates, distances, distances_stride)
               : sz_dispatch_cpu_table.levenshtein_distances(engine, candidates, distances, distances_stride);
}
