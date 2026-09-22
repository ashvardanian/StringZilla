/**
 *  @file c/stringzilla/levenshtein.c
 *  @brief Per-domain dispatch shim for the Levenshtein edit distances (`sz_levenshtein_distance*`).
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/levenshtein.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_levenshtein_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->levenshtein_distance = sz_levenshtein_distance_serial;
    impl->levenshtein_distances = sz_levenshtein_distances_serial;
    impl->levenshtein_distance_utf8 = sz_levenshtein_distance_utf8_serial;
    impl->levenshtein_distances_utf8 = sz_levenshtein_distances_utf8_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->levenshtein_distances = sz_levenshtein_distances_haswell;
        impl->levenshtein_distances_utf8 = sz_levenshtein_distances_utf8_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->levenshtein_distances = sz_levenshtein_distances_skylake;
        impl->levenshtein_distances_utf8 = sz_levenshtein_distances_utf8_skylake;
    }
#endif

#if SZ_USE_ICELAKE
    // Runes never reach the byte lanes, so the rune verb stays whatever the Skylake tier bound above.
    if (caps & sz_cap_icelake_k) { impl->levenshtein_distances = sz_levenshtein_distances_icelake; }
#endif

    // Last, so a device outranks every CPU tier. Only the byte verbs: the rune path prepares a page table the
    // device tier does not build, so a UTF-8 round keeps the widest CPU backend installed above.
#if SZ_USE_CUDA
    if (caps & sz_cap_cuda_k) {
        impl->levenshtein_distance = sz_levenshtein_distance_cuda;
        impl->levenshtein_distances = sz_levenshtein_distances_cuda;
    }
#endif
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distance(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                   sz_memory_allocator_t *alloc, sz_size_t *distance) {
    return sz_dispatch_table.levenshtein_distance(a, a_length, b, b_length, alloc, distance);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distances(sz_cptr_t query, sz_size_t query_length,
                                                    sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                    sz_size_t *distances) {
    return sz_dispatch_table.levenshtein_distances(query, query_length, candidates, alloc, distances);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distance_utf8(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                        sz_size_t *distance) {
    return sz_dispatch_table.levenshtein_distance_utf8(a, a_length, b, b_length, alloc, distance);
}

SZ_API_RUNTIME sz_status_t sz_levenshtein_distances_utf8(sz_cptr_t query, sz_size_t query_length,
                                                         sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                         sz_size_t *distances) {
    return sz_dispatch_table.levenshtein_distances_utf8(query, query_length, candidates, alloc, distances);
}
