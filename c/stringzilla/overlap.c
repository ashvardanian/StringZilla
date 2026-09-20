/**
 *  @file c/stringzilla/overlap.c
 *  @brief Per-domain dispatch shim for window overlap (`sz_overlap_score`, `sz_overlap_scores`).
 *  @author Ash Vardanian
 *  @date January 27, 2024
 */
#include "dispatch.h"
#include <stringzilla/overlap.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_overlap_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->overlap_score = sz_overlap_score_serial;
    impl->overlap_scores = sz_overlap_scores_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->overlap_score = sz_overlap_score_haswell;
        impl->overlap_scores = sz_overlap_scores_haswell;
    }
#endif

#if SZ_USE_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->overlap_score = sz_overlap_score_skylake;
        impl->overlap_scores = sz_overlap_scores_skylake;
    }
#endif

    // Last, so a device outranks every CPU tier: the verbs installed here take the CPU backends' arguments and
    // stage whatever the device cannot already reach, so a caller holding host memory still gets an answer.
#if SZ_USE_CUDA
    if (caps & sz_cap_cuda_k) {
        impl->overlap_score = sz_overlap_score_cuda;
        impl->overlap_scores = sz_overlap_scores_cuda;
    }
#endif
}

SZ_API_RUNTIME sz_status_t sz_overlap_score(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                            sz_size_t candidate_length, sz_size_t const *widths, sz_size_t widths_count,
                                            sz_memory_allocator_t *alloc, sz_f32_t *scores) {
    return sz_dispatch_table.overlap_score(query, query_length, candidate, candidate_length, widths, widths_count,
                                           alloc, scores);
}

SZ_API_RUNTIME sz_status_t sz_overlap_scores(sz_cptr_t query, sz_size_t query_length, sz_sequence_t const *candidates,
                                             sz_size_t const *widths, sz_size_t widths_count,
                                             sz_memory_allocator_t *alloc, sz_f32_t *scores) {
    return sz_dispatch_table.overlap_scores(query, query_length, candidates, widths, widths_count, alloc, scores);
}
