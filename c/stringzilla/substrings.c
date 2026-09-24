/**
 *  @file c/stringzilla/substrings.c
 *  @brief Per-domain dispatch shim for multi-pattern search and scoring (`sz_substrings_*`).
 *  @author Ash Vardanian
 *  @date September 20, 2026
 */
#include <stringzilla/stringzilla.h> // `sz_capabilities`
#include <stringzilla/substrings.h>

#include "dispatch.h"

SZ_DISPATCH_INTERNAL void sz_dispatch_substrings_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;

    impl->substrings_counts = sz_substrings_counts_serial;
    impl->substrings_find = sz_substrings_find_serial;
    impl->substrings_replace = sz_substrings_replace_serial;
    impl->substrings_bm25_scores = sz_substrings_bm25_scores_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->substrings_counts = sz_substrings_counts_haswell;
        impl->substrings_find = sz_substrings_find_haswell;
        impl->substrings_replace = sz_substrings_replace_haswell;
        impl->substrings_bm25_scores = sz_substrings_bm25_scores_haswell;
    }
#endif
#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->substrings_counts = sz_substrings_counts_icelake;
        impl->substrings_find = sz_substrings_find_icelake;
        impl->substrings_replace = sz_substrings_replace_icelake;
        impl->substrings_bm25_scores = sz_substrings_bm25_scores_icelake;
    }
#endif
#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->substrings_counts = sz_substrings_counts_neon;
        impl->substrings_find = sz_substrings_find_neon;
        impl->substrings_replace = sz_substrings_replace_neon;
        impl->substrings_bm25_scores = sz_substrings_bm25_scores_neon;
    }
#endif
    sz_unused_(caps);
}

SZ_DISPATCH_INTERNAL void sz_dispatch_substrings_gpu_update_(void) {
    sz_implementations_gpu_t *impl = &sz_dispatch_gpu_table;

    impl->substrings_counts = SZ_NULL;
    impl->substrings_find = SZ_NULL;
    impl->substrings_replace = SZ_NULL;
    impl->substrings_bm25_scores = SZ_NULL;

    // Pure assignment either way, so a second call from a second `_init_gpu` costs one device query.
#if SZ_USE_CUDA
    if (sz_substrings_cuda_devices_() > 0) {
        impl->substrings_counts = sz_substrings_counts_cuda;
        impl->substrings_find = sz_substrings_find_cuda;
        impl->substrings_replace = sz_substrings_replace_cuda;
        impl->substrings_bm25_scores = sz_substrings_bm25_scores_cuda;
    }
#endif
}

SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_cpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         sz_substrings_engine_t *engine) {
    // The CPU table already holds the tier this machine answers with, so the engine records only which of
    // the two tables the compute verbs must then read.
    sz_capability_t const capability = (sz_capability_t)(sz_capabilities() & sz_caps_cpus_k);
    return sz_substrings_engine_build_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       capability, alloc, engine);
}

#if SZ_USE_CUDA
SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         void *stream, sz_substrings_engine_t *engine) {
    // The only caller that needs a device, so the only place the device table is filled.
    sz_dispatch_gpu_table_init();
    if (!sz_dispatch_gpu_table.substrings_counts) return sz_device_code_mismatch_k;
    return sz_substrings_engine_init_cuda(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       alloc, stream, engine);
}
#else
SZ_API_RUNTIME sz_status_t sz_substrings_engine_init_gpu(sz_sequence_t const *needles,
                                                         sz_substrings_case_sensitivity_t case_sensitivity,
                                                         sz_substrings_overlap_policy_t overlap_policy,
                                                         sz_size_t hot_states, sz_size_t matches_budget,
                                                         sz_memory_allocator_t *alloc,
                                                         void *stream, sz_substrings_engine_t *engine) {
    sz_unused_(needles), sz_unused_(case_sensitivity), sz_unused_(overlap_policy), sz_unused_(hot_states);
    sz_unused_(matches_budget), sz_unused_(alloc), sz_unused_(stream), sz_unused_(engine);
    return sz_device_code_mismatch_k;
}
#endif

SZ_API_RUNTIME void sz_substrings_engine_free(sz_substrings_engine_t *engine) { sz_substrings_engine_free_(engine); }

SZ_API_RUNTIME sz_status_t sz_substrings_counts(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                sz_size_t *counts, sz_size_t counts_stride) {
    return engine->capability & sz_caps_cuda_k
               ? sz_dispatch_gpu_table.substrings_counts(engine, haystacks, counts, counts_stride)
               : sz_dispatch_cpu_table.substrings_counts(engine, haystacks, counts, counts_stride);
}

SZ_API_RUNTIME sz_status_t sz_substrings_find(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                              sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                              sz_size_t *matches_offsets) {
    return engine->capability & sz_caps_cuda_k
               ? sz_dispatch_gpu_table.substrings_find(engine, haystacks, matches, matches_capacity, matches_offsets)
               : sz_dispatch_cpu_table.substrings_find(engine, haystacks, matches, matches_capacity, matches_offsets);
}

SZ_API_RUNTIME sz_status_t sz_substrings_replace(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                 sz_sequence_t const *replacements, sz_ptr_t tape,
                                                 sz_size_t tape_capacity, sz_size_t *offsets) {
    return engine->capability & sz_caps_cuda_k
               ? sz_dispatch_gpu_table.substrings_replace(engine, haystacks, replacements, tape, tape_capacity,
                                                          offsets)
               : sz_dispatch_cpu_table.substrings_replace(engine, haystacks, replacements, tape, tape_capacity,
                                                          offsets);
}

SZ_API_RUNTIME sz_status_t sz_substrings_bm25_scores(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_f32_t const *document_lengths,
                                                     sz_substrings_bm25_t const *parameters,
                                                     sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                     sz_size_t scores_stride) {
    return engine->capability & sz_caps_cuda_k
               ? sz_dispatch_gpu_table.substrings_bm25_scores(engine, haystacks, document_lengths, parameters,
                                                              needle_weights, scores, scores_stride)
               : sz_dispatch_cpu_table.substrings_bm25_scores(engine, haystacks, document_lengths, parameters,
                                                              needle_weights, scores, scores_stride);
}
