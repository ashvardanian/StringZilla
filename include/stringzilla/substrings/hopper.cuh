/**
 *  @file include/stringzilla/substrings/hopper.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Hopper tier of multi-pattern search: BM25 scoring that walks a short batch's haystacks
 *      with whole clusters of blocks, and the @c _hopper exports, over the CUDA host side.
 *
 *  From compute capability 9.0 a grid's blocks group into clusters, each block addressing the
 *  others' shared memory, so a batch of fewer haystacks than resident blocks walks each haystack
 *  with a whole cluster, every count landing in its first block's tally, rather than idling the
 *  device at a block each. Every other verb runs the CUDA host side over this unit's own copies of
 *  the shared kernels.
 *
 *  @sa include/stringzilla/substrings/cuda.cuh
 *  @sa include/stringzilla/substrings/simt.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_HOPPER_CUH_
#define STRINGZILLA_SUBSTRINGS_HOPPER_CUH_

#include "stringzilla/substrings/cuda.cuh"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_TARGET_HOPPER

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Clusters

/** This block's rank in its cluster, from Hopper's @c %cluster_ctarank . */
STRINGZILLA_DEVICE sz_u32_t sz_cluster_rank_hopper_(void) {
    sz_u32_t rank;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

/** Blocks in this block's cluster, from Hopper's @c %cluster_nctarank . */
STRINGZILLA_DEVICE sz_u32_t sz_cluster_size_hopper_(void) {
    sz_u32_t size;
    asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(size));
    return size;
}

/** Where @p pointer, into this block's shared memory, lands in the cluster's block at @p rank, with
 *  Hopper's @c mapa . */
STRINGZILLA_DEVICE void *sz_cluster_map_hopper_(void *pointer, sz_u32_t rank) {
    void *mapped;
    asm("mapa.u64 %0, %1, %2;" : "=l"(mapped) : "l"(pointer), "r"(rank));
    return mapped;
}

/** A barrier across every thread of the cluster, ordering each block's shared writes before it,
 *  with Hopper's @c barrier.cluster . */
STRINGZILLA_DEVICE void sz_cluster_sync_hopper_(void) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

#pragma endregion Clusters

#pragma region Scoring Kernel

/**
 *  @brief Scores one haystack per cluster: its blocks walk contiguous chunks into the first
 *      block's tally, which only that block then sums.
 *
 *  Launched only in clusters of @c sz_substrings_bm25_cluster_blocks_cuda_k blocks, each with the
 *  dynamic shared memory @ref sz_substrings_bm25_prepare_simt_ lays out.
 */
static __global__ void sz_substrings_bm25_hopper_kernel_(sz_substrings_engine_t engine, sz_sequence_t haystacks,
                                                         sz_f32_t const *document_lengths,
                                                         sz_substrings_bm25_t parameters,
                                                         sz_f32_t const *needle_weights, sz_u32_t *overflow_rows,
                                                         sz_f32_t *scores, sz_size_t scores_stride,
                                                         sz_u32_t staged_accepts_words, sz_u32_t staged_count) {
    sz_substrings_bm25_block_simt_t const block = sz_substrings_bm25_prepare_simt_(&engine, overflow_rows,
                                                                                   staged_accepts_words, staged_count);
    sz_size_t const needles_count = engine.needles_count;
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(needles_count);
    sz_u32_t const cluster_rank = sz_cluster_rank_hopper_(), cluster_blocks = sz_cluster_size_hopper_();
    sz_substrings_tally_simt_t leader;
    sz_size_t haystack_index;
    leader.layout = block.tally.layout;
    leader.counts = (sz_u32_t *)sz_cluster_map_hopper_(block.tally.counts, 0);
    leader.keys = block.tally.keys ? (sz_u32_t *)sz_cluster_map_hopper_(block.tally.keys, 0) : STRINGZILLA_NULL;
    leader.overflowed = (sz_u32_t *)sz_cluster_map_hopper_(block.tally.overflowed, 0);
    leader.overflow = block.tally.overflow ? block.tally.overflow - (sz_size_t)cluster_rank * needles_count
                                           : STRINGZILLA_NULL;

    for (haystack_index = blockIdx.x / cluster_blocks; haystack_index < haystacks.count;
         haystack_index += gridDim.x / cluster_blocks) {
        sz_cptr_t const haystack = sz_sequence_tape_start_simt_(haystacks.handle, haystack_index);
        sz_size_t const length = sz_sequence_tape_length_simt_(haystacks.handle, haystack_index);
        sz_f64_t const norm = sz_substrings_bm25_norm(
            &parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        // The first block's tally is clear before the cluster counts into it, and every count lands
        // before that block reads them.
        sz_cluster_sync_hopper_();
        sz_substrings_bm25_walk_simt_(&engine, &block.staged, haystack, length, haystack_index,
                                      (sz_size_t)cluster_rank * blockDim.x + threadIdx.x,
                                      (sz_size_t)cluster_blocks * blockDim.x, &leader);
        sz_cluster_sync_hopper_();
        if (cluster_rank == 0)
            sz_substrings_bm25_score_simt_(&block.tally, table_slots, needles_count, &parameters, norm, needle_weights,
                                           block.sum, scores + haystack_index * scores_stride);
    }
    // A block's shared memory outlives every count another block of its cluster sends into it.
    sz_cluster_sync_hopper_();
}

#pragma endregion Scoring Kernel

STRINGZILLA_API sz_status_t sz_substrings_engine_init_hopper(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_cuda_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, sz_cap_hopper_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_size_t *counts, sz_size_t counts_stride,
                                                        sz_stream_t stream) {
    return sz_substrings_counts_scoped_cuda_(engine, haystacks, counts, counts_stride, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                      sz_size_t *matches_offsets, sz_stream_t stream) {
    return sz_substrings_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_hopper(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                         sz_sequence_t const *replacements, sz_ptr_t target,
                                                         sz_size_t target_capacity, sz_size_t *offsets,
                                                         sz_stream_t stream) {
    return sz_substrings_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_hopper(sz_substrings_engine_t *engine,
                                                             sz_sequence_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_cuda_(engine, haystacks, document_lengths, parameters, needle_weights,
                                                  scores, scores_stride,
                                                  (void const *)sz_substrings_bm25_hopper_kernel_, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_HOPPER
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBSTRINGS_HOPPER_CUH_
