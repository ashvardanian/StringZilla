/**
 *  @file include/stringzilla/substrings/hopper.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Hopper tier of multi-pattern search: BM25 scoring that walks the haystacks a scoring
 *      round's pairs table missed with whole clusters of blocks, and the @c _hopper exports, over
 *      the CUDA host side.
 *
 *  From compute capability 9.0 a grid's blocks group into clusters, each block addressing the
 *  others' shared memory. A batch of fewer haystacks than resident blocks walks each haystack the
 *  table missed with a whole cluster, its tally sharded across the cluster's blocks by needle, so
 *  the counts spread over as many multiprocessors and each block sums the terms of its own shard.
 *  The tile scans look back once per cluster rather than once per block, over the aggregate of its
 *  tiles, and total each tile with @c redux.sync as the Ampere tier does. Every other verb and the
 *  round itself run the CUDA host side over this unit's own copies of the shared kernels.
 *
 *  @sa include/stringzilla/substrings/cuda.cuh
 *  @sa include/stringzilla/substrings/simt.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_HOPPER_CUH_
#define STRINGZILLA_SUBSTRINGS_HOPPER_CUH_

#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_block_sum_ampere_`, `sz_block_sum_bounded_ampere_`
#include "stringzilla/strings/hopper.cuh" // `sz_chain_look_back_hopper_`, `sz_chain_tile_hopper_` & kin

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_HOPPER_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_HOPPER

#pragma region Scanning Kernels

/** @ref sz_substrings_scan_tile_simt_ looking back once per cluster, each tile totalled through
 *  @ref sz_block_sum_ampere_ ; launched only in clusters of @c sz_cluster_blocks_cuda_k blocks. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_hopper_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                      sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                      sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale, sz_block_scan_cuda_,
                                  sz_block_sum_ampere_, sz_chain_tile_hopper_, sz_chain_look_back_hopper_);
}

/** @ref sz_substrings_scan_hopper_kernel_ in 32 bits, as a round's chunks and keep flags sum below
 *  2^32. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_bounded_hopper_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                              sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                              sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale,
                                  sz_block_scan_bounded_cuda_, sz_block_sum_bounded_ampere_, sz_chain_tile_hopper_,
                                  sz_chain_look_back_hopper_);
}

/** @ref sz_substrings_scan_sized_simt_ looking back once per cluster, each tile totalled through
 *  @ref sz_block_sum_ampere_ ; launched only in clusters of @c sz_cluster_blocks_cuda_k blocks. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_sized_hopper_kernel_(sz_size_t *chunk_slots, sz_size_t count, sz_size_t const *chunks_count,
                                            sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                            sz_u64_t *stale, sz_size_t matches_budget, sz_bool_t emitting,
                                            sz_substrings_report_t *report) {
    sz_substrings_scan_sized_simt_(chunk_slots, count, chunks_count, elements_per_tile, chain, states, stale,
                                   matches_budget, emitting, report, sz_block_scan_cuda_, sz_block_sum_ampere_,
                                   sz_chain_tile_hopper_, sz_chain_look_back_hopper_);
}

#pragma endregion Scanning Kernels

#pragma region Scoring Kernel

/**
 *  @brief Writes the score of every haystack the round's pairs table held whole, and scores each
 *      of the rest with a cluster: its blocks walk contiguous chunks, each needle counting into
 *      the tally of the block its index falls to modulo the cluster, then each block sums the
 *      terms of its own tally into the first block's sum, one atomic per warp.
 *
 *  Launched only in clusters of @c sz_cluster_blocks_cuda_k blocks, each with the dynamic shared
 *  memory @ref sz_substrings_bm25_prepare_simt_ lays out. A cluster whose haystacks the table held
 *  whole stages nothing. A haystack takes two cluster barriers: one between the walk and the
 *  terms, and one split around the next haystack's setup, after which the first block reads the
 *  sum. A shard's spill flag alternates between two words, so the terms clear the one the next
 *  haystack raises while the walkers of this one may still raise the other.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_hopper_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                      sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                      sz_f32_t const *needle_weights, sz_u32_t *overflow_rows, sz_f32_t *scores,
                                      sz_size_t scores_stride, sz_u32_t staged_accepts_words, sz_u32_t staged_count,
                                      unsigned long long const *sums, sz_size_t const *overflowed) {
    __shared__ sz_substrings_tally_simt_t shards[sz_cluster_blocks_cuda_k];
    __shared__ sz_u32_t spilled[2];
    sz_u32_t const cluster_rank = sz_cluster_rank_hopper_();
    // Every block of a cluster checks the same haystacks, so the cluster leaves or stays whole.
    if (!sz_substrings_bm25_finish_simt_(haystacks, scores, scores_stride, sums, overflowed,
                                         blockIdx.x / sz_cluster_blocks_cuda_k, gridDim.x / sz_cluster_blocks_cuda_k))
        return;
    sz_substrings_bm25_block_simt_t const block = sz_substrings_bm25_prepare_simt_(&engine, overflow_rows,
                                                                                   staged_accepts_words, staged_count);
    sz_size_t const needles_count = engine.needles_count;
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(needles_count);
    // A direct shard holds every eighth needle, a hashed one probes a whole table of its own.
    sz_size_t const shard_slots = block.tally.keys ? table_slots
                                                   : sz_size_divide_round_up(table_slots, sz_cluster_blocks_cuda_k);
    unsigned long long *const leader_sum = (unsigned long long *)sz_cluster_map_hopper_(block.sum, 0);
    sz_size_t haystack_index, pending = STRINGZILLA_SIZE_MAX, walked = 0;
    if (threadIdx.x < sz_cluster_blocks_cuda_k) {
        sz_substrings_tally_simt_t shard = block.tally;
        shard.counts = (sz_u32_t *)sz_cluster_map_hopper_(block.tally.counts, threadIdx.x);
        if (shard.keys) shard.keys = (sz_u32_t *)sz_cluster_map_hopper_(block.tally.keys, threadIdx.x);
        if (shard.overflow)
            shard.overflow = shard.overflow - cluster_rank * needles_count + threadIdx.x * needles_count;
        shard.overflowed = (sz_u32_t *)sz_cluster_map_hopper_(spilled, threadIdx.x);
        shards[threadIdx.x] = shard;
    }
    if (threadIdx.x == 0) spilled[0] = 0, spilled[1] = 0;
    // Every block's tally is clear and its shards laid out before any block counts into them.
    sz_cluster_sync_hopper_();

    for (haystack_index = blockIdx.x / sz_cluster_blocks_cuda_k; haystack_index < haystacks.count;
         haystack_index += gridDim.x / sz_cluster_blocks_cuda_k) {
        if (!overflowed[haystack_index]) continue;
        sz_cptr_t const haystack = sz_strings_start_simt_(haystacks, haystack_index);
        sz_size_t const length = sz_strings_length_simt_(haystacks, haystack_index);
        sz_f64_t const norm = sz_substrings_bm25_norm(
            &parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        sz_substrings_tally_simt_t own = block.tally;
        own.overflowed = spilled + (walked & 1);
        // The last haystack's terms landed in the first block's sum and every shard is clear again.
        if (pending != STRINGZILLA_SIZE_MAX) {
            sz_cluster_wait_hopper_();
            if (cluster_rank == 0 && threadIdx.x == 0)
                scores[pending * scores_stride] = sz_substrings_bm25_fixed_score_simt_(*block.sum), *block.sum = 0;
        }
        sz_substrings_bm25_walk_simt_(
            &engine, &block.staged, haystack, length, (sz_size_t)cluster_rank * blockDim.x + threadIdx.x,
            (sz_size_t)sz_cluster_blocks_cuda_k * blockDim.x, shards, sz_cluster_blocks_cuda_k);
        // Every count lands in its shard before that shard's block reads it.
        sz_cluster_sync_hopper_();
        {
            sz_i64_t const mine = sz_substrings_bm25_terms_simt_(&own, shard_slots, needles_count, &parameters, norm,
                                                                 needle_weights, threadIdx.x, blockDim.x, cluster_rank,
                                                                 sz_cluster_blocks_cuda_k);
            // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
            sz_u64_t const warp_total = sz_u64_lanes_sum_cuda_((sz_u64_t)mine);
            if ((threadIdx.x & 31u) == 0 && warp_total) atomicAdd(leader_sum, (unsigned long long)warp_total);
        }
        // Every read of the other flag ended before the last haystack's barrier, and no walker
        // raises it before this one's.
        if (threadIdx.x < sz_cluster_blocks_cuda_k)
            shards[threadIdx.x].overflowed = (sz_u32_t *)sz_cluster_map_hopper_(spilled + (~walked & 1), threadIdx.x);
        if (threadIdx.x == 0) spilled[~walked & 1] = 0;
        sz_cluster_arrive_hopper_();
        pending = haystack_index, ++walked;
    }
    if (pending != STRINGZILLA_SIZE_MAX) {
        sz_cluster_wait_hopper_();
        if (cluster_rank == 0 && threadIdx.x == 0)
            scores[pending * scores_stride] = sz_substrings_bm25_fixed_score_simt_(*block.sum);
    }
}

#pragma endregion Scoring Kernel

/** The Hopper tier's entry points, its tile scans looking back once per cluster. */
static sz_substrings_entry_points_cuda_t const sz_substrings_entry_points_hopper_ = {
    (void const *)sz_substrings_scan_hopper_kernel_,
    (void const *)sz_substrings_scan_bounded_hopper_kernel_,
    (void const *)sz_substrings_scan_sized_hopper_kernel_,
    (void const *)sz_substrings_rewrite_offsets_cuda_kernel_,
    sz_cluster_blocks_cuda_k,
    sz_cap_hopper_k,
};

STRINGZILLA_API sz_status_t sz_substrings_engine_init_hopper(
    sz_substrings_engine_t *engine, sz_strings_view_t const *needles, sz_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_cuda_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, &sz_substrings_entry_points_hopper_,
                                                  allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_contains_hopper(sz_substrings_engine_t *engine,
                                                          sz_strings_view_t const *haystacks, sz_u1x8_t *contained,
                                                          sz_stream_t stream) {
    return sz_substrings_contains_scoped_cuda_(engine, haystacks, contained, &sz_substrings_entry_points_hopper_,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_hopper(sz_substrings_engine_t *engine,
                                                        sz_strings_view_t const *haystacks, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_stream_t stream) {
    return sz_substrings_counts_scoped_cuda_(engine, haystacks, counts, counts_stride,
                                             &sz_substrings_entry_points_hopper_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_hopper(sz_substrings_engine_t *engine,
                                                      sz_strings_view_t const *haystacks, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_stream_t stream) {
    return sz_substrings_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                           &sz_substrings_entry_points_hopper_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_hopper(sz_substrings_engine_t *engine,
                                                         sz_strings_view_t const *haystacks,
                                                         sz_strings_view_t const *replacements, sz_ptr_t target,
                                                         sz_size_t target_capacity, sz_size_t *offsets,
                                                         sz_stream_t stream) {
    return sz_substrings_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              &sz_substrings_entry_points_hopper_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_hopper(sz_substrings_engine_t *engine,
                                                             sz_strings_view_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_cuda_(
        engine, haystacks, document_lengths, parameters, needle_weights, scores, scores_stride,
        (void const *)sz_substrings_bm25_hopper_kernel_, &sz_substrings_entry_points_hopper_, stream);
}

#endif // STRINGZILLA_TARGET_HOPPER

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_HOPPER_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBSTRINGS_HOPPER_CUH_
