/**
 *  @file include/stringzilla/overlap/simt.cuh
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief CUDA backend for window overlap: one thread per candidate, its chain walked through a
 *      ring of prefix hashes so every width is scored in one pass, and one prepared query's B-tree
 *      probed per block row.
 *
 *  The arithmetic is the serial tier's, reached from the device through `--expt-relaxed-constexpr`,
 *  so the scores are bit-identical rather than merely close. Integers, not doubles: both factors
 *  are below 2^32, so the product fits one @c u64 and the remainder by a constant lowers to a
 *  multiply-high - where an @c f64 reduction would run at the device's double-precision rate, a
 *  sixty-fourth of its single-precision one on consumer parts.
 *
 *  A candidate's chain is a dependent recurrence, so it stays on one thread; parallelism comes from
 *  the candidates, which is what the device has thousands of. The thread keeps the last
 *  @ref sz_overlap_simt_ring_span_k prefix hashes rather than the whole chain, so its footprint is
 *  constant in the candidate's length and every width is scored in the same pass over the text.
 *
 *  The query axis rides @c blockIdx.y, one tree per block, strided when a batch outruns a grid
 *  dimension. The chain is therefore walked once per query rather than once per candidate - the
 *  probe is a @c levels deep descent with a sixteen-key branch step at every node against two
 *  operations of chain, so sharing it would buy about a percent and cost a per-thread match counter
 *  per query, which no register file holds.
 *
 *  Nothing here crosses the bus during a round. The forest, the candidates and the scores are
 *  already where the device reaches them and the texts are read in place, so a round costs one
 *  launch rather than a round trip.
 *
 *  @sa include/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_SIMT_CUH_
#define STRINGZILLA_OVERLAP_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/overlap/serial.h"

#if STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
extern "C" {
#endif

#pragma region CUDA

/** Prefix hashes one thread keeps, one past the widest window so a window's start and end never
 *  alias; a power of two, so the ring index is a mask rather than a division. */
enum { sz_overlap_simt_ring_span_k = sz_overlap_simt_widest_window_k + 1 };

/** Candidates one block scores when the device cannot be asked; measured flat from 32 to 512 and
 *  off a cliff at 1024, so this is the ceiling of the flat range rather than a tuned figure. */
enum { sz_overlap_simt_candidates_per_block_k = 512 };

/** Queries one grid carries on @c blockIdx.y; a batch past it strides, since a grid
 *  dimension is bounded. */
enum { sz_overlap_simt_queries_per_grid_k = 65535 };

/** The share of a block's shared memory a staged tree may take. Staging is worth about half again
 *  while the tree is small, and stops paying past that: the descent is data-dependent, and shared
 *  memory's banks handle a scatter worse than L1 does with its sector reuse. A fraction rather than
 *  a byte count, because the ceiling itself moves
 *  - a hundred kilobytes per multiprocessor on consumer Ada against more than twice that on
 *    the datacenter parts. */
enum { sz_overlap_simt_shared_tree_share_k = 3 };

/**
 *  @brief What @ref sz_overlap_engine_init_simt_scoped_ resolved once, kept at the head of the
 *      engine's own block.
 *
 *  Every member costs a driver round trip to answer, and none of them moves between rounds, so a
 *  scoring verb reads them rather than asking again.
 */
typedef struct sz_overlap_simt_geometry_t {

    /** Threads one block runs, whichever count lands the most resident warps. */
    sz_size_t candidates_per_block;

    /** @c u32 entries a block stages, sized by the widest tree, or zero for none. */
    sz_size_t staged_nodes_count;
} sz_overlap_simt_geometry_t;

/** Whether the prepared query holds one raw @p window_hash, walking the tree a level at a time. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_simt_btree_probe_(sz_overlap_btree_t const *btree, sz_u32_t window_hash) {
    sz_u32_t const key = window_hash ^ (sz_u32_t)sz_overlap_sign_flip_k;
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 != btree->levels; ++level) {
        sz_u32_t const *const separators = btree->nodes +
                                           (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k;
        node = node * sz_overlap_branches_per_node_k + sz_overlap_serial_branch_step_(separators, key);
    }
    sz_u32_t const *const leaf = btree->nodes +
                                 (btree->level_bases[btree->levels - 1] + node) * sz_overlap_keys_per_node_k;
    return sz_overlap_serial_leaf_step_(leaf, key);
}

/**
 *  @brief Scores one candidate against one prepared query, one score per width, on one thread.
 *
 *  The ring holds P(k) back to P(k − widest), so a window of any width up to that reads its
 *  start straight out of it and the text is walked once however many widths are asked for.
 *
 *  @p candidates ' accessors run here, on the device: one call per candidate, uniform across the
 *  warp, against multi-kilobyte texts. Nothing is flattened for the launch, so a caller whose
 *  sequence is some other layout entirely - a tape, a column, an index into someone else's arena -
 *  needs no conversion.
 */
STRINGZILLA_DEVICE void sz_overlap_simt_sweep_(sz_overlap_engine_t const *engine, sz_overlap_btree_t const *btree,
                                               sz_size_t query, sz_sequence_t const *candidates, sz_size_t candidate,
                                               sz_f32_t *scores, sz_size_t scores_query_stride,
                                               sz_size_t scores_candidate_stride) {
    sz_cptr_t const text = candidates->get_start(candidates->handle, candidate);
    sz_size_t const length = candidates->get_length(candidates->handle, candidate);
    sz_size_t const query_length = engine->lengths[query];
    sz_f32_t *const candidate_scores = scores + query * scores_query_stride + candidate * scores_candidate_stride;

    sz_u32_t matches[sz_overlap_simt_widths_max_k];
    for (sz_size_t index = 0; index != engine->widths_count; ++index) matches[index] = 0;

    // `ring[k & mask]` is `P(k)`, so a window ending at `k` reads its start at `(k - width) & mask`. The base is
    // `256 mod p`, which is 256 itself, the modulus being the wider of the two.
    sz_size_t const mask = sz_overlap_simt_ring_span_k - 1;
    sz_u32_t ring[sz_overlap_simt_ring_span_k];
    ring[0] = 0;
    sz_u64_t prior = 0;
    for (sz_size_t position = 0; position != length; ++position) {
        prior = (prior * 256 + (sz_u64_t)(sz_u8_t)text[position]) % (sz_u64_t)sz_overlap_modulus_k;
        sz_size_t const ending = position + 1;
        ring[ending & mask] = (sz_u32_t)prior;
        for (sz_size_t index = 0; index != engine->widths_count; ++index) {
            sz_size_t const width = engine->widths[index];
            if (!width || width > ending) continue;
            sz_u64_t const start = (sz_u64_t)ring[(ending - width) & mask];
            sz_u64_t const shifted = start * engine->powers[index] % (sz_u64_t)sz_overlap_modulus_k;
            sz_u64_t const residue = prior + (sz_u64_t)sz_overlap_modulus_k - shifted;
            matches[index] += (sz_u32_t)sz_overlap_simt_btree_probe_(
                btree, (sz_u32_t)(residue % (sz_u64_t)sz_overlap_modulus_k));
        }
    }

    for (sz_size_t index = 0; index != engine->widths_count; ++index) {
        sz_size_t const width = engine->widths[index];
        sz_bool_t const scored = width && width <= query_length && width <= length ? sz_true_k : sz_false_k;
        candidate_scores[index] = scored
                                      ? sz_overlap_share_(matches[index], length - width + 1, query_length - width + 1)
                                      : 0.0f;
    }
}

/**
 *  @brief One block row per prepared query, one thread per candidate, the tree staged when the
 *      launch asked for it.
 *
 *  No thread leaves the query loop early, because the staging barriers are collective: a candidate
 *  past the batch skips its sweep rather than returning, so every thread of the block reaches
 *  every @c __syncthreads.
 */
static __global__ void sz_overlap_simt_scores_kernel_(sz_overlap_engine_t engine, sz_sequence_t candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride, sz_size_t staged_nodes_count) {
    extern __shared__ sz_u32_t staged_nodes_[];
    sz_size_t const candidate = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (sz_size_t query = blockIdx.y; query < engine.count; query += gridDim.y) {
        sz_overlap_btree_t btree = sz_overlap_engine_row_(&engine, query);
        if (staged_nodes_count) {
            sz_size_t const entries = engine.nodes_offsets[query + 1] - engine.nodes_offsets[query];
            for (sz_size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
                staged_nodes_[entry] = btree.nodes[entry];
            __syncthreads();
            btree.nodes = staged_nodes_;
        }
        if (candidate < candidates.count)
            sz_overlap_simt_sweep_(&engine, &btree, query, &candidates, candidate, scores, scores_query_stride,
                                   scores_candidate_stride);
        if (staged_nodes_count) __syncthreads();
    }
}

/** Prepares the forest on the device the caller already made current, which is every step of
 *  @ref sz_overlap_engine_init_simt_scoped_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_simt_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator) {
    if (!sz_cuda_multiprocessors_()) return sz_missing_gpu_k;
    if (!window_widths_count || window_widths_count > sz_overlap_simt_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] > sz_overlap_simt_widest_window_k) return sz_unexpected_dimensions_k;

    sz_memory_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_memory_allocator_init_unified_(&unified, ordinal);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count,
                                                       sizeof(sz_overlap_simt_geometry_t), &unified, engine);
    if (opened != sz_success_k) return opened;
    if (!sz_memory_reaches_device_(engine->memory)) {
        sz_overlap_engine_close_(engine);
        return sz_device_memory_mismatch_k;
    }

    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    sz_size_t const chain_bytes = (longest_query + 1) * sizeof(sz_f64_t);
    sz_f64_t *const chain = (sz_f64_t *)host.allocate(chain_bytes, host.handle);
    if (!chain) {
        sz_overlap_engine_close_(engine);
        return sz_bad_alloc_k;
    }

    // The arena and the key counts stay writable until the engine is handed back; its readers see them const.
    sz_u32_t *const nodes = (sz_u32_t *)engine->nodes;
    sz_u32_t *const keys_counts = (sz_u32_t *)engine->keys_counts;
    sz_size_t widest_nodes = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        sz_size_t const length = engine->lengths[index];
        sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
        sz_size_t const entries = engine->nodes_offsets[index + 1] - engine->nodes_offsets[index];
        if (entries > widest_nodes) widest_nodes = entries;
        chain[0] = 0.0;
        sz_f64_t prior = 0.0;
        for (sz_size_t position = 0; position != length; ++position)
            prior = sz_overlap_f64x1_prefix_hash_step_serial_(prior, text + position, chain + position + 1);

        sz_size_t written = 0;
        for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
            sz_size_t const width = engine->widths[width_index];
            if (!width || width > length) continue;
            sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
            sz_size_t const query_windows = length - width + 1;
            for (sz_size_t window = 0; window != query_windows; ++window)
                sz_overlap_f64x1_window_hash_step_serial(chain + window, chain + window + width, power,
                                                         arena + written + window);
            written += query_windows;
        }
        sz_overlap_btree_t btree;
        sz_size_t const distinct = sz_overlap_u32x1_btree_sort_serial_(arena, written);
        sz_overlap_btree_prepare_(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }
    host.free(chain, chain_bytes, host.handle);

    // What the bound device will hand one block, not what the part this was tuned on would have. Every block
    // stages one tree, so the widest of them is what the launch has to fit and what the occupancy walk is told.
    sz_overlap_simt_geometry_t *const geometry = (sz_overlap_simt_geometry_t *)sz_overlap_engine_head_(engine);
    geometry->staged_nodes_count = 0;
    if (widest_nodes * sizeof(sz_u32_t) * sz_overlap_simt_shared_tree_share_k <= sz_cuda_shared_bytes_per_block_())
        geometry->staged_nodes_count = widest_nodes;
    geometry->candidates_per_block = sz_cuda_block_size_((void const *)sz_overlap_simt_scores_kernel_,
                                                         geometry->staged_nodes_count * sizeof(sz_u32_t),
                                                         STRINGZILLA_SIZE_MAX, sz_overlap_simt_candidates_per_block_k);

    engine->capability = STRINGZILLA_ARCH_ROCM_ ? sz_cap_rocm_k : sz_cap_cuda_k, engine->ordinal = ordinal;
    return sz_success_k;
}

/**
 *  @brief Prepares every query of @p queries into one block on device @p ordinal and resolves
 *      the launch geometry.
 *
 *  The trees are laid out by the host, because a sort is neither a scan nor a map and a
 *  hand-written device radix sort would replace a host sort of a few tens of thousands of keys. The
 *  chain that feeds them is host working space no kernel ever reads, so it comes from the host
 *  allocator rather than from @p allocator.
 *
 *  @param[in] queries Read on the @b host, so its accessors must be host-callable, unlike
 *      a round's candidates.
 *  @param[in] candidates_budget Ignored, as rounds keep no per-candidate state on this backend.
 *  @param[in] allocator Unified and device-reachable, or @c STRINGZILLA_NULL for unified memory on
 *      @p ordinal.
 *  @param[in] stream Checked to belong to @p ordinal; the host lays the forest out, so nothing this
 *      call does is scheduled.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for a bad count or width of windows,
 *      @c sz_device_memory_mismatch_k when @p allocator hands back memory the device cannot reach or
 *      @p stream is another device's, or @c sz_missing_gpu_k without a device.
 *  @sa sz_overlap_engine_init
 *
 *  A window count is bad when zero or above @ref sz_overlap_simt_widths_max_k, and a width when it
 *  is past @ref sz_overlap_simt_widest_window_k, the per-thread ring's compile-time bound.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_simt_scoped_(sz_overlap_engine_t *engine,
                                                                   sz_sequence_t const *queries,
                                                                   sz_size_t const *window_widths,
                                                                   sz_size_t window_widths_count,
                                                                   sz_size_t candidates_budget, sz_size_t ordinal,
                                                                   sz_memory_allocator_t *allocator, void *stream) {
    int caller = 0;
    sz_unused_(candidates_budget);
    sz_status_t status = sz_cuda_device_enter_(ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_engine_init_simt_(engine, queries, window_widths, window_widths_count, ordinal, allocator);
    sz_cuda_device_leave_(caller);
    return status;
}

/** Enqueues one round on the device the caller already made current, which is every step of
 *  @ref sz_overlap_scores_simt_scoped_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_simt_(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                       sz_f32_t *scores, sz_size_t scores_query_stride,
                                                       sz_size_t scores_candidate_stride, void *stream) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    // The handle is checked, never the accessors: those are the device's to call, so the host must not, and a
    // pointer is all this side can inspect. That the texts they answer are device-reachable is the caller's word.
    if (!sz_memory_reaches_device_(scores)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device_(candidates->handle)) return sz_device_memory_mismatch_k;

    sz_overlap_simt_geometry_t const *const geometry = (sz_overlap_simt_geometry_t const *)sz_overlap_engine_head_(
        engine);
    sz_size_t const per_block = geometry->candidates_per_block;
    sz_size_t const blocks = sz_size_divide_round_up(candidates->count, per_block);
    sz_size_t const rows = engine->count < sz_overlap_simt_queries_per_grid_k ? engine->count
                                                                              : sz_overlap_simt_queries_per_grid_k;
    sz_size_t staged_nodes_count = geometry->staged_nodes_count;
    sz_size_t const staged_bytes = staged_nodes_count * sizeof(sz_u32_t);

    // The engine travels by value with its host-only members cleared: an allocator's function pointers would ride
    // into constant memory on every launch and no kernel can call them.
    sz_overlap_engine_t launched_engine = *engine;
    launched_engine.allocator.allocate = STRINGZILLA_NULL;
    launched_engine.allocator.free = STRINGZILLA_NULL;
    launched_engine.allocator.handle = STRINGZILLA_NULL;
    launched_engine.memory = STRINGZILLA_NULL, launched_engine.memory_bytes = 0;
    launched_engine.scratch = STRINGZILLA_NULL, launched_engine.scratch_bytes = 0;
    sz_sequence_t launched_candidates = *candidates;

    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)rows, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[6];
    arguments[0] = &launched_engine, arguments[1] = &launched_candidates, arguments[2] = &scores;
    arguments[3] = &scores_query_stride, arguments[4] = &scores_candidate_stride, arguments[5] = &staged_nodes_count;
    return sz_cuda_launch_((void const *)sz_overlap_simt_scores_kernel_, grid, block, arguments, staged_bytes, stream);
}

/**
 *  @brief The CUDA and ROCm kernel of @ref sz_overlap_scores, on the engine's device.
 *  @pre @p candidates carries @b device accessors, as @ref sz_cuda_sequence_from_string_views binds
 *      them, because the kernel is what calls them, once per candidate, uniform across the warp.
 *  @return @c sz_success_k, @c sz_device_memory_mismatch_k when the scores or the first candidate
 *      is host memory or @p stream is another device's, or @c sz_device_code_mismatch_k when the
 *      launch itself is refused.
 *  @note Enqueues on @p stream and returns; the caller joins it before reading @p scores.
 *
 *  Only the handle can be checked from this side, so host accessors reach the device as an invalid
 *  address rather than a status.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_simt_scoped_(sz_overlap_engine_t *engine,
                                                              sz_sequence_t const *candidates, sz_f32_t *scores,
                                                              sz_size_t scores_query_stride,
                                                              sz_size_t scores_candidate_stride, void *stream) {
    int caller = 0;
    sz_status_t status = sz_cuda_device_enter_(engine->ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride, stream);
    sz_cuda_device_leave_(caller);
    return status;
}

#pragma endregion CUDA

#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_overlap_engine_init_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_size_t ordinal,
                                                        sz_memory_allocator_t *allocator, void *stream) {
    return sz_overlap_engine_init_simt_scoped_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               ordinal, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, void *stream) {
    return sz_overlap_scores_simt_scoped_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                                          stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#if STRINGZILLA_TARGET_ROCM

STRINGZILLA_API sz_status_t sz_overlap_engine_init_rocm(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_size_t ordinal,
                                                        sz_memory_allocator_t *allocator, void *stream) {
    return sz_overlap_engine_init_simt_scoped_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               ordinal, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_rocm(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, void *stream) {
    return sz_overlap_scores_simt_scoped_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                                          stream);
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM
#endif // STRINGZILLA_OVERLAP_SIMT_CUH_
