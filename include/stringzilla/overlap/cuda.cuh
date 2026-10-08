/**
 *  @file include/stringzilla/overlap/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of window overlap: the forest laid out on the host, the launch geometry
 *      resolved once per engine, and the @c _cuda exports, over the walk of `overlap/simt.cuh`.
 *
 *  The trees are laid out by the host, because a sort is neither a scan nor a map and a
 *  hand-written device radix sort would replace a host sort of a few tens of thousands of keys.
 *
 *  Everything but the CUDA tier's own kernel and exports serves the Blackwell tier too, which
 *  launches its own kernel through the same host side.
 *
 *  @sa include/stringzilla/overlap/simt.cuh
 *  @sa include/stringzilla/overlap/blackwell.cuh
 *  @sa include/stringzilla/overlap/rocm.cuh
 */
#ifndef STRINGZILLA_OVERLAP_CUDA_CUH_
#define STRINGZILLA_OVERLAP_CUDA_CUH_

#include "stringzilla/cuda.cuh"
#include "stringzilla/overlap/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_

#ifdef __cplusplus
extern "C" {
#endif

/** Candidates one block's tile holds, and the threads it cuts their bytes across, when the device
 *  cannot be asked; measured flat from 32 to 512 and off a cliff at 1024, so this is the ceiling of
 *  the flat range rather than a tuned figure, and the widest tile the occupancy walk considers. */
enum { sz_overlap_candidates_per_block_cuda_k = 512 };

/** Queries one grid carries on @c blockIdx.y; a batch past it strides, since a grid
 *  dimension is bounded. */
enum { sz_overlap_queries_per_grid_cuda_k = 65535 };

/** The share of a block's shared memory a staged tree may take. Staging is worth about half again
 *  while the tree is small, and stops paying past that: the descent is data-dependent, and shared
 *  memory's banks handle a scatter worse than L1 does with its sector reuse. A fraction rather than
 *  a byte count, because the ceiling itself moves
 *  - a hundred kilobytes per multiprocessor on consumer Ada against more than twice that on
 *    the datacenter parts. */
enum { sz_overlap_shared_tree_share_cuda_k = 3 };

/**
 *  @brief What @ref sz_overlap_engine_init_scoped_cuda_ resolved once, kept at the head of the
 *      engine's own block.
 *
 *  Every member costs a driver round trip to answer, and none of them moves between rounds, so a
 *  scoring verb reads them rather than asking again.
 */
typedef struct sz_overlap_geometry_cuda_t {

    /** Threads one block runs, whichever count lands the most resident warps. */
    sz_size_t candidates_per_block;

    /** @c u32 entries a block stages, sized by the widest tree, or zero for none. */
    sz_size_t staged_nodes_count;
} sz_overlap_geometry_cuda_t;

/** Dynamic shared memory one block of @p threads takes: the staged tree, rounded up to a vector,
 *  then a count per width and a first chunk per candidate of its tile, and the tile's total. */
STRINGZILLA_INLINE sz_size_t sz_overlap_shared_bytes_cuda_(sz_size_t staged_nodes_count, sz_size_t threads,
                                                           sz_size_t widths_count) {
    return (sz_size_divide_round_up(staged_nodes_count, 4) * 4 + threads * (widths_count + 1) + 1) * sizeof(sz_u32_t);
}

/** Prepares the forest on the device the caller already made current, for the tier whose @p kernel
 *  the rounds will launch and whose @p capability the engine records, which is every step of
 *  @ref sz_overlap_engine_init_scoped_cuda_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_cuda_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, void const *kernel,
                                                            sz_capability_t capability, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;
    if (!window_widths_count || window_widths_count > sz_overlap_gpu_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] > sz_overlap_gpu_widest_window_k) return sz_unexpected_dimensions_k;

    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_cuda_(&unified);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count,
                                                       sizeof(sz_overlap_geometry_cuda_t), &unified, stream, engine);
    if (opened != sz_success_k) return opened;
    if (!sz_memory_accessible_cuda_(engine->memory)) {
        sz_overlap_engine_close_(engine, stream);
        return sz_device_memory_mismatch_k;
    }

    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_allocator_t host;
    sz_status_t const status = sz_allocator_init_heap(&host);
    if (status != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return status;
    }
    sz_size_t const chain_bytes = (longest_query + 1) * sizeof(sz_f64_t);
    sz_f64_t *const chain = (sz_f64_t *)host.allocate(chain_bytes, host.handle, stream);
    if (!chain) {
        sz_overlap_engine_close_(engine, stream);
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
    host.free(chain, chain_bytes, host.handle, stream);

    // What the bound device will hand one block, not what the part this was tuned on would have. Every block
    // stages one tree, so the widest of them is what the launch has to fit and what the occupancy
    // walk is told, beside the counts of the widest tile the walk may pick.
    sz_overlap_geometry_cuda_t *const geometry = (sz_overlap_geometry_cuda_t *)sz_overlap_engine_head_(engine);
    sz_size_t const shared_ceiling = sz_device_shared_bytes_per_block_cuda_();
    geometry->staged_nodes_count = 0;
    if (widest_nodes * sizeof(sz_u32_t) * sz_overlap_shared_tree_share_cuda_k <= shared_ceiling &&
        sz_overlap_shared_bytes_cuda_(widest_nodes, sz_overlap_candidates_per_block_cuda_k, engine->widths_count) <=
            shared_ceiling)
        geometry->staged_nodes_count = widest_nodes;
    geometry->candidates_per_block = sz_block_size_cuda_(
        kernel,
        sz_overlap_shared_bytes_cuda_(geometry->staged_nodes_count, sz_overlap_candidates_per_block_cuda_k,
                                      engine->widths_count),
        sz_overlap_candidates_per_block_cuda_k, sz_overlap_candidates_per_block_cuda_k);

    engine->capability = capability;
    return sz_success_k;
}

/**
 *  @brief Prepares every query of @p queries into one block on the device of @p stream and
 *      resolves the launch geometry.
 *
 *  The chain that feeds the trees is host working space no kernel ever reads, so it comes from the
 *  host allocator rather than from @p allocator.
 *
 *  @param[in] queries Read on the @b host, so its accessors must be host-callable, unlike
 *      a round's candidates.
 *  @param[in] candidates_budget Ignored, as rounds keep no per-candidate state on this backend.
 *  @param[in] kernel The tier's kernel, which the launch geometry is resolved for.
 *  @param[in] capability The tier's capability, which the engine records.
 *  @param[in] allocator Unified and device-reachable, or @c STRINGZILLA_NULL for unified memory on
 *      the device of @p stream.
 *  @param[in] stream Names the device; the host lays the forest out, so the call schedules nothing.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for a bad count or width of windows,
 *      @c sz_device_memory_mismatch_k when @p allocator hands back memory the device cannot reach
 *      or @p stream is one the runtime cannot place, or @c sz_missing_gpu_k without a device.
 *  @sa sz_overlap_engine_init
 *
 *  A window count is bad when zero or above @ref sz_overlap_gpu_widths_max_k, and a width when it
 *  is past @ref sz_overlap_gpu_widest_window_k, which bounds the bytes a chunk re-walks to start.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_scoped_cuda_(
    sz_overlap_engine_t *engine, sz_sequence_t const *queries, sz_size_t const *window_widths,
    sz_size_t window_widths_count, sz_size_t candidates_budget, void const *kernel, sz_capability_t capability,
    sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(candidates_budget);
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_engine_init_cuda_(engine, queries, window_widths, window_widths_count, kernel, capability,
                                          allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

/** Enqueues one round of the tier's @p kernel on the device the caller already made current,
 *  which is every step of @ref sz_overlap_scores_scoped_cuda_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_cuda_(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                       sz_f32_t *scores, sz_size_t scores_query_stride,
                                                       sz_size_t scores_candidate_stride, void const *kernel,
                                                       sz_stream_t stream) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    if (!sz_memory_accessible_cuda_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(scores)) return sz_device_memory_mismatch_k;
    if (candidates->get_start != sz_sequence_tape_start || candidates->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_cuda_(candidates->handle))
        return sz_device_memory_mismatch_k;

    sz_overlap_geometry_cuda_t const *const geometry = (sz_overlap_geometry_cuda_t const *)sz_overlap_engine_head_(
        engine);
    sz_size_t const per_block = geometry->candidates_per_block;
    sz_size_t const blocks = sz_size_divide_round_up(candidates->count, per_block);
    sz_size_t const rows = engine->count < sz_overlap_queries_per_grid_cuda_k ? engine->count
                                                                              : sz_overlap_queries_per_grid_cuda_k;
    sz_size_t staged_nodes_count = geometry->staged_nodes_count;
    sz_size_t const shared_bytes = sz_overlap_shared_bytes_cuda_(staged_nodes_count, per_block, engine->widths_count);

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
    return sz_launch_cuda_(kernel, grid, block, arguments, shared_bytes, stream);
}

/**
 *  @brief The tier's @p kernel of @ref sz_overlap_scores, on the engine's device.
 *  @pre @p candidates carries @b device accessors, as @ref sz_sequence_realloc_best binds them,
 *      because the kernel is what calls them, once per candidate, uniform across the warp.
 *  @return @c sz_success_k, @c sz_device_memory_mismatch_k when the engine, the scores or the
 *      candidates are memory the device of @p stream cannot reach, or @c sz_device_code_mismatch_k
 *      when the launch itself is refused.
 *  @note Enqueues on @p stream and returns; the caller joins it before reading @p scores.
 *
 *  Only the handle can be checked from this side, so host accessors reach the device as an invalid
 *  address rather than a status.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_scoped_cuda_(sz_overlap_engine_t *engine,
                                                              sz_sequence_t const *candidates, sz_f32_t *scores,
                                                              sz_size_t scores_query_stride,
                                                              sz_size_t scores_candidate_stride, void const *kernel,
                                                              sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_scores_cuda_(engine, candidates, scores, scores_query_stride, scores_candidate_stride, kernel,
                                     stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#if STRINGZILLA_TARGET_CUDA

static __global__ void sz_overlap_scores_cuda_kernel_(sz_overlap_engine_t engine, sz_sequence_t candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride, sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cuda_, sz_tile_queue_open_simt_,
                            sz_tile_queue_next_simt_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_cuda_kernel_, sz_cap_cuda_k, allocator,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_cuda(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                   sz_f32_t *scores, sz_size_t scores_query_stride,
                                                   sz_size_t scores_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_cuda_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                                          (void const *)sz_overlap_scores_cuda_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_OVERLAP_CUDA_CUH_
