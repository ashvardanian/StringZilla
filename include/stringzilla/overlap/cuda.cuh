/**
 *  @file include/stringzilla/overlap/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of window overlap: the forest laid out on the host, the launch geometry
 *      resolved once per engine, and the @c _cuda exports, over the walk of <tt>overlap/simt.cuh</tt>.
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

/** Candidates one block's tile holds, and the threads it cuts their bytes across, at most, which
 *  each tier's kernel compiles to beside its blocks per multiprocessor. */
enum { sz_overlap_candidates_per_block_cuda_k = 128 };

/** Blocks one multiprocessor keeps resident, which caps the walk at 64 registers where it would
 *  take 72. A register budget per multiprocessor, so it holds on any count of them and on MIG
 *  slices alike; eight is what every code this tier compiles holds, Turing's 1,024 threads
 *  included, and the Blackwell tier keeps twelve. */
enum { sz_overlap_blocks_per_multiprocessor_cuda_k = 8 };

/** Waves of resident blocks a grid carries before further queries fold into each block's own loop,
 *  which lays a tile's chunks out once for all of them. */
enum { sz_overlap_grid_waves_cuda_k = 8 };

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

    /** Blocks the device keeps resident at that geometry, or zero where it cannot be asked. */
    sz_size_t resident_blocks;
} sz_overlap_geometry_cuda_t;

/** Dynamic shared memory one block of @p threads takes: the staged tree, rounded up to a vector,
 *  then a count per width of one walk and a first chunk per candidate of its tile, the tile's
 *  total, and two 16-bit owners per thread. */
STRINGZILLA_INLINE sz_size_t sz_overlap_shared_bytes_cuda_(sz_size_t staged_nodes_count, sz_size_t threads,
                                                           sz_size_t widths_count) {
    sz_size_t const walk_widths = sz_min_of_two(widths_count, (sz_size_t)sz_overlap_gpu_widths_per_walk_k);
    return (sz_size_divide_round_up(staged_nodes_count, 4) * 4 + threads * (walk_widths + 2) + 1) * sizeof(sz_u32_t);
}

/** Prepares the forest on the device the caller already made current, for the tier whose @p kernel
 *  the rounds will launch and whose @p capability the engine records, which is every step of
 *  @ref sz_overlap_engine_init_scoped_cuda_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_cuda_(sz_overlap_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, void const *kernel,
                                                            sz_capability_t capability, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;

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

    sz_overlap_engine_fill_serial_(engine, queries, chain);
    sz_size_t widest_nodes = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        widest_nodes = sz_max_of_two(widest_nodes, engine->nodes_offsets[index + 1] - engine->nodes_offsets[index]);
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
    geometry->resident_blocks = sz_device_multiprocessors_cuda_() *
                                sz_resident_blocks_cuda_(kernel, geometry->candidates_per_block,
                                                         sz_overlap_shared_bytes_cuda_(geometry->staged_nodes_count,
                                                                                       geometry->candidates_per_block,
                                                                                       engine->widths_count));

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
 *  @param[in] queries Read on the @b host, so in any layout, unlike a round's candidates.
 *  @param[in] candidates_budget Ignored, as rounds keep no per-candidate state on this backend.
 *  @param[in] kernel The tier's kernel, which the launch geometry is resolved for.
 *  @param[in] capability The tier's capability, which the engine records.
 *  @param[in] allocator Unified and device-reachable, or @c STRINGZILLA_NULL for unified memory on
 *      the device of @p stream.
 *  @param[in] stream Names the device; the host lays the forest out, so the call schedules nothing.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for zero widths,
 *      @c sz_device_memory_mismatch_k when @p allocator hands back memory the device cannot reach
 *      or @p stream is one the runtime cannot place, or @c sz_missing_gpu_k without a device.
 *  @sa sz_overlap_engine_init
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_scoped_cuda_(
    sz_overlap_engine_t *engine, sz_strings_view_t const *queries, sz_size_t const *window_widths,
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
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_cuda_(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                       sz_f32_t radius, sz_u32_t *matches,
                                                       sz_size_t matches_query_stride,
                                                       sz_size_t matches_candidate_stride, void const *kernel,
                                                       void const *bounded_kernel, sz_stream_t stream) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, matches_query_stride,
                                                              matches_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    if (!sz_memory_accessible_cuda_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(matches)) return sz_device_memory_mismatch_k;
    if (!sz_strings_accessible_cuda_(candidates)) return sz_device_memory_mismatch_k;

    sz_overlap_geometry_cuda_t const *const geometry = (sz_overlap_geometry_cuda_t const *)sz_overlap_engine_head_(
        engine);
    sz_size_t const per_block = geometry->candidates_per_block;
    sz_size_t const blocks = sz_size_divide_round_up(candidates->count, per_block);
    sz_size_t rows = sz_min_of_two(engine->count, (sz_size_t)sz_overlap_queries_per_grid_cuda_k);
    if (geometry->resident_blocks)
        rows = sz_min_of_two(rows,
                             sz_size_divide_round_up(sz_overlap_grid_waves_cuda_k * geometry->resident_blocks, blocks));
    sz_size_t staged_nodes_count = geometry->staged_nodes_count;

    // The engine travels by value with its host-only members cleared: an allocator's function pointers would ride
    // into constant memory on every launch and no kernel can call them.
    sz_overlap_engine_t launched_engine = *engine;
    launched_engine.allocator.allocate = STRINGZILLA_NULL;
    launched_engine.allocator.reallocate = STRINGZILLA_NULL;
    launched_engine.allocator.free = STRINGZILLA_NULL;
    launched_engine.allocator.handle = STRINGZILLA_NULL;
    launched_engine.memory = STRINGZILLA_NULL, launched_engine.memory_bytes = 0;
    launched_engine.scratch = STRINGZILLA_NULL, launched_engine.scratch_bytes = 0;

    sz_strings_view_t launched_candidates = *candidates;
    sz_u32_t *launched_matches = matches;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)rows, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    sz_u32_t misses_bound = sz_overlap_misses_bound_(radius);
    void *arguments[7];
    arguments[0] = &launched_engine, arguments[1] = &launched_candidates, arguments[2] = &launched_matches;
    arguments[3] = &matches_query_stride, arguments[4] = &matches_candidate_stride, arguments[5] = &staged_nodes_count;
    arguments[6] = &misses_bound;

    // One launch per group of widths one walk rolls, each over its own slice of every row; a
    // bounded round sums a pair's misses over every width, so it takes them all in one launch.
    sz_bool_t const bounded = radius < 1.0f ? sz_true_k : sz_false_k;
    sz_size_t const widths_per_launch = bounded ? engine->widths_count : (sz_size_t)sz_overlap_gpu_widths_per_walk_k;
    void const *const launched_kernel = bounded ? bounded_kernel : kernel;
    sz_status_t status = sz_success_k;
    for (sz_size_t first_width = 0; status == sz_success_k && first_width < engine->widths_count;
         first_width += widths_per_launch) {
        launched_engine.widths = engine->widths + first_width, launched_engine.powers = engine->powers + first_width;
        launched_engine.widths_count = sz_min_of_two(engine->widths_count - first_width, widths_per_launch);
        launched_matches = matches + first_width;
        sz_size_t const shared_bytes = sz_overlap_shared_bytes_cuda_(staged_nodes_count, per_block,
                                                                     launched_engine.widths_count);
        status = sz_launch_cuda_(launched_kernel, grid, block, arguments, shared_bytes, stream);
    }
    return status;
}

/**
 *  @brief The tier's @p kernel of @ref sz_overlap_scores, or its @p bounded_kernel under a radius,
 *      on the engine's device.
 *  @pre @p candidates is a tape the device reaches, as @ref sz_strings_to_tape_best writes one with
 *      a unified allocator.
 *  @return @c sz_success_k, @c sz_device_memory_mismatch_k when the engine, the matches or the
 *      candidates are memory the device of @p stream cannot reach, or @c sz_device_code_mismatch_k
 *      when the launch itself is refused.
 *  @note Enqueues on @p stream and returns; the caller joins it before reading @p matches.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_scoped_cuda_(sz_overlap_engine_t *engine,
                                                              sz_strings_view_t const *candidates, sz_f32_t radius,
                                                              sz_u32_t *matches, sz_size_t matches_query_stride,
                                                              sz_size_t matches_candidate_stride, void const *kernel,
                                                              void const *bounded_kernel, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_scores_cuda_(engine, candidates, radius, matches, matches_query_stride,
                                     matches_candidate_stride, kernel, bounded_kernel, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#if STRINGZILLA_TARGET_CUDA

static __global__ __launch_bounds__(sz_overlap_candidates_per_block_cuda_k,
                                    sz_overlap_blocks_per_multiprocessor_cuda_k) void sz_overlap_scores_cuda_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cuda_, sz_tile_queue_open_simt_,
                            sz_tile_queue_next_simt_);
}

static __global__ __launch_bounds__(
    sz_overlap_candidates_per_block_cuda_k) void sz_overlap_bounded_scores_cuda_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count, sz_u32_t misses_bound) {
    sz_overlap_bounded_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                                    staged_nodes_count, misses_bound, sz_group_shuffle_index_cuda_,
                                    sz_tile_queue_open_simt_, sz_tile_queue_next_simt_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_cuda(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_cuda_kernel_, sz_cap_cuda_k, allocator,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_cuda(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                   sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                   sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_cuda_(engine, candidates, radius, matches, matches_query_stride,
                                          matches_candidate_stride, (void const *)sz_overlap_scores_cuda_kernel_,
                                          (void const *)sz_overlap_bounded_scores_cuda_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_OVERLAP_CUDA_CUH_
