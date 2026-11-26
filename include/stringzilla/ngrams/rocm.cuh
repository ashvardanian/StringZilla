/**
 *  @file include/stringzilla/ngrams/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of N-grams: the tables filled on the host, the launch geometry resolved
 *      once per engine, and the @c _rocm exports, over the walks of <tt>ngrams/simt.cuh</tt>.
 *
 *  The tables are filled by the host, as filing a few thousand N-grams per query is no launch's
 *  worth and the host already hashes them for every other tier. Every walk stays inside its own 32
 *  lanes, so a 64-wide wavefront runs two of them side by side.
 *
 *  @sa include/stringzilla/ngrams/simt.cuh
 *  @sa include/stringzilla/ngrams/cuda.cuh
 */
#ifndef STRINGZILLA_NGRAMS_ROCM_CUH_
#define STRINGZILLA_NGRAMS_ROCM_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_

#include "stringzilla/rocm.cuh"
#include "stringzilla/ngrams/simt.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Candidates one block's tile holds, a warp scoring one at a time, when the device cannot be
 *  asked, and the widest tile the occupancy walk considers: past it a block schedules too coarsely
 *  for what residency returns. */
enum { sz_ngrams_candidates_per_block_rocm_k = 512 };

/** Queries one grid carries on @c blockIdx.y ; a batch past it strides, since a grid dimension is
 *  bounded. */
enum { sz_ngrams_queries_per_grid_rocm_k = 65535 };

/** Waves of resident blocks a grid carries before further queries fold into each block's own loop,
 *  which keeps a block's warps on one tile for all of them. */
enum { sz_ngrams_grid_waves_rocm_k = 8 };

/** The share of a block's shared memory a staged table may take, so the hit words and the resident
 *  blocks keep the rest. A fraction rather than a byte count, because the ceiling itself moves
 *  between parts, which a byte count would hard-code. */
enum { sz_ngrams_shared_table_share_rocm_k = 3 };

/** Threads one sketching block runs at most, a group of 32 lanes sketching one text at a time over
 *  its own ranks; fewer where the ranks of four groups outgrow the block's shared memory. */
enum { sz_ngrams_sketch_threads_per_block_rocm_k = 128 };

/** Blocks one sketching grid carries at most; a batch past them is covered by each group's loop. */
enum { sz_ngrams_sketch_blocks_per_grid_rocm_k = 0x7FFFFFFF };

/**
 *  @brief What @ref sz_ngrams_engine_init_scoped_rocm_ resolved once, kept at the head of the
 *      engine's own block.
 *
 *  Every member costs a driver round trip to answer, and none of them moves between rounds, so a
 *  verb reads them rather than asking again.
 */
typedef struct sz_ngrams_geometry_rocm_t {

    /** Threads one block runs, whichever count lands the most resident threads. */
    sz_size_t candidates_per_block;

    /** Slots a block stages, sized by the widest table, or zero for none. */
    sz_size_t staged_slots_count;

    /** Hit words each warp keeps, the widest query's N-grams in bits. */
    sz_size_t hit_words_per_warp;

    /** Blocks the device keeps resident at that geometry, or zero where it cannot be asked. */
    sz_size_t resident_blocks;

    /** Shared memory the device hands one block, which bounds a sketch's ranks. */
    sz_size_t shared_bytes_per_block;
} sz_ngrams_geometry_rocm_t;

/** Dynamic shared memory one block of @p threads takes: the staged table, then each warp's hits. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_shared_bytes_rocm_(sz_size_t staged_slots_count, sz_size_t threads,
                                                          sz_size_t hit_words_per_warp) {
    return staged_slots_count * sizeof(sz_u64_t) +
           sz_size_divide_round_up(threads, 32) * hit_words_per_warp * sizeof(sz_u32_t);
}

/** Prepares the tables on the device the caller already made current, for the tier whose @p kernel
 *  the rounds will launch and whose @p capability the engine records, which is every step of
 *  @ref sz_ngrams_engine_init_scoped_rocm_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_init_rocm_(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count, void const *kernel,
                                                           sz_capability_t capability, sz_allocator_t *allocator,
                                                           sz_stream_t stream) {
    if (!sz_device_multiprocessors_rocm_()) return sz_missing_gpu_k;

    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_rocm_(&unified);
    sz_status_t const opened = sz_ngrams_engine_open_(queries, window_widths, window_widths_count,
                                                      sizeof(sz_ngrams_geometry_rocm_t), &unified, stream, engine);
    if (opened != sz_success_k) return opened;
    if (!sz_memory_accessible_rocm_(engine->memory)) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_device_memory_mismatch_k;
    }

    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        longest_query = sz_max_of_two(longest_query, (sz_size_t)engine->lengths[index]);
    sz_allocator_t host;
    sz_status_t const status = sz_allocator_init_heap(&host);
    if (status != sz_success_k) {
        sz_ngrams_engine_close_(engine, stream);
        return status;
    }
    sz_size_t const prefixes_bytes = (longest_query + 1) * sizeof(sz_f64_t) +
                                     (sz_ngrams_count(longest_query, engine->widths, engine->widths_count) + 1) *
                                         sizeof(sz_u32_t);
    sz_f64_t *const prefixes = (sz_f64_t *)host.allocate(prefixes_bytes, host.handle, stream);
    if (!prefixes) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_bad_alloc_k;
    }
    sz_ngrams_engine_fill_serial_(engine, queries, prefixes, (sz_u32_t *)(prefixes + longest_query + 1));
    host.free(prefixes, prefixes_bytes, host.handle, stream);
    sz_size_t widest_slots = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        widest_slots = sz_max_of_two(widest_slots, engine->slots_offsets[index + 1] - engine->slots_offsets[index]);

    // What the bound device will hand one block, not what the part this was tuned on would have.
    // Every block stages one table and every warp keeps hit words for the widest query, so a query
    // whose words outgrow one warp's share of the block is refused here, not at the first round.
    sz_ngrams_geometry_rocm_t *const geometry = (sz_ngrams_geometry_rocm_t *)sz_ngrams_engine_head_(engine);
    sz_size_t const shared_ceiling = sz_device_shared_bytes_per_block_rocm_();
    sz_size_t const hit_words = sz_size_divide_round_up(sz_ngrams_engine_widest_ngrams_(engine), 32);
    if (sz_ngrams_shared_bytes_rocm_(0, 32, hit_words) > shared_ceiling) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_unexpected_dimensions_k;
    }
    geometry->shared_bytes_per_block = shared_ceiling;
    geometry->hit_words_per_warp = hit_words;
    geometry->staged_slots_count = 0;
    if (widest_slots * sizeof(sz_u64_t) * sz_ngrams_shared_table_share_rocm_k <= shared_ceiling &&
        sz_ngrams_shared_bytes_rocm_(widest_slots, sz_ngrams_candidates_per_block_rocm_k, hit_words) <= shared_ceiling)
        geometry->staged_slots_count = widest_slots;
    sz_size_t widest_block = sz_ngrams_candidates_per_block_rocm_k;
    while (widest_block > 32 &&
           sz_ngrams_shared_bytes_rocm_(geometry->staged_slots_count, widest_block, hit_words) > shared_ceiling)
        widest_block /= 2;
    geometry->candidates_per_block = sz_block_size_rocm_(
        kernel, sz_ngrams_shared_bytes_rocm_(geometry->staged_slots_count, widest_block, hit_words), widest_block,
        widest_block);
    geometry->resident_blocks = sz_device_multiprocessors_rocm_() *
                                sz_resident_blocks_rocm_(
                                    kernel, geometry->candidates_per_block,
                                    sz_ngrams_shared_bytes_rocm_(geometry->staged_slots_count,
                                                                 geometry->candidates_per_block, hit_words));

    engine->capability = capability;
    return sz_success_k;
}

/**
 *  @brief Prepares every query of @p queries into one block on the device of @p stream and
 *      resolves the launch geometry.
 *
 *  The prefixes that feed the tables are host working space no kernel ever reads, so they come from
 *  the host allocator rather than from @p allocator .
 *
 *  @param[in] queries Read on the @b host , so in any layout, unlike a round's candidates; null or
 *      empty for an engine that only sketches.
 *  @param[in] candidates_budget Ignored, as rounds keep no per-candidate state on this backend.
 *  @param[in] kernel The tier's overlaps kernel, which the launch geometry is resolved for.
 *  @param[in] capability The tier's capability, which the engine records.
 *  @param[in] allocator Device-reachable, or @c STRINGZILLA_NULL for unified memory of @p stream .
 *  @param[in] stream Names the device; the host fills the tables, so the call schedules nothing.
 *  @return @c sz_success_k , @c sz_unexpected_dimensions_k for widths the host refuses or a query
 *      whose hit words outgrow a warp's shared memory, @c sz_device_memory_mismatch_k when
 *      @p allocator hands back memory the device cannot reach or @p stream is one the runtime
 *      cannot place, or @c sz_missing_gpu_k without a device.
 *  @sa sz_ngrams_engine_init
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_engine_init_scoped_rocm_(
    sz_ngrams_engine_t *engine, sz_strings_view_t const *queries, sz_size_t const *window_widths,
    sz_size_t window_widths_count, sz_size_t candidates_budget, void const *kernel, sz_capability_t capability,
    sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(candidates_budget);
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_ngrams_engine_init_rocm_(engine, queries, window_widths, window_widths_count, kernel, capability,
                                         allocator, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

/** Enqueues one round of the tier's @p kernel on the device the caller already made current,
 *  which is every step of @ref sz_ngrams_overlaps_scoped_rocm_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_overlaps_rocm_(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                        sz_f32_t radius, sz_u32_t *matches,
                                                        sz_size_t matches_query_stride,
                                                        sz_size_t matches_candidate_stride, void const *kernel,
                                                        sz_stream_t stream) {
    sz_status_t const dimensions = sz_ngrams_engine_strides_(engine, candidates->count, matches_query_stride,
                                                             matches_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    if (!sz_memory_accessible_rocm_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_rocm_(matches)) return sz_device_memory_mismatch_k;
    if (!sz_strings_accessible_rocm_(candidates)) return sz_device_memory_mismatch_k;

    sz_ngrams_geometry_rocm_t const *const geometry = (sz_ngrams_geometry_rocm_t const *)sz_ngrams_engine_head_(engine);
    sz_size_t const per_block = geometry->candidates_per_block;
    sz_size_t const blocks = sz_size_divide_round_up(candidates->count, per_block);
    sz_size_t rows = sz_min_of_two(engine->count, (sz_size_t)sz_ngrams_queries_per_grid_rocm_k);
    if (geometry->resident_blocks)
        rows = sz_min_of_two(rows,
                             sz_size_divide_round_up(sz_ngrams_grid_waves_rocm_k * geometry->resident_blocks, blocks));
    sz_size_t staged_slots_count = geometry->staged_slots_count, hit_words_per_warp = geometry->hit_words_per_warp;
    // The engine travels by value with its host-only members cleared: an allocator's function
    // pointers would ride into constant memory on every launch, and no kernel can call them.
    sz_ngrams_engine_t launched_engine = *engine;
    launched_engine.allocator.allocate = STRINGZILLA_NULL;
    launched_engine.allocator.reallocate = STRINGZILLA_NULL;
    launched_engine.allocator.free = STRINGZILLA_NULL;
    launched_engine.allocator.handle = STRINGZILLA_NULL;
    launched_engine.memory = STRINGZILLA_NULL, launched_engine.memory_bytes = 0;
    launched_engine.scratch = STRINGZILLA_NULL, launched_engine.scratch_bytes = 0;

    sz_strings_view_t launched_candidates = *candidates;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)rows, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    sz_u32_t misses_bound = sz_ngrams_misses_bound_(radius);
    void *arguments[8];
    arguments[0] = &launched_engine, arguments[1] = &launched_candidates, arguments[2] = &matches;
    arguments[3] = &matches_query_stride, arguments[4] = &matches_candidate_stride, arguments[5] = &staged_slots_count;
    arguments[6] = &hit_words_per_warp, arguments[7] = &misses_bound;
    return sz_launch_rocm_(kernel, grid, block, arguments,
                           sz_ngrams_shared_bytes_rocm_(staged_slots_count, per_block, hit_words_per_warp), stream);
}

/**
 *  @brief The tier's @p kernel of @ref sz_ngrams_overlaps on the engine's device.
 *  @pre @p candidates is a tape the device reaches, as @ref sz_strings_to_tape_best writes one with
 *      a unified allocator.
 *  @return @c sz_success_k , @c sz_device_memory_mismatch_k when the engine, the matches or the
 *      candidates are memory the device of @p stream cannot reach, or @c sz_device_code_mismatch_k
 *      when the launch itself is refused.
 *  @note Enqueues on @p stream and returns; the caller joins it before reading @p matches .
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_overlaps_scoped_rocm_(sz_ngrams_engine_t *engine,
                                                               sz_strings_view_t const *candidates, sz_f32_t radius,
                                                               sz_u32_t *matches, sz_size_t matches_query_stride,
                                                               sz_size_t matches_candidate_stride, void const *kernel,
                                                               sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_ngrams_overlaps_rocm_(engine, candidates, radius, matches, matches_query_stride,
                                      matches_candidate_stride, kernel, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

/** Enqueues one sketching round of the tier's @p kernel on the device the caller already made
 *  current, which is every step of @ref sz_ngrams_sketches_scoped_rocm_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_ngrams_sketches_rocm_(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                        sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                        void const *kernel, sz_stream_t stream) {
    sz_status_t const dimensions = sz_ngrams_sketch_dimensions_(engine->widths, engine->widths_count, slots);
    if (dimensions != sz_success_k) return dimensions;
    if (sketches_stride < slots) return sz_unexpected_dimensions_k;
    // A warp's ranks live in shared memory, so a sketch outgrowing a block's is refused unlaunched.
    sz_ngrams_geometry_rocm_t const *const geometry = (sz_ngrams_geometry_rocm_t const *)sz_ngrams_engine_head_(engine);
    sz_size_t const warp_bytes = slots * sizeof(sz_u32_t);
    if (warp_bytes > geometry->shared_bytes_per_block) return sz_unexpected_dimensions_k;
    if (!texts->count) return sz_success_k;
    if (!sz_memory_accessible_rocm_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_rocm_(sketches)) return sz_device_memory_mismatch_k;
    if (!sz_strings_accessible_rocm_(texts)) return sz_device_memory_mismatch_k;

    sz_size_t const warps = sz_min_of_two((sz_size_t)sz_ngrams_sketch_threads_per_block_rocm_k / 32,
                                          geometry->shared_bytes_per_block / warp_bytes);
    sz_size_t const blocks = sz_min_of_two(sz_size_divide_round_up(texts->count, warps),
                                           (sz_size_t)sz_ngrams_sketch_blocks_per_grid_rocm_k);
    sz_size_t const *widths = engine->widths;
    sz_size_t widths_count = engine->widths_count;
    sz_strings_view_t launched_texts = *texts;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = (unsigned)(warps * 32), block.y = 1, block.z = 1;
    void *arguments[6];
    arguments[0] = &widths, arguments[1] = &widths_count, arguments[2] = &launched_texts;
    arguments[3] = &slots, arguments[4] = &sketches, arguments[5] = &sketches_stride;
    return sz_launch_rocm_(kernel, grid, block, arguments, warps * warp_bytes, stream);
}

/**
 *  @brief The tier's @p kernel of @ref sz_ngrams_sketches on the engine's device.
 *  @pre @p texts is a tape the device reaches, as @ref sz_strings_to_tape_best writes one with
 *      a unified allocator.
 *  @return @c sz_success_k , @c sz_unexpected_dimensions_k for no slots, a stride under them or
 *      ranks past one block's shared memory, @c sz_device_memory_mismatch_k when the engine, the
 *      sketches or the texts are memory the device of @p stream cannot reach, or
 *      @c sz_device_code_mismatch_k when the launch itself is refused.
 *  @note Enqueues on @p stream and returns; the caller joins it before reading @p sketches .
 */
STRINGZILLA_INLINE sz_status_t sz_ngrams_sketches_scoped_rocm_(sz_ngrams_engine_t *engine,
                                                               sz_strings_view_t const *texts, sz_size_t slots,
                                                               sz_u8_t *sketches, sz_size_t sketches_stride,
                                                               void const *kernel, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_ngrams_sketches_rocm_(engine, texts, slots, sketches, sketches_stride, kernel, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

#if STRINGZILLA_TARGET_ROCM

static __global__ void __launch_bounds__(sz_ngrams_candidates_per_block_rocm_k)
    sz_ngrams_overlaps_rocm_kernel_(sz_ngrams_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches,
                                    sz_size_t matches_query_stride, sz_size_t matches_candidate_stride,
                                    sz_size_t staged_slots_count, sz_size_t hit_words_per_warp, sz_u32_t misses_bound) {
    sz_simt_group_t const group = {32,
                                   threadIdx.x & 31u,
                                   sz_lanes_any_rocm_,
                                   sz_lanes_ballot_rocm_,
                                   sz_lanes_first_rocm_,
                                   sz_lanes_below_rocm_,
                                   sz_shuffle_index_rocm_,
                                   sz_shuffle_up_rocm_,
                                   sz_u32_lanes_prefix_sum_rocm_,
                                   sz_i32_lanes_prefix_min_rocm_,
                                   sz_u64_lanes_sum_rocm_,
                                   sz_group_sync_rocm_};
    sz_ngrams_overlaps_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                             staged_slots_count, hit_words_per_warp, misses_bound, &group);
}

static __global__ void __launch_bounds__(sz_ngrams_sketch_threads_per_block_rocm_k)
    sz_ngrams_sketches_rocm_kernel_(sz_size_t const *widths, sz_size_t widths_count, sz_strings_view_t texts,
                                    sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride) {
    sz_ngrams_sketches_simt_(widths, widths_count, texts, slots, sketches, sketches_stride);
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                       sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                       sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                       sz_stream_t stream) {
    return sz_ngrams_engine_init_scoped_rocm_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                              (void const *)sz_ngrams_overlaps_rocm_kernel_, sz_cap_rocm_k, allocator,
                                              stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_ngrams_overlaps_scoped_rocm_(engine, candidates, radius, matches, matches_query_stride,
                                           matches_candidate_stride, (void const *)sz_ngrams_overlaps_rocm_kernel_,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_rocm(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                    sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                    sz_stream_t stream) {
    return sz_ngrams_sketches_scoped_rocm_(engine, texts, slots, sketches, sketches_stride,
                                           (void const *)sz_ngrams_sketches_rocm_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_ROCM

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_NGRAMS_ROCM_CUH_
