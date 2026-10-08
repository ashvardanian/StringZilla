/**
 *  @file include/stringzilla/substrings/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of multi-pattern search: the arena a round runs out of, the launches that
 *      walk, cover, rewrite and score, and the @c _cuda exports, over the kernels of
 *      `substrings/simt.cuh`.
 *
 *  No compute verb joins the stream. Every size a launch needs is either fixed when the engine is
 *  built - the chunk and match budgets - or derived on the device from one the host never sees, and
 *  what a round discovered reaches the caller through @c sz_substrings_report_t after the caller's
 *  own join. When a round outruns its match budget the writing walk, the cover, the compaction and
 *  the boundaries kernel each retire at their first instruction, so an output is left untouched
 *  rather than truncated.
 *
 *  Everything but the exports serves the Hopper tier too, whose BM25 scoring launches in clusters,
 *  so a batch of fewer haystacks than resident blocks walks each of them with a whole cluster
 *  rather than idling the device.
 *
 *  @sa include/stringzilla/substrings/simt.cuh
 *  @sa include/stringzilla/substrings/hopper.cuh
 *  @sa include/stringzilla/substrings/rocm.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_CUDA_CUH_
#define STRINGZILLA_SUBSTRINGS_CUDA_CUH_

#include "stringzilla/cuda.cuh"
#include "stringzilla/substrings/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Host Plumbing

/** Blocks one launch covers the device with, per multiprocessor, when the work is grid-strided. */
enum { sz_substrings_blocks_per_multiprocessor_cuda_k = 8 };

/** Blocks a BM25 cluster spans on Hopper's tier, which is the portable ceiling. */
enum { sz_substrings_bm25_cluster_blocks_cuda_k = 8 };

/** Whether the caller reads the emitted matches, or only the boundaries the sizing
 *  walk already scanned. */
typedef enum sz_substrings_matches_cuda_t {

    /** Counting under an overlapping policy: the scanned chunk slots are the whole answer. */
    sz_substrings_matches_unneeded_cuda_k = 0,

    /** Finding, rewriting, or any cover: the list has to exist before anything can read
     *  or thin it. */
    sz_substrings_matches_needed_cuda_k = 1,
} sz_substrings_matches_cuda_t;

/** The walk kernel of @p engine's case sensitivity. */
STRINGZILLA_INLINE void const *sz_substrings_walk_kernel_cuda_(sz_substrings_engine_t const *engine) {
    return engine->case_sensitivity == sz_substrings_uncased_k ? (void const *)sz_substrings_walk_uncased_simt_kernel_
                                                               : (void const *)sz_substrings_walk_cased_simt_kernel_;
}

/** Words of the acceptance bitmap, one bit per double-array slot. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_accepts_words_cuda_(sz_substrings_engine_t const *engine) {
    return sz_size_divide_round_up(engine->slots_count, 32);
}

/** Multiprocessors on the current device, or one when the driver will not say, since chunk widths
 *  divide by what this scales. */
STRINGZILLA_INLINE sz_size_t sz_substrings_multiprocessors_cuda_(void) {
    sz_size_t const multiprocessors = sz_device_multiprocessors_cuda_();
    return sz_max_of_two(multiprocessors, (sz_size_t)1);
}

/** Blocks of @p kernel this device holds resident per multiprocessor at @p shared_bytes
 *  of dynamic shared, and never fewer than one. */
STRINGZILLA_INLINE sz_size_t sz_substrings_resident_blocks_cuda_(void const *kernel, sz_size_t shared_bytes) {
    sz_size_t const blocks = sz_resident_blocks_cuda_(kernel, sz_substrings_threads_per_block_simt_k, shared_bytes);
    return sz_max_of_two(blocks, (sz_size_t)1);
}

/** Threads the walk keeps resident across the whole device, which is what a chunk width
 *  is derived from. */
STRINGZILLA_INLINE sz_size_t sz_substrings_resident_threads_cuda_(void const *kernel, sz_size_t shared_bytes) {
    return sz_substrings_multiprocessors_cuda_() * sz_substrings_resident_blocks_cuda_(kernel, shared_bytes) *
           sz_substrings_threads_per_block_simt_k;
}

/**
 *  @brief Grid for a grid-strided kernel over @p items, covering the device without
 *      exceeding the work.
 *
 *  Capped at what @p kernel actually keeps resident rather than at a fixed
 *  blocks-per-multiprocessor guess, since a kernel's residency moves with its registers and its
 *  dynamic shared memory.
 */
STRINGZILLA_INLINE unsigned sz_substrings_grid_for_cuda_(void const *kernel, sz_size_t shared_bytes, sz_size_t items) {
    sz_size_t blocks = sz_size_divide_round_up(items, sz_substrings_threads_per_block_simt_k);
    sz_size_t const covering = sz_substrings_multiprocessors_cuda_() *
                               sz_substrings_resident_blocks_cuda_(kernel, shared_bytes);
    if (blocks == 0) blocks = 1;
    return (unsigned)sz_min_of_two(blocks, covering);
}

/** Grid for the flat helper kernels, none of which takes dynamic shared memory. */
STRINGZILLA_INLINE unsigned sz_substrings_grid_cuda_(sz_size_t items) {
    sz_size_t blocks = sz_size_divide_round_up(items, sz_substrings_threads_per_block_simt_k);
    sz_size_t const covering = sz_substrings_multiprocessors_cuda_() * sz_substrings_blocks_per_multiprocessor_cuda_k;
    if (blocks == 0) blocks = 1;
    return (unsigned)sz_min_of_two(blocks, covering);
}

/** Launches @p kernel over @p blocks blocks of the tier's fixed block size, on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_substrings_launch_cuda_(void const *kernel, unsigned blocks, void **arguments,
                                                          sz_size_t shared_bytes, sz_stream_t stream) {
    dim3 grid, block;
    grid.x = blocks, grid.y = 1, grid.z = 1;
    block.x = sz_substrings_threads_per_block_simt_k, block.y = 1, block.z = 1;
    return sz_launch_cuda_(kernel, grid, block, arguments, shared_bytes, stream);
}

/** Tiles the tier's fixed block size covers @p count elements in. */
STRINGZILLA_INLINE sz_size_t sz_substrings_tiles_cuda_(sz_size_t count) {
    return sz_size_divide_round_up(count, sz_substrings_threads_per_block_simt_k);
}

/**
 *  @brief Turns @p values into its own exclusive prefix sum, leaving the grand total
 *      in @c values[count].
 *  @param[in] live How many leading values the device wrote, scanned with one more slot for their
 *      total, or @c STRINGZILLA_NULL to scan all @p count; the grid covers @p count either way.
 *  @param[in] scan_chain Zeroed slot the tiles chain their totals through, zeroed again on return.
 *
 *  One launch: each tile sums itself, waits for the tiles before it to chain theirs, then scans.
 */
STRINGZILLA_INLINE sz_status_t sz_substrings_scan_cuda_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                                        sz_size_t *scan_chain, sz_stream_t stream) {
    sz_size_t const tiles = sz_min_of_two(sz_substrings_tiles_cuda_(count), (sz_size_t)sz_chain_tiles_max_k);
    sz_size_t counted = count;
    sz_size_t elements_per_tile;
    void *arguments[5];
    if (!count) return sz_success_k;
    elements_per_tile = sz_size_divide_round_up(count, tiles);
    arguments[0] = &values, arguments[1] = &counted, arguments[2] = &live, arguments[3] = &elements_per_tile;
    arguments[4] = &scan_chain;
    return sz_substrings_launch_cuda_((void const *)sz_substrings_scan_simt_kernel_, (unsigned)tiles, arguments, 0,
                                      stream);
}

/** Scans the sizing walk's @p chunk_slots as @ref sz_substrings_scan_cuda_ does, its last tile then
 *  publishing into @p report what the walk found and whether it fit @p matches_budget. */
STRINGZILLA_INLINE sz_status_t sz_substrings_scan_sized_cuda_(sz_size_t *chunk_slots, sz_size_t count,
                                                              sz_size_t *scan_chain, sz_size_t matches_budget,
                                                              sz_bool_t emitting, sz_substrings_report_t *report,
                                                              sz_stream_t stream) {
    sz_size_t const tiles = sz_min_of_two(sz_substrings_tiles_cuda_(count), (sz_size_t)sz_chain_tiles_max_k);
    sz_size_t elements_per_tile = sz_size_divide_round_up(count, tiles);
    void *arguments[] = {&chunk_slots, &count, &elements_per_tile, &scan_chain, &matches_budget, &emitting, &report};
    return sz_substrings_launch_cuda_((void const *)sz_substrings_scan_sized_simt_kernel_, (unsigned)tiles, arguments,
                                      0, stream);
}

/** How many hot rows a kernel may stage in shared memory. */
typedef enum sz_substrings_staging_cuda_t {

    /** All of the hot tier or none of it: the walk pays no bounds test for a prefix
     *  most steps miss. */
    sz_substrings_stage_whole_cuda_k = 0,

    /** The longest prefix that fits beside the reservation, for a kernel that shares
     *  its memory anyway. */
    sz_substrings_stage_prefix_cuda_k = 1,
} sz_substrings_staging_cuda_t;

/**
 *  @brief Hot rows one block of @p kernel stages beside @p reserved_bytes of its own
 *      dynamic shared memory.
 *
 *  Rows are staged only while they displace no resident block: the walk is latency-bound, so
 *  warps are worth more than rows. The ceiling is the block's default, since raising it
 *  needs @c cudaFuncSetAttribute per kernel, and a launch asking for more than the default
 *  is rejected outright.
 */
STRINGZILLA_INLINE sz_u32_t sz_substrings_staged_rows_cuda_(sz_substrings_engine_t const *engine, void const *kernel,
                                                            sz_size_t reserved_bytes,
                                                            sz_substrings_staging_cuda_t staging) {
    sz_size_t const row_bytes = engine->classes_count * sizeof(sz_u32_t);
    sz_size_t const unstaged_blocks = sz_substrings_resident_blocks_cuda_(kernel, reserved_bytes);
    sz_size_t const ceiling = sz_device_shared_bytes_per_block_cuda_();
    sz_size_t fitting, low, high;
    if (!engine->hot_count) return 0;
    if (reserved_bytes >= ceiling) return 0;
    fitting = sz_min_of_two((ceiling - reserved_bytes) / row_bytes, (sz_size_t)engine->hot_count);
    if (staging == sz_substrings_stage_whole_cuda_k) {
        if (fitting < engine->hot_count) return 0;
        return sz_substrings_resident_blocks_cuda_(kernel, reserved_bytes + fitting * row_bytes) < unstaged_blocks
                   ? 0
                   : engine->hot_count;
    }
    // The longest prefix keeping the block count, found by bisection since residency falls monotonically.
    low = 0, high = fitting;
    while (low < high) {
        sz_size_t const middle = low + sz_size_divide_round_up(high - low, 2);
        if (sz_substrings_resident_blocks_cuda_(kernel, reserved_bytes + middle * row_bytes) < unstaged_blocks)
            high = middle - 1;
        else low = middle;
    }
    return (sz_u32_t)low;
}

/** Hot rows this tier's walk stages, which fixes both its shared memory and its residency. */
STRINGZILLA_INLINE sz_u32_t sz_substrings_walk_rows_cuda_(sz_substrings_engine_t const *engine) {
    return sz_substrings_staged_rows_cuda_(engine, sz_substrings_walk_kernel_cuda_(engine), 0,
                                           sz_substrings_stage_whole_cuda_k);
}

/**
 *  @brief Global tally rows a hashed BM25 launch may spill into, which is zero for a direct one.
 *
 *  Occupancy is measured without the shared memory the launch will actually reserve, so the
 *  count is an upper bound on the blocks any later round can run - the one property an arena
 *  sized once needs.
 */
STRINGZILLA_INLINE sz_size_t sz_substrings_bm25_rows_cuda_(sz_substrings_engine_t const *engine) {
    if (engine->needles_count <= (sz_u32_t)sz_substrings_tally_slots_simt_k) return 0;
    return sz_substrings_multiprocessors_cuda_() *
           sz_substrings_resident_blocks_cuda_((void const *)sz_substrings_bm25_simt_kernel_, 0);
}

/**
 *  @brief Byte offsets of the one arena every device round runs out of.
 *
 *  Everything past the boundaries is sized by the engine's budgets rather than by what a round
 *  discovers, so no launch waits on a count to reach the host first. Those two budget-sized
 *  halves are the whole reason the three readbacks that used to feed host allocations and grid
 *  dimensions are gone.
 */
typedef struct sz_substrings_arena_cuda_t {

    /** Offset of the round's report, which is the only thing a caller reads after its own join. */
    sz_size_t report;

    /** Offset of the @b [haystacks + 1] exclusive chunk boundaries, per haystack. */
    sz_size_t chunk_offsets;

    /** Offset of the @b [haystacks + 1] boundaries of the reported matches, per haystack. */
    sz_size_t haystack_offsets;

    /** Offset of the @b [slots_count] per-chunk counts, which the scan turns
     *  into output offsets. */
    sz_size_t chunk_slots;

    /** Offset of the one slot every scan chains its tile totals through. */
    sz_size_t scan_chain;

    /** Offset of the @b [matches_budget] emitted matches, before any cover thins them. */
    sz_size_t emitted;

    /** Offset of the cover's survivors, equal to @c emitted when no cover runs. */
    sz_size_t reported;

    /** Offset of the @b [matches_budget + 1] scanned keep flags, unallocated without a cover. */
    sz_size_t keep_offsets;

    /** Offset of the @b [matches_budget] rewrite drifts, unallocated without a cover. */
    sz_size_t gap_offsets;

    /** Offset of the hashed BM25 tally rows, unallocated for a vocabulary a block's
     *  own table holds. */
    sz_size_t overflow_rows;

    /** Entries of @c chunk_slots, which bounds the chunk count by construction rather
     *  than by hope. */
    sz_size_t slots_count;

    /** Rows at @c overflow_rows, which caps how many blocks a hashed BM25 launch may run. */
    sz_size_t overflow_count;

    /** Bytes the whole arena takes. */
    sz_size_t total;
} sz_substrings_arena_cuda_t;

/** Lays the device arena out for one round over @p haystacks_count texts. */
STRINGZILLA_INLINE sz_substrings_arena_cuda_t sz_substrings_arena_cuda_(sz_substrings_engine_t const *engine,
                                                                        sz_size_t haystacks_count) {
    sz_size_t const boundaries = haystacks_count + 1;
    sz_size_t const matches = engine->matches_budget;
    sz_size_t const match_bytes = matches * sizeof(sz_substrings_match_t);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_substrings_arena_cuda_t arena;
    // A chunk is at least the corpus over the budget wide, so the corpus contributes at most `chunk_budget`
    // chunks and each haystack's own remainder at most one more.
    arena.slots_count = engine->chunk_budget + haystacks_count + 1;
    arena.overflow_count = sz_substrings_bm25_rows_cuda_(engine);
    arena.report = 0;
    arena.chunk_offsets = arena.report + sizeof(sz_substrings_report_t);
    arena.haystack_offsets = arena.chunk_offsets + boundaries * sizeof(sz_size_t);
    arena.chunk_slots = arena.haystack_offsets + boundaries * sizeof(sz_size_t);
    arena.scan_chain = arena.chunk_slots + arena.slots_count * sizeof(sz_size_t);
    arena.emitted = arena.scan_chain + sizeof(sz_size_t);
    arena.reported = arena.emitted + match_bytes;
    arena.keep_offsets = arena.reported + (covering ? match_bytes : 0);
    arena.gap_offsets = arena.keep_offsets + (covering ? (matches + 1) * sizeof(sz_size_t) : 0);
    arena.overflow_rows = arena.gap_offsets + (covering ? matches * sizeof(sz_size_t) : 0);
    arena.total = arena.overflow_rows + arena.overflow_count * engine->needles_count * sizeof(sz_u32_t);
    return arena;
}

/** What one round's launches read out of the arena, every pointer of it device-resident. */
typedef struct sz_substrings_round_cuda_t {

    /** The @b [haystacks + 1] exclusive chunk boundaries. */
    sz_size_t *chunk_offsets;

    /** The @b [haystacks + 1] boundaries of the reported matches, which a verb may redirect
     *  to its output. */
    sz_size_t *haystack_offsets;

    /** One slot per chunk: its match count from the sizing pass, then its
     *  exclusive output offset. */
    sz_size_t *chunk_slots;

    /** The slot every scan chains its tile totals through, zero between scans. */
    sz_size_t *scan_chain;

    /** Every emitted match, before any cover thins them. */
    sz_substrings_match_t *emitted;

    /** The matches a cover kept, which is @c emitted itself when no cover ran. */
    sz_substrings_match_t *reported;

    /** The scanned keep flags, or @c STRINGZILLA_NULL when every match is reported. */
    sz_size_t *keep_offsets;

    /** One rewrite drift per reported match, or @c STRINGZILLA_NULL when no rewrite can run. */
    sz_size_t *gap_offsets;

    /** Entries of @c chunk_slots, so the scan's grand total is its last one. */
    sz_size_t slots_count;
} sz_substrings_round_cuda_t;

/** Binds one round's pointers onto the engine's arena, which a compute verb does
 *  before it launches. */
STRINGZILLA_INLINE void sz_substrings_round_bind_cuda_(sz_substrings_engine_t const *engine, sz_size_t haystacks_count,
                                                       sz_substrings_round_cuda_t *round) {
    sz_substrings_arena_cuda_t const arena = sz_substrings_arena_cuda_(engine, haystacks_count);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    round->chunk_offsets = (sz_size_t *)(block + arena.chunk_offsets);
    round->haystack_offsets = (sz_size_t *)(block + arena.haystack_offsets);
    round->chunk_slots = (sz_size_t *)(block + arena.chunk_slots);
    round->scan_chain = (sz_size_t *)(block + arena.scan_chain);
    round->emitted = (sz_substrings_match_t *)(block + arena.emitted);
    round->reported = covering ? (sz_substrings_match_t *)(block + arena.reported) : round->emitted;
    round->keep_offsets = covering ? (sz_size_t *)(block + arena.keep_offsets) : STRINGZILLA_NULL;
    round->gap_offsets = covering ? (sz_size_t *)(block + arena.gap_offsets) : STRINGZILLA_NULL;
    round->slots_count = arena.slots_count;
}

/** Takes the one arena every round of this engine runs out of, sized for its haystacks budget, so
 *  no compute verb ever allocates. */
STRINGZILLA_INLINE sz_status_t sz_substrings_arena_reserve_cuda_(sz_substrings_engine_t *engine, sz_stream_t stream) {
    sz_substrings_arena_cuda_t const arena = sz_substrings_arena_cuda_(engine, engine->haystacks_budget);
    sz_allocator_t *const allocator = &engine->allocator;
    void *const block = allocator->allocate(arena.total, allocator->handle, stream);
    if (!block) return sz_bad_alloc_k;
    if (!sz_memory_accessible_cuda_(block)) {
        allocator->free(block, arena.total, allocator->handle, stream);
        return sz_device_memory_mismatch_k;
    }
    engine->scratch = block, engine->scratch_bytes = arena.total;
    engine->report = (sz_substrings_report_t *)((sz_ptr_t)block + arena.report);
    return sz_success_k;
}

/** Zeroes everything a round reads before it writes: the report, the counters
 *  and the boundaries. */
STRINGZILLA_INLINE sz_status_t sz_substrings_arena_clear_cuda_(sz_substrings_engine_t const *engine,
                                                               sz_size_t haystacks_count, sz_stream_t stream) {
    sz_substrings_arena_cuda_t const arena = sz_substrings_arena_cuda_(engine, haystacks_count);
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    // The head of the arena only, so no round pays a memset proportional to a budget it did not spend.
    return sz_fill_cuda_(block + arena.report, arena.emitted - arena.report, 0, stream);
}

/**
 *  @brief Walks every chunk and settles the cover, leaving @p round holding what the verbs read.
 *
 *  Every size this needs is either fixed at construction or derived on the device from one the host
 *  never sees, so the whole pass enqueues without a single join: the corpus total feeds a chunk
 *  width the device computes, the chunk count is bounded by the engine's own budget, and the
 *  emitted count reaches the later launches through `engine->report`, not through a copy back.
 *
 *  @param[in] wanted Whether the caller reads the matches, since an overlapping count answers from
 *      the boundaries its sizing walk already scanned and never touches the match arena at all.
 *  @param[out] haystack_offsets Where the per-haystack boundaries land, or @c STRINGZILLA_NULL to
 *      leave them in the arena, as every verb but @ref sz_substrings_find_scoped_cuda_ does.
 */
STRINGZILLA_INLINE sz_status_t sz_substrings_walk_cuda_(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_substrings_matches_cuda_t wanted,
                                                        sz_size_t *haystack_offsets, sz_substrings_round_cuda_t *round,
                                                        sz_stream_t stream) {
    sz_u32_t staged_rows = sz_substrings_walk_rows_cuda_(engine);
    sz_size_t const staged_bytes = (sz_size_t)staged_rows * engine->classes_count * sizeof(sz_u32_t);
    sz_size_t boundaries = haystacks->count + 1;
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    // A cover is decided between matches, so counting one costs what finding one costs; only an
    // overlapping count can answer from the boundaries its sizing walk already scanned.
    sz_bool_t emitting = (sz_bool_t)(covering || wanted == sz_substrings_matches_needed_cuda_k);
    sz_size_t matches_budget = engine->matches_budget;
    sz_size_t longest = engine->max_source_match_bytes;
    sz_substrings_overlap_policy_t overlap_policy = engine->overlap_policy;
    sz_substrings_report_t *report;
    sz_sequence_t launched_haystacks = *haystacks;
    sz_substrings_engine_t launched_engine;
    sz_substrings_gpu_pass_t walk_pass = sz_substrings_gpu_sizing_k;
    sz_size_t *emitted_at;
    void *arguments[8];
    sz_status_t status;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;
    sz_substrings_round_bind_cuda_(engine, haystacks->count, round);
    if (haystack_offsets) round->haystack_offsets = haystack_offsets;
    report = engine->report, launched_engine = *engine;
    emitted_at = round->chunk_slots + round->slots_count - 1;
    status = sz_substrings_arena_clear_cuda_(engine, haystacks->count, stream);

    // Round one: how many chunks each haystack owns, scanned into the range that haystack's chunks take.
    if (status == sz_success_k) {
        arguments[0] = &launched_engine, arguments[1] = &launched_haystacks, arguments[2] = &round->chunk_offsets;
        status = sz_substrings_launch_cuda_((void const *)sz_substrings_chunk_counts_simt_kernel_,
                                            sz_substrings_grid_cuda_(haystacks->count), arguments, 0, stream);
    }
    if (status == sz_success_k)
        status = sz_substrings_scan_cuda_(round->chunk_offsets, boundaries, STRINGZILLA_NULL, round->scan_chain,
                                          stream);

    // Round two: the sizing walk, launched against the chunk budget rather than a discovered chunk count.
    if (status == sz_success_k) {
        arguments[0] = &launched_engine, arguments[1] = &staged_rows, arguments[2] = &launched_haystacks;
        arguments[3] = &round->chunk_offsets, arguments[4] = &report, arguments[5] = &round->chunk_slots;
        arguments[6] = &round->emitted, arguments[7] = &walk_pass;
        status = sz_substrings_launch_cuda_(
            sz_substrings_walk_kernel_cuda_(engine),
            sz_substrings_grid_for_cuda_(sz_substrings_walk_kernel_cuda_(engine), staged_bytes, round->slots_count),
            arguments, staged_bytes, stream);
    }
    if (status == sz_success_k)
        status = sz_substrings_scan_sized_cuda_(round->chunk_slots, round->slots_count, round->scan_chain,
                                                matches_budget, emitting, report, stream);

    // Round three: the writing walk, the cover over what it wrote, and the per-haystack boundaries both
    // feed. Each retires at its first instruction when the sizing walk outran the budget.
    if (status == sz_success_k && emitting) {
        walk_pass = sz_substrings_gpu_writing_k;
        arguments[0] = &launched_engine, arguments[1] = &staged_rows, arguments[2] = &launched_haystacks;
        arguments[3] = &round->chunk_offsets, arguments[4] = &report, arguments[5] = &round->chunk_slots;
        arguments[6] = &round->emitted, arguments[7] = &walk_pass;
        status = sz_substrings_launch_cuda_(
            sz_substrings_walk_kernel_cuda_(engine),
            sz_substrings_grid_for_cuda_(sz_substrings_walk_kernel_cuda_(engine), staged_bytes, round->slots_count),
            arguments, staged_bytes, stream);
    }
    if (status == sz_success_k && covering) {
        arguments[0] = &round->emitted, arguments[1] = &report, arguments[2] = &longest;
        arguments[3] = &overlap_policy, arguments[4] = &round->keep_offsets;
        status = sz_substrings_launch_cuda_((void const *)sz_substrings_cover_simt_kernel_,
                                            sz_substrings_grid_cuda_(matches_budget), arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_substrings_scan_cuda_(round->keep_offsets, matches_budget + 1, emitted_at, round->scan_chain,
                                              stream);
        if (status == sz_success_k) {
            arguments[0] = &round->emitted, arguments[1] = &report;
            arguments[2] = &round->keep_offsets, arguments[3] = &round->reported;
            status = sz_substrings_launch_cuda_((void const *)sz_substrings_compact_simt_kernel_,
                                                sz_substrings_grid_cuda_(matches_budget), arguments, 0, stream);
        }
    }
    if (status == sz_success_k) {
        sz_size_t *keep_offsets = round->keep_offsets;
        arguments[0] = &round->chunk_offsets, arguments[1] = &round->chunk_slots, arguments[2] = &keep_offsets;
        arguments[3] = &report, arguments[4] = &round->haystack_offsets, arguments[5] = &boundaries;
        status = sz_substrings_launch_cuda_((void const *)sz_substrings_haystack_offsets_simt_kernel_,
                                            sz_substrings_grid_cuda_(boundaries), arguments, 0, stream);
    }
    return status;
}

/** Whether every argument the device verbs read or write is one a kernel can address. */
STRINGZILLA_INLINE sz_bool_t sz_substrings_resident_cuda_(sz_substrings_engine_t const *engine,
                                                          sz_sequence_t const *haystacks) {
    // An array the kernel dereferences, rather than the owning handle it never touches, so a caller
    // holding a borrowed view of a resident engine is not refused for a null owner.
    if (!sz_memory_accessible_cuda_(engine->base)) return sz_false_k;
    return haystacks->get_start == sz_sequence_tape_start && haystacks->get_length == sz_sequence_tape_length &&
                   sz_memory_accessible_cuda_(haystacks->handle)
               ? sz_true_k
               : sz_false_k;
}

#pragma endregion Host Plumbing

#pragma region Construction

/** Compiles @p needles on the device the caller already made current, for the tier
 *  @p capability names, which is every step of @ref sz_substrings_engine_init_scoped_cuda_ but the
 *  device scope; the host compiles the vocabulary in place, so nothing here is scheduled. */
STRINGZILLA_INLINE sz_status_t sz_substrings_engine_init_cuda_(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_capability_t capability, sz_allocator_t *allocator, sz_stream_t stream) {
    sz_allocator_t unified;
    sz_size_t staged_bytes;
    sz_status_t status;
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;
    if (!allocator) {
        sz_allocator_init_unified_cuda_(&unified);
        allocator = &unified;
    }
    if (!matches_budget) matches_budget = (sz_size_t)sz_substrings_gpu_matches_budget_default_k;
    if (!haystacks_budget) haystacks_budget = (sz_size_t)sz_substrings_gpu_haystacks_budget_default_k;
    status = sz_substrings_engine_compile_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                           capability, allocator, stream, engine);
    if (status != sz_success_k) return status;
    // The host builder writes the block in place, so a device-only allocation cannot serve as the vocabulary.
    if (!sz_memory_accessible_cuda_(engine->memory)) {
        sz_substrings_engine_free_(engine, stream);
        return sz_device_memory_mismatch_k;
    }
    staged_bytes = (sz_size_t)sz_substrings_walk_rows_cuda_(engine) * engine->classes_count * sizeof(sz_u32_t);
    engine->chunk_budget = sz_substrings_resident_threads_cuda_(sz_substrings_walk_kernel_cuda_(engine), staged_bytes) *
                           sz_substrings_chunks_per_thread_simt_k;
    engine->haystacks_budget = haystacks_budget;
    status = sz_substrings_arena_reserve_cuda_(engine, stream);
    if (status != sz_success_k) sz_substrings_engine_free_(engine, stream);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_substrings_engine_init_scoped_cuda_(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_capability_t capability, sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_substrings_engine_init_cuda_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                             matches_budget, haystacks_budget, capability, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#pragma endregion Construction

#pragma region Backends

STRINGZILLA_INLINE sz_status_t sz_substrings_counts_cuda_(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_stream_t stream) {
    sz_substrings_round_cuda_t round;
    sz_size_t haystacks_count = haystacks->count;
    void *arguments[4];
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_resident_cuda_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(counts)) return sz_device_memory_mismatch_k;

    status = sz_substrings_walk_cuda_(engine, haystacks, sz_substrings_matches_unneeded_cuda_k, STRINGZILLA_NULL,
                                      &round, stream);
    if (status != sz_success_k) return status;

    arguments[0] = &round.haystack_offsets, arguments[1] = &counts, arguments[2] = &counts_stride;
    arguments[3] = &haystacks_count;
    return sz_substrings_launch_cuda_((void const *)sz_substrings_counts_simt_kernel_,
                                      sz_substrings_grid_cuda_(haystacks_count), arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_substrings_find_cuda_(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                        sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_substrings_round_cuda_t round;
    sz_substrings_report_t *report;
    void *arguments[4];
    sz_status_t status;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_resident_cuda_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(matches_offsets)) return sz_device_memory_mismatch_k;
    if (matches_capacity && !sz_memory_accessible_cuda_(matches)) return sz_device_memory_mismatch_k;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;
    // The offsets kernel retires when the matches did not fit, so the caller's array is zeroed rather than
    // left holding whatever it held before.
    status = sz_fill_cuda_(matches_offsets, (haystacks->count + 1) * sizeof(sz_size_t), 0, stream);
    if (status != sz_success_k) return status;

    // The boundaries land straight in the caller's array, which is the same shape a rewrite already takes.
    status = sz_substrings_walk_cuda_(engine, haystacks, sz_substrings_matches_needed_cuda_k, matches_offsets, &round,
                                      stream);
    if (status != sz_success_k) return status;

    // The survivor count lives on the device, so the clip at the capacity is a kernel rather than a copy.
    report = engine->report;
    arguments[0] = &round.reported, arguments[1] = &report, arguments[2] = &matches, arguments[3] = &matches_capacity;
    status = sz_substrings_launch_cuda_((void const *)sz_substrings_store_matches_simt_kernel_,
                                        sz_substrings_grid_cuda_(matches_capacity), arguments, 0, stream);
    if (status != sz_success_k) return status;
    arguments[0] = &report, arguments[1] = &matches_capacity;
    return sz_substrings_launch_cuda_((void const *)sz_substrings_stored_simt_kernel_, 1, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_substrings_replace_cuda_(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_sequence_t const *replacements, sz_ptr_t target,
                                                           sz_size_t target_capacity, sz_size_t *offsets,
                                                           sz_stream_t stream) {
    sz_substrings_round_cuda_t round;
    sz_substrings_report_t *report;
    sz_sequence_t launched_haystacks, launched_replacements;
    sz_size_t boundaries = haystacks->count + 1;
    sz_size_t target_ceiling = target_capacity;
    sz_size_t *rewritten_at;
    void *arguments[8];
    sz_status_t status;
    // A substitution over matches that share bytes is not a function, so there is no cover to apply.
    if (engine->overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != engine->needles_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_resident_cuda_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (replacements->get_start != sz_sequence_tape_start || replacements->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_cuda_(replacements->handle))
        return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(offsets)) return sz_device_memory_mismatch_k;
    if (target_capacity && !sz_memory_accessible_cuda_(target)) return sz_device_memory_mismatch_k;

    status = sz_substrings_walk_cuda_(engine, haystacks, sz_substrings_matches_needed_cuda_k, STRINGZILLA_NULL, &round,
                                      stream);
    if (status != sz_success_k) return status;
    // The offsets kernel retires when the matches did not fit, so the caller's array is zeroed rather than
    // left holding whatever it held before.
    status = sz_fill_cuda_(offsets, boundaries * sizeof(sz_size_t), 0, stream);
    if (status != sz_success_k) return status;

    report = engine->report;
    rewritten_at = offsets + haystacks->count;
    launched_haystacks = *haystacks, launched_replacements = *replacements;

    // One block per haystack, so the drift scan a rewrite needs stays inside one block's carry.
    arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
    arguments[2] = &round.haystack_offsets, arguments[3] = &round.reported, arguments[4] = &report;
    arguments[5] = &round.gap_offsets, arguments[6] = &offsets;
    status = sz_substrings_launch_cuda_(
        (void const *)sz_substrings_rewrite_offsets_simt_kernel_,
        sz_substrings_grid_cuda_(haystacks->count * sz_substrings_threads_per_block_simt_k), arguments, 0, stream);
    if (status == sz_success_k)
        status = sz_substrings_scan_cuda_(offsets, boundaries, STRINGZILLA_NULL, round.scan_chain, stream);
    if (status == sz_success_k) {
        arguments[0] = &rewritten_at, arguments[1] = &target_ceiling, arguments[2] = &report;
        status = sz_substrings_launch_cuda_((void const *)sz_substrings_target_simt_kernel_, 1, arguments, 0, stream);
    }
    // The copy's grid comes from the caller's own capacity, which is the last host number a rewrite needs.
    if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
        arguments[2] = &round.haystack_offsets, arguments[3] = &round.reported, arguments[4] = &round.gap_offsets;
        arguments[5] = &offsets, arguments[6] = &report, arguments[7] = &target;
        status = sz_substrings_launch_cuda_((void const *)sz_substrings_rewrite_copy_simt_kernel_,
                                            sz_substrings_grid_cuda_(target_ceiling), arguments, 0, stream);
    }
    return status;
}

/**
 *  @brief Scores every haystack with a block of its own, or, given @p clustered, with a whole
 *      cluster of blocks where the batch holds too few haystacks to fill the device.
 *  @param[in] clustered The kernel walking one haystack per cluster, which Hopper's tier passes,
 *      or @c STRINGZILLA_NULL.
 */
STRINGZILLA_INLINE sz_status_t sz_substrings_bm25_scores_cuda_(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride,
    void const *clustered, sz_stream_t stream) {
    void const *kernel = (void const *)sz_substrings_bm25_simt_kernel_;
    sz_substrings_arena_cuda_t const arena = sz_substrings_arena_cuda_(engine, haystacks->count);
    sz_size_t const needles_count = engine->needles_count;
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(needles_count);
    sz_substrings_tally_layout_simt_t const layout = needles_count <= sz_substrings_tally_slots_simt_k
                                                         ? sz_substrings_tally_direct_simt_k
                                                         : sz_substrings_tally_hashed_simt_k;
    sz_size_t const table_bytes = (layout == sz_substrings_tally_hashed_simt_k ? 2 : 1) * table_slots *
                                  sizeof(sz_u32_t);
    sz_size_t const accepts_bytes = sz_substrings_accepts_words_cuda_(engine) * sizeof(sz_u32_t);
    sz_substrings_engine_t engine_copy;
    sz_sequence_t haystacks_copy = *haystacks;
    sz_substrings_bm25_t parameters_copy;
    sz_u32_t *overflow_rows = STRINGZILLA_NULL;
    sz_size_t reserved_bytes = table_bytes, shared_bytes, blocks, resident, clusters, cluster_blocks = 1;
    sz_u32_t staged_accepts_words = 0, staged_count;
    dim3 grid, block;
    void *arguments[10];
    sz_status_t status = sz_substrings_bm25_check(parameters, needle_weights);
    if (status != sz_success_k) return status;
    if (!scores_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_resident_cuda_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(scores)) return sz_device_memory_mismatch_k;
    if (needles_count && !sz_memory_accessible_cuda_(needle_weights)) return sz_device_memory_mismatch_k;
    if (document_lengths && !sz_memory_accessible_cuda_(document_lengths)) return sz_device_memory_mismatch_k;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;
    engine_copy = *engine;

    // The acceptance bitmap is read on every step, so it is staged whenever it fits without costing a block.
    if (table_bytes + accepts_bytes <= sz_device_shared_bytes_per_block_cuda_() &&
        sz_substrings_resident_blocks_cuda_(kernel, table_bytes + accepts_bytes) >=
            sz_substrings_resident_blocks_cuda_(kernel, table_bytes))
        staged_accepts_words = (sz_u32_t)(accepts_bytes / sizeof(sz_u32_t)), reserved_bytes += accepts_bytes;
    staged_count = sz_substrings_staged_rows_cuda_(engine, kernel, reserved_bytes, sz_substrings_stage_prefix_cuda_k);
    shared_bytes = reserved_bytes + (sz_size_t)staged_count * engine->classes_count * sizeof(sz_u32_t);
    resident = sz_substrings_multiprocessors_cuda_() * sz_substrings_resident_blocks_cuda_(kernel, shared_bytes);
    blocks = sz_min_of_two(haystacks->count, resident);
    // Fewer haystacks than resident blocks would idle the device at a block each, so where the tier
    // groups blocks, each haystack takes a whole cluster instead.
    clusters = clustered && haystacks->count < resident
                   ? sz_resident_clusters_cuda_(clustered, sz_substrings_threads_per_block_simt_k, shared_bytes,
                                                sz_substrings_bm25_cluster_blocks_cuda_k)
                   : 0;
    if (layout == sz_substrings_tally_hashed_simt_k)
        clusters = sz_min_of_two(clusters, arena.overflow_count / sz_substrings_bm25_cluster_blocks_cuda_k);
    if (clusters) {
        kernel = clustered, cluster_blocks = sz_substrings_bm25_cluster_blocks_cuda_k;
        blocks = sz_min_of_two(haystacks->count, clusters) * cluster_blocks;
    }

    // A hashed tally spills into one arena row per block, so the block count is capped by the rows the
    // arena was sized for rather than by what the device happens to have free at this moment.
    if (layout == sz_substrings_tally_hashed_simt_k) {
        sz_size_t const overflow_bytes = arena.overflow_count * needles_count * sizeof(sz_u32_t);
        if (cluster_blocks == 1) blocks = sz_max_of_two(sz_min_of_two(blocks, arena.overflow_count), (sz_size_t)1);
        overflow_rows = (sz_u32_t *)((sz_ptr_t)engine->scratch + arena.overflow_rows);
        status = sz_fill_cuda_(overflow_rows, overflow_bytes, 0, stream);
        if (status != sz_success_k) return status;
    }
    status = sz_fill_cuda_(engine->report, sizeof(sz_substrings_report_t), 0, stream);
    if (status != sz_success_k) return status;

    parameters_copy = *parameters;
    arguments[0] = &engine_copy, arguments[1] = &haystacks_copy, arguments[2] = &document_lengths;
    arguments[3] = &parameters_copy, arguments[4] = &needle_weights, arguments[5] = &overflow_rows;
    arguments[6] = &scores, arguments[7] = &scores_stride, arguments[8] = &staged_accepts_words;
    arguments[9] = &staged_count;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = sz_substrings_threads_per_block_simt_k, block.y = 1, block.z = 1;
    return sz_launch_clustered_cuda_(kernel, grid, block, arguments, shared_bytes, cluster_blocks, stream);
}

/*  Each launcher makes the engine's device current around its helper, and restores the caller's. */

STRINGZILLA_INLINE sz_status_t sz_substrings_counts_scoped_cuda_(sz_substrings_engine_t *engine,
                                                                 sz_sequence_t const *haystacks, sz_size_t *counts,
                                                                 sz_size_t counts_stride, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_substrings_counts_cuda_(engine, haystacks, counts, counts_stride, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_substrings_find_scoped_cuda_(sz_substrings_engine_t *engine,
                                                               sz_sequence_t const *haystacks,
                                                               sz_substrings_match_t *matches,
                                                               sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                               sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_substrings_find_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_substrings_replace_scoped_cuda_(sz_substrings_engine_t *engine,
                                                                  sz_sequence_t const *haystacks,
                                                                  sz_sequence_t const *replacements, sz_ptr_t target,
                                                                  sz_size_t target_capacity, sz_size_t *offsets,
                                                                  sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_substrings_replace_cuda_(engine, haystacks, replacements, target, target_capacity, offsets, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_substrings_bm25_scores_scoped_cuda_(
    sz_substrings_engine_t *engine, sz_sequence_t const *haystacks, sz_f32_t const *document_lengths,
    sz_substrings_bm25_t const *parameters, sz_f32_t const *needle_weights, sz_f32_t *scores, sz_size_t scores_stride,
    void const *clustered, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_substrings_bm25_scores_cuda_(engine, haystacks, document_lengths, parameters, needle_weights, scores,
                                             scores_stride, clustered, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#pragma endregion Backends

#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_substrings_engine_init_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                           sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_cuda_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, sz_cap_cuda_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    return sz_substrings_counts_scoped_cuda_(engine, haystacks, counts, counts_stride, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets, sz_stream_t stream) {
    return sz_substrings_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t target,
                                                       sz_size_t target_capacity, sz_size_t *offsets,
                                                       sz_stream_t stream) {
    return sz_substrings_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_cuda_(engine, haystacks, document_lengths, parameters, needle_weights,
                                                  scores, scores_stride, STRINGZILLA_NULL, stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBSTRINGS_CUDA_CUH_
