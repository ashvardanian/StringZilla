/**
 *  @file include/stringzilla/overlap/simt.cuh
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief CUDA backend for window overlap: a block's tile of candidates cut into chunks of equal
 *      bytes, one per thread, each width's window hash rolled through a chunk in one pass, and one
 *      prepared query's B-tree probed per block row.
 *
 *  The arithmetic is the serial tier's, reached from the device through `--expt-relaxed-constexpr`,
 *  so the scores are bit-identical rather than merely close. Integers, not doubles: both factors
 *  are below 2^32, so the product fits one @c u64 and the remainder by a constant lowers to a
 *  multiply-high - where an @c f64 reduction would run at the device's double-precision rate, a
 *  sixty-fourth of its single-precision one on consumer parts.
 *
 *  A chain of hashes is a dependent recurrence, but a window's hash depends only on its own bytes,
 *  so a candidate can be cut anywhere: a chunk primes itself from the widest window's bytes before
 *  its start and counts only the windows ending inside it. A block cuts its tile's bytes evenly
 *  across its threads, so a warp costs its share rather than its longest candidate, a long one is
 *  spread across as many threads as it holds shares, and the counts meet in shared memory before
 *  one thread per candidate scores it.
 *
 *  The query axis rides @c blockIdx.y, one tree per block, strided when a batch outruns a grid
 *  dimension. The text is therefore walked once per query rather than once per candidate - the
 *  probe is a @c levels deep descent with a sixteen-key node halved at every level against two
 *  operations of chain, so sharing it would buy about a percent and cost a per-thread match counter
 *  per query, which no register file holds.
 *
 *  On Blackwell a block done with its tile takes over one whose block has not started, keeping its
 *  staged tree when the query is the same.
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

/** Candidates one block's tile holds, and the threads it cuts their bytes across, when the device
 *  cannot be asked; measured flat from 32 to 512 and off a cliff at 1024, so this is the ceiling of
 *  the flat range rather than a tuned figure, and the widest tile the occupancy walk considers. */
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

/** How many of a node's sixteen ascending entries sit below the flipped @p key, up to fifteen, by
 *  halving: four dependent loads where a scan takes sixteen, and only a sixteenth entry below the
 *  key goes uncounted. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_simt_rank_(sz_u32_t const *node, sz_i32_t key) {
    sz_u32_t rank = (sz_i32_t)node[7] < key ? 8u : 0u;
    rank += (sz_i32_t)node[rank + 3] < key ? 4u : 0u;
    rank += (sz_i32_t)node[rank + 1] < key ? 2u : 0u;
    rank += (sz_i32_t)node[rank] < key ? 1u : 0u;
    return rank;
}

/** Whether the prepared query holds one raw @p window_hash, halving one node per level. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_simt_btree_probe_(sz_overlap_btree_t const *btree, sz_u32_t window_hash) {
    sz_i32_t const key = (sz_i32_t)(window_hash ^ (sz_u32_t)sz_overlap_sign_flip_k);
    sz_u32_t const last = sz_overlap_keys_per_node_k - 1;
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 < btree->levels; ++level) {
        sz_u32_t const *const separators = btree->nodes +
                                           (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k;
        sz_u32_t const rank = sz_overlap_simt_rank_(separators, key);
        node = node * sz_overlap_branches_per_node_k + rank + (rank == last && (sz_i32_t)separators[last] < key);
    }
    // The leaves lead the arena, so the last level's base is zero.
    sz_u32_t const *const leaf = btree->nodes + node * sz_overlap_keys_per_node_k;
    return leaf[sz_overlap_simt_rank_(leaf, key)] == (sz_u32_t)key;
}

/**
 *  @brief Counts the windows of one query's widths that end inside one chunk of a candidate, on one
 *      thread, and adds them to the candidate's shared counts.
 *
 *  A window's hash depends only on its own bytes, so a chunk primes itself from @p warm_up bytes
 *  before its own start and every window is counted by the one chunk it ends in. Each width rolls
 *  its own hash a byte at a time, `H' = H * 256 + in - out * 256^w`, the byte leaving the window
 *  read again from the text rather than kept: the hashes stay in registers, where a ring of prefix
 *  hashes indexed by width would take a stack frame, and each byte costs one reduction per width
 *  instead of three. The residues are the serial tier's, as both are exact modulo the same prime.
 *
 *  @p candidates ' accessors run here, on the device, once per chunk. Nothing is flattened for the
 *  launch, so a caller whose sequence is some other layout entirely - a tape, a column, an index
 *  into someone else's arena - needs no conversion.
 */
STRINGZILLA_DEVICE void sz_overlap_simt_chunk_(sz_overlap_engine_t const *engine, sz_overlap_btree_t const *btree,
                                               sz_sequence_t const *candidates, sz_size_t candidate,
                                               sz_size_t chunk_begin, sz_size_t chunk_bytes, sz_size_t warm_up,
                                               sz_u32_t *counts) {
    sz_cptr_t const text = candidates->get_start(candidates->handle, candidate);
    sz_size_t const length = candidates->get_length(candidates->handle, candidate);
    sz_size_t const walk_begin = chunk_begin > warm_up ? chunk_begin - warm_up : 0;
    sz_size_t const walk_bytes = sz_min_of_two(chunk_begin + chunk_bytes, length) - walk_begin;
    sz_size_t const warm_bytes = chunk_begin - walk_begin;
    sz_size_t const widths_count = engine->widths_count;
    sz_u8_t const *const walk = (sz_u8_t const *)text + walk_begin;

    // Unrolled to the ceiling and guarded, so the hashes and counters stay in registers.
    sz_u32_t hashes[sz_overlap_simt_widths_max_k], matches[sz_overlap_simt_widths_max_k];
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_simt_widths_max_k; ++index) hashes[index] = 0, matches[index] = 0;

    // Bytes before the walk count as zeros, so a window not yet full holds its own bytes' hash.
    for (sz_size_t ending = 1; ending <= walk_bytes; ++ending) {
        sz_u64_t const incoming = walk[ending - 1];
#pragma unroll
        for (sz_size_t index = 0; index != sz_overlap_simt_widths_max_k; ++index) {
            if (index >= widths_count) break;
            sz_size_t const width = engine->widths[index];
            if (!width) continue;
            sz_u64_t const outgoing = ending > width ? walk[ending - width - 1] : 0;
            sz_u64_t const rolled = (sz_u64_t)hashes[index] * 256 + incoming + (sz_u64_t)sz_overlap_modulus_k * 256 -
                                    outgoing * engine->powers[index];
            hashes[index] = (sz_u32_t)(rolled % (sz_u64_t)sz_overlap_modulus_k);
            if (ending <= warm_bytes || width > ending) continue;
            matches[index] += (sz_u32_t)sz_overlap_simt_btree_probe_(btree, hashes[index]);
        }
    }
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_simt_widths_max_k; ++index)
        if (index < widths_count && matches[index]) atomicAdd(counts + index, matches[index]);
}

/** Index of the last entry at or below @p value in an ascending array; zero when none is. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_simt_last_not_above_(sz_u32_t const *ascending, sz_size_t count,
                                                             sz_size_t value) {
    sz_size_t low = 0, high = count;
    while (low + 1 < high) {
        sz_size_t const middle = low + (high - low) / 2;
        if (ascending[middle] <= value) low = middle;
        else high = middle;
    }
    return low;
}

/**
 *  @brief Cuts one block's tile of candidates into chunks of about equal bytes, a share per thread,
 *      returning the bytes a chunk takes and leaving each candidate's first in @p chunk_offsets.
 *
 *  A thread per candidate costs a warp its longest one, and a single long candidate pins a single
 *  thread. Cutting the tile's bytes evenly across the block's threads instead bounds a thread's
 *  work by one share, however the lengths are spread, and a long candidate is split across as many
 *  threads as it holds shares.
 */
STRINGZILLA_DEVICE sz_size_t sz_overlap_simt_tile_chunks_(sz_sequence_t const *candidates, sz_size_t tile_first,
                                                          sz_size_t tile_count, sz_size_t warm_up,
                                                          unsigned long long *tile_bytes, sz_u32_t *chunk_offsets) {
    sz_size_t mine = 0, candidate;
    if (threadIdx.x == 0) *tile_bytes = 0;
    __syncthreads();
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        mine += candidates->get_length(candidates->handle, tile_first + candidate);
    if (mine) atomicAdd(tile_bytes, (unsigned long long)mine);
    __syncthreads();

    // Never so narrow that priming a chunk costs more than a quarter of walking it.
    sz_size_t const chunk_bytes = sz_max_of_two(sz_size_divide_round_up((sz_size_t)*tile_bytes, blockDim.x),
                                                4 * sz_max_of_two(warm_up, (sz_size_t)1));
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        chunk_offsets[candidate] = (sz_u32_t)sz_size_divide_round_up(
            candidates->get_length(candidates->handle, tile_first + candidate), chunk_bytes);
    __syncthreads();

    // One warp scans the tile's chunk counts, each lane a run of them, so the scan is two passes
    // and a shuffle ladder rather than a block-wide collective.
    if (threadIdx.x < 32) {
        unsigned const lane = threadIdx.x;
        sz_size_t const run = sz_size_divide_round_up(tile_count + 1, 32);
        sz_size_t const first = sz_min_of_two(lane * run, tile_count + 1),
                        last = sz_min_of_two(first + run, tile_count + 1);
        sz_u32_t total = 0, index;
        for (index = first; index != last; ++index) total += index < tile_count ? chunk_offsets[index] : 0;
        sz_u32_t inclusive = total;
#pragma unroll
        for (unsigned offset = 1; offset != 32; offset <<= 1) {
            sz_u32_t const below = sz_shuffle_up_simt_(inclusive, offset);
            if (lane >= offset) inclusive += below;
        }
        sz_u32_t running = inclusive - total;
        for (index = first; index != last; ++index) {
            sz_u32_t const own = index < tile_count ? chunk_offsets[index] : 0;
            chunk_offsets[index] = running, running += own;
        }
    }
    __syncthreads();
    return chunk_bytes;
}

/**
 *  @brief One block per tile of candidates and per prepared query, the tile's bytes cut into chunks
 *      its threads share, the tree staged when the launch asked for it.
 *
 *  No thread leaves the query loop early, because the barriers are collective: the staging, the
 *  counts zeroed before the chunks add to them, and the scores read off them after.
 *
 *  Dynamic shared memory holds the staged tree, then each candidate's count per width, then each
 *  candidate's first chunk.
 */
static __global__ void sz_overlap_simt_scores_kernel_(sz_overlap_engine_t engine, sz_sequence_t candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride, sz_size_t staged_nodes_count) {
    extern __shared__ sz_u32_t sz_overlap_simt_shared_[];
    __shared__ unsigned long long tile_bytes;
    __shared__ sz_tile_queue_t queue;
    sz_size_t const widths_count = engine.widths_count;
    sz_u32_t *const staged_nodes = sz_overlap_simt_shared_;
    sz_u32_t *const counts = staged_nodes + sz_size_divide_round_up(staged_nodes_count, 4) * 4;
    sz_u32_t *const chunk_offsets = counts + (sz_size_t)blockDim.x * widths_count;
    sz_size_t warm_up = 0, staged_query = STRINGZILLA_SIZE_MAX, chunk_bytes = 0, slot, chunk, candidate;
    sz_u32_t tile_x = blockIdx.x, tile_y = blockIdx.y, laid_x = 0xFFFFFFFFu;
    for (slot = 0; slot != widths_count; ++slot) warm_up = sz_max_of_two(warm_up, (sz_size_t)engine.widths[slot]);
    warm_up = warm_up ? warm_up - 1 : 0;
    sz_tile_queue_open_simt_(&queue, 1, gridDim.x, 0);

    // A tile taken over from another block of the same grid keeps the chunks or the tree already
    // laid out whenever it shares the candidates or the query.
    do {
        sz_size_t const tile_first = (sz_size_t)tile_x * blockDim.x;
        sz_size_t const tile_count = sz_min_of_two((sz_size_t)blockDim.x, candidates.count - tile_first);
        if (tile_x != laid_x) {
            chunk_bytes = sz_overlap_simt_tile_chunks_(&candidates, tile_first, tile_count, warm_up, &tile_bytes,
                                                       chunk_offsets);
            laid_x = tile_x;
        }
        sz_size_t const chunks = chunk_offsets[tile_count];

        for (sz_size_t query = tile_y; query < engine.count; query += gridDim.y) {
            sz_overlap_btree_t btree = sz_overlap_engine_row_(&engine, query);
            sz_size_t const query_length = engine.lengths[query];
            if (staged_nodes_count && query != staged_query) {
                sz_size_t const entries = engine.nodes_offsets[query + 1] - engine.nodes_offsets[query];
                for (sz_size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
                    staged_nodes[entry] = btree.nodes[entry];
                staged_query = query;
            }
            if (staged_nodes_count) btree.nodes = staged_nodes;
            for (slot = threadIdx.x; slot < tile_count * widths_count; slot += blockDim.x) counts[slot] = 0;
            __syncthreads();

            for (chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
                candidate = sz_overlap_simt_last_not_above_(chunk_offsets, tile_count, chunk);
                sz_overlap_simt_chunk_(&engine, &btree, &candidates, tile_first + candidate,
                                       (chunk - chunk_offsets[candidate]) * chunk_bytes, chunk_bytes, warm_up,
                                       counts + candidate * widths_count);
            }
            __syncthreads();

            for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x) {
                sz_size_t const length = candidates.get_length(candidates.handle, tile_first + candidate);
                sz_f32_t *const candidate_scores = scores + query * scores_query_stride +
                                                   (tile_first + candidate) * scores_candidate_stride;
                for (slot = 0; slot != widths_count; ++slot) {
                    sz_size_t const width = engine.widths[slot];
                    sz_bool_t const scored = width && width <= query_length && width <= length ? sz_true_k : sz_false_k;
                    candidate_scores[slot] = scored ? sz_overlap_share_(counts[candidate * widths_count + slot],
                                                                        length - width + 1, query_length - width + 1)
                                                    : 0.0f;
                }
            }
            // The next query restages the tree and rezeroes the counts these reads still hold.
            __syncthreads();
        }
    } while (sz_tile_queue_next_simt_(&queue, &tile_x, &tile_y));
}

/** Dynamic shared memory one block of @p threads takes: the staged tree, rounded up to a vector,
 *  then a count per width and a first chunk per candidate of its tile, and the tile's total. */
STRINGZILLA_INLINE sz_size_t sz_overlap_simt_shared_bytes_(sz_size_t staged_nodes_count, sz_size_t threads,
                                                           sz_size_t widths_count) {
    return (sz_size_divide_round_up(staged_nodes_count, 4) * 4 + threads * (widths_count + 1) + 1) * sizeof(sz_u32_t);
}

/** Prepares the forest on the device the caller already made current, which is every step of
 *  @ref sz_overlap_engine_init_simt_scoped_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_simt_(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator) {
    if (!sz_device_multiprocessors_()) return sz_missing_gpu_k;
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
    // stages one tree, so the widest of them is what the launch has to fit and what the occupancy
    // walk is told, beside the counts of the widest tile the walk may pick.
    sz_overlap_simt_geometry_t *const geometry = (sz_overlap_simt_geometry_t *)sz_overlap_engine_head_(engine);
    sz_size_t const shared_ceiling = sz_device_shared_bytes_per_block_();
    geometry->staged_nodes_count = 0;
    if (widest_nodes * sizeof(sz_u32_t) * sz_overlap_simt_shared_tree_share_k <= shared_ceiling &&
        sz_overlap_simt_shared_bytes_(widest_nodes, sz_overlap_simt_candidates_per_block_k, engine->widths_count) <=
            shared_ceiling)
        geometry->staged_nodes_count = widest_nodes;
    geometry->candidates_per_block = sz_device_block_size_(
        (void const *)sz_overlap_simt_scores_kernel_,
        sz_overlap_simt_shared_bytes_(geometry->staged_nodes_count, sz_overlap_simt_candidates_per_block_k,
                                      engine->widths_count),
        sz_overlap_simt_candidates_per_block_k, sz_overlap_simt_candidates_per_block_k);

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
 *  is past @ref sz_overlap_simt_widest_window_k, which bounds the bytes a chunk re-walks to start.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_init_simt_scoped_(sz_overlap_engine_t *engine,
                                                                   sz_sequence_t const *queries,
                                                                   sz_size_t const *window_widths,
                                                                   sz_size_t window_widths_count,
                                                                   sz_size_t candidates_budget, sz_size_t ordinal,
                                                                   sz_memory_allocator_t *allocator, void *stream) {
    int caller = 0;
    sz_unused_(candidates_budget);
    sz_status_t status = sz_device_enter_(ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_engine_init_simt_(engine, queries, window_widths, window_widths_count, ordinal, allocator);
    sz_device_leave_(caller);
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
    sz_size_t const shared_bytes = sz_overlap_simt_shared_bytes_(staged_nodes_count, per_block, engine->widths_count);

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
    return sz_device_launch_((void const *)sz_overlap_simt_scores_kernel_, grid, block, arguments, shared_bytes,
                             stream);
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
    sz_status_t status = sz_device_enter_(engine->ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_overlap_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride, stream);
    sz_device_leave_(caller);
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
