/**
 *  @file include/stringzilla/overlap/simt.cuh
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief The walk CUDA and ROCm share for window overlap: a block's tile of candidates cut into
 *      chunks of equal bytes, one per thread, each width's window hash rolled through a chunk in
 *      one pass, and one prepared query's B-tree probed per block row.
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
 *  Only device code lives here; each vendor wraps the walk in its own kernel and prepares the
 *  forest, sizes the grid and launches from its own host side, in `cuda.cuh` and `rocm.cuh`.
 *
 *  @sa include/stringzilla/overlap.h
 *  @sa include/stringzilla/overlap/cuda.cuh
 *  @sa include/stringzilla/overlap/rocm.cuh
 */
#ifndef STRINGZILLA_OVERLAP_SIMT_CUH_
#define STRINGZILLA_OVERLAP_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/overlap/serial.h"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Scores Walk

/** @p value modulo @c sz_overlap_modulus_k without a 64-bit division: the Barrett quotient from
 *  the high half of one product with the reciprocal is short by at most one, which one conditional
 *  subtraction settles, so the residue is exactly the serial tier's. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_reduce_simt_(sz_u64_t value) {
    sz_u64_t const modulus = sz_overlap_modulus_k, reciprocal = ~(sz_u64_t)0 / modulus;
    sz_u64_t const remainder = value - __umul64hi(value, reciprocal) * modulus;
    return (sz_u32_t)(remainder >= modulus ? remainder - modulus : remainder);
}

/** How many of a node's sixteen ascending entries sit below the flipped @p key, up to fifteen, by
 *  halving: four dependent loads where a scan takes sixteen, and only a sixteenth entry below the
 *  key goes uncounted. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_rank_simt_(sz_u32_t const *node, sz_i32_t key) {
    sz_u32_t rank = (sz_i32_t)node[7] < key ? 8u : 0u;
    rank += (sz_i32_t)node[rank + 3] < key ? 4u : 0u;
    rank += (sz_i32_t)node[rank + 1] < key ? 2u : 0u;
    rank += (sz_i32_t)node[rank] < key ? 1u : 0u;
    return rank;
}

/** One prepared query's B-tree as a block walks it, its level bases in the block's shared memory,
 *  where indexing them by level costs no stack frame. */
typedef struct sz_overlap_btree_simt_t {

    /** The tree's nodes, in shared memory when the block staged them. */
    sz_u32_t const *nodes;

    /** Levels from the root down to the leaves. */
    sz_size_t levels;

    /** First node of each level, root first, in shared memory. */
    sz_size_t const *level_bases;
} sz_overlap_btree_simt_t;

/** Lays query @p query 's level bases into @p level_bases, as @ref sz_overlap_engine_row_ does, and
 *  returns its levels. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_level_bases_simt_(sz_overlap_engine_t const *engine, sz_size_t query,
                                                          sz_size_t *level_bases) {
    sz_size_t const leaves = sz_overlap_btree_leaves_(engine->keys_counts[query]);
    sz_size_t levels = 1, base = 0, level_nodes = leaves;
    for (sz_size_t nodes = leaves; nodes != 1; nodes = sz_size_divide_round_up(nodes, sz_overlap_branches_per_node_k))
        ++levels;
    level_bases[levels - 1] = 0;
    for (sz_size_t below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = sz_size_divide_round_up(level_nodes, sz_overlap_branches_per_node_k);
        level_bases[below - 1] = base;
    }
    return levels;
}

/** Whether the prepared query holds one raw @p window_hash, halving one node per level. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_btree_probe_simt_(sz_overlap_btree_simt_t const *btree, sz_u32_t window_hash) {
    sz_i32_t const key = (sz_i32_t)(window_hash ^ (sz_u32_t)sz_overlap_sign_flip_k);
    sz_u32_t const last = sz_overlap_keys_per_node_k - 1;
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 < btree->levels; ++level) {
        sz_u32_t const *const separators = btree->nodes +
                                           (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k;
        sz_u32_t const rank = sz_overlap_rank_simt_(separators, key);
        node = node * sz_overlap_branches_per_node_k + rank + (rank == last && (sz_i32_t)separators[last] < key);
    }
    // The leaves lead the arena, so the last level's base is zero.
    sz_u32_t const *const leaf = btree->nodes + node * sz_overlap_keys_per_node_k;
    return leaf[sz_overlap_rank_simt_(leaf, key)] == (sz_u32_t)key;
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
STRINGZILLA_DEVICE void sz_overlap_chunk_simt_(sz_overlap_engine_t const *engine, sz_overlap_btree_simt_t const *btree,
                                               sz_sequence_t const *candidates, sz_size_t candidate,
                                               sz_size_t chunk_begin, sz_size_t chunk_bytes, sz_size_t warm_up,
                                               sz_u32_t *counts) {
    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates->handle, candidate);
    sz_size_t const length = sz_sequence_tape_length_simt_(candidates->handle, candidate);
    sz_size_t const walk_begin = chunk_begin > warm_up ? chunk_begin - warm_up : 0;
    sz_size_t const walk_bytes = sz_min_of_two(chunk_begin + chunk_bytes, length) - walk_begin;
    sz_size_t const warm_bytes = chunk_begin - walk_begin;
    sz_size_t const widths_count = engine->widths_count;
    sz_u8_t const *const walk = (sz_u8_t const *)text + walk_begin;

    // Unrolled to the ceiling and guarded, so the hashes and counters stay in registers.
    sz_u32_t hashes[sz_overlap_gpu_widths_max_k], matches[sz_overlap_gpu_widths_max_k];
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_gpu_widths_max_k; ++index) hashes[index] = 0, matches[index] = 0;

    // Bytes before the walk count as zeros, so a window not yet full holds its own bytes' hash.
    for (sz_size_t ending = 1; ending <= walk_bytes; ++ending) {
        sz_u64_t const incoming = __ldg(walk + ending - 1);
#pragma unroll
        for (sz_size_t index = 0; index != sz_overlap_gpu_widths_max_k; ++index) {
            if (index >= widths_count) break;
            sz_size_t const width = engine->widths[index];
            if (!width) continue;
            sz_u64_t const outgoing = ending > width ? __ldg(walk + ending - width - 1) : 0;
            sz_u64_t const rolled = (sz_u64_t)hashes[index] * 256 + incoming + (sz_u64_t)sz_overlap_modulus_k * 256 -
                                    outgoing * engine->powers[index];
            hashes[index] = sz_overlap_reduce_simt_(rolled);
            if (ending <= warm_bytes || width > ending) continue;
            matches[index] += (sz_u32_t)sz_overlap_btree_probe_simt_(btree, hashes[index]);
        }
    }
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_gpu_widths_max_k; ++index)
        if (index < widths_count && matches[index]) atomicAdd(counts + index, matches[index]);
}

/** Index of the last entry at or below @p value in an ascending array; zero when none is. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_last_not_above_simt_(sz_u32_t const *ascending, sz_size_t count,
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
STRINGZILLA_DEVICE sz_size_t sz_overlap_tile_chunks_simt_(sz_sequence_t const *candidates, sz_size_t tile_first,
                                                          sz_size_t tile_count, sz_size_t warm_up,
                                                          unsigned long long *tile_bytes, sz_u32_t *chunk_offsets,
                                                          sz_u32_t (*shuffle_up)(sz_u32_t, unsigned)) {
    sz_size_t mine = 0, candidate;
    if (threadIdx.x == 0) *tile_bytes = 0;
    __syncthreads();
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        mine += sz_sequence_tape_length_simt_(candidates->handle, tile_first + candidate);
    if (mine) atomicAdd(tile_bytes, (unsigned long long)mine);
    __syncthreads();

    // Never so narrow that priming a chunk costs more than a quarter of walking it.
    sz_size_t const chunk_bytes = sz_max_of_two(sz_size_divide_round_up((sz_size_t)*tile_bytes, blockDim.x),
                                                4 * sz_max_of_two(warm_up, (sz_size_t)1));
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        chunk_offsets[candidate] = (sz_u32_t)sz_size_divide_round_up(
            sz_sequence_tape_length_simt_(candidates->handle, tile_first + candidate), chunk_bytes);
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
            sz_u32_t const below = shuffle_up(inclusive, offset);
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
 *
 *  @param[in] shuffle_up The vendor's cross-lane step.
 *  @param[in] open, next The tile queue's steps, which differ per generation.
 */
STRINGZILLA_DEVICE void sz_overlap_scores_simt_(sz_overlap_engine_t engine, sz_sequence_t candidates, sz_f32_t *scores,
                                                sz_size_t scores_query_stride, sz_size_t scores_candidate_stride,
                                                sz_size_t staged_nodes_count,
                                                sz_u32_t (*shuffle_up)(sz_u32_t, unsigned),
                                                void (*open)(sz_tile_queue_t *, sz_size_t, sz_size_t, sz_size_t),
                                                int (*next)(sz_tile_queue_t *, sz_u32_t *, sz_u32_t *)) {
    extern __shared__ sz_u32_t sz_overlap_shared_simt_[];
    __shared__ unsigned long long tile_bytes;
    __shared__ sz_tile_queue_t queue;
    __shared__ sz_size_t level_bases[sz_overlap_btree_levels_max_k];
    __shared__ sz_size_t levels;
    sz_size_t const widths_count = engine.widths_count;
    sz_u32_t *const staged_nodes = sz_overlap_shared_simt_;
    sz_u32_t *const counts = staged_nodes + sz_size_divide_round_up(staged_nodes_count, 4) * 4;
    sz_u32_t *const chunk_offsets = counts + (sz_size_t)blockDim.x * widths_count;
    sz_size_t warm_up = 0, staged_query = STRINGZILLA_SIZE_MAX, chunk_bytes = 0, slot, chunk, candidate;
    sz_u32_t tile_x = blockIdx.x, tile_y = blockIdx.y, laid_x = 0xFFFFFFFFu;
    for (slot = 0; slot != widths_count; ++slot) warm_up = sz_max_of_two(warm_up, (sz_size_t)engine.widths[slot]);
    warm_up = warm_up ? warm_up - 1 : 0;
    open(&queue, 1, gridDim.x, 0);

    // A tile taken over from another block of the same grid keeps the chunks or the tree already
    // laid out whenever it shares the candidates or the query.
    do {
        sz_size_t const tile_first = (sz_size_t)tile_x * blockDim.x;
        sz_size_t const tile_count = sz_min_of_two((sz_size_t)blockDim.x, candidates.count - tile_first);
        if (tile_x != laid_x) {
            chunk_bytes = sz_overlap_tile_chunks_simt_(&candidates, tile_first, tile_count, warm_up, &tile_bytes,
                                                       chunk_offsets, shuffle_up);
            laid_x = tile_x;
        }
        sz_size_t const chunks = chunk_offsets[tile_count];

        for (sz_size_t query = tile_y; query < engine.count; query += gridDim.y) {
            sz_u32_t const *const query_nodes = engine.nodes + engine.nodes_offsets[query];
            sz_size_t const query_length = engine.lengths[query];
            sz_overlap_btree_simt_t btree;
            if (threadIdx.x == 0) levels = sz_overlap_level_bases_simt_(&engine, query, level_bases);
            if (staged_nodes_count && query != staged_query) {
                sz_size_t const entries = engine.nodes_offsets[query + 1] - engine.nodes_offsets[query];
                for (sz_size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
                    staged_nodes[entry] = __ldg(query_nodes + entry);
                staged_query = query;
            }
            for (slot = threadIdx.x; slot < tile_count * widths_count; slot += blockDim.x) counts[slot] = 0;
            __syncthreads();
            btree.nodes = staged_nodes_count ? staged_nodes : query_nodes;
            btree.levels = levels, btree.level_bases = level_bases;

            for (chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
                candidate = sz_overlap_last_not_above_simt_(chunk_offsets, tile_count, chunk);
                sz_overlap_chunk_simt_(&engine, &btree, &candidates, tile_first + candidate,
                                       (chunk - chunk_offsets[candidate]) * chunk_bytes, chunk_bytes, warm_up,
                                       counts + candidate * widths_count);
            }
            __syncthreads();

            for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x) {
                sz_size_t const length = sz_sequence_tape_length_simt_(candidates.handle, tile_first + candidate);
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
    } while (next(&queue, &tile_x, &tile_y));
}

#pragma endregion Scores Walk

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_OVERLAP_SIMT_CUH_
