/**
 *  @file include/stringzilla/overlap/simt.cuh
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief The walk CUDA and ROCm share for window overlap: a block's tile of candidates cut into
 *      chunks of equal bytes, one per thread, each width's window hash rolled through a chunk in
 *      one pass, and one prepared query's B-tree probed per block row.
 *
 *  The arithmetic is the serial tier's, reached from the device through
 *  @c --expt-relaxed-constexpr, so the counts are exactly serial's rather than merely close.
 *  Integers, not doubles: both factors are below 2^32, so the product fits one @c u64 and the
 *  remainder by a constant lowers to a multiply-high - where an @c f64 reduction would run at the
 *  device's double-precision rate, a sixty-fourth of its single-precision one on consumer parts.
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
 *  Nothing here crosses the bus during a round. The forest, the candidates and the matches are
 *  already where the device reaches them and the texts are read in place, so a round costs one
 *  launch rather than a round trip.
 *
 *  A walk scoring one query against scattered candidates prepares that query itself instead, a warp
 *  laying its tree into a span the caller owns, and counts each candidate's matches a warp at a
 *  time, through the same roll and probe.
 *
 *  Only device code lives here; each vendor wraps the walk in its own kernel and prepares the
 *  forest, sizes the grid and launches from its own host side, in <tt>cuda.cuh</tt> and <tt>rocm.cuh</tt>.
 *
 *  @sa c/stringzilla/overlap.h
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

/** Entry @p index of a tree's nodes, read from shared memory where the block @p staged them and
 *  through the read-only path otherwise, rather than by a generic load that serves both. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_entry_simt_(sz_u32_t const *nodes, sz_size_t index, sz_bool_t staged) {
    return staged ? nodes[index] : __ldg(nodes + index);
}

/** How many of a node's sixteen ascending entries sit below the flipped @p key, up to fifteen, by
 *  halving: four dependent loads where a scan takes sixteen, and only a sixteenth entry below the
 *  key goes uncounted. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_rank_simt_(sz_u32_t const *node, sz_i32_t key, sz_bool_t staged) {
    sz_u32_t rank = (sz_i32_t)sz_overlap_entry_simt_(node, 7, staged) < key ? 8u : 0u;
    rank += (sz_i32_t)sz_overlap_entry_simt_(node, rank + 3, staged) < key ? 4u : 0u;
    rank += (sz_i32_t)sz_overlap_entry_simt_(node, rank + 1, staged) < key ? 2u : 0u;
    rank += (sz_i32_t)sz_overlap_entry_simt_(node, rank, staged) < key ? 1u : 0u;
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
STRINGZILLA_DEVICE sz_size_t sz_overlap_btree_probe_simt_(sz_overlap_btree_simt_t const *btree, sz_u32_t window_hash,
                                                          sz_bool_t staged) {
    sz_i32_t const key = (sz_i32_t)(window_hash ^ (sz_u32_t)sz_overlap_sign_flip_k);
    sz_u32_t const last = sz_overlap_keys_per_node_k - 1;
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 < btree->levels; ++level) {
        sz_u32_t const *const separators = btree->nodes +
                                           (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k;
        sz_u32_t const rank = sz_overlap_rank_simt_(separators, key, staged);
        node = node * sz_overlap_branches_per_node_k + rank +
               (rank == last && (sz_i32_t)sz_overlap_entry_simt_(separators, last, staged) < key);
    }
    // The leaves lead the arena, so the last level's base is zero.
    sz_u32_t const *const leaf = btree->nodes + node * sz_overlap_keys_per_node_k;
    return sz_overlap_entry_simt_(leaf, sz_overlap_rank_simt_(leaf, key, staged), staged) == (sz_u32_t)key;
}

/** One byte's step of a rolling window hash, <tt>H' = H * 256 + incoming - outgoing * 256^w</tt>, with
 *  @p power being 256^w and the bytes before a text counting as zeros. The residues are the serial
 *  tier's, as both are exact modulo the same prime. */
STRINGZILLA_DEVICE sz_u32_t sz_overlap_roll_step_simt_(sz_u32_t hash, sz_u32_t incoming, sz_u32_t outgoing,
                                                       sz_u32_t power) {
    return sz_overlap_reduce_simt_((sz_u64_t)hash * 256 + incoming + (sz_u64_t)sz_overlap_modulus_k * 256 -
                                   (sz_u64_t)outgoing * power);
}

/**
 *  @brief Rolls each width's window hash through the endings past @p from and up to @p until of
 *      @p walk, on one thread, and adds every window the tree holds to that width's @p matches.
 *
 *  Each width rolls its own hash a byte at a time, the byte leaving the window read again from the
 *  text rather than kept: the hashes stay in registers, where a ring of prefix hashes indexed by
 *  width would take a stack frame and bound the width, and each byte costs one reduction per width
 *  instead of three. The first @p warm_bytes endings only prime the hashes, and a window not yet
 *  full is never probed. A caller resumes a walk by passing the @p hashes this one left.
 *
 *  @param[in,out] hashes, matches One entry per width, up to @ref sz_overlap_gpu_widths_per_walk_k.
 */
STRINGZILLA_DEVICE void sz_overlap_roll_simt_(sz_u8_t const *walk, sz_size_t from, sz_size_t until,
                                              sz_size_t warm_bytes, sz_u32_t const *widths, sz_u32_t const *powers,
                                              sz_size_t widths_count, sz_overlap_btree_simt_t const *btree,
                                              sz_bool_t staged, sz_u32_t *hashes, sz_u32_t *matches) {
    for (sz_size_t ending = from + 1; ending <= until; ++ending) {
        sz_u32_t const incoming = __ldg(walk + ending - 1);
#pragma unroll
        for (sz_size_t index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) {
            if (index >= widths_count) break;
            sz_size_t const width = widths[index];
            if (!width) continue;
            sz_u32_t const outgoing = ending > width ? __ldg(walk + ending - width - 1) : 0;
            hashes[index] = sz_overlap_roll_step_simt_(hashes[index], incoming, outgoing, powers[index]);
            if (ending <= warm_bytes || width > ending) continue;
            matches[index] += (sz_u32_t)sz_overlap_btree_probe_simt_(btree, hashes[index], staged);
        }
    }
}

/**
 *  @brief Counts the windows of one query's widths that end inside one chunk of a candidate, on one
 *      thread, and adds them to the candidate's shared counts.
 *
 *  A window's hash depends only on its own bytes, so a chunk primes itself from @p warm_up bytes
 *  before its own start and every window is counted by the one chunk it ends in.
 */
STRINGZILLA_DEVICE void sz_overlap_chunk_simt_(sz_overlap_engine_t const *engine, sz_overlap_btree_simt_t const *btree,
                                               sz_strings_view_t const *candidates, sz_size_t candidate,
                                               sz_size_t chunk_begin, sz_size_t chunk_bytes, sz_size_t warm_up,
                                               sz_u32_t *counts, sz_size_t widths_count, sz_bool_t staged) {
    sz_cptr_t const text = sz_strings_start_simt_(*candidates, candidate);
    sz_size_t const length = sz_strings_length_simt_(*candidates, candidate);
    sz_size_t const walk_begin = chunk_begin > warm_up ? chunk_begin - warm_up : 0;
    sz_size_t const walk_bytes = sz_min_of_two(chunk_begin + chunk_bytes, length) - walk_begin;
    sz_size_t const warm_bytes = chunk_begin - walk_begin;

    // Unrolled to one walk's widths and guarded, so the hashes and counters stay in registers.
    sz_u32_t hashes[sz_overlap_gpu_widths_per_walk_k], matches[sz_overlap_gpu_widths_per_walk_k];
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) hashes[index] = 0, matches[index] = 0;
    sz_overlap_roll_simt_((sz_u8_t const *)text + walk_begin, 0, walk_bytes, warm_bytes, engine->widths, engine->powers,
                          widths_count, btree, staged, hashes, matches);
#pragma unroll
    for (sz_size_t index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index)
        if (index < widths_count && matches[index]) atomicAdd(counts + index, matches[index]);
}

/** Every chunk of a tile this thread takes against one query's tree, the counts landing per
 *  candidate. The walk passes @p widths_count and @p staged as literals, so each copy unrolls one
 *  width loop and loads from one memory space, rather than through a generic load serving both. */
STRINGZILLA_DEVICE void sz_overlap_chunks_simt_(sz_overlap_engine_t const *engine, sz_overlap_btree_simt_t const *btree,
                                                sz_strings_view_t const *candidates, sz_size_t tile_first,
                                                sz_size_t chunks, sz_u32_t const *chunk_offsets,
                                                sz_u16_t const *chunk_owners, sz_size_t chunk_bytes, sz_size_t warm_up,
                                                sz_u32_t *counts, sz_size_t widths_count, sz_bool_t staged) {
    for (sz_size_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sz_size_t const candidate = chunk_owners[chunk];
        sz_overlap_chunk_simt_(engine, btree, candidates, tile_first + candidate,
                               (chunk - chunk_offsets[candidate]) * chunk_bytes, chunk_bytes, warm_up,
                               counts + candidate * widths_count, widths_count, staged);
    }
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
STRINGZILLA_DEVICE sz_size_t sz_overlap_tile_chunks_simt_(sz_strings_view_t const *candidates, sz_size_t tile_first,
                                                          sz_size_t tile_count, sz_size_t warm_up,
                                                          unsigned long long *tile_bytes, sz_u32_t *chunk_offsets,
                                                          sz_u32_t (*shuffle_up)(sz_u32_t, unsigned)) {
    sz_size_t mine = 0, candidate;
    if (threadIdx.x == 0) *tile_bytes = 0;
    __syncthreads();
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        mine += sz_strings_length_simt_(*candidates, tile_first + candidate);
    if (mine) atomicAdd(tile_bytes, (unsigned long long)mine);
    __syncthreads();

    // Never so narrow that priming a chunk costs more than a quarter of walking it.
    sz_size_t const chunk_bytes = sz_max_of_two(sz_size_divide_round_up((sz_size_t)*tile_bytes, blockDim.x),
                                                4 * sz_max_of_two(warm_up, (sz_size_t)1));
    for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
        chunk_offsets[candidate] = (sz_u32_t)sz_size_divide_round_up(
            sz_strings_length_simt_(*candidates, tile_first + candidate), chunk_bytes);
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

/** Endings a lane rolls between two checks of whether a candidate can still come within its radius:
 *  a check is ten shuffles, against a few hundred operations per ending at even one width. */
enum { sz_overlap_gpu_endings_per_check_k = 32 };

/** The sum of @p value over this lane's aligned group of @p lanes lanes, a power of two up to a
 *  warp, in every lane of the group, by halves; groups may diverge from one another. */
STRINGZILLA_DEVICE sz_size_t sz_overlap_lanes_sum_simt_(sz_size_t value, unsigned lane, unsigned lanes,
                                                        sz_u32_t (*shuffle_index)(sz_u32_t, unsigned, unsigned)) {
    for (unsigned offset = lanes / 2; offset != 0; offset >>= 1) {
        sz_u32_t const low = shuffle_index((sz_u32_t)value, lane ^ offset, lanes);
        sz_u32_t const high = shuffle_index((sz_u32_t)((sz_u64_t)value >> 32), lane ^ offset, lanes);
        value += (sz_size_t)(((sz_u64_t)high << 32) | low);
    }
    return value;
}

/**
 *  @brief Counts the windows of one candidate, at every width, that one query's tree holds, every
 *      lane of an aligned group of @p lanes taking part, and gives up once even every window left
 *      matching would leave more than @p radius of the longer side's windows unmatched.
 *
 *  Each lane of the group rolls an even run of the candidate's bytes, primed from the widest
 *  window's bytes before it, as the batch walk's chunks are. Lanes compare their totals every
 *  @ref sz_overlap_gpu_endings_per_check_k endings, and a candidate with too few windows to come
 *  within @p radius is answered before a byte is read.
 *
 *  With @p counts, lane zero also writes each width's matches there once its group of widths is
 *  rolled, as @ref sz_overlap_scores does, and a pair whose misses pass @p radius, however the walk
 *  ended, ends with zeros at every width.
 *
 *  @param[in] btree The tree, read through plain loads when @p staged, so the same kernel may have
 *      written it, and through the read-only path otherwise.
 *  @param[in] query_length The query's bytes, which only the counts read.
 *  @param[in] query_windows The query's windows over every width.
 *  @param[in] lane The calling lane within its group, below @p lanes.
 *  @param[in] lanes The group's lanes, a power of two up to a warp; a group needs as many as its
 *      runs need to cover the widest window, past which more lanes only roll more priming bytes.
 *  @param[in] shuffle_index The vendor's cross-lane step within a group, which groups may take
 *      apart from one another.
 *  @param[out] counts The pair's @b [widths_count] matches, or null for their sum alone.
 *  @return The same count in every lane: exact whenever it leaves at most @p radius misses, and
 *      otherwise a count that leaves more.
 */
STRINGZILLA_DEVICE sz_size_t sz_overlap_candidate_walk_simt_(sz_overlap_btree_simt_t const *btree, sz_bool_t staged,
                                                             sz_u32_t const *widths, sz_u32_t const *powers,
                                                             sz_size_t widths_count, sz_size_t query_length,
                                                             sz_size_t query_windows, sz_cptr_t text, sz_size_t length,
                                                             sz_size_t radius, unsigned lane, unsigned lanes,
                                                             sz_u32_t (*shuffle_index)(sz_u32_t, unsigned, unsigned),
                                                             sz_u32_t *counts) {
    sz_size_t const candidate_windows = sz_overlap_windows(length, widths, widths_count);
    sz_size_t const longer = sz_max_of_two(candidate_windows, query_windows);
    sz_size_t const needed = longer > radius ? longer - radius : 0;
    sz_size_t const run = sz_size_divide_round_up(length, lanes);
    sz_size_t const run_begin = sz_min_of_two(lane * run, length);
    sz_size_t const run_end = sz_min_of_two(run_begin + run, length);
    sz_size_t pending = sz_overlap_windows(run_end, widths, widths_count) -
                        sz_overlap_windows(run_begin, widths, widths_count);
    sz_size_t matched = 0, scored = 0;

    // A candidate too short to come within the radius walks no group and lands in the epilogue.
    sz_size_t first_width = candidate_windows < needed ? widths_count : 0;
    for (; first_width < widths_count; first_width += sz_overlap_gpu_widths_per_walk_k) {
        sz_u32_t const *const group_widths = widths + first_width;
        sz_size_t const group_count = sz_min_of_two(widths_count - first_width,
                                                    (sz_size_t)sz_overlap_gpu_widths_per_walk_k);
        sz_size_t warm_up = 0, index;
        for (index = 0; index != group_count; ++index) warm_up = sz_max_of_two(warm_up, (sz_size_t)group_widths[index]);
        warm_up = warm_up ? warm_up - 1 : 0;
        sz_size_t const walk_begin = run_begin > warm_up ? run_begin - warm_up : 0;
        sz_size_t const walk_bytes = run_end != run_begin ? run_end - walk_begin : 0;
        sz_size_t const warm_bytes = run_begin - walk_begin;

        sz_u32_t hashes[sz_overlap_gpu_widths_per_walk_k], matches[sz_overlap_gpu_widths_per_walk_k];
#pragma unroll
        for (index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) hashes[index] = 0, matches[index] = 0;
        // As many passes in every lane, so the checks between them see the whole warp.
        sz_size_t const passes = sz_size_divide_round_up(run + warm_up, sz_overlap_gpu_endings_per_check_k);
        sz_size_t reached = 0, pass = 0, group_matched = 0;
        for (; pass != passes; ++pass) {
            sz_size_t const until = sz_min_of_two(reached + sz_overlap_gpu_endings_per_check_k, walk_bytes);
            if (reached < until) {
                sz_overlap_roll_simt_((sz_u8_t const *)text + walk_begin, reached, until, warm_bytes, group_widths,
                                      powers + first_width, group_count, btree, staged, hashes, matches);
                // Only the windows ending past the run's start were probed, the rest having primed.
                sz_size_t const probed_from = sz_max_of_two(walk_begin + reached, run_begin);
                sz_size_t const probed_until = sz_max_of_two(walk_begin + until, run_begin);
                pending -= sz_overlap_windows(probed_until, group_widths, group_count) -
                           sz_overlap_windows(probed_from, group_widths, group_count);
                reached = until;
            }
            if (!needed) continue;
            group_matched = 0;
#pragma unroll
            for (index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) group_matched += matches[index];
            if (sz_overlap_lanes_sum_simt_(matched + group_matched + pending, lane, lanes, shuffle_index) < needed)
                break;
        }
        group_matched = 0;
#pragma unroll
        for (index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) group_matched += matches[index];
        matched += group_matched;
        if (pass != passes) break;
#pragma unroll
        for (index = 0; index != sz_overlap_gpu_widths_per_walk_k; ++index) {
            if (!counts || index >= group_count) break;
            sz_size_t const width = group_widths[index];
            sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
            sz_size_t const own_query_windows = width && width <= query_length ? query_length - width + 1 : 0;
            sz_size_t const total = sz_overlap_lanes_sum_simt_(matches[index], lane, lanes, shuffle_index);
            sz_size_t const counted = windows && own_query_windows ? total : 0;
            scored += counted;
            if (lane == 0) counts[first_width + index] = (sz_u32_t)counted;
        }
    }

    // Whichever check stopped the pair, past the radius every width counts zero.
    if (counts && longer - scored > radius)
        for (sz_size_t index = lane; index < widths_count; index += lanes) counts[index] = 0;
    return sz_overlap_lanes_sum_simt_(matched, lane, lanes, shuffle_index);
}

/**
 *  @brief One block per tile of candidates and per prepared query, the tile's bytes cut into chunks
 *      its threads share, the tree staged when the launch asked for it.
 *
 *  No thread leaves the query loop early, because the barriers are collective: the staging, the
 *  counts zeroed before the chunks add to them, and the matches read off them after.
 *
 *  The @p engine holds at most @ref sz_overlap_gpu_widths_per_walk_k widths, and @p matches starts
 *  at its first one: the host launches once per group of an engine's widths, each over its own
 *  slice of every row, so the walk keeps the registers of one group alone.
 *
 *  Dynamic shared memory holds the staged tree, then each candidate's count per width, then each
 *  candidate's first chunk, then each chunk's candidate as a 16-bit index: a tile cuts into at most
 *  two chunks per thread, as a chunk takes at least the tile's bytes over the block's threads.
 *
 *  @param[in] shuffle_up The vendor's cross-lane step.
 *  @param[in] open, next The tile queue's steps, which differ per generation.
 */
STRINGZILLA_DEVICE void sz_overlap_scores_simt_(sz_overlap_engine_t engine, sz_strings_view_t candidates,
                                                sz_u32_t *matches, sz_size_t matches_query_stride,
                                                sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count,
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
    sz_u16_t *const chunk_owners = (sz_u16_t *)(chunk_offsets + blockDim.x + 1);
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
            // Once per tile, not a binary search per chunk and query: a short candidate's chunk is
            // about as many bytes as that search takes steps.
            for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x)
                for (chunk = chunk_offsets[candidate]; chunk != chunk_offsets[candidate + 1]; ++chunk)
                    chunk_owners[chunk] = (sz_u16_t)candidate;
            __syncthreads();
            laid_x = tile_x;
        }
        sz_size_t const chunks = chunk_offsets[tile_count];

        for (sz_size_t query = tile_y; query < engine.count; query += gridDim.y) {
            sz_u32_t const *const query_nodes = engine.nodes + engine.nodes_offsets[query];
            sz_size_t const query_length = engine.lengths[query];
            if (threadIdx.x == 0) levels = sz_overlap_level_bases_simt_(&engine, query, level_bases);
            if (staged_nodes_count && query != staged_query) {
                sz_size_t const entries = engine.nodes_offsets[query + 1] - engine.nodes_offsets[query];
                for (sz_size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
                    staged_nodes[entry] = __ldg(query_nodes + entry);
                staged_query = query;
            }
            for (slot = threadIdx.x; slot < tile_count * widths_count; slot += blockDim.x) counts[slot] = 0;
            __syncthreads();
            // One tree per memory space, so no walk reads through a pointer that may be either.
            sz_overlap_btree_simt_t const staged_tree = {staged_nodes, levels, level_bases};
            sz_overlap_btree_simt_t const global_tree = {query_nodes, levels, level_bases};
            if (staged_nodes_count && widths_count == 1)
                sz_overlap_chunks_simt_(&engine, &staged_tree, &candidates, tile_first, chunks, chunk_offsets,
                                        chunk_owners, chunk_bytes, warm_up, counts, 1, sz_true_k);
            else if (staged_nodes_count)
                sz_overlap_chunks_simt_(&engine, &staged_tree, &candidates, tile_first, chunks, chunk_offsets,
                                        chunk_owners, chunk_bytes, warm_up, counts, widths_count, sz_true_k);
            else if (widths_count == 1)
                sz_overlap_chunks_simt_(&engine, &global_tree, &candidates, tile_first, chunks, chunk_offsets,
                                        chunk_owners, chunk_bytes, warm_up, counts, 1, sz_false_k);
            else
                sz_overlap_chunks_simt_(&engine, &global_tree, &candidates, tile_first, chunks, chunk_offsets,
                                        chunk_owners, chunk_bytes, warm_up, counts, widths_count, sz_false_k);
            __syncthreads();

            for (candidate = threadIdx.x; candidate < tile_count; candidate += blockDim.x) {
                sz_size_t const length = sz_strings_length_simt_(candidates, tile_first + candidate);
                sz_u32_t *const candidate_matches = matches + query * matches_query_stride +
                                                    (tile_first + candidate) * matches_candidate_stride;
                for (slot = 0; slot != widths_count; ++slot) {
                    sz_size_t const width = engine.widths[slot];
                    sz_bool_t const scored = width && width <= query_length && width <= length ? sz_true_k : sz_false_k;
                    candidate_matches[slot] = scored ? counts[candidate * widths_count + slot] : 0;
                }
            }
            // The next query restages the tree and rezeroes the counts these reads still hold.
            __syncthreads();
        }
    } while (next(&queue, &tile_x, &tile_y));
}

/**
 *  @brief One block per tile of candidates and per prepared query, as @ref sz_overlap_scores_simt_
 *      walks them, each warp scoring a candidate of the tile at a time within its budget.
 *
 *  A pair's misses sum over every width, so the launch takes them all, and each pair is one call of
 *  @ref sz_overlap_candidate_walk_simt_ under @p misses_bound of its own longer side, which stops
 *  it once it is past that budget and writes its row either way; no chunks are cut and no counts
 *  meet in shared memory, which holds only the staged tree. The barriers around the staging are
 *  collective, so no thread leaves the query loop before the others.
 *
 *  @param[in] misses_bound The radius as @ref sz_overlap_misses_bound_ rounds it.
 *  @param[in] shuffle_index The vendor's cross-lane step within a group of lanes.
 *  @param[in] open, next The tile queue's steps, which differ per generation.
 */
STRINGZILLA_DEVICE void sz_overlap_bounded_scores_simt_(
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count, sz_u32_t misses_bound,
    sz_u32_t (*shuffle_index)(sz_u32_t, unsigned, unsigned),
    void (*open)(sz_tile_queue_t *, sz_size_t, sz_size_t, sz_size_t),
    int (*next)(sz_tile_queue_t *, sz_u32_t *, sz_u32_t *)) {
    extern __shared__ sz_u32_t sz_overlap_shared_simt_[];
    __shared__ sz_tile_queue_t queue;
    __shared__ sz_size_t level_bases[sz_overlap_btree_levels_max_k];
    __shared__ sz_size_t levels;
    sz_u32_t *const staged_nodes = sz_overlap_shared_simt_;
    unsigned const lane = threadIdx.x & 31u, warps = blockDim.x / 32;
    sz_size_t staged_query = STRINGZILLA_SIZE_MAX;
    sz_u32_t tile_x = blockIdx.x, tile_y = blockIdx.y;
    open(&queue, 1, gridDim.x, 0);
    do {
        sz_size_t const tile_first = (sz_size_t)tile_x * blockDim.x;
        sz_size_t const tile_count = sz_min_of_two((sz_size_t)blockDim.x, candidates.count - tile_first);
        for (sz_size_t query = tile_y; query < engine.count; query += gridDim.y) {
            sz_u32_t const *const query_nodes = engine.nodes + engine.nodes_offsets[query];
            sz_size_t const query_length = engine.lengths[query];
            sz_size_t const query_windows = sz_overlap_windows(query_length, engine.widths, engine.widths_count);
            if (threadIdx.x == 0) levels = sz_overlap_level_bases_simt_(&engine, query, level_bases);
            if (staged_nodes_count && query != staged_query) {
                sz_size_t const entries = engine.nodes_offsets[query + 1] - engine.nodes_offsets[query];
                for (sz_size_t entry = threadIdx.x; entry < entries; entry += blockDim.x)
                    staged_nodes[entry] = __ldg(query_nodes + entry);
                staged_query = query;
            }
            __syncthreads();
            sz_overlap_btree_simt_t const staged_tree = {staged_nodes, levels, level_bases};
            sz_overlap_btree_simt_t const global_tree = {query_nodes, levels, level_bases};
            for (sz_size_t candidate = threadIdx.x / 32; candidate < tile_count; candidate += warps) {
                sz_cptr_t const text = sz_strings_start_simt_(candidates, tile_first + candidate);
                sz_size_t const length = sz_strings_length_simt_(candidates, tile_first + candidate);
                sz_u32_t *const candidate_matches = matches + query * matches_query_stride +
                                                    (tile_first + candidate) * matches_candidate_stride;
                sz_size_t const longer = sz_max_of_two(sz_overlap_windows(length, engine.widths, engine.widths_count),
                                                       query_windows);
                sz_size_t const bound = (sz_size_t)(((sz_u64_t)misses_bound * longer) >> 24);
                if (staged_nodes_count)
                    sz_overlap_candidate_walk_simt_(&staged_tree, sz_true_k, engine.widths, engine.powers,
                                                    engine.widths_count, query_length, query_windows, text, length,
                                                    bound, lane, 32, shuffle_index, candidate_matches);
                else
                    sz_overlap_candidate_walk_simt_(&global_tree, sz_false_k, engine.widths, engine.powers,
                                                    engine.widths_count, query_length, query_windows, text, length,
                                                    bound, lane, 32, shuffle_index, candidate_matches);
            }
            // The next query restages the tree these walks still read.
            __syncthreads();
        }
    } while (next(&queue, &tile_x, &tile_y));
}

#pragma endregion Scores Walk

#pragma region Prepared Queries

/**
 *  @brief Prepares one query into a warp's @p span, every lane of the warp taking part, so a device
 *      walk prepares each query where it scores it.
 *
 *  The steps are @ref sz_overlap_engine_init 's, and so is every entry either writes: every width's
 *  window hashes, each lane rolling its own run of windows, a bitonic sort a lane per pair of every
 *  step, the repeats dropped by a ballot per 32 keys, and the tree laid out over the leaves, a lane
 *  per separator. The warp is synchronized on return.
 *
 *  @param[in] text The query, read through plain loads, so it may be written by the same kernel.
 *  @param[in] widths The @b [widths_count] window widths, kept where every walk reads them.
 *  @param[out] span Four-byte aligned, @ref sz_overlap_query_span_bytes long: the tree's entries,
 *      then a power per width.
 *  @param[out] query Points into @p span and at @p widths.
 *  @param[in] lane The calling lane, below 32.
 *  @param[in] shuffle_up, shuffle_index, ballot The vendor's cross-lane steps.
 */
STRINGZILLA_DEVICE void sz_overlap_query_prepare_simt_(sz_cptr_t text, sz_size_t length, sz_u32_t const *widths,
                                                       sz_size_t widths_count, sz_ptr_t span, sz_overlap_query_t *query,
                                                       unsigned lane, sz_u32_t (*shuffle_up)(sz_u32_t, unsigned),
                                                       sz_u32_t (*shuffle_index)(sz_u32_t, unsigned),
                                                       sz_u32_t (*ballot)(int)) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_u32_t *const keys = (sz_u32_t *)span;
    sz_size_t const windows = sz_overlap_windows(length, widths, widths_count);
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity_(windows);
    sz_u32_t *const powers = keys + sz_overlap_btree_entries_(windows);
    sz_size_t index, entry;
    // The span may still serve the previous query's readers until every lane gets here.
    __syncwarp();
    for (index = lane; index < widths_count; index += 32)
        powers[index] = (sz_u32_t)sz_overlap_window_power(widths[index]);
    __syncwarp();

    // Each lane rolls a run of one width's windows, priming its hash on the run's first window.
    sz_size_t written = 0;
    for (index = 0; index != widths_count; ++index) {
        sz_size_t const width = widths[index];
        if (!width || width > length) continue;
        sz_u32_t const power = powers[index];
        sz_size_t const width_windows = length - width + 1;
        sz_size_t const run = sz_size_divide_round_up(width_windows, 32);
        sz_size_t const first = sz_min_of_two(lane * run, width_windows);
        sz_size_t const last = sz_min_of_two(first + run, width_windows);
        sz_u32_t hash = 0;
        if (first != last)
            for (sz_size_t ending = 1; ending != last - first + width; ++ending) {
                sz_u32_t const outgoing = ending > width ? bytes[first + ending - width - 1] : 0;
                hash = sz_overlap_roll_step_simt_(hash, bytes[first + ending - 1], outgoing, power);
                if (ending >= width) keys[written + first + ending - width] = hash;
            }
        written += width_windows;
    }
    for (entry = windows + lane; entry < capacity; entry += 32) keys[entry] = sz_overlap_padding_key_k;
    __syncwarp();

    // The serial sort's network, each step's pairs spread across the lanes.
    for (sz_size_t phase = 2; phase <= capacity; phase *= 2) {
        for (entry = lane; entry < capacity / 2; entry += 32) {
            sz_size_t const offset = entry & (phase / 2 - 1), start = (entry - offset) * 2;
            sz_overlap_exchange_serial_(keys + start + offset, keys + start + phase - 1 - offset);
        }
        __syncwarp();
        for (sz_size_t distance = phase / 4; distance != 0; distance /= 2) {
            for (entry = lane; entry < capacity / 2; entry += 32) {
                sz_size_t const offset = entry & (distance - 1), start = (entry - offset) * 2;
                sz_overlap_exchange_serial_(keys + start + offset, keys + start + distance + offset);
            }
            __syncwarp();
        }
    }

    // A key survives unless it repeats the one before, which lane zero takes from the last pass.
    sz_size_t distinct = 0;
    sz_u32_t carried = 0;
    for (sz_size_t base = 0; base < windows; base += 32) {
        sz_size_t const position = base + lane;
        sz_u32_t const key = position < windows ? keys[position] : 0;
        sz_u32_t const below = shuffle_up(key, 1);
        int const kept = position < windows && (position == 0 || key != (lane ? below : carried));
        sz_u32_t const survivors = ballot(kept);
        __syncwarp();
        if (kept) keys[distinct + __popc(survivors & ((1u << lane) - 1u))] = key;
        distinct += __popc(survivors);
        carried = shuffle_index(key, 31);
    }
    __syncwarp();

    // The leaves flip and pad in place, then each separator reads the first key of the child to its
    // right, as @ref sz_overlap_btree_prepare lays them.
    sz_u32_t const padding = (sz_u32_t)sz_overlap_padding_key_k ^ (sz_u32_t)sz_overlap_sign_flip_k;
    sz_size_t const leaves = sz_overlap_btree_leaves_(distinct);
    for (entry = lane; entry < leaves * sz_overlap_keys_per_node_k; entry += 32)
        keys[entry] = entry < distinct ? keys[entry] ^ (sz_u32_t)sz_overlap_sign_flip_k : padding;
    __syncwarp();
    sz_size_t levels = 1, base = 0, level_nodes = leaves, leaves_per_child = 1;
    for (sz_size_t nodes = leaves; nodes != 1; nodes = sz_size_divide_round_up(nodes, sz_overlap_branches_per_node_k))
        ++levels;
    if (lane == 0) query->btree.level_bases[levels - 1] = 0;
    for (sz_size_t below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = sz_size_divide_round_up(level_nodes, sz_overlap_branches_per_node_k);
        if (lane == 0) query->btree.level_bases[below - 1] = base;
        for (entry = lane; entry < level_nodes * sz_overlap_keys_per_node_k; entry += 32) {
            sz_size_t const node = entry / sz_overlap_keys_per_node_k, separator = entry % sz_overlap_keys_per_node_k;
            sz_size_t const first_leaf = (node * sz_overlap_branches_per_node_k + separator + 1) * leaves_per_child;
            sz_u32_t const first_key = first_leaf < leaves ? keys[first_leaf * sz_overlap_keys_per_node_k] : padding;
            keys[base * sz_overlap_keys_per_node_k + entry] = first_key - 1;
        }
        leaves_per_child *= sz_overlap_branches_per_node_k;
    }
    if (lane == 0) {
        query->btree.nodes = keys;
        query->btree.levels = levels;
        query->widths = widths;
        query->powers = powers;
        query->widths_count = widths_count;
        query->windows = windows;
    }
    __syncwarp();
}

/**
 *  @brief Counts the windows of one candidate, at every width, that a prepared query holds, every
 *      lane of an aligned group of @p lanes taking part, and gives up once even every window left
 *      matching would leave more than @p radius of the longer side's windows unmatched.
 *
 *  Each lane rolls an even run of the candidate's bytes, primed from the widest window's bytes
 *  before it, as the batch walk's chunks are. Lanes compare their totals every
 *  @ref sz_overlap_gpu_endings_per_check_k endings, and a candidate with too few windows to come
 *  within @p radius is answered before a byte is read.
 *
 *  With @c longer the greater of the candidate's @ref sz_overlap_windows and the query's window
 *  count, the misses are @c longer less the matches, and a radius admits @c longer times its
 *  @ref sz_overlap_misses_bound_, shifted down by 24, of them, which is the @p radius to pass; a
 *  @c longer of zero is the caller's to score.
 *
 *  @param[in] query Prepared by @ref sz_overlap_query_prepare_simt_, its span read through plain
 *      loads, so the same kernel may have written it.
 *  @param[in] radius Misses past which counting stops; @c STRINGZILLA_SIZE_MAX bounds nothing.
 *  @param[in] lane The calling lane within its group, below @p lanes.
 *  @param[in] lanes The group's lanes, a power of two up to a warp.
 *  @param[in] shuffle_index The vendor's cross-lane step within a group, such as
 *      @c sz_group_shuffle_index_cuda_ .
 *  @return The same count in every lane of the group: exact whenever it leaves at most @p radius
 *      misses, and otherwise a count that leaves more.
 */
STRINGZILLA_DEVICE sz_size_t sz_overlap_candidate_matches_simt_(
    sz_overlap_query_t const *query, sz_cptr_t text, sz_size_t length, sz_size_t radius, unsigned lane, unsigned lanes,
    sz_u32_t (*shuffle_index)(sz_u32_t, unsigned, unsigned)) {
    sz_overlap_btree_simt_t const btree = {query->btree.nodes, query->btree.levels, query->btree.level_bases};
    return sz_overlap_candidate_walk_simt_(&btree, sz_true_k, query->widths, query->powers, query->widths_count, 0,
                                           query->windows, text, length, radius, lane, lanes, shuffle_index,
                                           STRINGZILLA_NULL);
}

#pragma endregion Prepared Queries

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_OVERLAP_SIMT_CUH_
