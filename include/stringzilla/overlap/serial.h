/**
 *  @file include/stringzilla/overlap/serial.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Serial backend for window overlap: the prefix chain one byte at a time, the window hashes
 *      at any width and offset, and the prepared-query B-tree every candidate probes, plus the
 *      constants its SIMD backends share.
 *
 *  @sa c/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_SERIAL_H_
#define STRINGZILLA_OVERLAP_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`, `sz_cap_cpus_k`

#ifdef __cplusplus
extern "C" {
#endif

/** The 32-bit prime every window hash is reduced by; the SIMD backends hold it as an @c f64 and
 *  reduce through a rounded reciprocal. Its cheapest signed base-256 multiple within a machine word
 *  has digit weight twenty-four. */
static sz_u32_t const sz_overlap_modulus_k = 4026525731u;

/** 256ᵏ mod p for k ∈ [0, 8]: the multipliers that carry a prefix k bytes on. */
static sz_f64_t const sz_overlap_powers_of_256_k[9] = {1.0,         256.0,       65536.0,     16777216.0,  268441565.0,
                                                       270103213.0, 695485101.0, 877053692.0, 3066829947.0};

/** Keys one node holds on every backend, and the children one branch node routes to. */
enum { sz_overlap_keys_per_node_k = 16, sz_overlap_branches_per_node_k = sz_overlap_keys_per_node_k + 1 };

/** Levels a tree can span: every key is a distinct u32 residue and a node holds sixteen. */
enum { sz_overlap_btree_levels_max_k = 8 };

/** Pads the sorted key buffer: above every residue, so it sorts last. */
static sz_u32_t const sz_overlap_padding_key_k = 0xFFFFFFFFu;

/** The tree stores every key with this bit flipped, so signed compares order them. */
static sz_u32_t const sz_overlap_sign_flip_k = 0x80000000u;

/** Candidate chains advanced together in the scoring loops: one modular multiply-add step is about
 *  fifty cycles of latency against about a dozen of issue, so four independent chains fill the pipe
 *  and a fifth gains nothing. */
enum { sz_overlap_interleaved_chains_k = 4 };

/** Widths one GPU walk rolls together, their hashes and match counters held in registers; an engine
 *  with more walks each stretch of text once per group of them. */
enum { sz_overlap_gpu_widths_per_walk_k = 8 };

#pragma region Generic Public Helpers

/** 256 raised to @p width, mod p: the multiplier that shifts a prefix past that many bytes. */
STRINGZILLA_CONSTEXPR sz_f64_t sz_overlap_window_power(sz_size_t width) {
    sz_u64_t const prime = sz_overlap_modulus_k;
    sz_u64_t result = 1, base = 256ull % prime, exponent = width;
    while (exponent) {
        if (exponent & 1) result = result * base % prime;
        base = base * base % prime;
        exponent >>= 1;
    }
    return (sz_f64_t)result;
}

/**
 *  @brief One query's window hashes laid out as the tree every candidate probes.
 *
 *  The leaves hold the sorted keys, sixteen to a node, sign-flipped and padded above every residue;
 *  each branch level above holds separators, one below the first key of the child to its right, so
 *  a signed @c key > separator count is the child ordinal. The leaves lead the arena and every
 *  branch level follows, the root last.
 */
typedef struct sz_overlap_btree_t {

    /** The @b [nodes,keys_per_node] slots, leaves first, keys sign-flipped. */
    sz_u32_t const *nodes;

    /** Levels including the leaves. */
    sz_size_t levels;

    /** First node of each level, the root at index zero; a level holds only the nodes
     *  its leaves need. */
    sz_size_t level_bases[sz_overlap_btree_levels_max_k];
} sz_overlap_btree_t;

STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_btree_sorted_capacity_(sz_size_t count) {
    sz_size_t capacity = 64;
    while (capacity < count) capacity *= 2;
    return capacity;
}

/** Slots the sort pads @p count keys up to: the least power of two at or above @p count and
 *  at least 64. */
STRINGZILLA_INLINE sz_size_t sz_overlap_btree_sorted_capacity(sz_size_t count) {
    return sz_overlap_btree_sorted_capacity_(count);
}

/** Leaves a tree of @p keys_count keys needs; never zero, so an empty tree still has
 *  one padding leaf. */
STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_btree_leaves_(sz_size_t keys_count) {
    sz_size_t const leaves = sz_size_divide_round_up(keys_count, sz_overlap_keys_per_node_k);
    return leaves ? leaves : 1;
}

STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_btree_entries_(sz_size_t keys_count) {
    sz_size_t const sorted = sz_overlap_btree_sorted_capacity_(keys_count);
    sz_size_t nodes = 0;
    for (sz_size_t level_nodes = sz_overlap_btree_leaves_(keys_count);;
         level_nodes = sz_size_divide_round_up(level_nodes, sz_overlap_branches_per_node_k)) {
        nodes += level_nodes;
        if (level_nodes == 1) break;
    }
    sz_size_t const tree = nodes * sz_overlap_keys_per_node_k;
    return sorted > tree ? sorted : tree;
}

/** @c u32 entries the arena must hold for @p keys_count keys written at its start: the sort's
 *  padded capacity or every level's nodes, whichever is larger. */
STRINGZILLA_INLINE sz_size_t sz_overlap_btree_entries(sz_size_t keys_count) {
    return sz_overlap_btree_entries_(keys_count);
}

/** Windows a text of @p length bytes holds, summed over @p widths_count @p widths; a width of zero
 *  or past the text holds none. */
STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_windows(sz_size_t length, sz_u32_t const *widths, sz_size_t widths_count) {
    sz_size_t windows = 0;
    for (sz_size_t index = 0; index != widths_count; ++index)
        if (widths[index] && widths[index] <= length) windows += length - widths[index] + 1;
    return windows;
}

/** One query prepared where a device walk scores it, inside a span its caller owns: the tree
 *  over its window hashes at every width, and what a candidate's share needs beside it. */
typedef struct sz_overlap_query_t {

    /** The tree over the query's distinct window hashes, at the head of the span. */
    sz_overlap_btree_t btree;

    /** The @b [widths_count] window widths, in bytes, read where the caller keeps them. */
    sz_u32_t const *widths;

    /** The @b [widths_count] @ref sz_overlap_window_power values, at the tail of the span. */
    sz_u32_t const *powers;

    /** Entries in @c widths and @c powers. */
    sz_size_t widths_count;

    /** The query's windows summed over every width, repeats included, as @ref sz_overlap_windows
     *  counts them. */
    sz_size_t windows;
} sz_overlap_query_t;

/** Bytes of the span a device walk prepares one query of @p length bytes into at @p widths: the
 *  tree's @ref sz_overlap_btree_entries over every width's windows, then a power per width. */
STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_query_span_bytes(sz_size_t length, sz_u32_t const *widths,
                                                            sz_size_t widths_count) {
    return (sz_overlap_btree_entries_(sz_overlap_windows(length, widths, widths_count)) + widths_count) *
           sizeof(sz_u32_t);
}

/** Drops repeats from @p count ascending keys in place, answering how many distinct ones remain. */
STRINGZILLA_INLINE sz_size_t sz_overlap_btree_unique_(sz_u32_t *keys, sz_size_t count) {
    sz_size_t distinct = count != 0;
    for (sz_size_t index = 1; index < count; ++index)
        if (keys[index] != keys[distinct - 1]) keys[distinct++] = keys[index];
    return distinct;
}

STRINGZILLA_INLINE sz_status_t sz_overlap_btree_prepare_(sz_u32_t *nodes, sz_size_t keys_count,
                                                         sz_overlap_btree_t *btree) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_u32_t const padding = (sz_u32_t)sz_overlap_padding_key_k ^ (sz_u32_t)sz_overlap_sign_flip_k;
    sz_size_t const leaves = sz_overlap_btree_leaves_(keys_count);
    sz_size_t levels = 1;
    for (sz_size_t level_nodes = leaves; level_nodes != 1;
         level_nodes = sz_size_divide_round_up(level_nodes, branches_per_node))
        ++levels;
    sz_assert_(levels <= sz_overlap_btree_levels_max_k);

    for (sz_size_t index = 0; index != keys_count; ++index) nodes[index] ^= sz_overlap_sign_flip_k;
    for (sz_size_t index = keys_count; index != leaves * keys_per_node; ++index) nodes[index] = padding;

    // A separator is the first flipped key of the child to its right, less one; the first leaf is never a right
    // child, so the smallest flipped key never wraps, and a missing child takes the padding's predecessor.
    sz_size_t base = 0, level_nodes = leaves, leaves_per_child = 1;
    btree->level_bases[levels - 1] = 0;
    for (sz_size_t below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = sz_size_divide_round_up(level_nodes, branches_per_node);
        btree->level_bases[below - 1] = base;
        for (sz_size_t node = 0; node != level_nodes; ++node)
            for (sz_size_t separator = 0; separator != keys_per_node; ++separator) {
                sz_size_t const first_leaf = (node * branches_per_node + separator + 1) * leaves_per_child;
                sz_u32_t const first_key = first_leaf < leaves ? nodes[first_leaf * keys_per_node] : padding;
                nodes[(base + node) * keys_per_node + separator] = first_key - 1;
            }
        leaves_per_child *= branches_per_node;
    }

    btree->nodes = nodes;
    btree->levels = levels;
    return sz_success_k;
}

/**
 *  @brief Lays out the tree over the @p keys_count sorted distinct keys leading @c nodes: flips the
 *      leaves in place, pads the last leaf, and writes the branch levels after the leaves. Zero
 *      keys give one padding leaf.
 *  @param[in,out] nodes Caller-owned, @ref sz_overlap_btree_entries entries, the keys at its start.
 *  @param[in] keys_count Number of sorted distinct keys leading @p nodes.
 *  @param[out] btree The forest view over @p nodes, with its level count.
 *  @return @c sz_success_k always; the level count is an invariant of sixteen keys per node,
 *      asserted, never reported.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_btree_prepare(sz_u32_t *nodes, sz_size_t keys_count,
                                                        sz_overlap_btree_t *btree) {
    return sz_overlap_btree_prepare_(nodes, keys_count, btree);
}

/** The share of its longer side's windows a pair may miss within @p radius, rounded up to 24
 *  fractional bits and full at 1 or more, so a pair's budget of misses is this times its longer
 *  side's windows, shifted down by 24, in integers alone. */
STRINGZILLA_INLINE sz_u32_t sz_overlap_misses_bound_(sz_f32_t radius) {
    if (!(radius < 1.0f)) return 1u << 24;
    if (!(radius > 0.0f)) return 0;
    sz_f32_t const scaled = radius * 16777216.0f;
    return (sz_u32_t)scaled + ((sz_f32_t)(sz_u32_t)scaled != scaled);
}

#pragma endregion Generic Public Helpers

#pragma region Engine

/**
 *  @brief A forest of prepared query trees over one block, and the widths every one of
 *      them scores at.
 *
 *  The first six members are the @ref sz_overlap_btree_t forest in tensor form: every query's slots
 *  back to back, which a kernel indexes by arithmetic where an array of structs would need a
 *  pointer chase. Whatever a tier needs beyond them - a launch geometry, a stream - sits in
 *  @c memory 's head, which only that tier reads.
 */
typedef struct sz_overlap_engine_t {

    /** Every query's tree slots, back to back, @c nodes_offsets addressing them. */
    sz_u32_t const *nodes;

    /** The @b [count+1] slot offsets into @c nodes, the last being its length. */
    sz_size_t const *nodes_offsets;

    /** The @b [count] keys each tree holds, from which its levels follow. */
    sz_u32_t const *keys_counts;

    /** The @b [widths_count] window widths, in bytes. */
    sz_u32_t const *widths;

    /** The @b [widths_count] @ref sz_overlap_window_power values. */
    sz_u32_t const *powers;

    /** The @b [count] bytes each query spans, a share's denominator. */
    sz_u32_t const *lengths;

    /** Queries prepared, which is the first axis of every output. */
    sz_size_t count;

    /** Entries in @c widths and @c powers, the last axis of an output. */
    sz_size_t widths_count;

    /** The capability whose init kernel prepared the forest, which picks every round's kernel. */
    sz_capability_t capability;

    /** What built both blocks below and what grows the second. */
    sz_allocator_t allocator;

    /** The forest's block, fixed for the engine's life, its head tier-private. */
    void *memory;

    /** Bytes of that block. */
    sz_size_t memory_bytes;

    /** The round's chains and window hashes, grown and never shrunk. */
    void *scratch;

    /** Bytes of that block, zero until the first round sizes it. */
    sz_size_t scratch_bytes;
} sz_overlap_engine_t;

/** Materializes one query's tree: the levels and their bases are a closed form of its key count. */
STRINGZILLA_CONSTEXPR sz_overlap_btree_t sz_overlap_engine_row_(sz_overlap_engine_t const *engine, sz_size_t index) {
    sz_size_t const branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t const leaves = sz_overlap_btree_leaves_(engine->keys_counts[index]);
    sz_size_t levels = 1;
    for (sz_size_t level_nodes = leaves; level_nodes != 1;
         level_nodes = sz_size_divide_round_up(level_nodes, branches_per_node))
        ++levels;

    sz_overlap_btree_t btree = {STRINGZILLA_NULL, 0, {0}};
    btree.nodes = engine->nodes + engine->nodes_offsets[index];
    btree.levels = levels;
    btree.level_bases[levels - 1] = 0;
    sz_size_t base = 0, level_nodes = leaves;
    for (sz_size_t below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = sz_size_divide_round_up(level_nodes, branches_per_node);
        btree.level_bases[below - 1] = base;
    }
    return btree;
}

/** The tier's own record at the head of @p engine 's block, where
 *  @ref sz_overlap_engine_open_ reserved it. */
STRINGZILLA_INLINE void *sz_overlap_engine_head_(sz_overlap_engine_t const *engine) {
    return (void *)(((sz_size_t)engine->memory + 63) & ~(sz_size_t)63);
}

/** The length of the longest of @p strings, which sizes a round's chains. */
STRINGZILLA_INLINE sz_size_t sz_overlap_longest_(sz_strings_view_t const *strings) {
    sz_size_t longest = 0;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < strings->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(strings, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position)
            longest = sz_max_of_two(longest, views[position].length);
    }
    return longest;
}

/**
 *  @brief Takes one block for the forest of @p queries and points every view of @p engine into it.
 *
 *  Leaves @c nodes uninitialized and @c keys_counts zero: a tier fills both with its own hashing
 *  and sort, and records the capability it did so at.
 *
 *  @param[in] head_bytes Bytes the tier keeps for itself before the arena, rounded up to
 *      a cache line.
 *  @param[in] allocator Whose residency every view of the engine inherits; stored by value, so
 *      @c _free needs no other.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k for zero widths, or @c sz_bad_alloc_k
 *      when the block cannot be taken.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_open_(sz_strings_view_t const *queries, sz_size_t const *window_widths,
                                                       sz_size_t window_widths_count, sz_size_t head_bytes,
                                                       sz_allocator_t const *allocator, sz_stream_t stream,
                                                       sz_overlap_engine_t *engine) {
    if (!window_widths_count) return sz_unexpected_dimensions_k;
    for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
        if (window_widths[width_index] > 0xFFFFFFFFu) return sz_unexpected_dimensions_k;
    sz_size_t const count = queries->count;
    sz_size_t const head = (head_bytes + 63) & ~(sz_size_t)63;

    // Every width's window hashes of one query share one tree, so its arena spans their sum rather than their max.
    sz_size_t nodes_count = 0;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(queries, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const length = views[position].length;
            if (length > 0xFFFFFFFFu) return sz_unexpected_dimensions_k;
            sz_size_t keys = 0;
            for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
                if (window_widths[width_index] && window_widths[width_index] <= length)
                    keys += length - window_widths[width_index] + 1;
            nodes_count += sz_overlap_btree_entries_(keys);
        }
    }

    sz_size_t const total_bytes = 63 + head + (count + 1) * sizeof(sz_size_t) +
                                  (nodes_count + 2 * count + 2 * window_widths_count) * sizeof(sz_u32_t);
    sz_ptr_t const allocation = (sz_ptr_t)allocator->allocate(total_bytes, allocator->handle, stream);
    if (!allocation) return sz_bad_alloc_k;

    sz_ptr_t const aligned = (sz_ptr_t)(((sz_size_t)allocation + 63) & ~(sz_size_t)63);
    sz_size_t *const nodes_offsets = (sz_size_t *)(aligned + head);
    sz_u32_t *const nodes = (sz_u32_t *)(nodes_offsets + count + 1);
    sz_u32_t *const keys_counts = nodes + nodes_count;
    sz_u32_t *const lengths = keys_counts + count;
    sz_u32_t *const widths = lengths + count;
    sz_u32_t *const powers = widths + window_widths_count;

    sz_size_t written = 0;
    for (sz_size_t block_start = 0; block_start < count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(queries, block_start, views);
        for (sz_size_t position = 0; position != block_count; ++position) {
            sz_size_t const index = block_start + position;
            sz_size_t const length = views[position].length;
            sz_size_t keys = 0;
            for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
                if (window_widths[width_index] && window_widths[width_index] <= length)
                    keys += length - window_widths[width_index] + 1;
            nodes_offsets[index] = written;
            written += sz_overlap_btree_entries_(keys);
            keys_counts[index] = 0;
            lengths[index] = (sz_u32_t)length;
        }
    }
    nodes_offsets[count] = written;
    for (sz_size_t index = 0; index != window_widths_count; ++index) {
        widths[index] = (sz_u32_t)window_widths[index];
        powers[index] = (sz_u32_t)sz_overlap_window_power(window_widths[index]);
    }

    engine->nodes = nodes, engine->nodes_offsets = nodes_offsets, engine->keys_counts = keys_counts;
    engine->widths = widths, engine->powers = powers, engine->lengths = lengths;
    engine->count = count, engine->widths_count = window_widths_count;
    engine->capability = 0;
    engine->allocator = *allocator;
    engine->memory = allocation, engine->memory_bytes = total_bytes;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
    return sz_success_k;
}

/** Bytes one round's @p chains interleaved chains need over its longest candidate, its window
 *  hashes at every width of @p engine, width after width, and the @p order_entries entries its
 *  hashing order is built in. */
STRINGZILLA_INLINE sz_size_t sz_overlap_engine_round_bytes_(sz_overlap_engine_t const *engine,
                                                            sz_size_t longest_candidate, sz_size_t chains,
                                                            sz_size_t order_entries) {
    sz_size_t const hashes_bytes = (longest_candidate + 1) * chains * sizeof(sz_f64_t) +
                                   (sz_overlap_windows(longest_candidate, engine->widths, engine->widths_count) + 1) *
                                       sizeof(sz_u32_t);
    return sz_size_divide_round_up(hashes_bytes, sizeof(sz_size_t)) * sizeof(sz_size_t) +
           order_entries * sizeof(sz_size_t);
}

/** Grows @p engine 's round block to @p bytes on the device of @p stream, keeping whatever it
 *  already holds when that is enough. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_grow_(sz_overlap_engine_t *engine, sz_size_t bytes,
                                                       sz_stream_t stream) {
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    if (engine->scratch)
        engine->allocator.free(engine->scratch, engine->scratch_bytes, engine->allocator.handle, stream);
    engine->scratch = engine->allocator.allocate(bytes, engine->allocator.handle, stream);
    engine->scratch_bytes = engine->scratch ? bytes : 0;
    return engine->scratch ? sz_success_k : sz_bad_alloc_k;
}

/** Refuses a stride under the axis it spans, so no row can be written into its neighbour's. */
STRINGZILLA_INLINE sz_status_t sz_overlap_engine_strides_(sz_overlap_engine_t const *engine, sz_size_t candidates_count,
                                                          sz_size_t matches_query_stride,
                                                          sz_size_t matches_candidate_stride) {
    if (matches_candidate_stride < engine->widths_count) return sz_unexpected_dimensions_k;
    if (matches_query_stride < candidates_count * matches_candidate_stride) return sz_unexpected_dimensions_k;
    return sz_success_k;
}

/** Signature of every backend's tree probe, such as @ref sz_overlap_u32x1_btree_probe_serial. */
typedef sz_size_t (*sz_overlap_btree_probe_t)(sz_overlap_btree_t const *, sz_u32_t const *, sz_size_t);

/** Keys a bounded probe takes at least between two checks of its pair's misses, so a pair close to
 *  its bound still probes whole registers rather than a key at a time. */
enum { sz_overlap_keys_per_check_k = 64 };

/**
 *  @brief Counts one candidate's matches against one @p query of @p engine at every width, as
 *      @ref sz_overlap_scores counts them.
 *
 *  The pair may miss @p misses_bound of its own longer side, as @c sz_overlap_misses_bound_ rounds
 *  its radius, and its windows go to @p probe in runs of at most the misses it may still take, so
 *  that budget is checked once per run rather than once per window. Windows past either text are
 *  misses before any probe, so a pair whose lengths alone exhaust its budget probes none.
 *
 *  @param[in] window_hashes The candidate's window hashes, width after width, as
 *      @ref sz_overlap_windows counts them.
 *  @param[out] row The pair's count at width zero, the other widths following it.
 */
STRINGZILLA_INLINE void sz_overlap_pair_scores_(sz_overlap_engine_t const *engine, sz_size_t query,
                                                sz_u32_t const *window_hashes, sz_size_t length, sz_u32_t misses_bound,
                                                sz_overlap_btree_probe_t probe, sz_u32_t *row) {
    sz_size_t const widths_count = engine->widths_count;
    sz_size_t const query_length = engine->lengths[query];
    sz_size_t longer = 0, reachable = 0, width_index;
    for (width_index = 0; width_index != widths_count; ++width_index) {
        sz_size_t const width = engine->widths[width_index];
        sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
        sz_size_t const query_windows = width && width <= query_length ? query_length - width + 1 : 0;
        longer += sz_max_of_two(windows, query_windows);
        reachable += query_windows ? windows : 0;
    }

    sz_size_t const bound = (sz_size_t)(((sz_u64_t)misses_bound * longer) >> 24);
    sz_overlap_btree_t const btree = sz_overlap_engine_row_(engine, query);
    sz_size_t misses = longer - reachable;
    sz_u32_t const *keys = window_hashes;
    for (width_index = 0; width_index != widths_count && misses <= bound; ++width_index) {
        sz_size_t const width = engine->widths[width_index];
        sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
        sz_size_t const probed_windows = width && width <= query_length ? windows : 0;
        // Within the slack a run takes every window left, past it the slack and one more.
        sz_size_t found = 0, probed = 0;
        while (probed != probed_windows && misses + probed - found <= bound) {
            sz_size_t const slack = bound - (misses + probed - found);
            sz_size_t const run = sz_min_of_two(probed_windows - probed,
                                                sz_max_of_two(slack + 1, (sz_size_t)sz_overlap_keys_per_check_k));
            found += probe(&btree, keys + probed, run);
            probed += run;
        }
        misses += probed - found;
        row[width_index] = (sz_u32_t)found;
        keys += windows;
    }
    // Whichever check stopped the pair, past its budget every width counts zero.
    if (misses > bound)
        for (width_index = 0; width_index != widths_count; ++width_index) row[width_index] = 0;
}

/** What one round hashes, and where it counts each hashed candidate against every query. */
typedef struct sz_overlap_round_t {

    /** The candidates' indices in the order the round hashes them. */
    sz_size_t const *order;

    /** Candidates the round hashes. */
    sz_size_t count;

    /** The misses the radius allows, as @c sz_overlap_misses_bound_ rounds it. */
    sz_u32_t misses_bound;

    /** Where the counts go. */
    sz_u32_t *matches;

    /** Entries from one query's plane of @c matches to the next. */
    sz_size_t matches_query_stride;

    /** Entries from one candidate's row of @c matches to the next. */
    sz_size_t matches_candidate_stride;
} sz_overlap_round_t;

/** Counts @p candidate of @p round 's batch against every query, from its @p window_hashes. */
STRINGZILLA_INLINE void sz_overlap_round_score_(sz_overlap_engine_t const *engine, sz_overlap_round_t const *round,
                                                sz_size_t candidate, sz_u32_t const *window_hashes, sz_size_t length,
                                                sz_overlap_btree_probe_t probe) {
    for (sz_size_t query = 0; query != engine->count; ++query)
        sz_overlap_pair_scores_(
            engine, query, window_hashes, length, round->misses_bound, probe,
            round->matches + query * round->matches_query_stride + candidate * round->matches_candidate_stride);
}

/**
 *  @brief Grows @p engine 's round block for the cross product of @p candidates and lays out a
 *      @p round that hashes them ascending by length, so the chains one group interleaves end
 *      together and no long remainder steps alone.
 *  @param[out] longest_candidate The longest of @p candidates, which strides the chains.
 *  @return @c sz_success_k, with an empty @p round when either batch is empty;
 *      @c sz_unexpected_dimensions_k when either stride is under the axis it spans; or
 *      @c sz_bad_alloc_k.
 */
STRINGZILLA_INLINE sz_status_t sz_overlap_scores_round_(sz_overlap_engine_t *engine,
                                                        sz_strings_view_t const *candidates, sz_f32_t radius,
                                                        sz_u32_t *matches, sz_size_t matches_query_stride,
                                                        sz_size_t matches_candidate_stride, sz_size_t chains,
                                                        sz_stream_t stream, sz_overlap_round_t *round,
                                                        sz_size_t *longest_candidate) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, matches_query_stride,
                                                              matches_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    round->count = 0;
    if (!candidates->count || !engine->count) return sz_success_k;
    *longest_candidate = sz_overlap_longest_(candidates);
    sz_status_t const grown = sz_overlap_engine_grow_(
        engine, sz_overlap_engine_round_bytes_(engine, *longest_candidate, chains, 3 * candidates->count), stream);
    if (grown != sz_success_k) return grown;
    sz_size_t *const area = (sz_size_t *)((sz_ptr_t)engine->scratch +
                                          sz_overlap_engine_round_bytes_(engine, *longest_candidate, chains, 0));
    round->order = sz_strings_order_by_length_(candidates, area);
    round->count = candidates->count;
    round->misses_bound = sz_overlap_misses_bound_(radius);
    round->matches = matches;
    round->matches_query_stride = matches_query_stride, round->matches_candidate_stride = matches_candidate_stride;
    return sz_success_k;
}

/** Returns both of @p engine 's blocks to the allocator they were built with, once the work queued
 *  on @p stream is done with them, and leaves it empty. */
STRINGZILLA_INLINE void sz_overlap_engine_close_(sz_overlap_engine_t *engine, sz_stream_t stream) {
    if (engine->scratch)
        engine->allocator.free(engine->scratch, engine->scratch_bytes, engine->allocator.handle, stream);
    if (engine->memory) engine->allocator.free(engine->memory, engine->memory_bytes, engine->allocator.handle, stream);
    engine->nodes = STRINGZILLA_NULL, engine->nodes_offsets = STRINGZILLA_NULL, engine->keys_counts = STRINGZILLA_NULL;
    engine->widths = STRINGZILLA_NULL, engine->powers = STRINGZILLA_NULL, engine->lengths = STRINGZILLA_NULL;
    engine->count = 0, engine->widths_count = 0, engine->capability = 0;
    engine->memory = STRINGZILLA_NULL, engine->memory_bytes = 0;
    engine->scratch = STRINGZILLA_NULL, engine->scratch_bytes = 0;
}

#pragma endregion Engine

#pragma region Serial

/** Positions one serial step advances the chain by. */
enum { sz_overlap_f64x1_positions_per_step_serial_k = 1 };

/** (multiplier · multiplicand + addend) mod p, in [0, p), exact for every input below 2³²: the
 *  product fits one @c u64, and the remainder by a constant becomes a multiply-high and a shift. */
STRINGZILLA_CONSTEXPR sz_f64_t sz_overlap_multiply_add_serial_(sz_f64_t multiplier, sz_f64_t multiplicand,
                                                               sz_f64_t addend) {
    sz_u64_t const product = (sz_u64_t)multiplier * (sz_u64_t)multiplicand + (sz_u64_t)addend;
    return (sz_f64_t)(product % (sz_u64_t)sz_overlap_modulus_k);
}

STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x1_prefix_hash_step_serial_(sz_f64_t prior, sz_cptr_t text,
                                                                      sz_f64_t *prefix_hashes) {
    sz_f64_t const next = sz_overlap_multiply_add_serial_(prior, sz_overlap_powers_of_256_k[1],
                                                          (sz_f64_t)(sz_u8_t)text[0]);
    prefix_hashes[0] = next;
    return next;
}

/** Advances the chain over one byte, writing @c P(k+1) and answering it for the next step. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x1_prefix_hash_step_serial(sz_f64_t prior, sz_cptr_t text,
                                                                     sz_f64_t *prefix_hashes) {
    return sz_overlap_f64x1_prefix_hash_step_serial_(prior, text, prefix_hashes);
}

/** The last step over @p count positions, fewer than a full step's. */
STRINGZILLA_INLINE sz_f64_t sz_overlap_f64x1_prefix_hash_step_tail_serial(sz_f64_t prior, sz_cptr_t text,
                                                                          sz_size_t count, sz_f64_t *prefix_hashes) {
    return count ? sz_overlap_f64x1_prefix_hash_step_serial_(prior, text, prefix_hashes) : prior;
}

/**
 *  @brief One position's window hash: H(i, w) = P(i + w) − P(i) · bʷ, a full-width hash
 *      under the modulus.
 *  @param[in] prefix_hashes_at_start The prefix hash at the window's first position.
 *  @param[in] prefix_hashes_at_end The prefix hash past the window's last position.
 *  @param[in] window_power @ref sz_overlap_window_power for this window's width.
 *  @param[out] window_hashes The window's hash.
 */
STRINGZILLA_CONSTEXPR void sz_overlap_f64x1_window_hash_step_serial(sz_f64_t const *prefix_hashes_at_start,
                                                                    sz_f64_t const *prefix_hashes_at_end,
                                                                    sz_f64_t window_power, sz_u32_t *window_hashes) {
    sz_f64_t const shifted = sz_overlap_multiply_add_serial_(prefix_hashes_at_start[0], window_power, 0.0);
    sz_f64_t residue = prefix_hashes_at_end[0] - shifted;
    if (residue < 0.0) residue += (sz_f64_t)sz_overlap_modulus_k;
    window_hashes[0] = (sz_u32_t)residue;
}

/** The last step over @p count positions, fewer than a full step's. */
STRINGZILLA_INLINE void sz_overlap_f64x1_window_hash_step_tail_serial(sz_f64_t const *prefix_hashes_at_start,
                                                                      sz_f64_t const *prefix_hashes_at_end,
                                                                      sz_f64_t window_power, sz_size_t count,
                                                                      sz_u32_t *window_hashes) {
    if (count)
        sz_overlap_f64x1_window_hash_step_serial(prefix_hashes_at_start, prefix_hashes_at_end, window_power,
                                                 window_hashes);
}

/** Branchless compare-exchange: @p lower keeps the smaller key, unsigned. */
STRINGZILLA_CONSTEXPR void sz_overlap_exchange_serial_(sz_u32_t *lower, sz_u32_t *upper) {
    sz_u32_t const first = *lower, second = *upper;
    *lower = first < second ? first : second;
    *upper = first < second ? second : first;
}

STRINGZILLA_INLINE sz_size_t sz_overlap_u32x1_btree_sort_serial_(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity_(count);
    for (sz_size_t position = count; position != capacity; ++position) keys[position] = sz_overlap_padding_key_k;

    // Every merge opens mirrored, key `i` against key `phase - 1 - i`, so no run is ever descending.
    for (sz_size_t phase = 2; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; ++offset)
                sz_overlap_exchange_serial_(keys + start + offset, keys + start + phase - 1 - offset);
        for (sz_size_t distance = phase / 4; distance != 0; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; ++offset)
                    sz_overlap_exchange_serial_(keys + start + offset, keys + start + distance + offset);
    }
    return sz_overlap_btree_unique_(keys, count);
}

/** Sorts @p count keys ascending in place, unsigned, and drops repeats, answering how many remain;
 *  the buffer holds @ref sz_overlap_btree_sorted_capacity entries. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x1_btree_sort_serial(sz_u32_t *keys, sz_size_t count) {
    return sz_overlap_u32x1_btree_sort_serial_(keys, count);
}

/** One branch level: the child ordinal a flipped @p key descends into, the count of
 *  separators below it. */
STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_branch_step_serial_(sz_u32_t const *node, sz_u32_t key) {
    sz_size_t child = 0;
    for (sz_size_t separator = 0; separator != sz_overlap_keys_per_node_k; ++separator)
        child += (sz_i32_t)node[separator] < (sz_i32_t)key;
    return child;
}

/** The leaf compare: one when the flipped @p key sits in this node, zero otherwise. */
STRINGZILLA_CONSTEXPR sz_size_t sz_overlap_leaf_step_serial_(sz_u32_t const *node, sz_u32_t key) {
    sz_size_t found = 0;
    for (sz_size_t position = 0; position != sz_overlap_keys_per_node_k; ++position) found |= node[position] == key;
    return found;
}

STRINGZILLA_INLINE sz_size_t sz_overlap_u32x1_btree_probe_serial_(sz_overlap_btree_t const *btree, sz_u32_t const *keys,
                                                                  sz_size_t count) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t matches = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_u32_t const key = keys[index] ^ sz_overlap_sign_flip_k;
        sz_size_t node = 0;
        for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
            node = node * branches_per_node +
                   sz_overlap_branch_step_serial_(btree->nodes + (btree->level_bases[level] + node) * keys_per_node,
                                                  key);
        matches += sz_overlap_leaf_step_serial_(
            btree->nodes + (btree->level_bases[btree->levels - 1] + node) * keys_per_node, key);
    }
    return matches;
}

/** Counts how many of @p count raw keys of one candidate the tree holds. */
STRINGZILLA_INLINE sz_size_t sz_overlap_u32x1_btree_probe_serial(sz_overlap_btree_t const *btree, sz_u32_t const *keys,
                                                                 sz_size_t count) {
    return sz_overlap_u32x1_btree_probe_serial_(btree, keys, count);
}

/**
 *  @brief Hashes every query of @p queries and sorts its windows into its tree, on the host.
 *
 *  Every tier builds its forest this way, since a sort is neither a scan nor a map; only where the
 *  forest lives differs, so the device tiers call this over a block their kernels reach.
 *
 *  @param[in] queries Read on the host, in any layout over host-readable texts.
 *  @param[out] chain Host working space of the longest query's length plus one @c f64 entries.
 */
STRINGZILLA_INLINE void sz_overlap_engine_fill_serial_(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                       sz_f64_t *chain) {
    // The arena and the key counts stay writable until the engine is handed back; its readers see them const.
    sz_u32_t *const nodes = (sz_u32_t *)engine->nodes;
    sz_u32_t *const keys_counts = (sz_u32_t *)engine->keys_counts;
    sz_string_view_t views[64];
    for (sz_size_t block_start = 0; block_start < engine->count; block_start += 64) {
        sz_size_t const block_count = sz_strings_get_block_(queries, block_start, views);
        for (sz_size_t block_position = 0; block_position != block_count; ++block_position) {
            sz_size_t const index = block_start + block_position;
            sz_cptr_t const text = views[block_position].start;
            sz_size_t const length = engine->lengths[index];
            sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
            chain[0] = 0.0;
            sz_f64_t prior = 0.0;
            for (sz_size_t position = 0; position != length; ++position)
                prior = sz_overlap_f64x1_prefix_hash_step_serial_(prior, text + position, chain + position + 1);

            // Every width's window hashes land in one tree. A candidate window hash carries its own
            // width, so a hit is attributed to it and a cross-width coincidence costs `2^-32`.
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
    }
}

#if STRINGZILLA_TARGET_SERIAL

/**
 *  @brief Prepares every query of @p queries into one block, hashing and sorting on
 *      the serial tier.
 *  @param[in] candidates_budget Ignored, as a host round keeps no state per candidate.
 *  @param[in] allocator Where the forest's block comes from, or @c STRINGZILLA_NULL for the
 *      default host allocator.
 *  @sa sz_overlap_engine_init
 */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_serial(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_unused_(candidates_budget);
    sz_allocator_t host;
    if (allocator) host = *allocator;
    else {
        sz_status_t const status = sz_allocator_init_heap(&host);
        if (status != sz_success_k) return status;
    }
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count, 0, &host, stream,
                                                       engine);
    if (opened != sz_success_k) return opened;

    // The chain is the engine's own round block, so the first round reuses what the longest query
    // already asked for.
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_status_t const grown = sz_overlap_engine_grow_(engine, (longest_query + 1) * sizeof(sz_f64_t), stream);
    if (grown != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return grown;
    }

    sz_overlap_engine_fill_serial_(engine, queries, (sz_f64_t *)engine->scratch);
    engine->capability = sz_cap_serial_k;
    return sz_success_k;
}

/** Hashes the candidates of @p round, four chains at a time, and scores each as @p round says. */
STRINGZILLA_INLINE void sz_overlap_round_serial_(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                 sz_size_t longest_candidate, sz_overlap_round_t const *round) {
    sz_size_t const chains = sz_overlap_interleaved_chains_k;
    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefix_hashes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);

    // One chain per candidate, not one per candidate per width, and four chains at a time: each
    // step's modular multiply-add waits on the last, so four independent chains fill the pipeline
    // one alone leaves idle. A short last group repeats its first candidate in the spare chains, so
    // every group runs the same four-chain loop.
    for (sz_size_t first = 0; first < round->count; first += chains) {
        sz_size_t const interleaved = sz_min_of_two(round->count - first, chains);
        sz_cptr_t texts[sz_overlap_interleaved_chains_k];
        sz_size_t lengths[sz_overlap_interleaved_chains_k];
        sz_f64_t priors[sz_overlap_interleaved_chains_k];
        sz_size_t indices[sz_overlap_interleaved_chains_k];
        sz_string_view_t views[sz_overlap_interleaved_chains_k];
        sz_size_t chain, shortest = STRINGZILLA_SIZE_MAX;
        for (chain = 0; chain != chains; ++chain)
            indices[chain] = round->order[first + (chain < interleaved ? chain : 0)];
        sz_strings_get_(candidates, indices, chains, views);
        for (chain = 0; chain != chains; ++chain) {
            texts[chain] = views[chain].start, lengths[chain] = views[chain].length;
            priors[chain] = prefix_hashes[chain * chain_stride] = 0.0;
            shortest = sz_min_of_two(shortest, lengths[chain]);
        }
        sz_size_t position = 0;
        for (; position != shortest; ++position)
            for (chain = 0; chain != chains; ++chain)
                priors[chain] = sz_overlap_f64x1_prefix_hash_step_serial_(
                    priors[chain], texts[chain] + position, prefix_hashes + chain * chain_stride + position + 1);

        // One candidate's window hashes at every width serve every query's tree, so the hashing
        // runs once here and the probes run once per query over what it wrote.
        for (chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f64_t running = priors[chain];
            for (sz_size_t walked = position; walked != length; ++walked)
                running = sz_overlap_f64x1_prefix_hash_step_serial_(running, texts[chain] + walked,
                                                                    chain_prefix_hashes + walked + 1);
            sz_u32_t *keys = window_hashes;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                for (sz_size_t window = 0; window != windows; ++window)
                    sz_overlap_f64x1_window_hash_step_serial(
                        chain_prefix_hashes + window, chain_prefix_hashes + window + width, power, keys + window);
                keys += windows;
            }
            sz_overlap_round_score_(engine, round, indices[chain], window_hashes, length,
                                    sz_overlap_u32x1_btree_probe_serial_);
        }
    }
}

STRINGZILLA_API sz_status_t sz_overlap_scores_serial(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_overlap_round_t round;
    sz_size_t longest_candidate = 0;
    sz_status_t const prepared = sz_overlap_scores_round_(engine, candidates, radius, matches, matches_query_stride,
                                                          matches_candidate_stride, sz_overlap_interleaved_chains_k,
                                                          stream, &round, &longest_candidate);
    if (prepared != sz_success_k) return prepared;
    sz_overlap_round_serial_(engine, candidates, longest_candidate, &round);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#pragma endregion Serial

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_SERIAL_H_
