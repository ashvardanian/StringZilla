/**
 *  @brief Serial backend for window overlap: the prefix chain one byte at a time, the window hashes at any width
 *      and offset, and the prepared-query B-tree every candidate probes, plus the constants its SIMD backends share.
 *  @file include/stringzilla/overlap/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_SERIAL_H_
#define STRINGZILLA_OVERLAP_SERIAL_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** The 32-bit prime every window hash is reduced by; the SIMD backends hold it as an @c f64 and reduce through
 *  a rounded reciprocal. Its cheapest signed base-256 multiple within a machine word has digit weight twenty-four. */
static sz_u32_t const sz_overlap_modulus_k = 4026525731u;

/** @c 256^k mod p for @c k in @c 0…8: the multipliers that carry a prefix @c k bytes on. */
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

/** Candidate chains advanced together in the scoring loops: one modular multiply-add step is about fifty cycles
 *  of latency against about a dozen of issue, so four independent chains fill the pipe and a fifth gains nothing. */
enum { sz_overlap_interleaved_chains_k = 4 };

#pragma region Generic Public Helpers

/** @c 256^width mod p, the multiplier that shifts a prefix past a window of @p width bytes. */
SZ_HELPER_AUTO sz_f64_t sz_overlap_window_power(sz_size_t width) {
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
 *  The leaves hold the sorted keys, sixteen to a node, sign-flipped and padded above every residue; each branch
 *  level above holds separators, one below the first key of the child to its right, so a signed @c key > separator
 *  count is the child ordinal. The leaves lead the arena and every branch level follows, the root last.
 */
typedef struct sz_overlap_btree_t {
    sz_u32_t const *nodes; /**< The @b [nodes,keys_per_node] slots, leaves first, keys sign-flipped. */
    sz_size_t levels;      /**< Levels including the leaves. */
    /** First node of each level, the root at index zero; a level holds only the nodes its leaves need. */
    sz_size_t level_bases[sz_overlap_btree_levels_max_k];
} sz_overlap_btree_t;

/** Slots the sort pads @p count keys up to: the least power of two at or above @p count and at least 64. */
SZ_API_COMPTIME sz_size_t sz_overlap_btree_sorted_capacity(sz_size_t count) {
    sz_size_t capacity = 64;
    while (capacity < count) capacity *= 2;
    return capacity;
}

/** Leaves a tree of @p keys_count keys needs; never zero, so an empty tree still has one padding leaf. */
SZ_HELPER_AUTO sz_size_t sz_overlap_btree_leaves_(sz_size_t keys_count) {
    sz_size_t const leaves = (keys_count + sz_overlap_keys_per_node_k - 1) / sz_overlap_keys_per_node_k;
    return leaves ? leaves : 1;
}

/** @c u32 entries the arena must hold for @p keys_count keys written at its start: the sort's padded capacity
 *  or every level's nodes, whichever is larger. */
SZ_API_COMPTIME sz_size_t sz_overlap_btree_entries(sz_size_t keys_count) {
    sz_size_t const sorted = sz_overlap_btree_sorted_capacity(keys_count);
    sz_size_t nodes = 0;
    for (sz_size_t level_nodes = sz_overlap_btree_leaves_(keys_count);;
         level_nodes = (level_nodes + sz_overlap_branches_per_node_k - 1) / sz_overlap_branches_per_node_k) {
        nodes += level_nodes;
        if (level_nodes == 1) break;
    }
    sz_size_t const tree = nodes * sz_overlap_keys_per_node_k;
    return sorted > tree ? sorted : tree;
}

/** Drops repeats from @p count ascending keys in place, answering how many distinct ones remain. */
SZ_API_COMPTIME sz_size_t sz_overlap_btree_unique_(sz_u32_t *keys, sz_size_t count) {
    sz_size_t distinct = count != 0;
    for (sz_size_t index = 1; index < count; ++index)
        if (keys[index] != keys[distinct - 1]) keys[distinct++] = keys[index];
    return distinct;
}

/**
 *  @brief Lays out the tree over @p keys_count sorted distinct keys already at @c nodes[0…): flips the leaves in
 *      place, pads the last leaf, writes the branch levels after the leaves. Zero keys give one padding leaf.
 *  @param[in] nodes Caller-owned, @ref sz_overlap_btree_entries entries, the keys at its start.
 *  @retval sz_success_k always; the level count is an invariant of sixteen keys per node, asserted, never reported.
 */
SZ_API_COMPTIME sz_status_t sz_overlap_btree_prepare(sz_u32_t *nodes, sz_size_t keys_count, sz_overlap_btree_t *btree) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_u32_t const padding = (sz_u32_t)sz_overlap_padding_key_k ^ (sz_u32_t)sz_overlap_sign_flip_k;
    sz_size_t const leaves = sz_overlap_btree_leaves_(keys_count);
    sz_size_t levels = 1;
    for (sz_size_t level_nodes = leaves; level_nodes != 1;
         level_nodes = (level_nodes + branches_per_node - 1) / branches_per_node)
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
        level_nodes = (level_nodes + branches_per_node - 1) / branches_per_node;
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

/** @p matches over the longer side's window count, zero when both are empty; only the longer side keeps the
 *  share within @c [0, 1]. */
SZ_HELPER_AUTO sz_f32_t sz_overlap_share_(sz_size_t matches, sz_size_t candidate_windows, sz_size_t query_windows) {
    sz_size_t const longer = candidate_windows > query_windows ? candidate_windows : query_windows;
    return longer ? (sz_f32_t)((sz_f64_t)matches / (sz_f64_t)longer) : 0.0f;
}

#pragma endregion Generic Public Helpers

#pragma region Engine

/**
 *  @brief A forest of prepared query trees over one block, and the widths every one of them scores at.
 *
 *  The first six members are the @ref sz_overlap_btree_t forest in tensor form: every query's slots back to back,
 *  which a kernel indexes by arithmetic where an array of structs would need a pointer chase. Whatever a tier needs
 *  beyond them - a launch geometry, a stream - sits in @c memory 's head, which only that tier reads.
 */
typedef struct sz_overlap_engine_t {
    sz_u32_t const *nodes;          /**< Every query's tree slots, back to back, @c nodes_offsets addressing them. */
    sz_size_t const *nodes_offsets; /**< The @b [count+1] slot offsets into @c nodes, the last being its length. */
    sz_u32_t const *keys_counts;    /**< The @b [count] keys each tree holds, from which its levels follow. */
    sz_u32_t const *widths;         /**< The @b [widths_count] window widths, in bytes. */
    sz_u32_t const *powers;         /**< The @b [widths_count] @ref sz_overlap_window_power values. */
    sz_u32_t const *lengths;        /**< The @b [count] bytes each query spans, a share's denominator. */
    sz_size_t count;                /**< Queries prepared, which is the first axis of every output. */
    sz_size_t widths_count;         /**< Entries in @c widths and @c powers, the last axis of an output. */
    sz_capability_t capability;     /**< The tier @c _init_* resolved, and the only one that may score with it. */
    sz_memory_allocator_t alloc;    /**< What built both blocks below and what grows the second. */
    void *memory;                   /**< The forest's block, fixed for the engine's life, its head tier-private. */
    sz_size_t memory_bytes;         /**< Bytes of that block. */
    void *scratch;                  /**< The round's chains and window hashes, grown and never shrunk. */
    sz_size_t scratch_bytes;        /**< Bytes of that block, zero until the first round sizes it. */
} sz_overlap_engine_t;

/** Materializes one query's tree: the levels and their bases are a closed form of its key count. */
SZ_HELPER_AUTO sz_overlap_btree_t sz_overlap_engine_row_(sz_overlap_engine_t const *engine, sz_size_t index) {
    sz_size_t const branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t const leaves = sz_overlap_btree_leaves_(engine->keys_counts[index]);
    sz_size_t levels = 1;
    for (sz_size_t level_nodes = leaves; level_nodes != 1;
         level_nodes = (level_nodes + branches_per_node - 1) / branches_per_node)
        ++levels;

    sz_overlap_btree_t btree = {SZ_NULL, 0, {0}};
    btree.nodes = engine->nodes + engine->nodes_offsets[index];
    btree.levels = levels;
    btree.level_bases[levels - 1] = 0;
    sz_size_t base = 0, level_nodes = leaves;
    for (sz_size_t below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = (level_nodes + branches_per_node - 1) / branches_per_node;
        btree.level_bases[below - 1] = base;
    }
    return btree;
}

/** The tier's own record at the head of @p engine 's block, where @ref sz_overlap_engine_open_ reserved it. */
SZ_API_COMPTIME void *sz_overlap_engine_head_(sz_overlap_engine_t const *engine) {
    return (void *)(((sz_size_t)engine->memory + 63) & ~(sz_size_t)63);
}

/**
 *  @brief Takes one block for @p queries ' forest and points every view of @p engine into it.
 *
 *  Leaves @c nodes uninitialized and @c keys_counts zero: a tier fills both with its own hashing and sort, and
 *  records the capability it did so at.
 *  @param[in] head_bytes Bytes the tier keeps for itself before the arena, rounded up to a cache line.
 *  @param[in] alloc Whose residency every view of the engine inherits; stored by value, so @c _free needs no other.
 *  @retval sz_unexpected_dimensions_k for zero widths.
 *  @retval sz_bad_alloc_k when the block cannot be taken.
 */
SZ_API_COMPTIME sz_status_t sz_overlap_engine_open_(sz_sequence_t const *queries, sz_size_t const *window_widths,
                                                    sz_size_t window_widths_count, sz_size_t head_bytes,
                                                    sz_memory_allocator_t const *alloc, sz_overlap_engine_t *engine) {
    if (!window_widths_count) return sz_unexpected_dimensions_k;
    sz_size_t const count = queries->count;
    sz_size_t const head = (head_bytes + 63) & ~(sz_size_t)63;

    // Every width's window hashes of one query share one tree, so its arena spans their sum rather than their max.
    sz_size_t nodes_count = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_size_t const length = queries->get_length(queries->handle, index);
        sz_size_t keys = 0;
        for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
            if (window_widths[width_index] && window_widths[width_index] <= length)
                keys += length - window_widths[width_index] + 1;
        nodes_count += sz_overlap_btree_entries(keys);
    }

    sz_size_t const total_bytes = 63 + head + (count + 1) * sizeof(sz_size_t) +
                                  (nodes_count + 2 * count + 2 * window_widths_count) * sizeof(sz_u32_t);
    sz_ptr_t const allocation = (sz_ptr_t)alloc->allocate(total_bytes, alloc->handle);
    if (!allocation) return sz_bad_alloc_k;

    sz_ptr_t const aligned = (sz_ptr_t)(((sz_size_t)allocation + 63) & ~(sz_size_t)63);
    sz_size_t *const nodes_offsets = (sz_size_t *)(aligned + head);
    sz_u32_t *const nodes = (sz_u32_t *)(nodes_offsets + count + 1);
    sz_u32_t *const keys_counts = nodes + nodes_count;
    sz_u32_t *const lengths = keys_counts + count;
    sz_u32_t *const widths = lengths + count;
    sz_u32_t *const powers = widths + window_widths_count;

    sz_size_t written = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_size_t const length = queries->get_length(queries->handle, index);
        sz_size_t keys = 0;
        for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index)
            if (window_widths[width_index] && window_widths[width_index] <= length)
                keys += length - window_widths[width_index] + 1;
        nodes_offsets[index] = written;
        written += sz_overlap_btree_entries(keys);
        keys_counts[index] = 0;
        lengths[index] = (sz_u32_t)length;
    }
    nodes_offsets[count] = written;
    for (sz_size_t index = 0; index != window_widths_count; ++index) {
        widths[index] = (sz_u32_t)window_widths[index];
        powers[index] = (sz_u32_t)sz_overlap_window_power(window_widths[index]);
    }

    engine->nodes = nodes, engine->nodes_offsets = nodes_offsets, engine->keys_counts = keys_counts;
    engine->widths = widths, engine->powers = powers, engine->lengths = lengths;
    engine->count = count, engine->widths_count = window_widths_count;
    engine->capability = sz_caps_none_k;
    engine->alloc = *alloc;
    engine->memory = allocation, engine->memory_bytes = total_bytes;
    engine->scratch = SZ_NULL, engine->scratch_bytes = 0;
    return sz_success_k;
}

/** Bytes one round's @p chains interleaved chains and its window hashes need over its longest candidate. */
SZ_API_COMPTIME sz_size_t sz_overlap_engine_round_bytes_(sz_size_t longest_candidate, sz_size_t chains) {
    return (longest_candidate + 1) * chains * sizeof(sz_f64_t) + (longest_candidate + 1) * sizeof(sz_u32_t);
}

/** Grows @p engine 's round block to @p bytes, keeping whatever it already holds when that is enough. */
SZ_API_COMPTIME sz_status_t sz_overlap_engine_grow_(sz_overlap_engine_t *engine, sz_size_t bytes) {
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    if (engine->scratch) engine->alloc.free(engine->scratch, engine->scratch_bytes, engine->alloc.handle);
    engine->scratch = engine->alloc.allocate(bytes, engine->alloc.handle);
    engine->scratch_bytes = engine->scratch ? bytes : 0;
    return engine->scratch ? sz_success_k : sz_bad_alloc_k;
}

/** Refuses a stride under the axis it spans, so no row can be written into its neighbour's. */
SZ_API_COMPTIME sz_status_t sz_overlap_engine_strides_(sz_overlap_engine_t const *engine, sz_size_t candidates_count,
                                                       sz_size_t scores_query_stride,
                                                       sz_size_t scores_candidate_stride) {
    if (scores_candidate_stride < engine->widths_count) return sz_unexpected_dimensions_k;
    if (scores_query_stride < candidates_count * scores_candidate_stride) return sz_unexpected_dimensions_k;
    return sz_success_k;
}

/** Returns both of @p engine 's blocks to the allocator they were built with, and leaves it empty. */
SZ_API_COMPTIME void sz_overlap_engine_close_(sz_overlap_engine_t *engine) {
    if (engine->scratch) engine->alloc.free(engine->scratch, engine->scratch_bytes, engine->alloc.handle);
    if (engine->memory) engine->alloc.free(engine->memory, engine->memory_bytes, engine->alloc.handle);
    engine->nodes = SZ_NULL, engine->nodes_offsets = SZ_NULL, engine->keys_counts = SZ_NULL;
    engine->widths = SZ_NULL, engine->powers = SZ_NULL, engine->lengths = SZ_NULL;
    engine->count = 0, engine->widths_count = 0, engine->capability = sz_caps_none_k;
    engine->memory = SZ_NULL, engine->memory_bytes = 0;
    engine->scratch = SZ_NULL, engine->scratch_bytes = 0;
}

#pragma endregion Engine

#pragma region Serial

/** Positions one serial step advances the chain by. */
enum { sz_overlap_serial_f64x1_positions_per_step_k = 1 };

/** @c (multiplier · multiplicand + addend) mod p in @c [0, p), exact for every input below 2^32; the product fits
 *  one @c u64, and the remainder by a constant lowers to a multiply-high and a shift. */
SZ_HELPER_AUTO sz_f64_t sz_overlap_serial_multiply_add_(sz_f64_t multiplier, sz_f64_t multiplicand, sz_f64_t addend) {
    sz_u64_t const product = (sz_u64_t)multiplier * (sz_u64_t)multiplicand + (sz_u64_t)addend;
    return (sz_f64_t)(product % (sz_u64_t)sz_overlap_modulus_k);
}

/** Advances the chain over one byte, writing @c P(k+1) and answering it for the next step. */
SZ_API_COMPTIME sz_f64_t sz_overlap_f64x1_prefix_hash_step_serial(sz_f64_t prior, sz_cptr_t text,
                                                                  sz_f64_t *prefix_hashes) {
    sz_f64_t const next = sz_overlap_serial_multiply_add_(prior, sz_overlap_powers_of_256_k[1],
                                                          (sz_f64_t)(sz_u8_t)text[0]);
    prefix_hashes[0] = next;
    return next;
}

/** The last step over @p count positions, fewer than a full step's. */
SZ_API_COMPTIME sz_f64_t sz_overlap_f64x1_prefix_hash_step_tail_serial(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                                       sz_f64_t *prefix_hashes) {
    return count ? sz_overlap_f64x1_prefix_hash_step_serial(prior, text, prefix_hashes) : prior;
}

/**
 *  @brief One position's window hash: @c H(i,w) = P(i+w) - P(i)·b^w, a full-width hash under the modulus.
 *  @param[in] window_power @ref sz_overlap_window_power for this window's width.
 */
SZ_HELPER_AUTO void sz_overlap_f64x1_window_hash_step_serial(sz_f64_t const *prefix_hashes_at_start,
                                                             sz_f64_t const *prefix_hashes_at_end,
                                                             sz_f64_t window_power, sz_u32_t *window_hashes) {
    sz_f64_t const shifted = sz_overlap_serial_multiply_add_(prefix_hashes_at_start[0], window_power, 0.0);
    sz_f64_t residue = prefix_hashes_at_end[0] - shifted;
    if (residue < 0.0) residue += (sz_f64_t)sz_overlap_modulus_k;
    window_hashes[0] = (sz_u32_t)residue;
}

/** The last step over @p count positions, fewer than a full step's. */
SZ_API_COMPTIME void sz_overlap_f64x1_window_hash_step_tail_serial(sz_f64_t const *prefix_hashes_at_start,
                                                                   sz_f64_t const *prefix_hashes_at_end,
                                                                   sz_f64_t window_power, sz_size_t count,
                                                                   sz_u32_t *window_hashes) {
    if (count)
        sz_overlap_f64x1_window_hash_step_serial(prefix_hashes_at_start, prefix_hashes_at_end, window_power,
                                                 window_hashes);
}

/** Branchless compare-exchange: @p lower keeps the smaller key, unsigned. */
SZ_API_COMPTIME void sz_overlap_serial_exchange_(sz_u32_t *lower, sz_u32_t *upper) {
    sz_u32_t const first = *lower, second = *upper;
    *lower = first < second ? first : second;
    *upper = first < second ? second : first;
}

/** Sorts @p count keys ascending in place, unsigned, and drops repeats, answering how many remain; the buffer
 *  holds @ref sz_overlap_btree_sorted_capacity entries. */
SZ_API_COMPTIME sz_size_t sz_overlap_u32x1_btree_sort_serial(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity(count);
    for (sz_size_t position = count; position != capacity; ++position) keys[position] = sz_overlap_padding_key_k;

    // Every merge opens mirrored, key `i` against key `phase - 1 - i`, so no run is ever descending.
    for (sz_size_t phase = 2; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; ++offset)
                sz_overlap_serial_exchange_(keys + start + offset, keys + start + phase - 1 - offset);
        for (sz_size_t distance = phase / 4; distance != 0; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; ++offset)
                    sz_overlap_serial_exchange_(keys + start + offset, keys + start + distance + offset);
    }
    return sz_overlap_btree_unique_(keys, count);
}

/** One branch level: the child ordinal a flipped @p key descends into, the count of separators below it. */
SZ_HELPER_AUTO sz_size_t sz_overlap_serial_branch_step_(sz_u32_t const *node, sz_u32_t key) {
    sz_size_t child = 0;
    for (sz_size_t separator = 0; separator != sz_overlap_keys_per_node_k; ++separator)
        child += (sz_i32_t)node[separator] < (sz_i32_t)key;
    return child;
}

/** The leaf compare: one when the flipped @p key sits in this node, zero otherwise. */
SZ_HELPER_AUTO sz_size_t sz_overlap_serial_leaf_step_(sz_u32_t const *node, sz_u32_t key) {
    sz_size_t found = 0;
    for (sz_size_t position = 0; position != sz_overlap_keys_per_node_k; ++position) found |= node[position] == key;
    return found;
}

/** Counts how many of @p count raw keys of one candidate the tree holds. */
SZ_API_COMPTIME sz_size_t sz_overlap_u32x1_btree_probe_serial(sz_overlap_btree_t const *btree, sz_u32_t const *keys,
                                                              sz_size_t count) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t matches = 0;
    for (sz_size_t index = 0; index != count; ++index) {
        sz_u32_t const key = keys[index] ^ sz_overlap_sign_flip_k;
        sz_size_t node = 0;
        for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
            node = node * branches_per_node +
                   sz_overlap_serial_branch_step_(btree->nodes + (btree->level_bases[level] + node) * keys_per_node,
                                                  key);
        matches += sz_overlap_serial_leaf_step_(
            btree->nodes + (btree->level_bases[btree->levels - 1] + node) * keys_per_node, key);
    }
    return matches;
}

/**
 *  @brief Prepares every query of @p queries into one block, hashing and sorting on the serial tier.
 *  @param[in] alloc Where the forest's block comes from, or @c SZ_NULL for the default host allocator.
 *  @sa sz_overlap_engine_init_cpu
 */
SZ_API_COMPTIME sz_status_t sz_overlap_engine_init_serial(sz_sequence_t const *queries, sz_size_t const *window_widths,
                                                          sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                          sz_overlap_engine_t *engine) {
    sz_memory_allocator_t host;
    if (alloc) host = *alloc;
    else sz_memory_allocator_init_default(&host);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count, 0, &host, engine);
    if (opened != sz_success_k) return opened;

    // The chain is the engine's own round block, so the first round reuses what the longest query already asked for.
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_status_t const grown = sz_overlap_engine_grow_(engine, (longest_query + 1) * sizeof(sz_f64_t));
    if (grown != sz_success_k) {
        sz_overlap_engine_close_(engine);
        return grown;
    }

    // The arena and the key counts stay writable until the engine is handed back; its readers see them const.
    sz_u32_t *const nodes = (sz_u32_t *)engine->nodes;
    sz_u32_t *const keys_counts = (sz_u32_t *)engine->keys_counts;
    sz_f64_t *const chain = (sz_f64_t *)engine->scratch;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        sz_size_t const length = engine->lengths[index];
        sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
        chain[0] = 0.0;
        sz_f64_t prior = 0.0;
        for (sz_size_t position = 0; position != length; ++position)
            prior = sz_overlap_f64x1_prefix_hash_step_serial(prior, text + position, chain + position + 1);

        // Every width's window hashes land in one tree; a candidate window hash carries its own width, so a hit is
        // attributed to that width and a cross-width coincidence costs `2^-32`.
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
        sz_size_t const distinct = sz_overlap_u32x1_btree_sort_serial(arena, written);
        sz_overlap_btree_prepare(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }

    engine->capability = sz_cap_serial_k;
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_overlap_scores_serial(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                     sz_f32_t *scores, sz_size_t scores_query_stride,
                                                     sz_size_t scores_candidate_stride) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    sz_size_t const chains = sz_overlap_interleaved_chains_k;

    sz_size_t longest_candidate = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        if (length > longest_candidate) longest_candidate = length;
    }
    sz_status_t const grown = sz_overlap_engine_grow_(
        engine, sz_overlap_engine_round_bytes_(longest_candidate, chains));
    if (grown != sz_success_k) return grown;

    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefix_hashes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);

    // One chain per candidate, not one per candidate per width, and four chains at a time: each step's modular
    // multiply-add waits on the last, so four independent chains fill the pipeline one alone leaves idle.
    for (sz_size_t first = 0; first < candidates->count; first += chains) {
        sz_size_t const interleaved = candidates->count - first < chains ? candidates->count - first : chains;
        sz_cptr_t texts[sz_overlap_interleaved_chains_k];
        sz_size_t lengths[sz_overlap_interleaved_chains_k];
        sz_f64_t priors[sz_overlap_interleaved_chains_k];
        sz_size_t shortest = SZ_SIZE_MAX;
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            texts[chain] = candidates->get_start(candidates->handle, first + chain);
            lengths[chain] = candidates->get_length(candidates->handle, first + chain);
            priors[chain] = prefix_hashes[chain * chain_stride] = 0.0;
            if (lengths[chain] < shortest) shortest = lengths[chain];
        }
        sz_size_t position = 0;
        if (interleaved == chains)
            for (; position != shortest; ++position)
                for (sz_size_t chain = 0; chain != chains; ++chain)
                    priors[chain] = sz_overlap_f64x1_prefix_hash_step_serial(
                        priors[chain], texts[chain] + position, prefix_hashes + chain * chain_stride + position + 1);
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_f64_t running = priors[chain];
            for (sz_size_t walked = position; walked != lengths[chain]; ++walked)
                running = sz_overlap_f64x1_prefix_hash_step_serial(running, texts[chain] + walked,
                                                                   chain_prefix_hashes + walked + 1);
        }

        // One candidate's window hashes at one width serve every query's tree, so the hashing runs once here and
        // the probe runs `count` times over what it wrote.
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t const *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f32_t *const candidate_scores = scores + (first + chain) * scores_candidate_stride;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                for (sz_size_t window = 0; window != windows; ++window)
                    sz_overlap_f64x1_window_hash_step_serial(chain_prefix_hashes + window,
                                                             chain_prefix_hashes + window + width, power,
                                                             window_hashes + window);
                for (sz_size_t query = 0; query != engine->count; ++query) {
                    sz_size_t const query_length = engine->lengths[query];
                    sz_f32_t *const slot = candidate_scores + query * scores_query_stride + width_index;
                    if (!windows || width > query_length) {
                        *slot = 0.0f;
                        continue;
                    }
                    sz_overlap_btree_t const btree = sz_overlap_engine_row_(engine, query);
                    sz_size_t const matches = sz_overlap_u32x1_btree_probe_serial(&btree, window_hashes, windows);
                    *slot = sz_overlap_share_(matches, windows, query_length - width + 1);
                }
            }
        }
    }
    return sz_success_k;
}

#pragma endregion Serial

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_SERIAL_H_
