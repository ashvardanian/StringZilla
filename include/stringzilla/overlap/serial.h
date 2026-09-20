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
enum { sz_overlap_modulus_k = 4026525731u };

/** @c 256^k mod p for @c k in @c 0…8: the multipliers that carry a prefix @c k bytes on. */
static sz_f64_t const sz_overlap_powers_of_256_k[9] = {1.0,         256.0,       65536.0,     16777216.0,  268441565.0,
                                                       270103213.0, 695485101.0, 877053692.0, 3066829947.0};

/** Keys one node holds on every backend, and the children one branch node routes to. */
enum { sz_overlap_keys_per_node_k = 16, sz_overlap_branches_per_node_k = sz_overlap_keys_per_node_k + 1 };

/** Levels a tree can span: every key is a distinct u32 residue and a node holds sixteen. */
enum { sz_overlap_btree_levels_max_k = 8 };

/** Pads the sorted key buffer: above every residue, so it sorts last. */
enum { sz_overlap_padding_key_k = 0xFFFFFFFFu };

/** The tree stores every key with this bit flipped, so signed compares order them. */
enum { sz_overlap_sign_flip_k = 0x80000000u };

/** Candidate chains advanced together in the scoring loops: one modular multiply-add step is about fifty cycles
 *  of latency against about a dozen of issue, so four independent chains fill the pipe and a fifth gains nothing. */
enum { sz_overlap_interleaved_chains_k = 4 };

#pragma region Generic Public Helpers

/** @c 256^width mod p, the multiplier that shifts a prefix past a window of @p width bytes. */
SZ_API_COMPTIME sz_f64_t sz_overlap_window_power(sz_size_t width) {
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
SZ_API_COMPTIME sz_size_t sz_overlap_btree_leaves_(sz_size_t keys_count) {
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
SZ_API_COMPTIME sz_f32_t sz_overlap_share_(sz_size_t matches, sz_size_t candidate_windows, sz_size_t query_windows) {
    sz_size_t const longer = candidate_windows > query_windows ? candidate_windows : query_windows;
    return longer ? (sz_f32_t)((sz_f64_t)matches / (sz_f64_t)longer) : 0.0f;
}

#pragma endregion Generic Public Helpers

#pragma region Serial

/** Positions one serial step advances the chain by. */
enum { sz_overlap_serial_f64x1_positions_per_step_k = 1 };

/** @c (multiplier · multiplicand + addend) mod p in @c [0, p), exact for every input below 2^32; the product fits
 *  one @c u64, and the remainder by a constant lowers to a multiply-high and a shift. */
SZ_API_COMPTIME sz_f64_t sz_overlap_serial_multiply_add_(sz_f64_t multiplier, sz_f64_t multiplicand, sz_f64_t addend) {
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
SZ_API_COMPTIME void sz_overlap_f64x1_window_hash_step_serial(sz_f64_t const *prefix_hashes_at_start,
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
SZ_API_COMPTIME sz_size_t sz_overlap_serial_branch_step_(sz_u32_t const *node, sz_u32_t key) {
    sz_size_t child = 0;
    for (sz_size_t separator = 0; separator != sz_overlap_keys_per_node_k; ++separator)
        child += (sz_i32_t)node[separator] < (sz_i32_t)key;
    return child;
}

/** The leaf compare: one when the flipped @p key sits in this node, zero otherwise. */
SZ_API_COMPTIME sz_size_t sz_overlap_serial_leaf_step_(sz_u32_t const *node, sz_u32_t key) {
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

/** One round's scratch: the B-tree arena leads the allocation at its first 64-byte boundary; the window powers,
 *  the query's chain, the candidate chains and the window-hash buffer follow it. */
typedef struct sz_overlap_scratch_t {
    /** @c u32 entries of the arena, from @ref sz_overlap_btree_entries over every width's query window hashes. */
    sz_size_t nodes_count;
    sz_size_t total_bytes; /**< What the allocator is asked for, the alignment slack included. */
} sz_overlap_scratch_t;

/** Sizes one round's scratch: the B-tree arena, the window powers, the query's chain, @p chains candidate chains,
 *  the window hashes. */
SZ_API_COMPTIME sz_overlap_scratch_t sz_overlap_scratch(sz_size_t query_length, sz_size_t longest_candidate,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t chains) {
    // Every width's query window hashes share one tree, so the arena spans their sum rather than their max.
    sz_size_t query_keys = 0;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] && window_widths[index] <= query_length)
            query_keys += query_length - window_widths[index] + 1;

    sz_overlap_scratch_t scratch;
    scratch.nodes_count = sz_overlap_btree_entries(query_keys);
    scratch.total_bytes = 63 + (scratch.nodes_count + longest_candidate) * sizeof(sz_u32_t) +
                          (window_widths_count + query_length + 1 + (longest_candidate + 1) * chains) *
                              sizeof(sz_f64_t);
    return scratch;
}

SZ_API_COMPTIME sz_status_t sz_overlap_scores_serial(sz_cptr_t query, sz_size_t query_length,
                                                     sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                     sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                     sz_f32_t *scores) {
    if (!window_widths_count) return sz_unexpected_dimensions_k;
    sz_size_t const chains = sz_overlap_interleaved_chains_k;

    sz_size_t longest_candidate = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        if (length > longest_candidate) longest_candidate = length;
    }
    sz_overlap_scratch_t const scratch = sz_overlap_scratch(query_length, longest_candidate, window_widths,
                                                            window_widths_count, chains);
    sz_ptr_t const allocation = (sz_ptr_t)alloc->allocate(scratch.total_bytes, alloc->handle);
    if (!allocation) return sz_bad_alloc_k;

    sz_size_t const chain_stride = longest_candidate + 1;
    sz_u32_t *const nodes = (sz_u32_t *)(((sz_size_t)allocation + 63) & ~(sz_size_t)63);
    sz_f64_t *const window_powers = (sz_f64_t *)(nodes + scratch.nodes_count);
    sz_f64_t *const query_prefix_hashes = window_powers + window_widths_count;
    sz_f64_t *const prefix_hashes = query_prefix_hashes + query_length + 1;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);

    // The query's chain feeds every width's window hashes, so it is built once and read `window_widths_count` times.
    query_prefix_hashes[0] = 0.0;
    sz_f64_t prior = 0.0;
    for (sz_size_t position = 0; position != query_length; ++position)
        prior = sz_overlap_f64x1_prefix_hash_step_serial(prior, query + position, query_prefix_hashes + position + 1);

    // Every width's window hashes land in the arena and one tree; a candidate window hash carries its own width,
    // so a hit is attributed to that width and a cross-width coincidence costs `2^-32`.
    sz_size_t written = 0;
    for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index) {
        sz_size_t const width = window_widths[width_index];
        window_powers[width_index] = sz_overlap_window_power(width);
        if (!width || width > query_length) continue;
        sz_size_t const query_windows = query_length - width + 1;
        for (sz_size_t window = 0; window != query_windows; ++window)
            sz_overlap_f64x1_window_hash_step_serial(query_prefix_hashes + window, query_prefix_hashes + window + width,
                                                     window_powers[width_index], nodes + written + window);
        written += query_windows;
    }
    sz_overlap_btree_t btree;
    sz_overlap_btree_prepare(nodes, sz_overlap_u32x1_btree_sort_serial(nodes, written), &btree);

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

        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t const *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f32_t *const candidate_scores = scores + (first + chain) * window_widths_count;
            for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index) {
                sz_size_t const width = window_widths[width_index];
                if (!width || width > query_length || width > length) {
                    candidate_scores[width_index] = 0.0f;
                    continue;
                }
                sz_size_t const query_windows = query_length - width + 1, windows = length - width + 1;
                for (sz_size_t window = 0; window != windows; ++window)
                    sz_overlap_f64x1_window_hash_step_serial(chain_prefix_hashes + window,
                                                             chain_prefix_hashes + window + width,
                                                             window_powers[width_index], window_hashes + window);
                sz_size_t const matches = sz_overlap_u32x1_btree_probe_serial(&btree, window_hashes, windows);
                candidate_scores[width_index] = sz_overlap_share_(matches, windows, query_windows);
            }
        }
    }

    alloc->free(allocation, scratch.total_bytes, alloc->handle);
    return sz_success_k;
}

/** The query's and the candidate's chains advance together, so their latency-bound steps overlap. */
SZ_API_COMPTIME sz_status_t sz_overlap_score_serial(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                    sz_size_t candidate_length, sz_size_t const *window_widths,
                                                    sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                    sz_f32_t *scores) {
    if (!window_widths_count) return sz_unexpected_dimensions_k;
    sz_overlap_scratch_t const scratch = sz_overlap_scratch(query_length, candidate_length, window_widths,
                                                            window_widths_count, 1);
    sz_ptr_t const allocation = (sz_ptr_t)alloc->allocate(scratch.total_bytes, alloc->handle);
    if (!allocation) return sz_bad_alloc_k;

    sz_u32_t *const nodes = (sz_u32_t *)(((sz_size_t)allocation + 63) & ~(sz_size_t)63);
    sz_f64_t *const window_powers = (sz_f64_t *)(nodes + scratch.nodes_count);
    sz_f64_t *const query_prefix_hashes = window_powers + window_widths_count;
    sz_f64_t *const candidate_prefix_hashes = query_prefix_hashes + query_length + 1;
    sz_u32_t *const window_hashes = (sz_u32_t *)(candidate_prefix_hashes + candidate_length + 1);

    // Both chains advance together up to the shorter length, then the longer runs on alone.
    sz_size_t const shorter = query_length < candidate_length ? query_length : candidate_length;
    query_prefix_hashes[0] = candidate_prefix_hashes[0] = 0.0;
    sz_f64_t query_prior = 0.0, candidate_prior = 0.0;
    sz_size_t walked = 0;
    for (; walked != shorter; ++walked) {
        query_prior = sz_overlap_f64x1_prefix_hash_step_serial(query_prior, query + walked,
                                                               query_prefix_hashes + walked + 1);
        candidate_prior = sz_overlap_f64x1_prefix_hash_step_serial(candidate_prior, candidate + walked,
                                                                   candidate_prefix_hashes + walked + 1);
    }
    for (sz_size_t position = walked; position != query_length; ++position)
        query_prior = sz_overlap_f64x1_prefix_hash_step_serial(query_prior, query + position,
                                                               query_prefix_hashes + position + 1);
    for (sz_size_t position = walked; position != candidate_length; ++position)
        candidate_prior = sz_overlap_f64x1_prefix_hash_step_serial(candidate_prior, candidate + position,
                                                                   candidate_prefix_hashes + position + 1);

    sz_size_t written = 0;
    for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index) {
        sz_size_t const width = window_widths[width_index];
        window_powers[width_index] = sz_overlap_window_power(width);
        if (!width || width > query_length) continue;
        sz_size_t const query_windows = query_length - width + 1;
        for (sz_size_t window = 0; window != query_windows; ++window)
            sz_overlap_f64x1_window_hash_step_serial(query_prefix_hashes + window, query_prefix_hashes + window + width,
                                                     window_powers[width_index], nodes + written + window);
        written += query_windows;
    }
    sz_overlap_btree_t btree;
    sz_overlap_btree_prepare(nodes, sz_overlap_u32x1_btree_sort_serial(nodes, written), &btree);

    for (sz_size_t width_index = 0; width_index != window_widths_count; ++width_index) {
        sz_size_t const width = window_widths[width_index];
        if (!width || width > query_length || width > candidate_length) {
            scores[width_index] = 0.0f;
            continue;
        }
        sz_size_t const query_windows = query_length - width + 1, windows = candidate_length - width + 1;
        for (sz_size_t window = 0; window != windows; ++window)
            sz_overlap_f64x1_window_hash_step_serial(candidate_prefix_hashes + window,
                                                     candidate_prefix_hashes + window + width,
                                                     window_powers[width_index], window_hashes + window);
        sz_size_t const matches = sz_overlap_u32x1_btree_probe_serial(&btree, window_hashes, windows);
        scores[width_index] = sz_overlap_share_(matches, windows, query_windows);
    }

    alloc->free(allocation, scratch.total_bytes, alloc->handle);
    return sz_success_k;
}

#pragma endregion Serial

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_SERIAL_H_
