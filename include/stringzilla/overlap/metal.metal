/**
 *  @file include/stringzilla/overlap/metal.metal
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Window overlap on Apple GPUs of Metal family 7, M1 and newer: one thread per segment of a
 *      candidate, each width's window hash rolled through it in registers, and one prepared query's
 *      B-tree probed per threadgroup row.
 *
 *  Nothing here needs more than Apple7: 32-bit atomics, simdgroup prefix sums and 32 KB of
 *  threadgroup memory. A later family that earns its own kernels gets its own file beside this one.
 *
 *  @sa include/stringzilla/overlap/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/overlap/simt.cuh, the CUDA sibling this mirrors step for step
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  The CUDA kernel reaches the serial helpers through @c --expt-relaxed-constexpr; MSL cannot
 *  include the C headers, whose pointers carry no address space, so the four helpers this needs -
 *  the branch step, the leaf step, the tree's level bases and the misses bound - are mirrored here,
 *  and the tests hold every count to serial's.
 *
 *  Apple GPUs have no 64-bit divider either, so a remainder over @c ulong compiles to a long
 *  emulated sequence. The window hashes are reduced with Montgomery products instead, which need
 *  only the native 32-bit @c mul and @c mulhi.
 *
 *  The forest and the round block each point inside themselves by host address, turned into the
 *  kernel's own by @ref sz_reach_metal_ against the block's host address, while the candidates'
 *  tape and the matches are bound directly.
 *
 *  A round is three dispatches. One thread per candidate would leave most of a simdgroup idle while
 *  its longest lane finishes, so candidates are cut into segments of equal length instead: the
 *  first dispatch scans their segment counts into offsets, the second scores one segment per thread
 *  and adds its matches to the candidate's counters, and the third copies them out.
 */

/** The 32-bit prime every window hash is reduced by, as @c sz_overlap_modulus_k. */
constant uint sz_overlap_modulus_metal_k = 4026525731u;

/** −p⁻¹ mod 2³², R² mod p and 256 × R mod p, for Montgomery products with R = 2³². */
constant uint sz_overlap_modulus_inverse_metal_k = 574879861u;
constant uint sz_overlap_montgomery_square_metal_k = 3066829947u;
constant uint sz_overlap_byte_shift_metal_k = 270103213u;

/** Keys one node holds, and the children one branch node routes to. */
constant uint sz_overlap_keys_per_node_metal_k = 16, sz_overlap_branches_per_node_metal_k = 17;

/** Levels a tree can span. */
constant uint sz_overlap_btree_levels_max_metal_k = 8;

/** The bit every stored key has flipped, so signed compares order them. */
constant uint sz_overlap_sign_flip_metal_k = 0x80000000u;

/** Widths one walk of a segment rolls together, their hashes and match counters held per thread, as
 *  @c sz_overlap_gpu_widths_per_walk_k. */
constant uint sz_overlap_gpu_widths_per_walk_metal_k = 8;

/** Bytes of a candidate one thread scores; the windows ending inside it are its own. */
constant uint sz_overlap_segment_bytes_metal_k = 128;

/** Threads the segment scan runs, as one threadgroup of 32 simdgroups. */
constant uint sz_overlap_scan_threads_metal_k = 1024;

/** The launch record the host copies in: host addresses inside the forest and the round block, and
 *  the round's shape. */
struct sz_overlap_arguments_metal_t {
    ulong engine_host, scratch_host;
    ulong nodes, nodes_offsets, keys_counts, widths, powers, lengths;
    ulong segments, counts;
    ulong queries_count, widths_count, candidates_count, segments_count;
    ulong matches_query_stride, matches_candidate_stride, staged_nodes_count, widest_width, candidates_layout,
        misses_bound;
};

/**
 *  @brief a × b ÷ R mod p for a, b < p: with @p b in Montgomery form, as b × R mod p, a × b mod p.
 *
 *  The low halves of a × b and m × p sum to zero mod 2³² by the choice of m, so they carry exactly
 *  when the low half of a × b is nonzero, and the high halves plus that carry stay under 2p.
 */
inline uint sz_overlap_montgomery_metal_(uint a, uint b) {
    uint const low = a * b, high = mulhi(a, b);
    uint const correction = mulhi(low * sz_overlap_modulus_inverse_metal_k, sz_overlap_modulus_metal_k);
    ulong const sum = (ulong)high + correction + (low != 0);
    return (uint)(sum >= sz_overlap_modulus_metal_k ? sum - sz_overlap_modulus_metal_k : sum);
}

/** One byte's step of a rolling window hash, <tt>H' = H * 256 + incoming - outgoing * 256^w</tt>, with
 *  @p lifted_power being 256^w in Montgomery form, as @c sz_overlap_roll_step_simt_ steps: the same
 *  residue, as both are exact modulo the same prime. */
inline uint sz_overlap_roll_step_metal_(uint hash, uint incoming, uint outgoing, uint lifted_power) {
    // Under p + 256, which still fits 32 bits.
    uint const shifted = sz_overlap_montgomery_metal_(hash, sz_overlap_byte_shift_metal_k) + incoming;
    uint const leaving = sz_overlap_montgomery_metal_(outgoing, lifted_power);
    // Wrapping is exact here: the true difference is under p + 256, below 2³².
    uint residue = shifted - leaving + (shifted < leaving ? sz_overlap_modulus_metal_k : 0u);
    if (residue >= sz_overlap_modulus_metal_k) residue -= sz_overlap_modulus_metal_k;
    return residue;
}

/** How many of a node's sixteen ascending entries sit below the flipped @p key, up to fifteen, by
 *  halving: four dependent loads where a scan takes sixteen, as @c sz_overlap_rank_simt_ does. */
inline uint sz_overlap_rank_metal_(device uint const *node, int key) {
    uint rank = (int)node[7] < key ? 8u : 0u;
    rank += (int)node[rank + 3] < key ? 4u : 0u;
    rank += (int)node[rank + 1] < key ? 2u : 0u;
    rank += (int)node[rank] < key ? 1u : 0u;
    return rank;
}

/** Separators of one branch node under @p key, the child to descend into. */
inline uint sz_overlap_branch_step_metal_(device uint const *separators, int key) {
    uint const last = sz_overlap_keys_per_node_metal_k - 1;
    uint const rank = sz_overlap_rank_metal_(separators, key);
    return rank + (rank == last && (int)separators[last] < key);
}

/** The staged node is 16-byte aligned, so its sixteen keys come in four vector loads. */
inline uint sz_overlap_branch_step_metal_(threadgroup uint const *separators, int key) {
    threadgroup int4 const *quads = (threadgroup int4 const *)separators;
    int4 const keys = int4(key);
    uint4 const below = uint4(quads[0] < keys) + uint4(quads[1] < keys) + uint4(quads[2] < keys) +
                        uint4(quads[3] < keys);
    return below.x + below.y + below.z + below.w;
}

/** One when the flipped @p key sits in the leaf. */
inline uint sz_overlap_leaf_step_metal_(device uint const *leaf, uint key) {
    return leaf[sz_overlap_rank_metal_(leaf, (int)key)] == key ? 1u : 0u;
}

inline uint sz_overlap_leaf_step_metal_(threadgroup uint const *leaf, uint key) {
    threadgroup uint4 const *quads = (threadgroup uint4 const *)leaf;
    uint4 const keys = uint4(key);
    return any((quads[0] == keys) | (quads[1] == keys) | (quads[2] == keys) | (quads[3] == keys)) ? 1u : 0u;
}

/** Whether the tree over @p tree holds one raw @p window_hash, walking it a level at a time,
 *  reading the top levels staged from @p staged_from in @p staged_nodes and any lower levels in
 *  @p tree, as @c sz_overlap_btree_probe_simt_ does. */
inline uint sz_overlap_probe_metal_(device uint const *tree, threadgroup uint const *staged_nodes, uint staged_from,
                                    uint levels, thread uint const *level_bases, uint window_hash) {
    uint const key = window_hash ^ sz_overlap_sign_flip_metal_k;
    uint node = 0;
    for (uint level = 0; level + 1 != levels; ++level) {
        uint const offset = (level_bases[level] + node) * sz_overlap_keys_per_node_metal_k;
        uint const child = offset >= staged_from
                               ? sz_overlap_branch_step_metal_(staged_nodes + (offset - staged_from), (int)key)
                               : sz_overlap_branch_step_metal_(tree + offset, (int)key);
        node = node * sz_overlap_branches_per_node_metal_k + child;
    }
    uint const offset = (level_bases[levels - 1] + node) * sz_overlap_keys_per_node_metal_k;
    return offset >= staged_from ? sz_overlap_leaf_step_metal_(staged_nodes + (offset - staged_from), key)
                                 : sz_overlap_leaf_step_metal_(tree + offset, key);
}

/** The levels of a tree of @p keys_count keys and each level's first node, the root at zero, as
 *  @c sz_overlap_engine_row_ lays them out. @return The level count. */
inline uint sz_overlap_levels_metal_(uint keys_count, thread uint *level_bases) {
    uint const branches = sz_overlap_branches_per_node_metal_k;
    uint leaves = sz_size_divide_round_up_metal_<uint>(keys_count, sz_overlap_keys_per_node_metal_k);
    if (!leaves) leaves = 1;
    uint levels = 1;
    for (uint level_nodes = leaves; level_nodes != 1;
         level_nodes = sz_size_divide_round_up_metal_<uint>(level_nodes, branches))
        ++levels;
    level_bases[levels - 1] = 0;
    uint base = 0, level_nodes = leaves;
    for (uint below = levels - 1; below != 0; --below) {
        base += level_nodes;
        level_nodes = sz_size_divide_round_up_metal_<uint>(level_nodes, branches);
        level_bases[below - 1] = base;
    }
    return levels;
}

/** The most of its @p longer side's windows a pair may miss under @p misses_bound, the radius as
 *  @c sz_overlap_misses_bound_ rounds it, in integers alone. */
inline ulong sz_overlap_misses_bound_metal_(ulong misses_bound, ulong longer) { return (misses_bound * longer) >> 24; }

/** Segments a text of @p length bytes is cut into. */
inline uint sz_overlap_segments_of_metal_(ulong length) {
    return (uint)sz_size_divide_round_up_metal_<ulong>(length, sz_overlap_segment_bytes_metal_k);
}

/** The candidate owning global segment @p segment: offsets rise, and a text with no segments owns
 *  none, so the last offset at or under @p segment names it. */
inline ulong sz_overlap_owner_metal_(device uint const *offsets, ulong candidates_count, uint segment) {
    ulong low = 0, high = candidates_count;
    while (high - low > 1) {
        ulong const middle = (low + high) / 2;
        if (offsets[middle] <= segment) low = middle;
        else high = middle;
    }
    return low;
}

/** Whether a pair's lengths alone put it past its budget under @p misses_bound: a window past
 *  either text, and every window past the shorter one, is a miss no probe can match. */
inline bool sz_overlap_unreachable_metal_(device uint const *widths, ulong widths_count, ulong query_length,
                                          ulong length, ulong misses_bound) {
    ulong longer = 0, reachable = 0;
    for (ulong index = 0; index != widths_count; ++index) {
        ulong const width = widths[index];
        ulong const windows = width && width <= length ? length - width + 1 : 0;
        ulong const query_windows = width && width <= query_length ? query_length - width + 1 : 0;
        longer += max(windows, query_windows);
        reachable += query_windows ? windows : 0;
    }
    return longer - reachable > sz_overlap_misses_bound_metal_(misses_bound, longer);
}

/**
 *  @brief Counts one segment's windows of one candidate against one prepared query, adding them to
 *      the candidate's counters.
 *
 *  The windows ending inside the segment are its own. Hashing restarts up to the widest width
 *  before it, since a window needs all of its bytes, so every window is counted by exactly one
 *  thread, from the same residues serial computes.
 *
 *  Each width rolls its own hash a byte at a time, <tt>H' = H * 256 + in - out * 256^w</tt>, the byte
 *  leaving the window read again from the text, so no width is too wide for a thread to hold. An
 *  engine with more widths than one walk rolls walks the segment once per group of them.
 */
inline void sz_overlap_sweep_metal_(device uchar *engine, device uchar *scratch, sz_strings_tape_metal_t tape,
                                    constant sz_overlap_arguments_metal_t &arguments, device uint const *tree,
                                    threadgroup uint const *staged_nodes, uint staged_from, uint levels,
                                    thread uint const *level_bases, ulong query, uint segment, ulong candidate) {
    device uint const *offsets = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.segments);
    device uint const *widths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.widths);
    device uint const *powers = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.powers);
    device uchar const *text = sz_strings_start_metal_(tape, candidate);
    ulong const text_length = sz_strings_length_metal_(tape, candidate);
    ulong const segment_start = (ulong)(segment - offsets[candidate]) * sz_overlap_segment_bytes_metal_k;
    ulong const segment_end = min(text_length, segment_start + sz_overlap_segment_bytes_metal_k);
    ulong const origin = segment_start + 1 >= arguments.widest_width ? segment_start + 1 - arguments.widest_width : 0;

    device atomic_uint *counts = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.counts) +
                                 (query * arguments.candidates_count + candidate) * arguments.widths_count;

    // A pair its lengths alone put past its budget rolls nothing; the matches kernel zeroes it.
    device uint const *lengths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.lengths);
    if (sz_overlap_unreachable_metal_(widths, arguments.widths_count, lengths[query], text_length,
                                      arguments.misses_bound))
        return;

    for (ulong first_width = 0; first_width < arguments.widths_count;
         first_width += sz_overlap_gpu_widths_per_walk_metal_k) {
        ulong const group = min(arguments.widths_count - first_width, (ulong)sz_overlap_gpu_widths_per_walk_metal_k);

        // Widths and their powers stay in registers for the whole walk; each power is lifted once
        // into Montgomery form, 256^width × R mod p, so its product with a byte comes back plain.
        uint hashes[sz_overlap_gpu_widths_per_walk_metal_k], matches[sz_overlap_gpu_widths_per_walk_metal_k];
        uint window_widths[sz_overlap_gpu_widths_per_walk_metal_k];
        uint lifted_powers[sz_overlap_gpu_widths_per_walk_metal_k];
        for (ulong index = 0; index != group; ++index) {
            hashes[index] = 0, matches[index] = 0, window_widths[index] = widths[first_width + index];
            lifted_powers[index] = sz_overlap_montgomery_metal_(powers[first_width + index],
                                                                sz_overlap_montgomery_square_metal_k);
        }

        // Bytes before `origin` count as zeros, so a window not yet full holds its own bytes' hash.
        for (ulong position = origin; position != segment_end; ++position) {
            ulong const ending = position + 1;
            uint const incoming = text[position];
            for (ulong index = 0; index != group; ++index) {
                uint const width = window_widths[index];
                if (!width) continue;
                uint const outgoing = ending - origin > width ? text[position - width] : 0u;
                uint const residue = sz_overlap_roll_step_metal_(hashes[index], incoming, outgoing,
                                                                 lifted_powers[index]);
                hashes[index] = residue;
                if (ending <= segment_start || width > ending) continue;
                matches[index] += sz_overlap_probe_metal_(tree, staged_nodes, staged_from, levels, level_bases,
                                                          residue);
            }
        }
        for (ulong index = 0; index != group; ++index)
            if (matches[index])
                atomic_fetch_add_explicit(&counts[first_width + index], matches[index], memory_order_relaxed);
    }
}

/*  A walk scoring one query against scattered candidates prepares that query itself, a simdgroup
 *  laying its tree into a span the caller owns, and counts each candidate's matches a group of
 *  lanes at a time, through the same roll and probe, as the CUDA walk's prepared queries do. */

/** Endings a lane rolls between two checks of whether a candidate can still come within its limit,
 *  as @c sz_overlap_gpu_endings_per_check_k. */
constant ulong sz_overlap_gpu_endings_per_check_metal_k = 32;

/** Windows a text of @p length bytes holds, summed over @p count @p widths, as
 *  @c sz_overlap_windows counts them: a width of zero or past the text holds none. */
inline ulong sz_overlap_windows_metal_(ulong length, device uint const *widths, ulong count) {
    ulong windows = 0;
    for (ulong index = 0; index != count; ++index)
        if (widths[index] && widths[index] <= length) windows += length - widths[index] + 1;
    return windows;
}

/** One query prepared where a walk scores it, inside a span its caller owns, as
 *  @c sz_overlap_query_t lays it out: the tree over its window hashes at every width, and what a
 *  candidate's share needs beside it. Trivially copyable, so a walk may broadcast it. */
struct sz_overlap_query_metal_t {

    /** The tree's entries, leaves first, keys sign-flipped, at the head of the span. */
    device uint const *nodes;

    /** The @b [widths_count] window widths, in bytes, read where the caller keeps them. */
    device uint const *widths;

    /** The @b [widths_count] powers of 256 at each width, mod p, at the tail of the span. */
    device uint const *powers;

    /** Entries in @c widths and @c powers. */
    ulong widths_count;

    /** The query's windows summed over every width, repeats included. */
    ulong windows;

    /** The query's bytes, so a width past them probes nothing, as serial's pairs skip it. */
    ulong length;

    /** The distinct windows the tree holds, from which its levels follow, as an engine's
     *  @c keys_counts entry. */
    ulong keys_count;
};

/**
 *  @brief Prepares one query into a simdgroup's @p span, every lane of the simdgroup taking part,
 *      as @c sz_overlap_query_prepare_simt_ prepares it.
 *
 *  Every width's window hashes, each lane rolling its own run of windows from the same residues the
 *  serial tier computes, a bitonic sort a lane per pair of every step, the repeats dropped by a
 *  ballot per 32 keys, and the tree laid out over the leaves, a lane per separator. The simdgroup
 *  is synchronized on entry, as the span may still serve the previous query, and on return.
 *
 *  @param[in] text The query, read through plain loads.
 *  @param[in] widths The @b [widths_count] window widths, kept where every walk reads them.
 *  @param[out] span Eight-byte aligned, @c sz_overlap_query_span_bytes long: the tree's entries,
 *      then a power per width.
 *  @param[out] query Points into @p span and at @p widths, complete in every lane.
 *  @param[in] lane The calling lane, below 32.
 */
inline void sz_overlap_query_prepare_metal_(device uchar const *text, ulong length, device uint const *widths,
                                            ulong widths_count, device uchar *span,
                                            thread sz_overlap_query_metal_t &query, uint lane) {
    uint const lanes = 32, per_node = sz_overlap_keys_per_node_metal_k, branches = sz_overlap_branches_per_node_metal_k;
    uint const padding_key = 0xFFFFFFFFu, padding = padding_key ^ sz_overlap_sign_flip_metal_k;
    sz_group_metal_t const group = {lanes, lane};
    ulong const windows = sz_overlap_windows_metal_(length, widths, widths_count);
    // The sort pads to a power of two of at least 64 slots, and the powers follow the larger of
    // that and every level's nodes, as @c sz_overlap_btree_entries sizes the arena.
    ulong capacity = 64, nodes = 0;
    while (capacity < windows) capacity *= 2;
    for (ulong level_nodes = max(sz_size_divide_round_up_metal_<ulong>(windows, per_node), 1ul);;
         level_nodes = sz_size_divide_round_up_metal_<ulong>(level_nodes, branches)) {
        nodes += level_nodes;
        if (level_nodes == 1) break;
    }
    device uint *const keys = (device uint *)span;
    device uint *const powers = keys + max(capacity, nodes * per_node);
    group.sync();
    // 256 to each width by squaring in Montgomery form, from one's form, R mod p.
    for (ulong index = lane; index < widths_count; index += lanes) {
        uint power = sz_overlap_montgomery_metal_(1, sz_overlap_montgomery_square_metal_k);
        uint base = sz_overlap_byte_shift_metal_k;
        for (uint exponent = widths[index]; exponent; exponent >>= 1) {
            if (exponent & 1) power = sz_overlap_montgomery_metal_(power, base);
            base = sz_overlap_montgomery_metal_(base, base);
        }
        powers[index] = sz_overlap_montgomery_metal_(power, 1);
    }
    group.sync();

    // Each lane rolls a run of one width's windows, priming its hash on the run's first window.
    ulong written = 0;
    for (ulong index = 0; index != widths_count; ++index) {
        ulong const width = widths[index];
        if (!width || width > length) continue;
        uint const lifted_power = sz_overlap_montgomery_metal_(powers[index], sz_overlap_montgomery_square_metal_k);
        ulong const width_windows = length - width + 1;
        ulong const run = sz_size_divide_round_up_metal_<ulong>(width_windows, lanes);
        ulong const first = min((ulong)lane * run, width_windows), last = min(first + run, width_windows);
        // A lane past the last window rolls nothing.
        ulong const endings = first != last ? last - first + width - 1 : 0;
        uint hash = 0;
        for (ulong ending = 1; ending <= endings; ++ending) {
            uint const outgoing = ending > width ? text[first + ending - width - 1] : 0u;
            hash = sz_overlap_roll_step_metal_(hash, text[first + ending - 1], outgoing, lifted_power);
            if (ending >= width) keys[written + first + ending - width] = hash;
        }
        written += width_windows;
    }
    for (ulong entry = windows + lane; entry < capacity; entry += lanes) keys[entry] = padding_key;
    group.sync();

    // The serial sort's network, each step's pairs spread across the lanes.
    for (ulong phase = 2; phase <= capacity; phase *= 2) {
        for (ulong entry = lane; entry < capacity / 2; entry += lanes) {
            ulong const offset = entry & (phase / 2 - 1), start = (entry - offset) * 2;
            uint const lower = keys[start + offset], upper = keys[start + phase - 1 - offset];
            keys[start + offset] = min(lower, upper), keys[start + phase - 1 - offset] = max(lower, upper);
        }
        group.sync();
        for (ulong distance = phase / 4; distance != 0; distance /= 2) {
            for (ulong entry = lane; entry < capacity / 2; entry += lanes) {
                ulong const offset = entry & (distance - 1), start = (entry - offset) * 2;
                uint const lower = keys[start + offset], upper = keys[start + distance + offset];
                keys[start + offset] = min(lower, upper), keys[start + distance + offset] = max(lower, upper);
            }
            group.sync();
        }
    }

    // A key survives unless it repeats the one before, which lane zero takes from the last pass.
    ulong distinct = 0;
    uint carried = 0;
    for (ulong base = 0; base < windows; base += lanes) {
        ulong const position = base + lane;
        uint const key = position < windows ? keys[position] : 0u;
        uint const below = group.shuffle_up(key, 1);
        bool const kept = position < windows && (position == 0 || key != (lane ? below : carried));
        uint const survivors = group.ballot(kept);
        group.sync();
        if (kept) keys[distinct + popcount(survivors & ((1u << lane) - 1u))] = key;
        distinct += popcount(survivors);
        carried = group.shuffle(key, lanes - 1);
    }
    group.sync();

    // The leaves flip and pad in place, then each separator reads the first key of the child to its
    // right, as @c sz_overlap_btree_prepare lays them.
    ulong const leaves = max(sz_size_divide_round_up_metal_<ulong>(distinct, per_node), 1ul);
    for (ulong entry = lane; entry < leaves * per_node; entry += lanes)
        keys[entry] = entry < distinct ? keys[entry] ^ sz_overlap_sign_flip_metal_k : padding;
    group.sync();
    uint level_bases[sz_overlap_btree_levels_max_metal_k];
    uint const levels = sz_overlap_levels_metal_((uint)distinct, level_bases);
    ulong level_nodes = leaves, leaves_per_child = 1;
    for (uint below = levels - 1; below != 0; --below) {
        ulong const base = level_bases[below - 1];
        level_nodes = sz_size_divide_round_up_metal_<ulong>(level_nodes, branches);
        for (ulong entry = lane; entry < level_nodes * per_node; entry += lanes) {
            ulong const node = entry / per_node, separator = entry % per_node;
            ulong const first_leaf = (node * branches + separator + 1) * leaves_per_child;
            uint const first_key = first_leaf < leaves ? keys[first_leaf * per_node] : padding;
            keys[base * per_node + entry] = first_key - 1;
        }
        leaves_per_child *= branches;
    }
    group.sync();
    query.nodes = keys;
    query.widths = widths;
    query.powers = powers;
    query.widths_count = widths_count;
    query.windows = windows;
    query.length = length;
    query.keys_count = distinct;
}

/** Whether @p predicate holds on any lane of this lane's aligned group of @p lanes, read as
 *  nonzero, which other groups of the simdgroup may have left: the votes meet by halves, never past
 *  the group, as @c sz_group_any_cuda_ votes. */
inline bool sz_overlap_group_any_metal_(int predicate, uint lanes) {
    if (lanes == 32) return simd_any(predicate != 0);
    uint votes = predicate != 0 ? 1u : 0u;
    for (uint distance = 1; distance < lanes; distance *= 2) votes |= simd_shuffle_xor(votes, ushort(distance));
    return votes != 0;
}

/**
 *  @brief Rolls each width's window hash through the endings past @p from and up to @p until of
 *      @p walk, on one lane, and adds every window the tree holds to that width's @p matches, as
 *      @c sz_overlap_roll_simt_ rolls them.
 *
 *  Each width rolls its own hash a byte at a time, the byte leaving the window read again from the
 *  text rather than kept, so the hashes stay in registers. The first @p warm_bytes endings only
 *  prime the hashes, and a window not yet full is never probed. A caller resumes a walk by passing
 *  the @p hashes this one left. The tree is read through plain loads, never staged.
 *
 *  @param[in] lifted_powers One per width, 256 to the width lifted into Montgomery form.
 *  @param[in,out] hashes One entry per width, up to @c sz_overlap_gpu_widths_per_walk_metal_k.
 *  @param[in,out] matches One entry per width, up to @c sz_overlap_gpu_widths_per_walk_metal_k.
 */
inline void sz_overlap_roll_metal_(device uchar const *walk, ulong from, ulong until, ulong warm_bytes,
                                   thread uint const *widths, thread uint const *lifted_powers, ulong widths_count,
                                   device uint const *nodes, uint levels, thread uint const *level_bases,
                                   thread uint *hashes, thread uint *matches) {
    for (ulong ending = from + 1; ending <= until; ++ending) {
        uint const incoming = walk[ending - 1];
#pragma unroll
        for (uint index = 0; index != sz_overlap_gpu_widths_per_walk_metal_k; ++index) {
            if (index >= widths_count) break;
            ulong const width = widths[index];
            if (!width) continue;
            uint const outgoing = ending > width ? walk[ending - width - 1] : 0u;
            hashes[index] = sz_overlap_roll_step_metal_(hashes[index], incoming, outgoing, lifted_powers[index]);
            if (ending <= warm_bytes || width > ending) continue;
            matches[index] += sz_overlap_probe_metal_(nodes, nullptr, ~0u, levels, level_bases, hashes[index]);
        }
    }
}

/**
 *  @brief Counts the windows of one candidate, at every width, that a prepared query holds, every
 *      lane of an aligned group of @p lanes taking part, and gives up once even every window left
 *      matching would leave more than @p misses_limit of the longer side's windows unmatched, as
 *      @c sz_overlap_candidate_matches_simt_ counts them.
 *
 *  Each lane rolls an even run of the candidate's bytes, primed from the widest window's bytes
 *  before it, and each width rolls its own hash a byte at a time, the byte leaving the window read
 *  again from the text. Lanes compare their totals every
 *  @c sz_overlap_gpu_endings_per_check_metal_k endings, and a candidate with too few windows to
 *  come within @p misses_limit is answered before a byte is read. Every residue and every count is
 *  an integer, so a count within the limit is the sum of serial's counts over the widths. Unlike
 *  the CUDA twin, a width past the query's bytes probes nothing, as serial's pairs skip it.
 *
 *  @param[in] query Prepared by @ref sz_overlap_query_prepare_metal_, its span read through plain
 *      loads, so the same kernel may have written it.
 *  @param[in] misses_limit Misses past which counting stops; @c ULONG_MAX bounds nothing.
 *  @param[in] lane The calling lane within its group, below @p lanes.
 *  @param[in] lanes The group's lanes, one of 1, 4, 8, 16 and 32, aligned within the simdgroup.
 *  @return The same count in every lane of the group: exact whenever it leaves at most
 *      @p misses_limit misses, and otherwise a count that leaves more.
 */
inline ulong sz_overlap_candidate_matches_metal_(thread sz_overlap_query_metal_t const &query, device uchar const *text,
                                                 ulong length, ulong misses_limit, uint lane, uint lanes) {
    sz_group_metal_t const group = {lanes, lane};
    uint const per_walk = sz_overlap_gpu_widths_per_walk_metal_k;
    device uint const *const widths = query.widths;
    ulong const widths_count = query.widths_count;
    uint level_bases[sz_overlap_btree_levels_max_metal_k];
    uint const levels = sz_overlap_levels_metal_((uint)query.keys_count, level_bases);
    ulong const candidate_windows = sz_overlap_windows_metal_(length, widths, widths_count);
    ulong const longer = max(candidate_windows, query.windows);
    ulong const needed = longer > misses_limit ? longer - misses_limit : 0;
    ulong const run = sz_size_divide_round_up_metal_<ulong>(length, (ulong)lanes);
    ulong const run_begin = min((ulong)lane * run, length);
    ulong const run_end = min(run_begin + run, length);
    ulong pending = sz_overlap_windows_metal_(run_end, widths, widths_count) -
                    sz_overlap_windows_metal_(run_begin, widths, widths_count);
    ulong matched = 0;

    // A candidate too short to come within the limit walks no group of widths.
    ulong first_width = candidate_windows < needed ? widths_count : 0;
    for (; sz_overlap_group_any_metal_(first_width < widths_count, lanes); first_width += per_walk) {
        device uint const *const group_widths = widths + first_width;
        ulong const group_count = min(widths_count - first_width, (ulong)per_walk);
        // Widths and their powers stay in registers for the whole walk; each power is lifted once
        // into Montgomery form, so its product with a byte comes back plain. A width past the query
        // holds none of its windows, but one tree holds every width's, and a leading zero byte
        // leaves a hash unchanged, so serial probes none of them and neither does this.
        uint window_widths[sz_overlap_gpu_widths_per_walk_metal_k],
            lifted_powers[sz_overlap_gpu_widths_per_walk_metal_k];
        uint hashes[sz_overlap_gpu_widths_per_walk_metal_k], matches[sz_overlap_gpu_widths_per_walk_metal_k];
        ulong warm_up = 0;
#pragma unroll
        for (uint index = 0; index != per_walk; ++index) {
            bool const live = index < group_count && group_widths[index] <= query.length;
            window_widths[index] = live ? group_widths[index] : 0u;
            lifted_powers[index] = live ? sz_overlap_montgomery_metal_(query.powers[first_width + index],
                                                                       sz_overlap_montgomery_square_metal_k)
                                        : 0u;
            hashes[index] = 0, matches[index] = 0;
            warm_up = max(warm_up, (ulong)window_widths[index]);
        }
        warm_up = warm_up ? warm_up - 1 : 0;
        ulong const walk_begin = run_begin > warm_up ? run_begin - warm_up : 0;
        ulong const walk_bytes = run_end != run_begin ? run_end - walk_begin : 0;
        ulong const warm_bytes = run_begin - walk_begin;
        device uchar const *const walk = text + walk_begin;

        // As many passes in every lane, so the checks between them see the whole group.
        ulong const passes = sz_size_divide_round_up_metal_<ulong>(run + warm_up,
                                                                   sz_overlap_gpu_endings_per_check_metal_k);
        ulong reached = 0, pass = 0, group_matched = 0;
        for (; sz_overlap_group_any_metal_(pass != passes, lanes); ++pass) {
            ulong const until = min(reached + sz_overlap_gpu_endings_per_check_metal_k, walk_bytes);
            sz_overlap_roll_metal_(walk, reached, until, warm_bytes, window_widths, lifted_powers, group_count,
                                   query.nodes, levels, level_bases, hashes, matches);
            if (reached < until) {
                // Only the windows ending past the run's start were probed, the rest having primed.
                ulong const probed_from = max(walk_begin + reached, run_begin);
                ulong const probed_until = max(walk_begin + until, run_begin);
                pending -= sz_overlap_windows_metal_(probed_until, group_widths, group_count) -
                           sz_overlap_windows_metal_(probed_from, group_widths, group_count);
                reached = until;
            }
            if (!sz_overlap_group_any_metal_(needed != 0, lanes)) continue;
            group_matched = 0;
#pragma unroll
            for (uint index = 0; index != per_walk; ++index) group_matched += matches[index];
            if (sz_overlap_group_any_metal_(group.sum(matched + group_matched + pending) < needed, lanes)) break;
        }
        group_matched = 0;
#pragma unroll
        for (uint index = 0; index != per_walk; ++index) group_matched += matches[index];
        matched += group_matched;
        if (sz_overlap_group_any_metal_(pass != passes, lanes)) break;
    }
    return group.sum(matched);
}

/** Every candidate's first segment, and the total after the last, as one threadgroup's scan. */
kernel void sz_overlap_segments_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              device uchar const *candidates [[buffer(2)]],
                                              device uchar const *candidates_offsets [[buffer(3)]],
                                              constant sz_overlap_arguments_metal_t &arguments [[buffer(5)]],
                                              uint thread_index [[thread_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    sz_strings_tape_metal_t const tape = {candidates, candidates_offsets, arguments.candidates_layout};
    device uint *offsets = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.segments);
    ulong const count = arguments.candidates_count;
    ulong const chunk = sz_size_divide_round_up_metal_<ulong>(count, sz_overlap_scan_threads_metal_k);
    ulong const first = min(count, (ulong)thread_index * chunk), last = min(count, first + chunk);
    uint total = 0;
    for (ulong candidate = first; candidate != last; ++candidate)
        total += sz_overlap_segments_of_metal_(sz_strings_length_metal_(tape, candidate));

    threadgroup uint simdgroup_totals[sz_overlap_scan_threads_metal_k / 32];
    uint const within = simd_prefix_exclusive_sum(total);
    if (lane == 31) simdgroup_totals[simdgroup] = within + total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) simdgroup_totals[lane] = simd_prefix_exclusive_sum(simdgroup_totals[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint running = simdgroup_totals[simdgroup] + within;
    for (ulong candidate = first; candidate != last; ++candidate) {
        offsets[candidate] = running;
        running += sz_overlap_segments_of_metal_(sz_strings_length_metal_(tape, candidate));
    }
    if (last == count) offsets[count] = running; // Threads past the end agree on the total.
}

/**
 *  @brief One threadgroup row per prepared query, one thread per segment, the top of the tree
 *      staged in threadgroup memory when the launch asked for it.
 *
 *  No thread leaves the query loop early, because the staging barriers are collective: a segment
 *  past the round skips its sweep rather than returning.
 */
kernel void sz_overlap_scores_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device uchar const *candidates [[buffer(2)]], device uchar const *candidates_offsets [[buffer(3)]],
    constant sz_overlap_arguments_metal_t &arguments [[buffer(5)]], threadgroup uint *staged_nodes [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]) {
    uint const segment = group.x * group_size.x + thread_index;
    sz_strings_tape_metal_t const tape = {candidates, candidates_offsets, arguments.candidates_layout};
    device uint const *nodes = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.nodes);
    device ulong const *nodes_offsets = sz_reach_metal_<ulong>(engine, arguments.engine_host, arguments.nodes_offsets);
    device uint const *keys_counts = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.keys_counts);
    device uint const *offsets = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.segments);
    // A thread keeps its segment across the queries its row strides, so it finds the owner once.
    ulong const candidate = segment < arguments.segments_count
                                ? sz_overlap_owner_metal_(offsets, arguments.candidates_count, segment)
                                : 0;

    for (ulong query = group.y; query < arguments.queries_count; query += groups.y) {
        uint level_bases[sz_overlap_btree_levels_max_metal_k];
        uint const keys_count = keys_counts[query];
        uint const levels = sz_overlap_levels_metal_(keys_count, level_bases);
        device uint const *tree = nodes + nodes_offsets[query];
        ulong const entries = nodes_offsets[query + 1] - nodes_offsets[query];
        ulong const staged_from = entries > arguments.staged_nodes_count ? entries - arguments.staged_nodes_count : 0;
        ulong const staged_entries = entries - staged_from;
        for (ulong entry = thread_index; entry < staged_entries; entry += group_size.x)
            staged_nodes[entry] = tree[staged_from + entry];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (segment < arguments.segments_count)
            sz_overlap_sweep_metal_(engine, scratch, tape, arguments, tree, staged_nodes, (uint)staged_from, levels,
                                    level_bases, query, segment, candidate);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/** One thread per candidate and query: the counters land in the output, zeros for a pair past its
 *  budget, and are cleared for the next round. */
kernel void sz_overlap_matches_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                             device uchar const *candidates [[buffer(2)]],
                                             device uchar const *candidates_offsets [[buffer(3)]],
                                             device uint *matches [[buffer(4)]],
                                             constant sz_overlap_arguments_metal_t &arguments [[buffer(5)]],
                                             uint2 position [[thread_position_in_grid]]) {
    ulong const candidate = position.x, query = position.y;
    if (candidate >= arguments.candidates_count || query >= arguments.queries_count) return;
    sz_strings_tape_metal_t const tape = {candidates, candidates_offsets, arguments.candidates_layout};
    device uint const *widths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.widths);
    device uint const *lengths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.lengths);
    device uint *counts = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.counts) +
                          (query * arguments.candidates_count + candidate) * arguments.widths_count;
    device uint *candidate_matches = matches + query * arguments.matches_query_stride +
                                     candidate * arguments.matches_candidate_stride;
    ulong const length = sz_strings_length_metal_(tape, candidate), query_length = lengths[query];
    ulong longer = 0, misses = 0, index;
    for (index = 0; index != arguments.widths_count; ++index) {
        ulong const width = widths[index];
        ulong const windows = width && width <= length ? length - width + 1 : 0;
        ulong const query_windows = width && width <= query_length ? query_length - width + 1 : 0;
        longer += max(windows, query_windows);
        misses += max(windows, query_windows) - (windows && query_windows ? counts[index] : 0);
    }
    // One epilogue for every pair: exact counts within its budget, zeros past it.
    bool const within = misses <= sz_overlap_misses_bound_metal_(arguments.misses_bound, longer);
    for (index = 0; index != arguments.widths_count; ++index) {
        ulong const width = widths[index];
        bool const counted = within && width && width <= query_length && width <= length;
        candidate_matches[index] = counted ? counts[index] : 0u;
        counts[index] = 0;
    }
}
