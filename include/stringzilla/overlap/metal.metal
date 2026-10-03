/**
 *  @file include/stringzilla/overlap/metal.metal
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Window overlap on Apple GPUs of Metal family 7, M1 and newer: one thread per segment of a
 *      candidate, its chain walked through a ring of prefix hashes so every width is scored in one
 *      pass, and one prepared query's B-tree probed per threadgroup row.
 *
 *  Nothing here needs more than Apple7: 32-bit atomics, simdgroup prefix sums and 32 KB of
 *  threadgroup memory. A later family that earns its own kernels gets its own file beside this one.
 *
 *  @sa include/stringzilla/overlap/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/overlap/simt.cuh, the CUDA sibling this mirrors step for step
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  The CUDA kernel reaches the serial helpers through `--expt-relaxed-constexpr`; MSL cannot
 *  include the C headers, whose pointers carry no address space, so the four helpers this needs -
 *  the branch step, the leaf step, the tree's level bases and the share - are mirrored here, and
 *  the tests hold every score bit-identical to serial's. The share divides in @c f64, which Apple
 *  GPUs lack, so its double rounding is emulated with integer long division.
 *
 *  Apple GPUs have no 64-bit divider either, so a remainder over @c ulong compiles to a long
 *  emulated sequence. The window hashes are reduced with Montgomery products instead, which need
 *  only the native 32-bit @c mul and @c mulhi.
 *
 *  The forest and the round block each point inside themselves by host address, turned into the
 *  kernel's own by @ref sz_reach_metal_ against the block's host address, while the candidates'
 *  tape and the scores are bound directly.
 *
 *  A round is three dispatches. One thread per candidate would leave most of a simdgroup idle while
 *  its longest lane finishes, so candidates are cut into segments of equal length instead: the
 *  first dispatch scans their segment counts into offsets, the second scores one segment per thread
 *  and adds its matches to the candidate's counters, and the third turns them into shares.
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

/** Prefix hashes one thread keeps, a power of two so the ring index is a mask. */
constant uint sz_overlap_ring_span_metal_k = 32;

/** Widths one engine may hold, their match counters held per thread, as
 *  @c sz_overlap_gpu_widths_max_k. */
constant uint sz_overlap_gpu_widths_max_metal_k = 8;

/** Bytes of a candidate one thread scores; the windows ending inside it are its own. */
constant uint sz_overlap_segment_bytes_metal_k = 128;

/** Threads the segment scan runs, as one threadgroup of 32 simdgroups. */
constant uint sz_overlap_scan_threads_metal_k = 1024;

/** Words of the per-query membership filter: 2¹⁷ bits, 16 KB of threadgroup memory. */
constant uint sz_overlap_filter_words_metal_k = 4096;

/** The launch record the host copies in: host addresses inside the forest and the round block, and
 *  the round's shape. */
struct sz_overlap_arguments_metal_t {
    ulong engine_host, scratch_host;
    ulong nodes, nodes_offsets, keys_counts, widths, powers, lengths;
    ulong segments, counts;
    ulong queries_count, widths_count, candidates_count, segments_count;
    ulong scores_query_stride, scores_candidate_stride, staged_nodes_count, widest_width;
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

/** Separators of one branch node under @p key, the child to descend into. */
inline uint sz_overlap_branch_step_metal_(device uint const *separators, int key) {
    uint child = 0;
    for (uint separator = 0; separator != sz_overlap_keys_per_node_metal_k; ++separator)
        child += (int)separators[separator] < key;
    return child;
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
    uint found = 0;
    for (uint position = 0; position != sz_overlap_keys_per_node_metal_k; ++position) found |= leaf[position] == key;
    return found;
}

inline uint sz_overlap_leaf_step_metal_(threadgroup uint const *leaf, uint key) {
    threadgroup uint4 const *quads = (threadgroup uint4 const *)leaf;
    uint4 const keys = uint4(key);
    return any((quads[0] == keys) | (quads[1] == keys) | (quads[2] == keys) | (quads[3] == keys)) ? 1u : 0u;
}

/** Whether the tree over @p nodes holds one raw @p window_hash, walking it a level at a time, as
 *  @c sz_overlap_btree_probe_simt_ does. */
template <typename nodes_pointer_>
inline uint sz_overlap_probe_metal_(nodes_pointer_ nodes, uint levels, thread uint const *level_bases,
                                    uint window_hash) {
    uint const key = window_hash ^ sz_overlap_sign_flip_metal_k;
    uint node = 0;
    for (uint level = 0; level + 1 != levels; ++level) {
        uint const child = sz_overlap_branch_step_metal_(
            nodes + (level_bases[level] + node) * sz_overlap_keys_per_node_metal_k, (int)key);
        node = node * sz_overlap_branches_per_node_metal_k + child;
    }
    return sz_overlap_leaf_step_metal_(nodes + (level_bases[levels - 1] + node) * sz_overlap_keys_per_node_metal_k,
                                       key);
}

/**
 *  @brief Whether @p window_hash may be among the query's keys: the bits both of its halves pick
 *      are set. Never a false negative, so a miss skips the exact probe.
 *
 *  A simdgroup runs the probe when any of its 32 lanes needs it, so the filter pays only if nearly
 *  every lane misses; two bits per key keep false positives low enough for queries of thousands of
 *  windows, where one bit would still send most simdgroups down the tree.
 */
inline bool sz_overlap_filter_holds_metal_(threadgroup uint const *filter, uint window_hash) {
    uint const low = window_hash & 0x1FFFFu, high = window_hash >> 15;
    return ((filter[low >> 5] >> (low & 31)) & (filter[high >> 5] >> (high & 31)) & 1u) != 0;
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

/**
 *  @brief @p matches over the longer window count as @c sz_overlap_share_ rounds it: the quotient
 *      rounded to an @c f64, and that rounded again to an @c f32.
 *
 *  Long division yields the quotient's first 54 significant bits and a sticky bit, which is exactly
 *  what the @c f64 rounding looks at; the @c f32 rounding then reads the 53 bits it produced. Doing
 *  both steps keeps the rare double-rounding cases identical to the host's.
 */
inline float sz_overlap_share_metal_(ulong matches, ulong candidate_windows, ulong query_windows) {
    ulong const longer = candidate_windows > query_windows ? candidate_windows : query_windows;
    if (!longer || !matches) return 0.0f;

    // Matches never exceed the longer side, so doubling the remainder until it reaches `longer`
    // puts it in [longer, 2 × longer), where the first quotient bit is the leading one.
    int exponent = 0;
    ulong remainder = matches;
    while (remainder < longer) remainder <<= 1, --exponent;
    ulong bits = 0;
    for (uint step = 0; step != 54; ++step) {
        bits <<= 1;
        if (remainder >= longer) bits |= 1, remainder -= longer;
        remainder <<= 1;
    }
    bool const sticky = remainder != 0;

    // Round to 53 bits, nearest-even, as the `f64` quotient does.
    ulong significand = bits >> 1;
    bool const guard = (bits & 1) != 0;
    if (guard && (sticky || (significand & 1))) ++significand;
    if (significand == (1ul << 53)) significand >>= 1, ++exponent;

    // Round those 53 bits to 24, nearest-even, as the `f32` conversion does.
    ulong fraction = significand >> 29;
    ulong const dropped = significand & ((1ul << 29) - 1), tie = 1ul << 28;
    if (dropped > tie || (dropped == tie && (fraction & 1))) ++fraction;
    if (fraction == (1ul << 24)) fraction >>= 1, ++exponent;
    return as_type<float>((uint)((exponent + 127) << 23) | (uint)(fraction & 0x7FFFFFu));
}

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

/**
 *  @brief Counts one segment's windows of one candidate against one prepared query, adding them to
 *      the candidate's counters.
 *
 *  The windows ending inside the segment are its own. Hashing restarts up to the widest width
 *  before it, since a window needs the prefix at its start, so every window is counted by exactly
 *  one thread, from the same residues serial computes.
 */
template <typename nodes_pointer_>
inline void sz_overlap_sweep_metal_(device uchar *engine, device uchar *scratch, sz_sequence_tape_metal_t tape,
                                    constant sz_overlap_arguments_metal_t &arguments, nodes_pointer_ nodes,
                                    threadgroup uint const *filter, uint levels, thread uint const *level_bases,
                                    ulong query, uint segment) {
    device uint const *offsets = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.segments);
    device uint const *widths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.widths);
    device uint const *powers = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.powers);
    ulong const candidate = sz_overlap_owner_metal_(offsets, arguments.candidates_count, segment);
    device uchar const *text = sz_sequence_tape_start_metal_(tape, candidate);
    ulong const text_length = sz_sequence_tape_length_metal_(tape, candidate);
    ulong const segment_start = (ulong)(segment - offsets[candidate]) * sz_overlap_segment_bytes_metal_k;
    ulong const segment_end = min(text_length, segment_start + sz_overlap_segment_bytes_metal_k);
    ulong const origin = segment_start + 1 >= arguments.widest_width ? segment_start + 1 - arguments.widest_width : 0;

    // Widths and their powers stay in registers for the whole segment; each power is lifted once
    // into Montgomery form, 256^width × R mod p, so every window's product comes back plain.
    uint matches[sz_overlap_gpu_widths_max_metal_k], window_widths[sz_overlap_gpu_widths_max_metal_k];
    uint lifted_powers[sz_overlap_gpu_widths_max_metal_k];
    for (ulong index = 0; index != arguments.widths_count; ++index) {
        matches[index] = 0, window_widths[index] = widths[index];
        lifted_powers[index] = sz_overlap_montgomery_metal_(powers[index], sz_overlap_montgomery_square_metal_k);
    }

    // `ring[k & mask]` is the prefix hash from `origin` to `k`, so a window ending at `k` reads its
    // start at `(k - width) & mask`; the window's residue is the same whichever origin it rests on.
    ulong const mask = sz_overlap_ring_span_metal_k - 1;
    uint ring[sz_overlap_ring_span_metal_k];
    ring[origin & mask] = 0;
    uint prior = 0;
    for (ulong position = origin; position != segment_end; ++position) {
        // Under p + 256, which still fits 32 bits, so one subtraction reduces it.
        prior = sz_overlap_montgomery_metal_(prior, sz_overlap_byte_shift_metal_k) + text[position];
        if (prior >= sz_overlap_modulus_metal_k) prior -= sz_overlap_modulus_metal_k;
        ulong const ending = position + 1;
        ring[ending & mask] = prior;
        if (ending <= segment_start) continue;
        for (ulong index = 0; index != arguments.widths_count; ++index) {
            uint const width = window_widths[index];
            if (!width || width > ending) continue;
            uint const start = ring[(ending - width) & mask];
            uint const shifted = sz_overlap_montgomery_metal_(start, lifted_powers[index]);
            // Wrapping arithmetic is exact here: the true residue is under p, below 2³².
            uint const residue = prior - shifted + (prior < shifted ? sz_overlap_modulus_metal_k : 0u);
            if (sz_overlap_filter_holds_metal_(filter, residue))
                matches[index] += sz_overlap_probe_metal_(nodes, levels, level_bases, residue);
        }
    }

    device atomic_uint *counts = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.counts) +
                                 (query * arguments.candidates_count + candidate) * arguments.widths_count;
    for (ulong index = 0; index != arguments.widths_count; ++index)
        if (matches[index]) atomic_fetch_add_explicit(&counts[index], matches[index], memory_order_relaxed);
}

/** Every candidate's first segment, and the total after the last, as one threadgroup's scan. */
kernel void sz_overlap_segments_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              device ulong const *candidates [[buffer(2)]],
                                              constant sz_overlap_arguments_metal_t &arguments [[buffer(4)]],
                                              uint thread_index [[thread_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    sz_sequence_tape_metal_t const tape = {candidates};
    device uint *offsets = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.segments);
    ulong const count = arguments.candidates_count;
    ulong const chunk = sz_size_divide_round_up_metal_<ulong>(count, sz_overlap_scan_threads_metal_k);
    ulong const first = min(count, (ulong)thread_index * chunk), last = min(count, first + chunk);
    uint total = 0;
    for (ulong candidate = first; candidate != last; ++candidate)
        total += sz_overlap_segments_of_metal_(sz_sequence_tape_length_metal_(tape, candidate));

    threadgroup uint simdgroup_totals[sz_overlap_scan_threads_metal_k / 32];
    uint const within = simd_prefix_exclusive_sum(total);
    if (lane == 31) simdgroup_totals[simdgroup] = within + total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) simdgroup_totals[lane] = simd_prefix_exclusive_sum(simdgroup_totals[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint running = simdgroup_totals[simdgroup] + within;
    for (ulong candidate = first; candidate != last; ++candidate) {
        offsets[candidate] = running;
        running += sz_overlap_segments_of_metal_(sz_sequence_tape_length_metal_(tape, candidate));
    }
    if (last == count) offsets[count] = running; // Threads past the end agree on the total.
}

/**
 *  @brief One threadgroup row per prepared query, one thread per segment, the tree staged in
 *      threadgroup memory when the launch asked for it.
 *
 *  No thread leaves the query loop early, because the staging barriers are collective: a segment
 *  past the round skips its sweep rather than returning. Every query also gets its membership
 *  filter built from the leaves - the first @c keys_count entries of its tree - before any probe.
 */
kernel void sz_overlap_scores_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device ulong const *candidates [[buffer(2)]], constant sz_overlap_arguments_metal_t &arguments [[buffer(4)]],
    threadgroup uint *staged_nodes [[threadgroup(0)]], threadgroup uint *filter [[threadgroup(1)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 groups [[threadgroups_per_grid]],
    uint thread_index [[thread_index_in_threadgroup]], uint2 group_size [[threads_per_threadgroup]]) {
    uint const segment = group.x * group_size.x + thread_index;
    sz_sequence_tape_metal_t const tape = {candidates};
    device uint const *nodes = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.nodes);
    device ulong const *nodes_offsets = sz_reach_metal_<ulong>(engine, arguments.engine_host, arguments.nodes_offsets);
    device uint const *keys_counts = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.keys_counts);

    threadgroup atomic_uint *const filter_bits = (threadgroup atomic_uint *)filter;
    for (ulong query = group.y; query < arguments.queries_count; query += groups.y) {
        uint level_bases[sz_overlap_btree_levels_max_metal_k];
        uint const keys_count = keys_counts[query];
        uint const levels = sz_overlap_levels_metal_(keys_count, level_bases);
        device uint const *tree = nodes + nodes_offsets[query];
        for (uint word = thread_index; word < sz_overlap_filter_words_metal_k; word += group_size.x)
            atomic_store_explicit(&filter_bits[word], 0u, memory_order_relaxed);
        if (arguments.staged_nodes_count) {
            ulong const entries = nodes_offsets[query + 1] - nodes_offsets[query];
            for (ulong entry = thread_index; entry < entries; entry += group_size.x) staged_nodes[entry] = tree[entry];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint entry = thread_index; entry < keys_count; entry += group_size.x) {
            uint const window_hash = tree[entry] ^ sz_overlap_sign_flip_metal_k;
            uint const low = window_hash & 0x1FFFFu, high = window_hash >> 15;
            atomic_fetch_or_explicit(&filter_bits[low >> 5], 1u << (low & 31), memory_order_relaxed);
            atomic_fetch_or_explicit(&filter_bits[high >> 5], 1u << (high & 31), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (segment < arguments.segments_count) {
            if (arguments.staged_nodes_count)
                sz_overlap_sweep_metal_(engine, scratch, tape, arguments, (threadgroup uint const *)staged_nodes,
                                        filter, levels, level_bases, query, segment);
            else
                sz_overlap_sweep_metal_(engine, scratch, tape, arguments, tree, filter, levels, level_bases, query,
                                        segment);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/** One thread per candidate and query: the counters become shares, cleared for the next round. */
kernel void sz_overlap_shares_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                            device ulong const *candidates [[buffer(2)]],
                                            device float *scores [[buffer(3)]],
                                            constant sz_overlap_arguments_metal_t &arguments [[buffer(4)]],
                                            uint2 position [[thread_position_in_grid]]) {
    ulong const candidate = position.x, query = position.y;
    if (candidate >= arguments.candidates_count || query >= arguments.queries_count) return;
    sz_sequence_tape_metal_t const tape = {candidates};
    device uint const *widths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.widths);
    device uint const *lengths = sz_reach_metal_<uint>(engine, arguments.engine_host, arguments.lengths);
    device uint *counts = sz_reach_metal_<uint>(scratch, arguments.scratch_host, arguments.counts) +
                          (query * arguments.candidates_count + candidate) * arguments.widths_count;
    device float *candidate_scores = scores + query * arguments.scores_query_stride +
                                     candidate * arguments.scores_candidate_stride;
    ulong const length = sz_sequence_tape_length_metal_(tape, candidate), query_length = lengths[query];
    for (ulong index = 0; index != arguments.widths_count; ++index) {
        ulong const width = widths[index];
        bool const scored = width && width <= query_length && width <= length;
        candidate_scores[index] =
            scored ? sz_overlap_share_metal_(counts[index], length - width + 1, query_length - width + 1) : 0.0f;
        counts[index] = 0;
    }
}
