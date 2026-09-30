/**
 *  @file include/stringzilla/substrings/simt.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Multi-pattern search on Apple GPUs of Metal family 7, M1 and newer: one thread per
 *      haystack chunk, and the leftmost cover resolved after the walk rather than inside it.
 *
 *  @sa include/stringzilla/substrings/simt.h, which embeds and launches this source
 *  @sa include/stringzilla/substrings/simt.cuh, the CUDA sibling this mirrors kernel for kernel
 *
 *  MSL cannot include the C headers, so the transition, the acceptance bit and the BM25 terms are
 *  mirrored here. Apple GPUs have no @c f64, so a BM25 term is computed in @c f32 and summed in
 *  32.32 fixed point, which keeps a score independent of thread order; scores then agree with
 *  serial's to the tolerance the CUDA tier is held to rather than bit for bit.
 *
 *  Every pointer the host hands over is a host address inside the device's arena, turned into the
 *  kernel's own by @ref sz_reach_metal_. Metal has no 64-bit atomic add and nothing here needs
 *  one: every count and offset comes out of a scan, and every BM25 sum out of a tree over one
 *  threadgroup's partials.
 */
#include <metal_stdlib>

using namespace metal;

/** Divides rounding up, as @c sz_size_divide_round_up does in the C headers MSL cannot include. */
template <typename scalar_type_>
constexpr scalar_type_ sz_metal_divide_round_up_(scalar_type_ number, scalar_type_ divisor) {
    return (number + divisor - 1) / divisor;
}

/** Threads every threadgroup here runs, as the CUDA tier's blocks. */
constant uint sz_substrings_threads_simt_k = 256;

/** Candidates one thread scans quadratically before a cover segment falls back to emitted order. */
constant ulong sz_substrings_cover_segment_limit_simt_k = 4096;

/** Output bytes one threadgroup of a rewrite's copy owns. */
constant ulong sz_substrings_rewrite_tile_bytes_simt_k = 4096;

/** Slots one threadgroup's BM25 tally holds: half the CUDA tier's, as a key and a count each for
 *  every slot leaves room for nothing else in 32 KB of threadgroup memory. */
constant uint sz_substrings_tally_slot_bits_simt_k = 11;
constant uint sz_substrings_tally_slots_simt_k = 1u << sz_substrings_tally_slot_bits_simt_k;

/** Slots a hashed tally probes before spilling a needle to the threadgroup's overflow row. */
constant uint sz_substrings_tally_probes_simt_k = 16;

/** What a chunk walk does at each match, as the host's @c sz_substrings_simt_pass_t numbers it. */
constant ulong sz_substrings_sizing_simt_k = 0, sz_substrings_writing_simt_k = 1;

/** One output of a state, laid out as @c sz_substrings_output_t. */
struct sz_substrings_output_metal_t {
    uint needle_index, folded_match_bytes;
};

/** One match, laid out as @c sz_substrings_match_t. */
struct sz_substrings_match_metal_t {
    ulong haystack_index, needle_index, byte_offset, byte_length;
};

/** A round's report, laid out as @c sz_substrings_report_t. */
struct sz_substrings_report_metal_t {
    ulong matches_emitted, matches_stored, target_length, shortfall;
};

/** A haystack's or a replacement's text, laid out as @c sz_string_view_t. */
struct sz_view_metal_t {
    ulong start, length;
};

/** The launch record the host copies in: host addresses inside the arena, and the round's shape. */
struct sz_substrings_simt_arguments_t {
    ulong host_base;
    ulong hot_rows, byte_to_class, base, check, fail, accepts_words, outputs, outputs_counts, outputs_offsets;
    ulong hot_count, classes_count, root, needles_count, max_source_match_bytes, overlap_policy;
    ulong report, chunk_bytes, chunk_offsets, haystack_offsets, chunk_slots, tile_sums;
    ulong emitted, reported, keep_offsets, gap_offsets, overflow_rows;
    ulong slots_count, chunk_budget, chunk_floor, matches_budget, emitting;
    ulong haystacks, haystacks_count, counts, counts_stride, matches, matches_capacity;
    ulong replacements, target, target_ceiling, output_offsets;
    ulong document_lengths, needle_weights, scores, scores_stride, tally_hashed;
    float saturation, normalization, average_length, unused;
    ulong pass, scan_values, scan_count, scan_elements_per_tile, scan_tiles, clear_begin, clear_words;
};

/** The kernel's own address of @p host, a host address inside the arena. */
template <typename type_>
inline device type_ *sz_reach_metal_(device uchar *arena, ulong host_base, ulong host) {
    return (device type_ *)(arena + (host - host_base));
}

/** The automaton a walk steps, as @c sz_substrings_engine_t lays it out. */
struct sz_substrings_automaton_simt_t {
    device uint const *hot_rows;
    device uchar const *byte_to_class;
    device uint const *base, *check, *fail, *accepts_words, *outputs_counts;
    device sz_substrings_output_metal_t const *outputs;
    device ulong const *outputs_offsets;
    uint hot_count, classes_count, root, max_source_match_bytes;
};

inline sz_substrings_automaton_simt_t sz_substrings_automaton_simt_(
    device uchar *arena, constant sz_substrings_simt_arguments_t &arguments) {
    ulong const host_base = arguments.host_base;
    sz_substrings_automaton_simt_t automaton;
    automaton.hot_rows = sz_reach_metal_<uint>(arena, host_base, arguments.hot_rows);
    automaton.byte_to_class = sz_reach_metal_<uchar>(arena, host_base, arguments.byte_to_class);
    automaton.base = sz_reach_metal_<uint>(arena, host_base, arguments.base);
    automaton.check = sz_reach_metal_<uint>(arena, host_base, arguments.check);
    automaton.fail = sz_reach_metal_<uint>(arena, host_base, arguments.fail);
    automaton.accepts_words = sz_reach_metal_<uint>(arena, host_base, arguments.accepts_words);
    automaton.outputs_counts = sz_reach_metal_<uint>(arena, host_base, arguments.outputs_counts);
    automaton.outputs = sz_reach_metal_<sz_substrings_output_metal_t>(arena, host_base, arguments.outputs);
    automaton.outputs_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.outputs_offsets);
    automaton.hot_count = (uint)arguments.hot_count, automaton.classes_count = (uint)arguments.classes_count;
    automaton.root = (uint)arguments.root, automaton.max_source_match_bytes = (uint)arguments.max_source_match_bytes;
    return automaton;
}

/** Advances @p state by one byte, as @c sz_substrings_step does. */
inline uint sz_substrings_step_simt_(thread sz_substrings_automaton_simt_t const &automaton, uint state, uchar byte) {
    for (;;) {
        if (state < automaton.hot_count)
            return automaton.hot_rows[(ulong)state * automaton.classes_count + automaton.byte_to_class[byte]];
        ulong const candidate = (ulong)automaton.base[state] + byte;
        if (automaton.check[candidate] == state) return (uint)candidate;
        if (state == automaton.root) return automaton.root;
        state = automaton.fail[state];
    }
}

/** Collects what a sizing or writing walk finds: a count, and the matches
 *  themselves when written. */
struct sz_substrings_emitter_simt_t {
    device sz_substrings_match_metal_t *matches;
    ulong haystack_index, found;
    void report(uint needle, ulong offset, ulong length) thread {
        if (matches) matches[found] = {haystack_index, needle, offset, length};
        ++found;
    }
};

/** Counts what a scoring walk finds against each needle, in threadgroup slots first. */
struct sz_substrings_tallier_simt_t {
    threadgroup atomic_uint *keys, *counts, *overflowed;
    device atomic_uint *overflow;
    bool hashed;
    void report(uint needle, ulong offset, ulong length) thread {
        if (!hashed) {
            atomic_fetch_add_explicit(counts + needle, 1u, memory_order_relaxed);
            return;
        }
        uint const key = needle + 1;
        uint slot = (uint)(((ulong)key * 0x9E3779B97F4A7C15ul) >> (64 - sz_substrings_tally_slot_bits_simt_k));
        for (uint probe = 0; probe != sz_substrings_tally_probes_simt_k; ++probe) {
            // Metal's compare-exchange is the weak one, so a spurious failure retries until a
            // key is there.
            uint seated = atomic_load_explicit(keys + slot, memory_order_relaxed);
            while (seated == 0 &&
                   !atomic_compare_exchange_weak_explicit(keys + slot, &seated, key, memory_order_relaxed,
                                                          memory_order_relaxed) &&
                   seated == 0) {}
            if (seated == 0 || seated == key) {
                atomic_fetch_add_explicit(counts + slot, 1u, memory_order_relaxed);
                return;
            }
            slot = (slot + 1) & (sz_substrings_tally_slots_simt_k - 1);
        }
        atomic_fetch_add_explicit(overflow + needle, 1u, memory_order_relaxed);
        atomic_store_explicit(overflowed, 1u, memory_order_relaxed);
    }
};

/**
 *  @brief Walks one chunk of a haystack, reporting every match ending inside it to @p sink.
 *
 *  The warm-up primes the state from the @c max_source_match_bytes-1 bytes before the chunk,
 *  clamped to the haystack, and reports nothing, so every match is found by exactly one chunk.
 */
template <typename sink_>
inline void sz_substrings_walk_chunk_simt_(thread sz_substrings_automaton_simt_t const &automaton,
                                           device uchar const *haystack, ulong chunk_begin, ulong chunk_end,
                                           thread sink_ &sink) {
    ulong const warm_up = automaton.max_source_match_bytes > 0 ? automaton.max_source_match_bytes - 1 : 0;
    ulong const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    uint state = automaton.root;
    ulong position = walk_begin;
    for (; position < chunk_begin; ++position) state = sz_substrings_step_simt_(automaton, state, haystack[position]);
    for (; position < chunk_end; ++position) {
        state = sz_substrings_step_simt_(automaton, state, haystack[position]);
        if (!((automaton.accepts_words[state >> 5] >> (state & 31u)) & 1u)) continue;
        uint const count = automaton.outputs_counts[state];
        ulong const first = automaton.outputs_offsets[state];
        for (uint index = 0; index != count; ++index) {
            sz_substrings_output_metal_t const output = automaton.outputs[first + index];
            // `walk_begin` is clamped to the haystack's own start, so underflowing either
            // is one test.
            if (position + 1 - walk_begin < output.folded_match_bytes) continue;
            sink.report(output.needle_index, position + 1 - output.folded_match_bytes, output.folded_match_bytes);
        }
    }
}

/** The threadgroup's exclusive prefix sum of @p value, its own total left in @p total, over
 *  @p shared of one entry per thread. */
inline ulong sz_substrings_block_scan_simt_(ulong value, threadgroup ulong *shared, uint lane, uint width,
                                            thread ulong &total) {
    shared[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < width; offset *= 2) {
        ulong const addend = lane >= offset ? shared[lane - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lane] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong const inclusive = shared[lane];
    total = shared[width - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup); // ! The caller reuses `shared` for the next tile.
    return inclusive - value;
}

/** Index of the last entry at or below @p value in an ascending array; zero when none is. */
inline ulong sz_substrings_last_not_above_simt_(device ulong const *ascending, ulong count, ulong value) {
    ulong low = 0, high = count;
    while (low + 1 < high) {
        ulong const middle = low + (high - low) / 2;
        if (ascending[middle] <= value) low = middle;
        else high = middle;
    }
    return low;
}

/** One needle's BM25 term in 32.32 fixed point, its weight times the saturated frequency, as
 *  @c sz_substrings_bm25_term computes it in @c f64. */
inline long sz_substrings_bm25_fixed_simt_(float weight, uint frequency, float saturation, float norm) {
    float const term = weight * frequency * (saturation + 1) / (frequency + saturation * norm);
    return (long)rint(term * 4294967296.0f);
}

/** Zeroes @c clear_words words from @c clear_begin, which is how a round clears what
 *  it reads first. */
kernel void sz_substrings_clear_metal_kernel_(device uchar *arena [[buffer(0)]],
                                              constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                              uint position [[thread_position_in_grid]],
                                              uint threads [[threads_per_grid]]) {
    device ulong *words = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.clear_begin);
    for (ulong index = position; index < arguments.clear_words; index += threads) words[index] = 0;
}

/** Reduces one threadgroup's own contiguous tile of @c scan_values into @c tile_sums. */
kernel void sz_substrings_scan_reduce_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                    constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                    uint group [[threadgroup_position_in_grid]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_simt_k];
    device ulong const *values = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.scan_values);
    device ulong *tile_sums = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.tile_sums);
    ulong const begin = (ulong)group * arguments.scan_elements_per_tile;
    ulong const end = min(begin + arguments.scan_elements_per_tile, arguments.scan_count);
    ulong running = 0, total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        sz_substrings_block_scan_simt_(index < end ? values[index] : 0, shared, lane, width, total);
        running += total;
    }
    if (lane == 0) tile_sums[group] = running;
}

/** Scans @c tile_sums in place on one threadgroup, carrying a running offset across its tiles. */
kernel void sz_substrings_scan_carry_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                   constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                   uint lane [[thread_index_in_threadgroup]],
                                                   uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_simt_k];
    device ulong *tile_sums = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.tile_sums);
    ulong carry = 0, total;
    for (ulong first = 0; first < arguments.scan_tiles; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_substrings_block_scan_simt_(index < arguments.scan_tiles ? tile_sums[index] : 0,
                                                               shared, lane, width, total);
        if (index < arguments.scan_tiles) tile_sums[index] = carry + exclusive;
        carry += total;
    }
}

/** Scans one threadgroup's own tile of @c scan_values in place, seeded by the base
 *  the carry set. */
kernel void sz_substrings_scan_apply_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                   constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                   uint group [[threadgroup_position_in_grid]],
                                                   uint lane [[thread_index_in_threadgroup]],
                                                   uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_simt_k];
    device ulong *values = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.scan_values);
    device ulong const *tile_sums = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.tile_sums);
    ulong const begin = (ulong)group * arguments.scan_elements_per_tile;
    ulong const end = min(begin + arguments.scan_elements_per_tile, arguments.scan_count);
    ulong running = tile_sums[group], total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_substrings_block_scan_simt_(index < end ? values[index] : 0, shared, lane, width,
                                                               total);
        if (index < end) values[index] = running + exclusive;
        running += total;
    }
}

/**
 *  @brief Sums the haystacks' lengths on one threadgroup and derives this round's chunk width from
 *      them, so the host never reads the corpus.
 *
 *  There is no ceiling on the width, which is what makes the chunk budget a bound: a chunk holds at
 *  least @c total/budget bytes, so the corpus contributes at most @c budget chunks, and each
 *  haystack's own remainder at most one more.
 */
kernel void sz_substrings_chunk_bytes_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                    constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_simt_k];
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, arguments.host_base,
                                                                           arguments.haystacks);
    ulong mine = 0, total;
    for (ulong index = lane; index < arguments.haystacks_count; index += width) mine += views[index].length;
    sz_substrings_block_scan_simt_(mine, shared, lane, width, total);
    if (lane) return;
    ulong const budget = max(arguments.chunk_budget, 1ul);
    *sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.chunk_bytes) = max(
        max(sz_metal_divide_round_up_<ulong>(total, budget), arguments.chunk_floor), 1ul);
}

/** Writes how many chunks each haystack is cut into - at least one, so an empty haystack still gets
 *  a thread and lands its own boundary - which the scan then turns into its chunk range. */
kernel void sz_substrings_chunk_counts_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                     constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                     uint position [[thread_position_in_grid]],
                                                     uint threads [[threads_per_grid]]) {
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, arguments.host_base,
                                                                           arguments.haystacks);
    device ulong *chunk_offsets = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.chunk_offsets);
    ulong const chunk_bytes = *sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.chunk_bytes);
    for (ulong index = position; index < arguments.haystacks_count; index += threads) {
        ulong const length = views[index].length;
        chunk_offsets[index] = length == 0 ? 1 : sz_metal_divide_round_up_<ulong>(length, chunk_bytes);
    }
}

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in the pass @c pass names.
 *
 *  Both passes share @c chunk_slots: sizing writes each chunk's match count into its slot, and
 *  writing reads the exclusive offset the scan left there, so every chunk owns a private output
 *  range and a write needs no atomic.
 */
kernel void sz_substrings_walk_metal_kernel_(device uchar *arena [[buffer(0)]],
                                             constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                             uint position [[thread_position_in_grid]],
                                             uint threads [[threads_per_grid]]) {
    ulong const host_base = arguments.host_base;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base,
                                                                                                      arguments.report);
    // The writing pass has nowhere to write once the sizing pass outran the budget, so
    // it retires whole.
    if (arguments.pass == sz_substrings_writing_simt_k && report->shortfall) return;
    sz_substrings_automaton_simt_t const automaton = sz_substrings_automaton_simt_(arena, arguments);
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, host_base, arguments.haystacks);
    device ulong const *chunk_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.chunk_offsets);
    device ulong *chunk_slots = sz_reach_metal_<ulong>(arena, host_base, arguments.chunk_slots);
    device sz_substrings_match_metal_t *emitted = sz_reach_metal_<sz_substrings_match_metal_t>(arena, host_base,
                                                                                               arguments.emitted);
    ulong const chunk_count = chunk_offsets[arguments.haystacks_count];
    ulong const chunk_bytes = *sz_reach_metal_<ulong>(arena, host_base, arguments.chunk_bytes);

    for (ulong chunk_index = position; chunk_index < chunk_count; chunk_index += threads) {
        ulong const haystack_index = sz_substrings_last_not_above_simt_(chunk_offsets, arguments.haystacks_count,
                                                                        chunk_index);
        sz_view_metal_t const view = views[haystack_index];
        ulong const chunk_begin = (chunk_index - chunk_offsets[haystack_index]) * chunk_bytes;
        ulong const chunk_end = min(chunk_begin + chunk_bytes, view.length);
        sz_substrings_emitter_simt_t emitter;
        emitter.matches = arguments.pass == sz_substrings_writing_simt_k ? emitted + chunk_slots[chunk_index] : nullptr;
        emitter.haystack_index = haystack_index, emitter.found = 0;
        if (chunk_begin < chunk_end)
            sz_substrings_walk_chunk_simt_(automaton, sz_reach_metal_<uchar>(arena, host_base, view.start), chunk_begin,
                                           chunk_end, emitter);
        if (arguments.pass == sz_substrings_sizing_simt_k) chunk_slots[chunk_index] = emitter.found;
    }
}

/** Publishes what the sizing walk found, the one place a round learns whether it fit. The scanned
 *  chunk slots end in the emitted total, which every trailing zero carried. */
kernel void sz_substrings_sized_metal_kernel_(device uchar *arena [[buffer(0)]],
                                              constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        arena, arguments.host_base, arguments.report);
    device ulong const *chunk_slots = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.chunk_slots);
    ulong const emitted = chunk_slots[arguments.slots_count - 1];
    report->matches_emitted = emitted, report->matches_stored = emitted, report->target_length = 0;
    report->shortfall = arguments.emitting && emitted > arguments.matches_budget ? emitted - arguments.matches_budget
                                                                                 : 0;
}

/** Whether the boundary before @p index is real: nothing still to come starts before the maximum
 *  end already reached, and only matches ending within one match's length of it can. */
inline bool sz_substrings_boundary_before_simt_(device sz_substrings_match_metal_t const *matches, ulong count,
                                                ulong longest, ulong index) {
    if (index == 0 || matches[index - 1].haystack_index != matches[index].haystack_index) return true;
    ulong const reached = matches[index - 1].byte_offset + matches[index - 1].byte_length;
    for (ulong ahead = index; ahead < count; ++ahead) {
        if (matches[ahead].haystack_index != matches[index - 1].haystack_index) break;
        if (matches[ahead].byte_offset + matches[ahead].byte_length >= reached + longest) break;
        if (matches[ahead].byte_offset < reached) return false;
    }
    return true;
}

/** Decides which overlapping matches survive a leftmost cover, one segment per thread, as
 *  @c sz_substrings_simt_cover_kernel_ does on CUDA, falling back to emitted order past
 *  @ref sz_substrings_cover_segment_limit_simt_k candidates. */
kernel void sz_substrings_cover_metal_kernel_(device uchar *arena [[buffer(0)]],
                                              constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                              uint position [[thread_position_in_grid]],
                                              uint threads [[threads_per_grid]]) {
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        arena, arguments.host_base, arguments.report);
    if (report->shortfall) return;
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        arena, arguments.host_base, arguments.emitted);
    device ulong *keep = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.keep_offsets);
    ulong const count = report->matches_emitted, longest = arguments.max_source_match_bytes;
    bool const longest_first = arguments.overlap_policy == 1; // `sz_substrings_leftmost_longest_k`
    for (ulong index = position; index < count; index += threads) {
        // Only a segment's first match works; the rest are decided by whoever owns their segment.
        if (!sz_substrings_boundary_before_simt_(matches, count, longest, index)) continue;
        ulong segment_end = index + 1;
        while (segment_end < count && !sz_substrings_boundary_before_simt_(matches, count, longest, segment_end))
            ++segment_end;

        if (segment_end - index > sz_substrings_cover_segment_limit_simt_k) {
            ulong reached = 0;
            for (ulong slot = index; slot < segment_end; ++slot) {
                bool const accepted = matches[slot].byte_offset >= reached;
                keep[slot] = accepted;
                if (accepted) reached = matches[slot].byte_offset + matches[slot].byte_length;
            }
            continue;
        }

        // The greedy cover: take the earliest start at or past the cursor, breaking ties by
        // policy, and repeat.
        for (ulong slot = index; slot < segment_end; ++slot) keep[slot] = 0;
        for (ulong cursor = 0;;) {
            ulong chosen = segment_end;
            for (ulong slot = index; slot < segment_end; ++slot) {
                if (matches[slot].byte_offset < cursor) continue;
                if (chosen == segment_end) {
                    chosen = slot;
                    continue;
                }
                if (matches[slot].byte_offset != matches[chosen].byte_offset) {
                    if (matches[slot].byte_offset < matches[chosen].byte_offset) chosen = slot;
                    continue;
                }
                if (longest_first && matches[slot].byte_length != matches[chosen].byte_length) {
                    if (matches[slot].byte_length > matches[chosen].byte_length) chosen = slot;
                    continue;
                }
                if (matches[slot].needle_index < matches[chosen].needle_index) chosen = slot;
            }
            if (chosen == segment_end) break;
            keep[chosen] = 1;
            cursor = matches[chosen].byte_offset + matches[chosen].byte_length;
        }
    }
}

/** Gathers the surviving matches into their scanned slots, order preserved: a match was kept
 *  exactly when the scan steps across it. */
kernel void sz_substrings_compact_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                uint position [[thread_position_in_grid]],
                                                uint threads [[threads_per_grid]]) {
    ulong const host_base = arguments.host_base;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base,
                                                                                                      arguments.report);
    if (report->shortfall) return;
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(arena, host_base,
                                                                                                     arguments.emitted);
    device sz_substrings_match_metal_t *survivors = sz_reach_metal_<sz_substrings_match_metal_t>(arena, host_base,
                                                                                                 arguments.reported);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.keep_offsets);
    for (ulong index = position; index < report->matches_emitted; index += threads)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/** Publishes how many matches the cover kept, which every later boundary is read against. */
kernel void sz_substrings_covered_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        arena, arguments.host_base, arguments.report);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.keep_offsets);
    ulong const kept = keep_offsets[arguments.matches_budget];
    if (!report->shortfall) report->matches_stored = kept;
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
kernel void sz_substrings_haystack_offsets_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                         constant sz_substrings_simt_arguments_t &arguments
                                                         [[buffer(1)]],
                                                         uint position [[thread_position_in_grid]],
                                                         uint threads [[threads_per_grid]]) {
    ulong const host_base = arguments.host_base;
    if (sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base, arguments.report)->shortfall) return;
    device ulong const *chunk_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.chunk_offsets);
    device ulong const *chunk_slots = sz_reach_metal_<ulong>(arena, host_base, arguments.chunk_slots);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.keep_offsets);
    device ulong *haystack_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.haystack_offsets);
    for (ulong index = position; index <= arguments.haystacks_count; index += threads) {
        ulong const emitted_before = chunk_slots[chunk_offsets[index]];
        haystack_offsets[index] = arguments.keep_offsets ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
kernel void sz_substrings_counts_metal_kernel_(device uchar *arena [[buffer(0)]],
                                               constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                               uint position [[thread_position_in_grid]],
                                               uint threads [[threads_per_grid]]) {
    device ulong const *haystack_offsets = sz_reach_metal_<ulong>(arena, arguments.host_base,
                                                                  arguments.haystack_offsets);
    device ulong *counts = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.counts);
    for (ulong index = position; index < arguments.haystacks_count; index += threads)
        counts[index * arguments.counts_stride] = haystack_offsets[index + 1] - haystack_offsets[index];
}

/** Copies the surviving matches into the caller's array, clipped at a capacity only it knows. */
kernel void sz_substrings_store_matches_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                      constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                      uint position [[thread_position_in_grid]],
                                                      uint threads [[threads_per_grid]]) {
    ulong const host_base = arguments.host_base;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base,
                                                                                                      arguments.report);
    device sz_substrings_match_metal_t const *reported = sz_reach_metal_<sz_substrings_match_metal_t>(
        arena, host_base, arguments.reported);
    device sz_substrings_match_metal_t *matches = sz_reach_metal_<sz_substrings_match_metal_t>(arena, host_base,
                                                                                               arguments.matches);
    ulong const fitting = min(report->shortfall ? 0 : report->matches_stored, arguments.matches_capacity);
    for (ulong index = position; index < fitting; index += threads) matches[index] = reported[index];
}

/** Publishes what the store kept, once every thread of it has read the report it changes. */
kernel void sz_substrings_stored_metal_kernel_(device uchar *arena [[buffer(0)]],
                                               constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        arena, arguments.host_base, arguments.report);
    ulong const kept = report->shortfall ? 0 : report->matches_stored;
    report->matches_stored = min(kept, arguments.matches_capacity);
    if (kept > arguments.matches_capacity) report->shortfall = kept - arguments.matches_capacity;
}

/**
 *  @brief Writes where each match's preceding gap lands, and how long each haystack becomes, one
 *      threadgroup per haystack.
 *
 *  A rewrite is a tiling of gaps and replacements, and every boundary in it follows from how far
 *  the output has drifted from the input by the time a match is reached, so that drift is all this
 *  stores. Offsets are relative to the haystack's own start, since the base is only known after the
 *  scan across haystacks that this feeds.
 */
kernel void sz_substrings_rewrite_offsets_metal_kernel_(
    device uchar *arena [[buffer(0)]], constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_simt_k];
    ulong const host_base = arguments.host_base;
    if (sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base, arguments.report)->shortfall) return;
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, host_base, arguments.haystacks);
    device sz_view_metal_t const *replacements = sz_reach_metal_<sz_view_metal_t>(arena, host_base,
                                                                                  arguments.replacements);
    device ulong const *haystack_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.haystack_offsets);
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        arena, host_base, arguments.reported);
    device ulong *gap_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.gap_offsets);
    device ulong *output_sizes = sz_reach_metal_<ulong>(arena, host_base, arguments.output_offsets);

    for (ulong haystack_index = group; haystack_index < arguments.haystacks_count; haystack_index += groups) {
        ulong const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
        ulong drift_carry = 0, drift_in_tile;
        for (ulong tile_first = first; tile_first < last; tile_first += width) {
            ulong const match_index = tile_first + lane;
            bool const owns = match_index < last;
            ulong drift_here = 0, previous_end = 0;
            if (owns) {
                // Shrinking matches make this wrap, which is right: only the prefix sums
                // are ever read.
                drift_here = replacements[matches[match_index].needle_index].length - matches[match_index].byte_length;
                if (match_index != first)
                    previous_end = matches[match_index - 1].byte_offset + matches[match_index - 1].byte_length;
            }
            ulong const drift_before = sz_substrings_block_scan_simt_(drift_here, shared, lane, width, drift_in_tile);
            if (owns) gap_offsets[match_index] = previous_end + drift_carry + drift_before;
            drift_carry += drift_in_tile;
        }
        if (lane == 0) output_sizes[haystack_index] = views[haystack_index].length + drift_carry;
    }
}

/** Copies one stretch, clipped to the tile, with @p lane striding the surviving bytes. */
inline void sz_substrings_copy_clipped_simt_(device uchar *output, ulong tile_begin, ulong tile_end,
                                             ulong output_offset, device uchar const *source, ulong bytes, uint lane) {
    ulong const copy_end = min(output_offset + bytes, tile_end);
    for (ulong position = max(output_offset, tile_begin) + lane; position < copy_end; position += 32)
        output[position] = source[position - output_offset];
}

/** Publishes the bytes the rewrite needs, and whether the target could hold them. */
kernel void sz_substrings_target_metal_kernel_(device uchar *arena [[buffer(0)]],
                                               constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        arena, arguments.host_base, arguments.report);
    device ulong const *output_offsets = sz_reach_metal_<ulong>(arena, arguments.host_base, arguments.output_offsets);
    ulong const rewritten = output_offsets[arguments.haystacks_count];
    report->target_length = rewritten;
    if (!report->shortfall && rewritten > arguments.target_ceiling)
        report->shortfall = rewritten - arguments.target_ceiling;
}

/** Copies the rewritten target, one fixed-width output tile per threadgroup, one simdgroup per gap
 *  or replacement, so no threadgroup's work scales with one stretch's width. */
kernel void sz_substrings_rewrite_copy_metal_kernel_(device uchar *arena [[buffer(0)]],
                                                     constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                                     uint group [[threadgroup_position_in_grid]],
                                                     uint groups [[threadgroups_per_grid]],
                                                     uint simdgroup [[simdgroup_index_in_threadgroup]],
                                                     uint simdgroups [[simdgroups_per_threadgroup]],
                                                     uint lane [[thread_index_in_simdgroup]]) {
    ulong const host_base = arguments.host_base;
    // A target that cannot hold the whole rewrite is left untouched rather than holding a valid
    // prefix of one.
    if (sz_reach_metal_<sz_substrings_report_metal_t>(arena, host_base, arguments.report)->shortfall) return;
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, host_base, arguments.haystacks);
    device sz_view_metal_t const *replacements = sz_reach_metal_<sz_view_metal_t>(arena, host_base,
                                                                                  arguments.replacements);
    device ulong const *haystack_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.haystack_offsets);
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        arena, host_base, arguments.reported);
    device ulong const *gap_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.gap_offsets);
    device ulong const *output_offsets = sz_reach_metal_<ulong>(arena, host_base, arguments.output_offsets);
    device uchar *output = sz_reach_metal_<uchar>(arena, host_base, arguments.target);
    ulong const count = arguments.haystacks_count, total = output_offsets[count];
    ulong const tiles = sz_metal_divide_round_up_<ulong>(total, sz_substrings_rewrite_tile_bytes_simt_k);

    for (ulong tile = group; tile < tiles; tile += groups) {
        ulong const tile_begin = tile * sz_substrings_rewrite_tile_bytes_simt_k;
        ulong const tile_end = min(tile_begin + sz_substrings_rewrite_tile_bytes_simt_k, total);
        for (ulong haystack_index = sz_substrings_last_not_above_simt_(output_offsets, count + 1, tile_begin);
             haystack_index < count && output_offsets[haystack_index] < tile_end; ++haystack_index) {
            device uchar const *haystack = sz_reach_metal_<uchar>(arena, host_base, views[haystack_index].start);
            ulong const haystack_length = views[haystack_index].length, base = output_offsets[haystack_index];
            ulong const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
            ulong const wanted = tile_begin > base ? tile_begin - base : 0;
            ulong const skip = first == last
                                   ? 0
                                   : sz_substrings_last_not_above_simt_(gap_offsets + first, last - first, wanted);
            // Past the last match the drift is the whole haystack's, which its rewritten
            // length already names.
            ulong const total_drift = (output_offsets[haystack_index + 1] - base) - haystack_length;
            // Every match contributes a gap and a replacement; one more stretch
            // closes the haystack.
            for (ulong stretch = skip + simdgroup; stretch < last - first + 1; stretch += simdgroups) {
                ulong const match_index = first + stretch;
                bool const closes = match_index == last;
                ulong const previous_end = match_index == first ? 0
                                                                : matches[match_index - 1].byte_offset +
                                                                      matches[match_index - 1].byte_length;
                ulong const gap_source_end = closes ? haystack_length : matches[match_index].byte_offset;
                ulong const gap_begin = base + (closes ? previous_end + total_drift : gap_offsets[match_index]);
                sz_substrings_copy_clipped_simt_(output, tile_begin, tile_end, gap_begin, haystack + previous_end,
                                                 gap_source_end - previous_end, lane);
                if (closes) continue;
                sz_view_metal_t const replacement = replacements[matches[match_index].needle_index];
                sz_substrings_copy_clipped_simt_(
                    output, tile_begin, tile_end, gap_begin + (gap_source_end - previous_end),
                    sz_reach_metal_<uchar>(arena, host_base, replacement.start), replacement.length, lane);
            }
        }
    }
}

/**
 *  @brief Scores one haystack per threadgroup: its threads walk contiguous chunks into one
 *      threadgroup tally, then sum the tallied terms in 32.32 fixed point.
 *
 *  A threadgroup rather than a grid per haystack keeps the tally in threadgroup memory.
 *  A vocabulary wider than the slots hashes into them and spills into the threadgroup's
 *  own overflow row.
 */
kernel void sz_substrings_bm25_metal_kernel_(device uchar *arena [[buffer(0)]],
                                             constant sz_substrings_simt_arguments_t &arguments [[buffer(1)]],
                                             uint group [[threadgroup_position_in_grid]],
                                             uint groups [[threadgroups_per_grid]],
                                             uint lane [[thread_index_in_threadgroup]],
                                             uint width [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[sz_substrings_tally_slots_simt_k], keys[sz_substrings_tally_slots_simt_k];
    threadgroup atomic_uint overflowed;
    threadgroup long partials[sz_substrings_threads_simt_k];
    ulong const host_base = arguments.host_base, needles_count = arguments.needles_count;
    ulong const slots = min(needles_count, (ulong)sz_substrings_tally_slots_simt_k);
    sz_substrings_automaton_simt_t const automaton = sz_substrings_automaton_simt_(arena, arguments);
    device sz_view_metal_t const *views = sz_reach_metal_<sz_view_metal_t>(arena, host_base, arguments.haystacks);
    device float const *weights = sz_reach_metal_<float>(arena, host_base, arguments.needle_weights);
    device float const *lengths = sz_reach_metal_<float>(arena, host_base, arguments.document_lengths);
    device float *scores = sz_reach_metal_<float>(arena, host_base, arguments.scores);
    device atomic_uint *overflow = sz_reach_metal_<atomic_uint>(arena, host_base, arguments.overflow_rows) +
                                   group * needles_count;
    sz_substrings_tallier_simt_t tallier;
    tallier.keys = keys, tallier.counts = counts, tallier.overflowed = &overflowed, tallier.overflow = overflow;
    tallier.hashed = arguments.tally_hashed != 0;
    // Never narrower than the longest match, past which a chunk re-walks more warm-up than it owns.
    ulong const warm_up = max((ulong)automaton.max_source_match_bytes, 1ul);
    float const saturation = arguments.saturation, normalization = arguments.normalization;

    for (ulong slot = lane; slot < slots; slot += width) {
        atomic_store_explicit(&counts[slot], 0u, memory_order_relaxed);
        atomic_store_explicit(&keys[slot], 0u, memory_order_relaxed);
    }
    if (lane == 0) atomic_store_explicit(&overflowed, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong haystack_index = group; haystack_index < arguments.haystacks_count; haystack_index += groups) {
        sz_view_metal_t const view = views[haystack_index];
        ulong const chunk_bytes = max(sz_metal_divide_round_up_<ulong>(view.length, width), warm_up);
        ulong const chunk_begin = (ulong)lane * chunk_bytes;
        float const document_length = arguments.document_lengths ? lengths[haystack_index] : (float)view.length;
        float const norm = normalization > 0
                               ? 1 - normalization + normalization * document_length / arguments.average_length
                               : 1.0f;
        if (chunk_begin < view.length)
            sz_substrings_walk_chunk_simt_(automaton, sz_reach_metal_<uchar>(arena, host_base, view.start), chunk_begin,
                                           min(chunk_begin + chunk_bytes, view.length), tallier);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // Every slot and overflow entry is zeroed as it is read, so the next haystack starts from
        // a clean tally.
        long mine = 0;
        for (ulong slot = lane; slot < slots; slot += width) {
            uint const frequency = atomic_load_explicit(&counts[slot], memory_order_relaxed);
            if (!frequency) continue;
            ulong const needle = tallier.hashed ? atomic_load_explicit(&keys[slot], memory_order_relaxed) - 1 : slot;
            mine += sz_substrings_bm25_fixed_simt_(weights[needle], frequency, saturation, norm);
            atomic_store_explicit(&counts[slot], 0u, memory_order_relaxed);
            atomic_store_explicit(&keys[slot], 0u, memory_order_relaxed);
        }
        if (atomic_load_explicit(&overflowed, memory_order_relaxed))
            for (ulong needle = lane; needle < needles_count; needle += width) {
                uint const frequency = atomic_load_explicit(overflow + needle, memory_order_relaxed);
                if (!frequency) continue;
                mine += sz_substrings_bm25_fixed_simt_(weights[needle], frequency, saturation, norm);
                atomic_store_explicit(overflow + needle, 0u, memory_order_relaxed);
            }

        // A tree over the threadgroup's partials; integer addition commutes, so the
        // order never shows.
        partials[lane] = mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint half_width = width / 2; half_width != 0; half_width /= 2) {
            if (lane < half_width) partials[lane] += partials[lane + half_width];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            scores[haystack_index * arguments.scores_stride] = (float)partials[0] * (1.0f / 4294967296.0f);
            atomic_store_explicit(&overflowed, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}
