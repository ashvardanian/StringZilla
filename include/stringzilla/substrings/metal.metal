/**
 *  @file include/stringzilla/substrings/metal.metal
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Multi-pattern search on Apple GPUs of Metal family 7, M1 and newer: one thread per
 *      haystack chunk, and the leftmost cover resolved after the walk rather than inside it.
 *
 *  @sa include/stringzilla/substrings/metal.h, which embeds and launches this source
 *  @sa include/stringzilla/substrings/simt.cuh, the CUDA sibling this mirrors kernel for kernel
 *  @sa include/stringzilla/types.metal, the prelude compiled ahead of this source
 *
 *  MSL cannot include the C headers, so the transition, the acceptance bit and the BM25 terms are
 *  mirrored here. Apple GPUs have no @c f64, so a BM25 term is computed in @c f32 and summed in
 *  32.32 fixed point, which keeps a score independent of thread order; scores then agree with
 *  serial's to the tolerance the CUDA tier is held to rather than bit for bit.
 *
 *  The automaton and the round block each point inside themselves by host address, turned into the
 *  kernel's own by @ref sz_reach_metal_ against the block's host address; the tapes and the
 *  caller's arrays are bound directly, and a clear or a scan finds its target bound in a slot of
 *  its own. Metal has no 64-bit atomic add and nothing here needs one: every count and offset comes
 *  out of a scan, and every BM25 sum out of a tree over one threadgroup's partials.
 */

/** Threads every threadgroup here runs, as the CUDA tier's blocks. */
constant uint sz_substrings_threads_metal_k = 256;

/** Candidates one thread scans quadratically before a cover segment falls back to emitted order. */
constant ulong sz_substrings_cover_segment_limit_metal_k = 4096;

/** Output bytes one threadgroup of a rewrite's copy owns, as
 *  @c sz_substrings_gpu_rewrite_tile_bytes_k. */
constant ulong sz_substrings_gpu_rewrite_tile_bytes_metal_k = 4096;

/** Bits of a tally slot index: 2048 slots of a key and a count each leave room in 32 KB of
 *  threadgroup memory. */
constant uint sz_substrings_tally_slot_bits_metal_k = 11;

/** Slots one threadgroup's tally holds. */
constant uint sz_substrings_tally_slots_metal_k = 1u << sz_substrings_tally_slot_bits_metal_k;

/** Slots a hashed tally probes before spilling a needle to the threadgroup's overflow row. */
constant uint sz_substrings_tally_probes_metal_k = 16;

/** What a chunk walk does at each match, as @c sz_substrings_gpu_pass_t numbers it. */
constant ulong sz_substrings_gpu_sizing_metal_k = 0, sz_substrings_gpu_writing_metal_k = 1,
               sz_substrings_gpu_tallying_metal_k = 2;

/** Fraction bits of a BM25 sum, as @c sz_substrings_gpu_bm25_fraction_bits_k. */
constant uint sz_substrings_gpu_bm25_fraction_bits_metal_k = 32;

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

/** The launch record the host copies in: host addresses inside the automaton and the round block,
 *  and the round's shape. */
struct sz_substrings_arguments_metal_t {
    ulong engine_host, scratch_host;
    ulong hot_rows, byte_to_class, base, check, fail, accepts_words, outputs, outputs_counts, outputs_offsets;
    ulong hot_count, classes_count, root, needles_count, max_source_match_bytes, overlap_policy, case_sensitivity;
    ulong report, chunk_bytes, chunk_offsets, chunk_slots, tile_sums;
    ulong emitted, reported, keep_offsets, gap_offsets, overflow_rows;
    ulong slots_count, chunk_budget, chunk_floor, matches_budget, emitting;
    ulong haystacks_count, counts_stride, matches_capacity, target_ceiling;
    ulong document_lengths, scores_stride, tally_hashed;
    float saturation, normalization, average_length, unused;
    ulong pass, scan_count, scan_elements_per_tile, scan_tiles, clear_words;
};

/** The automaton a walk steps, as @c sz_substrings_engine_t lays it out. */
struct sz_substrings_automaton_metal_t {
    device uint const *hot_rows;
    device uchar const *byte_to_class;
    device uint const *base, *check, *fail, *accepts_words, *outputs_counts;
    device sz_substrings_output_metal_t const *outputs;
    device ulong const *outputs_offsets;
    uint hot_count, classes_count, root, max_source_match_bytes;
    bool uncased;
};

inline sz_substrings_automaton_metal_t sz_substrings_automaton_metal_(
    device uchar *engine, constant sz_substrings_arguments_metal_t &arguments) {
    ulong const engine_host = arguments.engine_host;
    sz_substrings_automaton_metal_t automaton;
    automaton.hot_rows = sz_reach_metal_<uint>(engine, engine_host, arguments.hot_rows);
    automaton.byte_to_class = sz_reach_metal_<uchar>(engine, engine_host, arguments.byte_to_class);
    automaton.base = sz_reach_metal_<uint>(engine, engine_host, arguments.base);
    automaton.check = sz_reach_metal_<uint>(engine, engine_host, arguments.check);
    automaton.fail = sz_reach_metal_<uint>(engine, engine_host, arguments.fail);
    automaton.accepts_words = sz_reach_metal_<uint>(engine, engine_host, arguments.accepts_words);
    automaton.outputs_counts = sz_reach_metal_<uint>(engine, engine_host, arguments.outputs_counts);
    automaton.outputs = sz_reach_metal_<sz_substrings_output_metal_t>(engine, engine_host, arguments.outputs);
    automaton.outputs_offsets = sz_reach_metal_<ulong>(engine, engine_host, arguments.outputs_offsets);
    automaton.hot_count = (uint)arguments.hot_count, automaton.classes_count = (uint)arguments.classes_count;
    automaton.root = (uint)arguments.root, automaton.max_source_match_bytes = (uint)arguments.max_source_match_bytes;
    automaton.uncased = arguments.case_sensitivity != 0;
    return automaton;
}

/** Advances @p state by one byte, as @c sz_substrings_step does. */
inline uint sz_substrings_step_metal_(thread sz_substrings_automaton_metal_t const &automaton, uint state, uchar byte) {
    for (;;) {
        if (state < automaton.hot_count)
            return automaton.hot_rows[(ulong)state * automaton.classes_count + automaton.byte_to_class[byte]];
        ulong const candidate = (ulong)automaton.base[state] + byte;
        if (automaton.check[candidate] == state) return (uint)candidate;
        if (state == automaton.root) return automaton.root;
        state = automaton.fail[state];
    }
}

/** A threadgroup's BM25 tally: its slots, the flag a spill raises, and its overflow row. */
struct sz_substrings_tally_metal_t {
    threadgroup atomic_uint *keys, *counts, *overflowed;
    device atomic_uint *overflow;
    bool hashed;
};

/** Counts one match of @p needle into @p tally, in its slots first, as @c sz_substrings_tally_simt_
 *  does. */
inline void sz_substrings_tally_metal_(thread sz_substrings_tally_metal_t const *tally, uint needle) {
    if (!tally->hashed) {
        atomic_fetch_add_explicit(tally->counts + needle, 1u, memory_order_relaxed);
        return;
    }
    uint const key = needle + 1;
    uint slot = (uint)(((ulong)key * 0x9E3779B97F4A7C15ul) >> (64 - sz_substrings_tally_slot_bits_metal_k));
    for (uint probe = 0; probe != sz_substrings_tally_probes_metal_k; ++probe) {
        // Metal's compare-exchange is the weak one, so a spurious failure retries until a
        // key is there.
        uint seated = atomic_load_explicit(tally->keys + slot, memory_order_relaxed);
        while (seated == 0 &&
               !atomic_compare_exchange_weak_explicit(tally->keys + slot, &seated, key, memory_order_relaxed,
                                                      memory_order_relaxed) &&
               seated == 0) {}
        if (seated == 0 || seated == key) {
            atomic_fetch_add_explicit(tally->counts + slot, 1u, memory_order_relaxed);
            return;
        }
        slot = (slot + 1) & (sz_substrings_tally_slots_metal_k - 1);
    }
    atomic_fetch_add_explicit(tally->overflow + needle, 1u, memory_order_relaxed);
    atomic_store_explicit(tally->overflowed, 1u, memory_order_relaxed);
}

constant uint sz_substrings_folded_image_max_metal_k = 9;

inline ulong sz_substrings_resolve_match_metal_(device uchar const *haystack, ulong source_end, uint trailing,
                                                uint folded_match_bytes, uint shift, thread bool &repeats) {
    uchar ring[sz_substrings_folded_image_max_metal_k] = {};
    ulong const wanted = (ulong)folded_match_bytes + shift;
    ulong position = source_end, start_here = ~0ul, start_earlier = ~0ul;
    ulong bytes = 0;
    uint pending = 0;
    bool periodic = shift != 0;
    repeats = false;
    for (ulong stepped = 0; stepped < trailing + wanted; ++stepped) {
        if (!pending) {
            if (!position) break;
            ulong candidate = position - 1;
            for (uint back = 0; back != 3 && candidate && (haystack[candidate] & 0xc0) == 0x80; ++back) --candidate;
            sz_utf8_folded_image_metal_t image;
            uint const source_bytes = sz_utf8_fold_next_metal_(haystack + candidate, haystack + position, image);
            if (!image.rune_ends || candidate + source_bytes != position) {
                bytes = haystack[--position];
                pending = 1;
            }
            else {
                bytes = image.bytes;
                pending = image.length;
                position = candidate;
            }
        }
        uchar const byte = uchar(bytes >> (8 * --pending));
        if (stepped < trailing) continue;
        ulong const taken = stepped - trailing + 1;
        if (shift) {
            if (taken > shift && ring[(taken - shift) % sz_substrings_folded_image_max_metal_k] != byte)
                periodic = false;
            ring[taken % sz_substrings_folded_image_max_metal_k] = byte;
        }
        if (taken == folded_match_bytes) start_here = position;
        if (taken == wanted) start_earlier = position;
        if (!periodic && taken >= folded_match_bytes) break;
    }
    repeats = periodic && start_here != ~0ul && start_here == start_earlier;
    return start_here != ~0ul ? start_here : source_end;
}

inline ulong sz_substrings_walk_chunk_uncased_metal_(thread sz_substrings_automaton_metal_t const &automaton,
                                                     device uchar const *haystack, ulong length, ulong chunk_begin,
                                                     ulong chunk_end, ulong haystack_index, ulong pass,
                                                     device sz_substrings_match_metal_t *matches_at_chunk,
                                                     thread sz_substrings_tally_metal_t const *tally) {
    ulong const warm_up = automaton.max_source_match_bytes ? automaton.max_source_match_bytes - 1 : 0;
    ulong walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    for (uint back = 0; back != 3 && walk_begin && (haystack[walk_begin] & 0xc0) == 0x80; ++back) --walk_begin;
    uint state = automaton.root;
    ulong folded = 0, last_break_folded_end = 0, found = 0;
    for (ulong position = walk_begin; position < length;) {
        sz_utf8_folded_image_metal_t image;
        uint const source_bytes = sz_utf8_fold_next_metal_(haystack + position, haystack + length, image);
        position += source_bytes;
        bool const breaks_boundary = popcount(image.rune_ends) != 1 || image.length != source_bytes;
        uint previous_rune_end = 0;
        for (uint image_index = 0; image_index != image.length;) {
            ++folded;
            if (!image.rune_ends) {
                ++image_index;
                state = automaton.root;
                continue;
            }
            state = sz_substrings_step_metal_(automaton, state, uchar(image.bytes >> (8 * image_index)));
            if (!((image.rune_ends >> image_index++) & 1u)) continue;
            uint const trailing = image.length - image_index;
            uint const shift = previous_rune_end ? image_index - previous_rune_end : 0;
            previous_rune_end = image_index;
            if (breaks_boundary) last_break_folded_end = folded + trailing;
            if (position > chunk_end) return found;
            if (position <= chunk_begin || !((automaton.accepts_words[state >> 5] >> (state & 31u)) & 1u)) continue;
            uint const count = automaton.outputs_counts[state];
            ulong const first = automaton.outputs_offsets[state];
            for (uint index = 0; index != count; ++index) {
                sz_substrings_output_metal_t const output = automaton.outputs[first + index];
                uint const folded_length = output.folded_match_bytes;
                if (folded < folded_length) continue;
                bool repeats = false;
                ulong const source_end = position - walk_begin;
                ulong const source_offset = folded - folded_length >= last_break_folded_end
                                                ? source_end - folded_length
                                                : sz_substrings_resolve_match_metal_(haystack + walk_begin, source_end,
                                                                                     trailing, folded_length, shift,
                                                                                     repeats);
                if (repeats) continue;
                if (pass == sz_substrings_gpu_writing_metal_k)
                    matches_at_chunk[found] = {haystack_index, output.needle_index, walk_begin + source_offset,
                                               source_end - source_offset};
                else if (pass == sz_substrings_gpu_tallying_metal_k)
                    sz_substrings_tally_metal_(tally, output.needle_index);
                ++found;
            }
        }
    }
    return found;
}

/**
 *  @brief Walks one chunk of a haystack in the pass @p pass names, sizing, writing to
 *      @p matches_at_chunk, or tallying into @p tally every match ending inside it.
 *  @return The matches the chunk holds.
 *
 *  The warm-up primes the state from the @c max_source_match_bytes-1 bytes before the chunk,
 *  clamped to the haystack, and reports nothing, so every match is found by exactly one chunk.
 */
inline ulong sz_substrings_walk_chunk_metal_(thread sz_substrings_automaton_metal_t const &automaton,
                                             device uchar const *haystack, ulong chunk_begin, ulong chunk_end,
                                             ulong haystack_index, ulong pass,
                                             device sz_substrings_match_metal_t *matches_at_chunk,
                                             thread sz_substrings_tally_metal_t const *tally) {
    ulong const warm_up = automaton.max_source_match_bytes > 0 ? automaton.max_source_match_bytes - 1 : 0;
    ulong const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    uint state = automaton.root;
    ulong position = walk_begin, found = 0;
    for (; position < chunk_begin; ++position) state = sz_substrings_step_metal_(automaton, state, haystack[position]);
    for (; position < chunk_end; ++position) {
        state = sz_substrings_step_metal_(automaton, state, haystack[position]);
        if (!((automaton.accepts_words[state >> 5] >> (state & 31u)) & 1u)) continue;
        uint const count = automaton.outputs_counts[state];
        ulong const first = automaton.outputs_offsets[state];
        for (uint index = 0; index != count; ++index) {
            sz_substrings_output_metal_t const output = automaton.outputs[first + index];
            // `walk_begin` is clamped to the haystack's own start, so underflowing either
            // is one test.
            if (position + 1 - walk_begin < output.folded_match_bytes) continue;
            if (pass == sz_substrings_gpu_writing_metal_k)
                matches_at_chunk[found] = {haystack_index, output.needle_index,
                                           position + 1 - output.folded_match_bytes, output.folded_match_bytes};
            else if (pass == sz_substrings_gpu_tallying_metal_k) sz_substrings_tally_metal_(tally, output.needle_index);
            ++found;
        }
    }
    return found;
}

/** The threadgroup's exclusive prefix sum of @p value, its own total left in @p total, over
 *  @p shared of one entry per thread. */
inline ulong sz_substrings_block_scan_metal_(ulong value, threadgroup ulong *shared, uint lane, uint width,
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
inline ulong sz_substrings_last_not_above_metal_(device ulong const *ascending, ulong count, ulong value) {
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
inline long sz_substrings_bm25_fixed_metal_(float weight, uint frequency, float saturation, float norm) {
    float const term = weight * frequency * (saturation + 1) / (frequency + saturation * norm);
    return (long)rint(term * (float)(1ul << sz_substrings_gpu_bm25_fraction_bits_metal_k));
}

/** Zeroes @c clear_words words of the cleared slot, which is how a round clears what it reads
 *  first. */
kernel void sz_substrings_clear_metal_kernel_(device ulong *words [[buffer(5)]],
                                              constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                              uint position [[thread_position_in_grid]],
                                              uint threads [[threads_per_grid]]) {
    for (ulong index = position; index < arguments.clear_words; index += threads) words[index] = 0;
}

/** Reduces one threadgroup's own contiguous tile of the scanned slot into @c tile_sums. */
kernel void sz_substrings_scan_reduce_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                    device ulong const *values [[buffer(4)]],
                                                    constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                    uint group [[threadgroup_position_in_grid]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_metal_k];
    device ulong *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong const begin = (ulong)group * arguments.scan_elements_per_tile;
    ulong const end = min(begin + arguments.scan_elements_per_tile, arguments.scan_count);
    ulong running = 0, total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        sz_substrings_block_scan_metal_(index < end ? values[index] : 0, shared, lane, width, total);
        running += total;
    }
    if (lane == 0) tile_sums[group] = running;
}

/** Scans @c tile_sums in place on one threadgroup, carrying a running offset across its tiles. */
kernel void sz_substrings_scan_carry_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                   constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                   uint lane [[thread_index_in_threadgroup]],
                                                   uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_metal_k];
    device ulong *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong carry = 0, total;
    for (ulong first = 0; first < arguments.scan_tiles; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_substrings_block_scan_metal_(index < arguments.scan_tiles ? tile_sums[index] : 0,
                                                                shared, lane, width, total);
        if (index < arguments.scan_tiles) tile_sums[index] = carry + exclusive;
        carry += total;
    }
}

/** Scans one threadgroup's own tile of the scanned slot in place, seeded by the base the carry
 *  set. */
kernel void sz_substrings_scan_apply_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                   device ulong *values [[buffer(4)]],
                                                   constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                   uint group [[threadgroup_position_in_grid]],
                                                   uint lane [[thread_index_in_threadgroup]],
                                                   uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_metal_k];
    device ulong const *tile_sums = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.tile_sums);
    ulong const begin = (ulong)group * arguments.scan_elements_per_tile;
    ulong const end = min(begin + arguments.scan_elements_per_tile, arguments.scan_count);
    ulong running = tile_sums[group], total;
    for (ulong first = begin; first < end; first += width) {
        ulong const index = first + lane;
        ulong const exclusive = sz_substrings_block_scan_metal_(index < end ? values[index] : 0, shared, lane, width,
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
kernel void sz_substrings_chunk_bytes_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                    device ulong const *haystacks [[buffer(2)]],
                                                    constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                    uint lane [[thread_index_in_threadgroup]],
                                                    uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_metal_k];
    sz_sequence_tape_metal_t const tape = {haystacks};
    ulong mine = 0, total;
    for (ulong index = lane; index < arguments.haystacks_count; index += width)
        mine += sz_sequence_tape_length_metal_(tape, index);
    sz_substrings_block_scan_metal_(mine, shared, lane, width, total);
    if (lane) return;
    ulong const budget = max(arguments.chunk_budget, 1ul);
    *sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.chunk_bytes) = max(
        max(sz_size_divide_round_up_metal_<ulong>(total, budget), arguments.chunk_floor), 1ul);
}

/** Writes how many chunks each haystack is cut into - at least one, so an empty haystack still gets
 *  a thread and lands its own boundary - which the scan then turns into its chunk range. */
kernel void sz_substrings_chunk_counts_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                     device ulong const *haystacks [[buffer(2)]],
                                                     constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                     uint position [[thread_position_in_grid]],
                                                     uint threads [[threads_per_grid]]) {
    sz_sequence_tape_metal_t const tape = {haystacks};
    device ulong *chunk_offsets = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.chunk_offsets);
    ulong const chunk_bytes = *sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.chunk_bytes);
    for (ulong index = position; index < arguments.haystacks_count; index += threads) {
        ulong const length = sz_sequence_tape_length_metal_(tape, index);
        chunk_offsets[index] = length == 0 ? 1 : sz_size_divide_round_up_metal_<ulong>(length, chunk_bytes);
    }
}

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in the pass @c pass names.
 *
 *  Both passes share @c chunk_slots: sizing writes each chunk's match count into its slot, and
 *  writing reads the exclusive offset the scan left there, so every chunk owns a private output
 *  range and a write needs no atomic.
 */
kernel void sz_substrings_walk_metal_kernel_(device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
                                             device ulong const *haystacks [[buffer(2)]],
                                             constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                             uint position [[thread_position_in_grid]],
                                             uint threads [[threads_per_grid]]) {
    ulong const scratch_host = arguments.scratch_host;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, scratch_host, arguments.report);
    // The writing pass has nowhere to write once the sizing pass outran the budget, so
    // it retires whole.
    if (arguments.pass == sz_substrings_gpu_writing_metal_k && report->shortfall) return;
    sz_substrings_automaton_metal_t const automaton = sz_substrings_automaton_metal_(engine, arguments);
    sz_sequence_tape_metal_t const tape = {haystacks};
    device ulong const *chunk_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.chunk_offsets);
    device ulong *chunk_slots = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.chunk_slots);
    device sz_substrings_match_metal_t *emitted = sz_reach_metal_<sz_substrings_match_metal_t>(scratch, scratch_host,
                                                                                               arguments.emitted);
    ulong const chunk_count = chunk_offsets[arguments.haystacks_count];
    ulong const chunk_bytes = *sz_reach_metal_<ulong>(scratch, scratch_host, arguments.chunk_bytes);

    for (ulong chunk_index = position; chunk_index < chunk_count; chunk_index += threads) {
        ulong const haystack_index = sz_substrings_last_not_above_metal_(chunk_offsets, arguments.haystacks_count,
                                                                         chunk_index);
        ulong const length = sz_sequence_tape_length_metal_(tape, haystack_index);
        ulong const chunk_begin = (chunk_index - chunk_offsets[haystack_index]) * chunk_bytes;
        ulong const chunk_end = min(chunk_begin + chunk_bytes, length);
        device sz_substrings_match_metal_t *matches_at_chunk = arguments.pass == sz_substrings_gpu_writing_metal_k
                                                                   ? emitted + chunk_slots[chunk_index]
                                                                   : nullptr;
        device uchar const *haystack = sz_sequence_tape_start_metal_(tape, haystack_index);
        ulong const found = chunk_begin >= chunk_end ? 0
                            : automaton.uncased
                                ? sz_substrings_walk_chunk_uncased_metal_(automaton, haystack, length, chunk_begin,
                                                                          chunk_end, haystack_index, arguments.pass,
                                                                          matches_at_chunk, nullptr)
                                : sz_substrings_walk_chunk_metal_(automaton, haystack, chunk_begin, chunk_end,
                                                                  haystack_index, arguments.pass, matches_at_chunk,
                                                                  nullptr);
        if (arguments.pass == sz_substrings_gpu_sizing_metal_k) chunk_slots[chunk_index] = found;
    }
}

/** Publishes what the sizing walk found, the one place a round learns whether it fit. The scanned
 *  chunk slots end in the emitted total, which every trailing zero carried. */
kernel void sz_substrings_sized_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, arguments.scratch_host, arguments.report);
    device ulong const *chunk_slots = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.chunk_slots);
    ulong const emitted = chunk_slots[arguments.slots_count - 1];
    report->matches_emitted = emitted, report->matches_stored = emitted, report->target_length = 0;
    report->shortfall = arguments.emitting && emitted > arguments.matches_budget ? emitted - arguments.matches_budget
                                                                                 : 0;
}

/** Whether the boundary before @p index is real: nothing still to come starts before the maximum
 *  end already reached, and only matches ending within one match's length of it can. */
inline bool sz_substrings_boundary_before_metal_(device sz_substrings_match_metal_t const *matches, ulong count,
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

/** Decides which overlapping matches survive a leftmost cover, one segment per thread, as the CUDA
 *  tier's cover does, falling back to emitted order past
 *  @ref sz_substrings_cover_segment_limit_metal_k candidates. */
kernel void sz_substrings_cover_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                              constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                              uint position [[thread_position_in_grid]],
                                              uint threads [[threads_per_grid]]) {
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, arguments.scratch_host, arguments.report);
    if (report->shortfall) return;
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        scratch, arguments.scratch_host, arguments.emitted);
    device ulong *keep = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.keep_offsets);
    ulong const count = report->matches_emitted, longest = arguments.max_source_match_bytes;
    bool const longest_first = arguments.overlap_policy == 1; // `sz_substrings_leftmost_longest_k`
    for (ulong index = position; index < count; index += threads) {
        // Only a segment's first match works; the rest are decided by whoever owns their segment.
        if (!sz_substrings_boundary_before_metal_(matches, count, longest, index)) continue;
        ulong segment_end = index + 1;
        while (segment_end < count && !sz_substrings_boundary_before_metal_(matches, count, longest, segment_end))
            ++segment_end;

        if (segment_end - index > sz_substrings_cover_segment_limit_metal_k) {
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
kernel void sz_substrings_compact_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                                uint position [[thread_position_in_grid]],
                                                uint threads [[threads_per_grid]]) {
    ulong const scratch_host = arguments.scratch_host;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, scratch_host, arguments.report);
    if (report->shortfall) return;
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        scratch, scratch_host, arguments.emitted);
    device sz_substrings_match_metal_t *survivors = sz_reach_metal_<sz_substrings_match_metal_t>(scratch, scratch_host,
                                                                                                 arguments.reported);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.keep_offsets);
    for (ulong index = position; index < report->matches_emitted; index += threads)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/** Publishes how many matches the cover kept, which every later boundary is read against. */
kernel void sz_substrings_covered_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, arguments.scratch_host, arguments.report);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(scratch, arguments.scratch_host, arguments.keep_offsets);
    ulong const kept = keep_offsets[arguments.matches_budget];
    if (!report->shortfall) report->matches_stored = kept;
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
kernel void sz_substrings_haystack_offsets_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                         device ulong *haystack_offsets [[buffer(3)]],
                                                         constant sz_substrings_arguments_metal_t &arguments
                                                         [[buffer(14)]],
                                                         uint position [[thread_position_in_grid]],
                                                         uint threads [[threads_per_grid]]) {
    ulong const scratch_host = arguments.scratch_host;
    if (sz_reach_metal_<sz_substrings_report_metal_t>(scratch, scratch_host, arguments.report)->shortfall) return;
    device ulong const *chunk_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.chunk_offsets);
    device ulong const *chunk_slots = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.chunk_slots);
    device ulong const *keep_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.keep_offsets);
    for (ulong index = position; index <= arguments.haystacks_count; index += threads) {
        ulong const emitted_before = chunk_slots[chunk_offsets[index]];
        haystack_offsets[index] = arguments.keep_offsets ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
kernel void sz_substrings_counts_metal_kernel_(device ulong const *haystack_offsets [[buffer(3)]],
                                               device ulong *counts [[buffer(6)]],
                                               constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
                                               uint position [[thread_position_in_grid]],
                                               uint threads [[threads_per_grid]]) {
    for (ulong index = position; index < arguments.haystacks_count; index += threads)
        counts[index * arguments.counts_stride] = haystack_offsets[index + 1] - haystack_offsets[index];
}

/** Copies the surviving matches into the caller's array, clipped at a capacity only it knows. */
kernel void sz_substrings_store_matches_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                                      device sz_substrings_match_metal_t *matches [[buffer(7)]],
                                                      constant sz_substrings_arguments_metal_t &arguments
                                                      [[buffer(14)]],
                                                      uint position [[thread_position_in_grid]],
                                                      uint threads [[threads_per_grid]]) {
    ulong const scratch_host = arguments.scratch_host;
    device sz_substrings_report_metal_t const *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, scratch_host, arguments.report);
    device sz_substrings_match_metal_t const *reported = sz_reach_metal_<sz_substrings_match_metal_t>(
        scratch, scratch_host, arguments.reported);
    ulong const fitting = min(report->shortfall ? 0 : report->matches_stored, arguments.matches_capacity);
    for (ulong index = position; index < fitting; index += threads) matches[index] = reported[index];
}

/** Publishes what the store kept, once every thread of it has read the report it changes. */
kernel void sz_substrings_stored_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                               constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, arguments.scratch_host, arguments.report);
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
    device uchar *scratch [[buffer(1)]], device ulong const *haystacks [[buffer(2)]],
    device ulong const *haystack_offsets [[buffer(3)]], device ulong const *replacements [[buffer(8)]],
    device ulong *output_sizes [[buffer(10)]], constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    threadgroup ulong shared[sz_substrings_threads_metal_k];
    ulong const scratch_host = arguments.scratch_host;
    if (sz_reach_metal_<sz_substrings_report_metal_t>(scratch, scratch_host, arguments.report)->shortfall) return;
    sz_sequence_tape_metal_t const haystacks_tape = {haystacks}, replacements_tape = {replacements};
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        scratch, scratch_host, arguments.reported);
    device ulong *gap_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.gap_offsets);

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
                drift_here = sz_sequence_tape_length_metal_(replacements_tape, matches[match_index].needle_index) -
                             matches[match_index].byte_length;
                if (match_index != first)
                    previous_end = matches[match_index - 1].byte_offset + matches[match_index - 1].byte_length;
            }
            ulong const drift_before = sz_substrings_block_scan_metal_(drift_here, shared, lane, width, drift_in_tile);
            if (owns) gap_offsets[match_index] = previous_end + drift_carry + drift_before;
            drift_carry += drift_in_tile;
        }
        if (lane == 0)
            output_sizes[haystack_index] = sz_sequence_tape_length_metal_(haystacks_tape, haystack_index) + drift_carry;
    }
}

/** Copies one stretch, clipped to the tile, with @p lane striding the surviving bytes. */
inline void sz_substrings_copy_clipped_metal_(device uchar *output, ulong tile_begin, ulong tile_end,
                                              ulong output_offset, device uchar const *source, ulong bytes, uint lane) {
    ulong const copy_end = min(output_offset + bytes, tile_end);
    for (ulong position = max(output_offset, tile_begin) + lane; position < copy_end; position += 32)
        output[position] = source[position - output_offset];
}

/** Publishes the bytes the rewrite needs, and whether the target could hold them. */
kernel void sz_substrings_target_metal_kernel_(device uchar *scratch [[buffer(1)]],
                                               device ulong const *output_offsets [[buffer(10)]],
                                               constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]]) {
    device sz_substrings_report_metal_t *report = sz_reach_metal_<sz_substrings_report_metal_t>(
        scratch, arguments.scratch_host, arguments.report);
    ulong const rewritten = output_offsets[arguments.haystacks_count];
    report->target_length = rewritten;
    if (!report->shortfall && rewritten > arguments.target_ceiling)
        report->shortfall = rewritten - arguments.target_ceiling;
}

/** Copies the rewritten target, one fixed-width output tile per threadgroup, one simdgroup per gap
 *  or replacement, so no threadgroup's work scales with one stretch's width. */
kernel void sz_substrings_rewrite_copy_metal_kernel_(
    device uchar *scratch [[buffer(1)]], device ulong const *haystacks [[buffer(2)]],
    device ulong const *haystack_offsets [[buffer(3)]], device ulong const *replacements [[buffer(8)]],
    device uchar *output [[buffer(9)]], device ulong const *output_offsets [[buffer(10)]],
    constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    ulong const scratch_host = arguments.scratch_host;
    // A target that cannot hold the whole rewrite is left untouched rather than holding a valid
    // prefix of one.
    if (sz_reach_metal_<sz_substrings_report_metal_t>(scratch, scratch_host, arguments.report)->shortfall) return;
    sz_sequence_tape_metal_t const haystacks_tape = {haystacks}, replacements_tape = {replacements};
    device sz_substrings_match_metal_t const *matches = sz_reach_metal_<sz_substrings_match_metal_t>(
        scratch, scratch_host, arguments.reported);
    device ulong const *gap_offsets = sz_reach_metal_<ulong>(scratch, scratch_host, arguments.gap_offsets);
    ulong const count = arguments.haystacks_count, total = output_offsets[count];
    ulong const tiles = sz_size_divide_round_up_metal_<ulong>(total, sz_substrings_gpu_rewrite_tile_bytes_metal_k);

    for (ulong tile = group; tile < tiles; tile += groups) {
        ulong const tile_begin = tile * sz_substrings_gpu_rewrite_tile_bytes_metal_k;
        ulong const tile_end = min(tile_begin + sz_substrings_gpu_rewrite_tile_bytes_metal_k, total);
        for (ulong haystack_index = sz_substrings_last_not_above_metal_(output_offsets, count + 1, tile_begin);
             haystack_index < count && output_offsets[haystack_index] < tile_end; ++haystack_index) {
            device uchar const *haystack = sz_sequence_tape_start_metal_(haystacks_tape, haystack_index);
            ulong const haystack_length = sz_sequence_tape_length_metal_(haystacks_tape, haystack_index);
            ulong const base = output_offsets[haystack_index];
            ulong const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
            ulong const wanted = tile_begin > base ? tile_begin - base : 0;
            ulong const skip = first == last
                                   ? 0
                                   : sz_substrings_last_not_above_metal_(gap_offsets + first, last - first, wanted);
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
                sz_substrings_copy_clipped_metal_(output, tile_begin, tile_end, gap_begin, haystack + previous_end,
                                                  gap_source_end - previous_end, lane);
                if (closes) continue;
                ulong const needle_index = matches[match_index].needle_index;
                sz_substrings_copy_clipped_metal_(
                    output, tile_begin, tile_end, gap_begin + (gap_source_end - previous_end),
                    sz_sequence_tape_start_metal_(replacements_tape, needle_index),
                    sz_sequence_tape_length_metal_(replacements_tape, needle_index), lane);
            }
        }
    }
}

/** Walks one haystack's share of chunks, the @p thread_index th of @p threads, into @p tally, as
 *  @c sz_substrings_bm25_walk_simt_ does. */
inline void sz_substrings_bm25_walk_metal_(thread sz_substrings_automaton_metal_t const &automaton,
                                           device uchar const *haystack, ulong length, ulong haystack_index,
                                           ulong thread_index, ulong threads,
                                           thread sz_substrings_tally_metal_t const *tally) {
    // Never narrower than the longest match, past which a chunk re-walks more warm-up than it owns.
    ulong const warm_up = max((ulong)automaton.max_source_match_bytes, 1ul);
    ulong const chunk_bytes = max(sz_size_divide_round_up_metal_<ulong>(length, threads), warm_up);
    ulong const chunk_begin = thread_index * chunk_bytes;
    if (chunk_begin >= length) return;
    ulong const chunk_end = min(chunk_begin + chunk_bytes, length);
    if (automaton.uncased)
        sz_substrings_walk_chunk_uncased_metal_(automaton, haystack, length, chunk_begin, chunk_end, haystack_index,
                                                sz_substrings_gpu_tallying_metal_k, nullptr, tally);
    else
        sz_substrings_walk_chunk_metal_(automaton, haystack, chunk_begin, chunk_end, haystack_index,
                                        sz_substrings_gpu_tallying_metal_k, nullptr, tally);
}

/** Sums the terms of the threadgroup's own @p tally in fixed point into @p score, zeroing every
 *  slot and overflow entry as it reads them, so the next haystack starts clean, as
 *  @c sz_substrings_bm25_score_simt_ does. */
inline void sz_substrings_bm25_score_metal_(thread sz_substrings_tally_metal_t const *tally, ulong table_slots,
                                            ulong needles_count, float saturation, float norm,
                                            device float const *needle_weights, threadgroup long *partials, uint lane,
                                            uint width, device float *score) {
    long mine = 0;
    for (ulong slot = lane; slot < table_slots; slot += width) {
        uint const frequency = atomic_load_explicit(tally->counts + slot, memory_order_relaxed);
        if (!frequency) continue;
        ulong const needle = tally->hashed ? atomic_load_explicit(tally->keys + slot, memory_order_relaxed) - 1 : slot;
        mine += sz_substrings_bm25_fixed_metal_(needle_weights[needle], frequency, saturation, norm);
        atomic_store_explicit(tally->counts + slot, 0u, memory_order_relaxed);
        atomic_store_explicit(tally->keys + slot, 0u, memory_order_relaxed);
    }
    if (atomic_load_explicit(tally->overflowed, memory_order_relaxed))
        for (ulong needle = lane; needle < needles_count; needle += width) {
            uint const frequency = atomic_load_explicit(tally->overflow + needle, memory_order_relaxed);
            if (!frequency) continue;
            mine += sz_substrings_bm25_fixed_metal_(needle_weights[needle], frequency, saturation, norm);
            atomic_store_explicit(tally->overflow + needle, 0u, memory_order_relaxed);
        }

    // A tree over the threadgroup's partials; integer addition commutes, so the order never shows.
    partials[lane] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint half_width = width / 2; half_width != 0; half_width /= 2) {
        if (lane < half_width) partials[lane] += partials[lane + half_width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        *score = (float)partials[0] / (float)(1ul << sz_substrings_gpu_bm25_fraction_bits_metal_k);
        atomic_store_explicit(tally->overflowed, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

/**
 *  @brief Scores one haystack per threadgroup: its threads walk contiguous chunks into one
 *      threadgroup tally, then sum the tallied terms in 32.32 fixed point.
 *
 *  A threadgroup rather than a grid per haystack keeps the tally in threadgroup memory.
 *  A vocabulary wider than the slots hashes into them and spills into the threadgroup's
 *  own overflow row.
 */
kernel void sz_substrings_bm25_metal_kernel_(
    device uchar *engine [[buffer(0)]], device uchar *scratch [[buffer(1)]],
    device ulong const *haystacks [[buffer(2)]], device float const *lengths [[buffer(11)]],
    device float const *weights [[buffer(12)]], device float *scores [[buffer(13)]],
    constant sz_substrings_arguments_metal_t &arguments [[buffer(14)]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]], uint lane [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[sz_substrings_tally_slots_metal_k], keys[sz_substrings_tally_slots_metal_k];
    threadgroup atomic_uint overflowed;
    threadgroup long partials[sz_substrings_threads_metal_k];
    ulong const needles_count = arguments.needles_count;
    ulong const table_slots = min(needles_count, (ulong)sz_substrings_tally_slots_metal_k);
    sz_substrings_automaton_metal_t const automaton = sz_substrings_automaton_metal_(engine, arguments);
    sz_sequence_tape_metal_t const tape = {haystacks};
    sz_substrings_tally_metal_t tally;
    tally.keys = keys, tally.counts = counts, tally.overflowed = &overflowed;
    tally.overflow = sz_reach_metal_<atomic_uint>(scratch, arguments.scratch_host, arguments.overflow_rows) +
                     group * needles_count;
    tally.hashed = arguments.tally_hashed != 0;
    float const saturation = arguments.saturation, normalization = arguments.normalization;

    for (ulong slot = lane; slot < table_slots; slot += width) {
        atomic_store_explicit(&counts[slot], 0u, memory_order_relaxed);
        atomic_store_explicit(&keys[slot], 0u, memory_order_relaxed);
    }
    if (lane == 0) atomic_store_explicit(&overflowed, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ulong haystack_index = group; haystack_index < arguments.haystacks_count; haystack_index += groups) {
        ulong const length = sz_sequence_tape_length_metal_(tape, haystack_index);
        float const document_length = arguments.document_lengths ? lengths[haystack_index] : (float)length;
        float const norm = normalization > 0
                               ? 1 - normalization + normalization * document_length / arguments.average_length
                               : 1.0f;
        sz_substrings_bm25_walk_metal_(automaton, sz_sequence_tape_start_metal_(tape, haystack_index), length,
                                       haystack_index, lane, width, &tally);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        sz_substrings_bm25_score_metal_(&tally, table_slots, needles_count, saturation, norm, weights, partials, lane,
                                        width, scores + haystack_index * arguments.scores_stride);
    }
}
