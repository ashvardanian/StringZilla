/**
 *  @file include/stringzilla/substrings/simt.cuh
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief The kernels CUDA and ROCm share for multi-pattern search: one thread per haystack chunk,
 *      the automaton's hot head staged in shared memory, and the leftmost cover resolved after the
 *      walk rather than inside it.
 *
 *  The transition is the serial tier's, reached from the device through `--expt-relaxed-constexpr`,
 *  so both sides answer the same matches rather than similar ones.
 *
 *  Parallelism comes from @b chunks rather than from haystacks: a corpus of one long document and a
 *  corpus of a million short ones then fill the device the same way. A chunk reports every match
 *  @b ending inside it, priming itself from the `max_source_match_bytes - 1` bytes before its own
 *  start - clamped to its haystack, never earlier - so every match is found exactly once and no
 *  chunk reads a neighbour's text.
 *
 *  A cover is a property of the matches, not of the bytes, so it is resolved after the walk. Inside
 *  the walk it would cost every thread a ring wide enough for the longest match, and a second walk
 *  to find a safe place to start; both are gone.
 *
 *  When a round outruns its match budget the writing walk, the cover, the compaction and the
 *  boundaries kernel each retire at their first instruction, so an output is left untouched rather
 *  than truncated. Each vendor's host side, in `substrings/cuda.cuh` and `substrings/rocm.cuh`,
 *  sizes and launches them.
 *
 *  @sa include/stringzilla/substrings.h
 */
#ifndef STRINGZILLA_SUBSTRINGS_SIMT_CUH_
#define STRINGZILLA_SUBSTRINGS_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/substrings/serial.h"
#include "stringzilla/utf8_uncased_fold/simt.cuh" // `sz_unicode_fold_image_simt_`

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Shapes

/** Threads every kernel here launches with; occupancy is shared-memory-bound rather than
 *  thread-bound, so a modest fixed block keeps the launch geometry simple and the block scan one
 *  power of two. */
enum { sz_substrings_threads_per_block_simt_k = 256 };

/** Candidates one thread scans quadratically before a cover segment falls back to emitted order. */
enum { sz_substrings_cover_segment_limit_simt_k = 4096 };

enum {

    /** Bits of a tally slot index: 4096 key-and-count slots fill 32 KB of static shared memory. */
    sz_substrings_tally_slot_bits_simt_k = 12,

    /** Slots one block's tally holds. */
    sz_substrings_tally_slots_simt_k = 1 << sz_substrings_tally_slot_bits_simt_k,

    /** Slots a hashed tally probes before spilling a needle to the block's overflow row. */
    sz_substrings_tally_probes_simt_k = 16,
};

/** How a tally maps a needle to a slot. */
typedef enum sz_substrings_tally_layout_simt_t {

    /** The vocabulary fits the slots, so a needle's index is its slot. */
    sz_substrings_tally_direct_simt_k = 0,

    /** A larger vocabulary, hashed into the slots with linear probing and an overflow
     *  row behind them. */
    sz_substrings_tally_hashed_simt_k = 1,
} sz_substrings_tally_layout_simt_t;

/** One block's per-needle counts for the haystack it is scoring. */
typedef struct sz_substrings_tally_simt_t {

    /** How @c counts is indexed. */
    sz_substrings_tally_layout_simt_t layout;

    /** In shared memory: each hashed slot's needle index plus one, zero while free;
     *  @c STRINGZILLA_NULL when direct. */
    sz_u32_t *keys;

    /** In shared memory: each slot's occurrences, one slot per needle when direct. */
    sz_u32_t *counts;

    /** The block's own @b [needles] global row for needles no probe could seat,
     *  @c STRINGZILLA_NULL when direct. */
    sz_u32_t *overflow;

    /** In shared memory: nonzero once anything reached @c overflow, so scoring scans
     *  it only then. */
    sz_u32_t *overflowed;
} sz_substrings_tally_simt_t;

#pragma endregion Shapes

#pragma region Device Helpers

/** Four haystack bytes as one read-only load; the address is peeled to its own alignment by the
 *  caller. */
STRINGZILLA_DEVICE sz_u32_t sz_substrings_load_quad_simt_(sz_u8_t const *pointer) {
    return __ldg((unsigned int const *)pointer);
}

/**
 *  @brief What a block keeps in shared memory beside the engine, which a walk reads in place of the
 *      engine's own arrays.
 *
 *  The engine stays the kernel's read-only parameter: rebinding its pointers to staged copies would
 *  make it a local variable, spilled to the stack and reloaded on every byte.
 */
typedef struct sz_substrings_staged_simt_t {

    /** The engine's class map, in shared memory. */
    sz_u8_t const *classes;

    /** The head of the hot tier, in shared memory. */
    sz_u32_t const *rows;

    /** States the staged head covers. */
    sz_u32_t rows_count;

    /** The acceptance bitmap, staged by the scoring kernel when it fits, else the engine's own. */
    sz_u32_t const *accepts_words;
} sz_substrings_staged_simt_t;

/**
 *  @brief One byte's transition, staged-shared-memory-first.
 *
 *  The probe is @ref sz_substrings_step's, spelled again so every automaton load takes the
 *  read-only global path: through the engine's plain pointers the device can only issue generic
 *  loads, which it cannot move past the walk's own match stores. A single cold lane still makes the
 *  whole warp pay that lane's failure-chase depth, which is the cost the staging exists to shrink.
 */
STRINGZILLA_DEVICE sz_u32_t sz_substrings_step_simt_(sz_substrings_engine_t const *engine,
                                                     sz_substrings_staged_simt_t const *staged, sz_u32_t state,
                                                     sz_u8_t byte) {
    sz_u32_t const byte_class = staged->classes[byte];
    // Asked on every chase step, so a failure link into the staged head stays in shared memory.
    for (;;) {
        if (state < staged->rows_count) return staged->rows[(sz_size_t)state * engine->classes_count + byte_class];
        if (state < engine->hot_count)
            return __ldg(engine->hot_rows + (sz_size_t)state * engine->classes_count + byte_class);
        sz_size_t const candidate = (sz_size_t)__ldg(engine->base + state) + byte;
        if (__ldg(engine->check + candidate) == state) return (sz_u32_t)candidate;
        if (state == engine->root) return engine->root;
        state = __ldg(engine->fail + state);
    }
}

/** Whether any needle ends on @p state, from the staged acceptance bitmap. */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_accepts_simt_(sz_substrings_staged_simt_t const *staged, sz_u32_t state) {
    return (sz_bool_t)((staged->accepts_words[state >> 5] >> (state & 31u)) & 1u);
}

/**
 *  @brief Cooperatively stages the class map and the head of the hot tier, once per block.
 *
 *  The hot tier's out-degree ordering makes its head the best prefix to stage, and every step
 *  reads the map.
 */
STRINGZILLA_DEVICE void sz_substrings_stage_simt_(sz_substrings_engine_t const *engine, sz_u8_t *staged_classes,
                                                  sz_u32_t *staged_rows, sz_u32_t staged_count) {
    sz_size_t const cells = (sz_size_t)staged_count * engine->classes_count;
    sz_size_t cell;
    for (cell = threadIdx.x; cell < STRINGZILLA_U8_MAX + 1; cell += blockDim.x)
        staged_classes[cell] = engine->byte_to_class[cell];
    for (cell = threadIdx.x; cell < cells; cell += blockDim.x) staged_rows[cell] = engine->hot_rows[cell];
    __syncthreads();
}

/** How many chunks of @p chunk_bytes a haystack of @p length bytes needs - at least one, so even an
 *  empty haystack still gets a thread and still lands its own boundary. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_chunks_for_simt_(sz_size_t length, sz_size_t chunk_bytes) {
    return length == 0 ? 1 : sz_size_divide_round_up(length, chunk_bytes);
}

/** Which haystack owns global chunk @p chunk_index, from the exclusive prefix sum
 *  of per-haystack counts. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_haystack_of_simt_(sz_size_t const *chunk_offsets, sz_size_t haystacks_count,
                                                             sz_size_t chunk_index) {
    sz_size_t low = 0, high = haystacks_count;
    while (low + 1 < high) {
        sz_size_t const middle = low + (high - low) / 2;
        if (chunk_offsets[middle] <= chunk_index) low = middle;
        else high = middle;
    }
    return low;
}

/** Index of the last entry at or below @p value in an ascending array; zero when none is. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_last_not_above_simt_(sz_size_t const *ascending, sz_size_t count,
                                                                sz_size_t value) {
    sz_size_t low = 0, high = count;
    while (low + 1 < high) {
        sz_size_t const middle = low + (high - low) / 2;
        if (ascending[middle] <= value) low = middle;
        else high = middle;
    }
    return low;
}

/** Counts one occurrence of @p needle, seating it in a free slot on first sight. */
STRINGZILLA_DEVICE void sz_substrings_tally_simt_(sz_substrings_tally_simt_t const *tally, sz_u32_t needle) {
    sz_u32_t const key = needle + 1;
    // Volatile, so a probe re-reads a slot a rival may have seated since.
    sz_u32_t const volatile *const seated_keys = tally->keys;
    sz_u32_t slot, probe;
    if (tally->layout == sz_substrings_tally_direct_simt_k) {
        atomicAdd(tally->counts + needle, 1u);
        return;
    }
    slot = (sz_u32_t)(((sz_u64_t)key * 0x9E3779B97F4A7C15ull) >> (64 - sz_substrings_tally_slot_bits_simt_k));
    for (probe = 0; probe != sz_substrings_tally_probes_simt_k; ++probe) {
        sz_u32_t seated = seated_keys[slot];
        if (seated == 0) seated = atomicCAS(tally->keys + slot, 0u, key);
        if (seated == 0 || seated == key) {
            atomicAdd(tally->counts + slot, 1u);
            return;
        }
        slot = (slot + 1) & (sz_substrings_tally_slots_simt_k - 1);
    }
    atomicAdd(tally->overflow + needle, 1u);
    *tally->overflowed = 1;
}

#pragma endregion Device Helpers

#pragma region Scan Kernels

/**
 *  @brief Scans one ticket's own tile in place, seeded by the totals the tiles before it chained
 *      through @p chain, so a whole scan is one launch.
 *  @param[in] live How many leading values the device wrote, scanned with one more slot for their
 *      total, or @c STRINGZILLA_NULL to scan all @p count.
 *  @return Whether this block scanned the last tile, whose values are final once it returns.
 *
 *  The chain slot starts zeroed and the last tile zeroes it again, so every scan of a round reuses
 *  the one slot the round's clear zeroed. Only blocks with values to scan take a ticket, so the
 *  last ticket is the slot's last touch however few of the launched blocks are live.
 */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_scan_tile_simt_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                                           sz_size_t elements_per_tile, sz_size_t *chain) {
    __shared__ sz_size_t shared[sz_substrings_threads_per_block_simt_k];
    // A live count past the buffer means its producer outran it and wrote none of it.
    sz_size_t const counted = !live ? count : *live < count ? *live + 1 : 0;
    sz_size_t const tiles = sz_size_divide_round_up(counted, elements_per_tile);
    sz_size_t tile, begin, end, running = 0, first;
    if (blockIdx.x >= tiles) return sz_false_k;
    tile = sz_chain_ticket_simt_(chain);
    begin = tile * elements_per_tile;
    end = sz_min_of_two(begin + elements_per_tile, counted);
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t total = 0;
        sz_block_scan_simt_(index < end ? values[index] : 0, shared, &total);
        running += total;
        __syncthreads();
    }
    running = sz_chain_offset_simt_(chain, tile, tiles, running);
    if (tile + 1 == tiles && threadIdx.x == 0) *chain = 0;
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < end ? values[index] : 0;
        sz_size_t total = 0;
        sz_size_t const exclusive = sz_block_scan_simt_(value, shared, &total);
        if (index < end) values[index] = running + exclusive;
        running += total;
        __syncthreads();
    }
    return tile + 1 == tiles ? sz_true_k : sz_false_k;
}

static __global__ void sz_substrings_scan_simt_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                                       sz_size_t elements_per_tile, sz_size_t *chain) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain);
}

/**
 *  @brief Scans the sizing walk's chunk slots, then publishes from the last tile what the walk
 *      found, which is the one place a round learns whether it fit.
 *  @param[in] emitting Whether the round reads the matches themselves, since a count that never
 *      does cannot overrun a match budget however many matches the corpus holds.
 */
static __global__ void sz_substrings_scan_sized_simt_kernel_(sz_size_t *chunk_slots, sz_size_t count,
                                                             sz_size_t elements_per_tile, sz_size_t *chain,
                                                             sz_size_t matches_budget, sz_bool_t emitting,
                                                             sz_substrings_report_t *report) {
    sz_size_t emitted;
    if (!sz_substrings_scan_tile_simt_(chunk_slots, count, STRINGZILLA_NULL, elements_per_tile, chain) ||
        threadIdx.x != 0)
        return;
    // The last slot carries the total through every trailing zero.
    emitted = chunk_slots[count - 1];
    report->matches_emitted = emitted;
    report->matches_stored = emitted;
    report->target_length = 0;
    report->shortfall = emitting && emitted > matches_budget ? emitted - matches_budget : 0;
}

#pragma endregion Scan Kernels

#pragma region Walk Kernels

/**
 *  @brief This round's chunk width: the tape's bytes over the engine's chunk budget, and at least
 *      four times the longest match, so a warm-up never outgrows a quarter of a chunk.
 *
 *  There is no ceiling on the width, and that is what makes the budget a bound rather than a hope:
 *  a chunk holds at least @c total/budget bytes, so the corpus contributes at most @c budget
 *  chunks, and each haystack's own remainder contributes at most one more. A ceiling would let a
 *  large corpus outrun any fixed budget, which is the readback this inversion exists to remove.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_chunk_bytes_simt_(sz_substrings_engine_t const *engine,
                                                             sz_sequence_t haystacks) {
    sz_u64_t const *offsets = (sz_u64_t const *)haystacks.handle;
    sz_size_t const budget = engine->chunk_budget ? engine->chunk_budget : 1;
    sz_size_t const share = sz_size_divide_round_up((sz_size_t)(offsets[haystacks.count] - offsets[0]), budget);
    return sz_max_of_two(sz_max_of_two(share, 4 * (sz_size_t)engine->max_source_match_bytes), (sz_size_t)1);
}

/** Writes how many chunks each haystack is cut into, which the scan then turns into
 *  its chunk range. */
static __global__ void sz_substrings_chunk_counts_simt_kernel_(sz_substrings_engine_t engine, sz_sequence_t haystacks,
                                                               sz_size_t *chunk_offsets) {
    sz_size_t const chunk_bytes = sz_substrings_chunk_bytes_simt_(&engine, haystacks);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < haystacks.count; index += stride)
        chunk_offsets[index] = sz_substrings_chunks_for_simt_(sz_sequence_tape_length_simt_(haystacks.handle, index),
                                                              chunk_bytes);
}

/**
 *  @brief Reports every match ending at @p delta, writing them when @p pass asks, and
 *      returns how many.
 *
 *  The acceptance bit answers "does anything end here" without touching the counts array, which at
 *  scale costs nearly as much as the haystack read itself.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_emit_simt_(sz_substrings_engine_t const *engine,
                                                      sz_substrings_staged_simt_t const *staged, sz_u32_t state,
                                                      sz_size_t walk_begin, sz_u32_t delta, sz_size_t haystack_index,
                                                      sz_substrings_gpu_pass_t pass, sz_substrings_match_t *matches_out,
                                                      sz_substrings_tally_simt_t const *tally) {
    sz_size_t output_offset, found = 0, index;
    sz_u32_t output_count;
    if (!sz_substrings_accepts_simt_(staged, state)) return 0;
    output_count = __ldg(engine->outputs_counts + state);
    output_offset = (sz_size_t)__ldg((unsigned long long const *)engine->outputs_offsets + state);
    for (index = 0; index != output_count; ++index) {
        sz_substrings_output_t const *const stored = engine->outputs + output_offset + index;
        sz_substrings_output_t output;
        output.needle_index = __ldg(&stored->needle_index);
        output.folded_match_bytes = __ldg(&stored->folded_match_bytes);
        // `walk_begin` is clamped to the haystack's own start, so underflowing the walk and underflowing
        // the haystack are the same test.
        if (delta + 1 < output.folded_match_bytes) continue;
        if (pass == sz_substrings_gpu_writing_k) {
            sz_substrings_match_t match;
            match.haystack_index = haystack_index;
            match.needle_index = output.needle_index;
            match.byte_offset = walk_begin + delta + 1 - output.folded_match_bytes;
            match.byte_length = output.folded_match_bytes;
            matches_out[found] = match;
        }
        else if (pass == sz_substrings_gpu_tallying_k) sz_substrings_tally_simt_(tally, output.needle_index);
        ++found;
    }
    return found;
}

/**
 *  @brief Walks one chunk of a byte-exact haystack, counting or writing every match
 *      ending inside it.
 *  @param[in] chunk_begin First byte of the chunk, relative to the haystack's own start.
 *  @return The number of matches the chunk holds.
 *
 *  The warm-up primes the state from before the chunk and reports nothing, so once it ends the emit
 *  test is gone from the loop rather than being re-asked on every byte.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_walk_chunk_cased_simt_(
    sz_substrings_engine_t const *engine, sz_substrings_staged_simt_t const *staged, sz_cptr_t haystack,
    sz_size_t length, sz_size_t chunk_begin, sz_size_t chunk_end, sz_size_t haystack_index,
    sz_substrings_gpu_pass_t pass, sz_substrings_match_t *matches_at_chunk, sz_substrings_tally_simt_t const *tally) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
    sz_size_t const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const walk_base = (sz_u8_t const *)haystack + walk_begin;
    // Every 64-bit quantity is resolved here, once; the per-byte loops below ride 32-bit deltas from it.
    sz_u32_t const walk_span = (sz_u32_t)(chunk_end - walk_begin);
    sz_u32_t const emit_from = (sz_u32_t)(chunk_begin - walk_begin);
    sz_size_t found = 0;
    sz_u32_t delta = 0, lane;
    sz_u32_t state = engine->root; // ? Fresh at `walk_begin`; no state crosses a haystack boundary.
    sz_unused_(length);

    for (; delta < emit_from; ++delta) state = sz_substrings_step_simt_(engine, staged, state, walk_base[delta]);

    // Peeled to the load's own alignment, so the body pays one four-byte load per four transitions; the
    // transition chain stays strictly serial, only the haystack reads widen.
    for (; delta < walk_span && (((sz_size_t)(walk_base + delta)) & 3u) != 0; ++delta) {
        state = sz_substrings_step_simt_(engine, staged, state, walk_base[delta]);
        found += sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta, haystack_index, pass,
                                          matches_at_chunk + found, tally);
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        sz_u32_t const quad = sz_substrings_load_quad_simt_(walk_base + delta);
#pragma unroll
        for (lane = 0; lane != 4; ++lane) {
            state = sz_substrings_step_simt_(engine, staged, state, (sz_u8_t)(quad >> (lane * 8)));
            found += sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta + lane, haystack_index, pass,
                                              matches_at_chunk + found, tally);
        }
    }
    for (; delta < walk_span; ++delta) {
        state = sz_substrings_step_simt_(engine, staged, state, walk_base[delta]);
        found += sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta, haystack_index, pass,
                                          matches_at_chunk + found, tally);
    }
    return found;
}

/**
 *  @brief The device twin of @c sz_substrings_folded_span, stepping back one folded codepoint at a
 *      time through the fold tables, as @c sz_substrings_resolve_match_metal_ does.
 *
 *  A byte that ends no well-formed codepoint passes through as itself and steps back by one, which
 *  keeps the backward byte stream the exact reverse of the forward walk's.
 */
STRINGZILLA_DEVICE sz_substrings_resolved_match_t
sz_substrings_folded_span_simt_(sz_cptr_t haystack, sz_substrings_folded_byte_t const *step, sz_size_t folded,
                                sz_size_t last_break_folded_end, sz_size_t folded_match_bytes) {
    sz_u8_t const *const haystack_bytes = (sz_u8_t const *)haystack;
    sz_size_t const trailing = step->trailing, shift = step->shift;
    sz_size_t const wanted = folded_match_bytes + shift;
    sz_size_t position = step->codepoint_end, stepped;
    sz_size_t start_here = STRINGZILLA_SIZE_MAX, start_earlier = STRINGZILLA_SIZE_MAX;
    sz_utf8_folded_image_t image;
    sz_u64_t ring = 0;
    sz_u8_t pending = 0;
    sz_bool_t periodic = (sz_bool_t)(shift != 0);
    sz_substrings_resolved_match_t resolved;
    resolved.source_offset = step->codepoint_end - folded_match_bytes;
    resolved.repeats = sz_false_k;
    if (folded - folded_match_bytes >= last_break_folded_end) return resolved;

    for (stepped = 0; stepped < trailing + wanted; ++stepped) {
        sz_u8_t byte;
        sz_size_t taken;
        if (!pending) {
            sz_size_t candidate, source_bytes;
            if (!position) break;
            candidate = position - 1;
            for (int back = 0; back != 3 && candidate && (haystack_bytes[candidate] & 0xC0) == 0x80; ++back)
                --candidate;
            source_bytes = sz_utf8_fold_next_simt_(haystack_bytes + candidate, haystack_bytes + position, &image);
            if (!image.rune_ends || candidate + source_bytes != position)
                image.bytes = haystack_bytes[--position], image.length = 1;
            else position = candidate;
            pending = image.length;
        }
        byte = (sz_u8_t)(image.bytes >> (8 * --pending));
        if (stepped < trailing) continue; // ? Folded bytes of the ending codepoint past the match's own end

        taken = stepped - trailing + 1;
        // A shift stays within one codepoint's image of at most six bytes, so eight ring bytes do.
        if (shift) {
            if (taken > shift && (sz_u8_t)(ring >> (8 * ((taken - shift) & 7))) != byte) periodic = sz_false_k;
            ring = (ring & ~(0xFFull << (8 * (taken & 7)))) | ((sz_u64_t)byte << (8 * (taken & 7)));
        }
        if (taken == folded_match_bytes) start_here = position;
        if (taken == wanted) start_earlier = position;
        if (!periodic && taken >= folded_match_bytes) break;
    }

    resolved.source_offset = start_here != STRINGZILLA_SIZE_MAX ? start_here : step->codepoint_end;
    resolved.repeats = (sz_bool_t)(periodic && start_here != STRINGZILLA_SIZE_MAX && start_here == start_earlier);
    return resolved;
}

/**
 *  @brief Walks one chunk as folded bytes, the case-insensitive twin of the walk above.
 *
 *  Folding makes a walk restart-safe only at a codepoint start, so the warm-up snaps back to one
 *  before it begins - three bytes at most, and always earlier, so the extra transitions only prime
 *  state further. Match ends are reported at the source codepoint's end, which keeps chunk
 *  ownership comparable against the unsnapped chunk bounds the planner handed out.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_walk_chunk_uncased_simt_(
    sz_substrings_engine_t const *engine, sz_substrings_staged_simt_t const *staged, sz_cptr_t haystack,
    sz_size_t length, sz_size_t chunk_begin, sz_size_t chunk_end, sz_size_t haystack_index,
    sz_substrings_gpu_pass_t pass, sz_substrings_match_t *matches_at_chunk, sz_substrings_tally_simt_t const *tally) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
    sz_size_t walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const haystack_bytes = (sz_u8_t const *)haystack;
    sz_substrings_folded_byte_t step;
    sz_utf8_folded_image_t image;
    sz_u32_t state = engine->root;
    sz_size_t folded = 0, last_break_folded_end = 0, found = 0, position;
    sz_u8_t image_index = 0, previous_rune_end = 0;
    sz_bool_t breaks_boundary = sz_false_k;
    // The aligned word under the cursor, so an ASCII run costs one load per four bytes.
    sz_u8_t const *loaded_word = STRINGZILLA_NULL;
    sz_u32_t loaded_quad = 0;

    // A fold is only restartable on a lead byte, so the walk snaps outward to one - never inward, which
    // would drop a match whose needle began mid-codepoint.
    while (walk_begin != 0 && haystack_bytes[walk_begin] >= 0x80 && (haystack_bytes[walk_begin] & 0xC0) == 0x80)
        --walk_begin;

    // Each codepoint folds into an image held in registers, which the automaton steps through a
    // byte at a time; only a byte ending a folded rune can end a match.
    image.length = 0;
    for (position = walk_begin;;) {
        sz_size_t output_offset, output_count, index;
        if (image_index == image.length) {
            sz_size_t source_bytes;
            if (position >= length) break;
            // An aligned word holding a haystack byte lies in a page the device may read whole.
            sz_u8_t const *const word = (sz_u8_t const *)((sz_size_t)(haystack_bytes + position) & ~(sz_size_t)3);
            if (word != loaded_word) loaded_word = word, loaded_quad = sz_substrings_load_quad_simt_(word);
            sz_u8_t const lead = (sz_u8_t)(loaded_quad >> (((sz_size_t)(haystack_bytes + position) & 3u) * 8));
            if (lead < 0x80)
                image.bytes = sz_ascii_fold_(lead), image.length = 1, image.rune_ends = 1, source_bytes = 1;
            else source_bytes = sz_utf8_fold_next_simt_(haystack_bytes + position, haystack_bytes + length, &image);
            breaks_boundary = (sz_bool_t)(__popc(image.rune_ends) != 1 || image.length != source_bytes);
            position += source_bytes;
            step.codepoint_end = position - walk_begin;
            image_index = 0, previous_rune_end = 0;
        }
        ++folded;
        // A byte beginning no codepoint is its own folded byte, and no match runs through it.
        if (!image.rune_ends) {
            ++image_index, state = engine->root;
            continue;
        }
        state = sz_substrings_step_simt_(engine, staged, state, (sz_u8_t)(image.bytes >> (8 * image_index)));
        if (!((image.rune_ends >> image_index++) & 1u)) continue;
        // A codepoint's first rune end has nothing before it to repeat, which a zero shift names.
        step.trailing = (sz_u8_t)(image.length - image_index);
        step.shift = previous_rune_end ? (sz_u8_t)(image_index - previous_rune_end) : 0;
        previous_rune_end = image_index;
        if (breaks_boundary) last_break_folded_end = folded + step.trailing;

        if (position > chunk_end) return found; // ? Past this chunk's share; the next chunk owns these ends.
        if (!sz_substrings_accepts_simt_(staged, state)) continue;
        if (position <= chunk_begin) continue; // ? Still warming up, where this chunk reports nothing.

        output_count = engine->outputs_counts[state];
        output_offset = engine->outputs_offsets[state];
        for (index = 0; index != output_count; ++index) {
            sz_substrings_output_t const output = engine->outputs[output_offset + index];
            sz_size_t const folded_length = output.folded_match_bytes;
            sz_substrings_resolved_match_t resolved;
            if (folded < folded_length) continue;
            resolved = sz_substrings_folded_span_simt_(haystack + walk_begin, &step, folded, last_break_folded_end,
                                                       folded_length);
            if (resolved.repeats) continue;
            if (pass == sz_substrings_gpu_writing_k) {
                matches_at_chunk[found].haystack_index = haystack_index;
                matches_at_chunk[found].needle_index = output.needle_index;
                matches_at_chunk[found].byte_offset = walk_begin + resolved.source_offset;
                matches_at_chunk[found].byte_length = step.codepoint_end - resolved.source_offset;
            }
            else if (pass == sz_substrings_gpu_tallying_k) sz_substrings_tally_simt_(tally, output.needle_index);
            ++found;
        }
    }
    return found;
}

/** Chunks the engine's budget cuts a round into per resident thread. A walk's cost follows its
 *  chunk's matches and failure chases, so more, smaller chunks spread a heavy region over more
 *  threads and leave the slowest thread less to finish alone. */
enum { sz_substrings_chunks_per_thread_simt_k = 8 };

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in whichever pass
 *      @p pass names.
 *
 *  Both passes share @p chunk_slots, because the host's in-place exclusive scan already makes them
 *  one allocation: sizing writes each chunk's match count into its slot, and writing reads the
 *  exclusive offset the scan left there. Every chunk owns a private, non-overlapping output range,
 *  so a write needs no atomic.
 */
STRINGZILLA_DEVICE void sz_substrings_walk_chunks_simt_(sz_substrings_engine_t const *engine,
                                                        sz_substrings_case_sensitivity_t case_sensitivity,
                                                        sz_u32_t staged_count, sz_sequence_t haystacks,
                                                        sz_size_t const *chunk_offsets,
                                                        sz_substrings_report_t const *report, sz_size_t *chunk_slots,
                                                        sz_substrings_match_t *matches, sz_substrings_gpu_pass_t pass) {
    extern __shared__ sz_u32_t sz_substrings_staged_simt_[];
    __shared__ sz_u8_t staged_classes[STRINGZILLA_U8_MAX + 1];
    sz_size_t const chunk_count = chunk_offsets[haystacks.count];
    sz_size_t const chunk_bytes = sz_substrings_chunk_bytes_simt_(engine, haystacks);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk_index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_substrings_staged_simt_t staged;
    // The writing pass has nowhere to write once the sizing pass outran the budget, so it retires whole.
    if (pass == sz_substrings_gpu_writing_k && report->shortfall) return;
    sz_substrings_stage_simt_(engine, staged_classes, sz_substrings_staged_simt_, staged_count);
    staged.classes = staged_classes, staged.rows = sz_substrings_staged_simt_, staged.rows_count = staged_count;
    staged.accepts_words = engine->accepts_words;

    for (; chunk_index < chunk_count; chunk_index += stride) {
        sz_size_t const haystack_index = sz_substrings_haystack_of_simt_(chunk_offsets, haystacks.count, chunk_index);
        sz_cptr_t const haystack = sz_sequence_tape_start_simt_(haystacks.handle, haystack_index);
        sz_size_t const length = sz_sequence_tape_length_simt_(haystacks.handle, haystack_index);
        sz_size_t const local_index = chunk_index - chunk_offsets[haystack_index];
        sz_size_t const chunk_begin = local_index * chunk_bytes;
        sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
        sz_substrings_match_t *const matches_at_chunk = pass == sz_substrings_gpu_writing_k
                                                            ? matches + chunk_slots[chunk_index]
                                                            : matches;
        sz_size_t found;
        if (chunk_begin >= chunk_end && length != 0) found = 0;
        else if (case_sensitivity == sz_substrings_uncased_k)
            found = sz_substrings_walk_chunk_uncased_simt_(engine, &staged, haystack, length, chunk_begin, chunk_end,
                                                           haystack_index, pass, matches_at_chunk, STRINGZILLA_NULL);
        else
            found = sz_substrings_walk_chunk_cased_simt_(engine, &staged, haystack, length, chunk_begin, chunk_end,
                                                         haystack_index, pass, matches_at_chunk, STRINGZILLA_NULL);
        if (pass == sz_substrings_gpu_sizing_k) chunk_slots[chunk_index] = found;
    }
}

/*  A byte-exact and a folded walk are compiled apart, so the folding cursor's registers never cut
 *  the residency of a walk that does not fold. */

static __global__ void sz_substrings_walk_cased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                                             sz_sequence_t haystacks, sz_size_t const *chunk_offsets,
                                                             sz_substrings_report_t const *report,
                                                             sz_size_t *chunk_slots, sz_substrings_match_t *matches,
                                                             sz_substrings_gpu_pass_t pass) {
    sz_substrings_walk_chunks_simt_(&engine, sz_substrings_cased_k, staged_count, haystacks, chunk_offsets, report,
                                    chunk_slots, matches, pass);
}

static __global__ void sz_substrings_walk_uncased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                                               sz_sequence_t haystacks, sz_size_t const *chunk_offsets,
                                                               sz_substrings_report_t const *report,
                                                               sz_size_t *chunk_slots, sz_substrings_match_t *matches,
                                                               sz_substrings_gpu_pass_t pass) {
    sz_substrings_walk_chunks_simt_(&engine, sz_substrings_uncased_k, staged_count, haystacks, chunk_offsets, report,
                                    chunk_slots, matches, pass);
}

#pragma endregion Walk Kernels

#pragma region Scoring Kernels

/** Slots a block's tally takes: one per needle when the vocabulary fits, a
 *  hashed table otherwise. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_tally_slots_for_simt_(sz_size_t needles_count) {
    return needles_count <= sz_substrings_tally_slots_simt_k ? needles_count : sz_substrings_tally_slots_simt_k;
}

/** Words of the acceptance bitmap, one bit per double-array slot. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_accepts_words_simt_(sz_substrings_engine_t const *engine) {
    return sz_size_divide_round_up(engine->slots_count, 32);
}

/** Walks one haystack's share of chunks, the @p thread th of @p threads, into @p tally. */
STRINGZILLA_DEVICE void sz_substrings_bm25_walk_simt_(sz_substrings_engine_t const *engine,
                                                      sz_substrings_staged_simt_t const *staged, sz_cptr_t haystack,
                                                      sz_size_t length, sz_size_t haystack_index, sz_size_t thread,
                                                      sz_size_t threads, sz_substrings_tally_simt_t const *tally) {
    // Never narrower than the longest match, past which a chunk re-walks more warm-up than it owns.
    sz_size_t const warm_up = sz_max_of_two((sz_size_t)engine->max_source_match_bytes, (sz_size_t)1);
    sz_size_t const chunk_bytes = sz_max_of_two(sz_size_divide_round_up(length, threads), warm_up);
    sz_size_t const chunk_begin = thread * chunk_bytes;
    if (chunk_begin >= length) return;
    sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
    if (engine->case_sensitivity == sz_substrings_uncased_k)
        sz_substrings_walk_chunk_uncased_simt_(engine, staged, haystack, length, chunk_begin, chunk_end, haystack_index,
                                               sz_substrings_gpu_tallying_k, STRINGZILLA_NULL, tally);
    else
        sz_substrings_walk_chunk_cased_simt_(engine, staged, haystack, length, chunk_begin, chunk_end, haystack_index,
                                             sz_substrings_gpu_tallying_k, STRINGZILLA_NULL, tally);
}

/** Sums the terms of the block's own @p tally in fixed point into @p score, zeroing every slot,
 *  overflow entry and the sum itself as it reads them, so the next haystack starts clean. */
STRINGZILLA_DEVICE void sz_substrings_bm25_score_simt_(sz_substrings_tally_simt_t const *tally, sz_size_t table_slots,
                                                       sz_size_t needles_count, sz_substrings_bm25_t const *parameters,
                                                       sz_f64_t norm, sz_f32_t const *needle_weights,
                                                       unsigned long long *block_sum, sz_f32_t *score) {
    sz_i64_t mine = 0;
    sz_size_t slot, needle;
    for (slot = threadIdx.x; slot < table_slots; slot += blockDim.x) {
        sz_u32_t const frequency = tally->counts[slot];
        if (!frequency) continue;
        needle = tally->keys ? (sz_size_t)tally->keys[slot] - 1 : slot;
        mine += __double2ll_rn(sz_substrings_bm25_term(parameters, norm, needle_weights[needle], frequency) *
                               (sz_f64_t)((sz_u64_t)1 << sz_substrings_gpu_bm25_fraction_bits_k));
        tally->counts[slot] = 0;
        if (tally->keys) tally->keys[slot] = 0;
    }
    if (*tally->overflowed)
        for (needle = threadIdx.x; needle < needles_count; needle += blockDim.x) {
            sz_u32_t const frequency = tally->overflow[needle];
            if (!frequency) continue;
            mine += __double2ll_rn(sz_substrings_bm25_term(parameters, norm, needle_weights[needle], frequency) *
                                   (sz_f64_t)((sz_u64_t)1 << sz_substrings_gpu_bm25_fraction_bits_k));
            tally->overflow[needle] = 0;
        }
    // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
    if (mine) atomicAdd(block_sum, (unsigned long long)mine);
    __syncthreads();
    if (threadIdx.x == 0) {
        *score = (sz_f32_t)((sz_f64_t)(sz_i64_t)*block_sum /
                            (sz_f64_t)((sz_u64_t)1 << sz_substrings_gpu_bm25_fraction_bits_k));
        *block_sum = 0, *tally->overflowed = 0;
    }
    __syncthreads();
}

/** What a scoring block keeps in its shared memory, which @ref sz_substrings_bm25_prepare_simt_
 *  lays out. */
typedef struct sz_substrings_bm25_block_simt_t {

    /** The block's own tally, zeroed. */
    sz_substrings_tally_simt_t tally;

    /** The class map, acceptance words and hot rows the block reads in place of the engine's. */
    sz_substrings_staged_simt_t staged;

    /** The block's fixed-point sum of one haystack's terms, zeroed. */
    unsigned long long *sum;
} sz_substrings_bm25_block_simt_t;

/**
 *  @brief Lays one scoring block's tally and staged rows out in its shared memory, zeroed and
 *      staged; every thread calls it, and it ends in a barrier.
 *  @param[in] overflow_rows The launch's spill rows, one per block, or @c STRINGZILLA_NULL for a
 *      vocabulary a block's own table holds.
 *  @param[in] staged_accepts_words Acceptance words staged in shared memory, all of them or zero.
 *  @param[in] staged_count Hot rows staged after them, a prefix of the tier.
 *
 *  Dynamic shared memory holds the counts, the keys when hashed, then the staged acceptance words
 *  and hot rows, in that order.
 */
STRINGZILLA_DEVICE sz_substrings_bm25_block_simt_t
sz_substrings_bm25_prepare_simt_(sz_substrings_engine_t const *engine, sz_u32_t *overflow_rows,
                                 sz_u32_t staged_accepts_words, sz_u32_t staged_count) {
    extern __shared__ sz_u32_t sz_substrings_scoring_simt_[];
    __shared__ sz_u8_t staged_classes[STRINGZILLA_U8_MAX + 1];
    __shared__ sz_u32_t overflowed;
    __shared__ unsigned long long block_sum;
    sz_size_t const needles_count = engine->needles_count;
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(needles_count);
    sz_substrings_bm25_block_simt_t block;
    sz_u32_t *staged_accepts, *staged_rows;
    sz_size_t slot;
    block.tally.layout = needles_count <= sz_substrings_tally_slots_simt_k ? sz_substrings_tally_direct_simt_k
                                                                           : sz_substrings_tally_hashed_simt_k;
    block.tally.counts = sz_substrings_scoring_simt_;
    block.tally.keys = block.tally.layout == sz_substrings_tally_hashed_simt_k ? block.tally.counts + table_slots
                                                                               : STRINGZILLA_NULL;
    block.tally.overflowed = &overflowed;
    block.tally.overflow = overflow_rows ? overflow_rows + (sz_size_t)blockIdx.x * needles_count : STRINGZILLA_NULL;
    block.sum = &block_sum;
    staged_accepts = block.tally.counts + (block.tally.keys ? 2 : 1) * table_slots;
    staged_rows = staged_accepts + staged_accepts_words;

    for (slot = threadIdx.x; slot < table_slots; slot += blockDim.x) {
        block.tally.counts[slot] = 0;
        if (block.tally.keys) block.tally.keys[slot] = 0;
    }
    for (slot = threadIdx.x; slot < staged_accepts_words; slot += blockDim.x)
        staged_accepts[slot] = engine->accepts_words[slot];
    if (threadIdx.x == 0) overflowed = 0, block_sum = 0;
    sz_substrings_stage_simt_(engine, staged_classes, staged_rows,
                              staged_count); // ! Ends in the barrier the zeroing needs.
    block.staged.classes = staged_classes, block.staged.rows = staged_rows, block.staged.rows_count = staged_count;
    block.staged.accepts_words = staged_accepts_words ? staged_accepts : engine->accepts_words;
    return block;
}

/**
 *  @brief Scores one haystack per block: its threads walk contiguous chunks into one shared tally,
 *      then sum the tallied terms in fixed point.
 *  @param[in] scores_stride Entries from one haystack's score to the next, so one launch
 *      writes one column.
 *
 *  A block rather than a grid per haystack keeps the tally in shared memory. Where a batch holds
 *  too few haystacks to fill the device, Hopper's tier walks each with a cluster of blocks instead.
 */
static __global__ void sz_substrings_bm25_simt_kernel_(sz_substrings_engine_t engine, sz_sequence_t haystacks,
                                                       sz_f32_t const *document_lengths,
                                                       sz_substrings_bm25_t parameters, sz_f32_t const *needle_weights,
                                                       sz_u32_t *overflow_rows, sz_f32_t *scores,
                                                       sz_size_t scores_stride, sz_u32_t staged_accepts_words,
                                                       sz_u32_t staged_count) {
    sz_substrings_bm25_block_simt_t const block = sz_substrings_bm25_prepare_simt_(&engine, overflow_rows,
                                                                                   staged_accepts_words, staged_count);
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(engine.needles_count);
    sz_size_t haystack_index;
    for (haystack_index = blockIdx.x; haystack_index < haystacks.count; haystack_index += gridDim.x) {
        sz_cptr_t const haystack = sz_sequence_tape_start_simt_(haystacks.handle, haystack_index);
        sz_size_t const length = sz_sequence_tape_length_simt_(haystacks.handle, haystack_index);
        sz_f64_t const norm = sz_substrings_bm25_norm(
            &parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        sz_substrings_bm25_walk_simt_(&engine, &block.staged, haystack, length, haystack_index, threadIdx.x, blockDim.x,
                                      &block.tally);
        __syncthreads();
        sz_substrings_bm25_score_simt_(&block.tally, table_slots, engine.needles_count, &parameters, norm,
                                       needle_weights, block.sum, scores + haystack_index * scores_stride);
    }
}

#pragma endregion Scoring Kernels

#pragma region Cover Kernels

/** Whether the boundary before @p index is real: nothing still to come starts before the
 *  maximum end already reached. Only matches ending within one match's length of it can, which
 *  bounds the look-ahead. */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_boundary_before_simt_(sz_substrings_match_t const *matches, sz_size_t count,
                                                                 sz_size_t longest, sz_size_t index) {
    sz_size_t reached, ahead;
    if (index == 0) return sz_true_k;
    if (matches[index - 1].haystack_index != matches[index].haystack_index) return sz_true_k;
    reached = matches[index - 1].byte_offset + matches[index - 1].byte_length;
    for (ahead = index; ahead < count; ++ahead) {
        if (matches[ahead].haystack_index != matches[index - 1].haystack_index) break;
        if (matches[ahead].byte_offset + matches[ahead].byte_length >= reached + longest) break;
        if (matches[ahead].byte_offset < reached) return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Decides which overlapping matches survive a leftmost cover, one segment per thread.
 *
 *  Within a haystack the walk emits in non-decreasing end order, so the running maximum end is
 *  simply the previous match's end, and a boundary sits where nothing still to come reaches back
 *  across it. Nothing before such a boundary can reach past it, so each segment resolves against a
 *  cursor of zero, independently of every other.
 *
 *  Segments are short in real text - a needle set drawn from a vocabulary leaves a median of one
 *  match between boundaries - so one thread takes a whole one. That is a measurement rather than a
 *  guarantee: a needle and its own suffixes over repetitive text make one segment of the whole
 *  document, and the greedy below is quadratic in a segment, so past
 *  @ref sz_substrings_cover_segment_limit_simt_k candidates a segment falls back to accepting in
 *  emitted order - the same cover whenever starts ascend with ends, and a documented approximation
 *  when they do not. Without the cap one thread could hold the grid.
 */
static __global__ void sz_substrings_cover_simt_kernel_(sz_substrings_match_t const *matches,
                                                        sz_substrings_report_t const *report, sz_size_t longest,
                                                        sz_substrings_overlap_policy_t policy, sz_size_t *keep) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Nothing was written, so there is nothing to cover; the report already names what did not fit.
    if (report->shortfall) return;
    for (; index < count; index += stride) {
        sz_size_t segment_end, slot, cursor;
        // Only a segment's first match works; the rest are decided by whoever owns their segment.
        if (!sz_substrings_boundary_before_simt_(matches, count, longest, index)) continue;
        for (segment_end = index + 1; segment_end < count; ++segment_end)
            if (sz_substrings_boundary_before_simt_(matches, count, longest, segment_end)) break;

        if (segment_end - index > sz_substrings_cover_segment_limit_simt_k) {
            sz_size_t reached = 0;
            for (slot = index; slot < segment_end; ++slot) {
                sz_bool_t const accepted = (sz_bool_t)(matches[slot].byte_offset >= reached);
                keep[slot] = accepted;
                if (accepted) reached = matches[slot].byte_offset + matches[slot].byte_length;
            }
            continue;
        }

        // The greedy cover: take the earliest start at or past the cursor, breaking ties by policy, and
        // repeat. Quadratic in the segment, which is why the segment is one thread's worth and no more.
        for (slot = index; slot < segment_end; ++slot) keep[slot] = 0;
        cursor = 0;
        for (;;) {
            sz_size_t chosen = segment_end;
            for (slot = index; slot < segment_end; ++slot) {
                if (matches[slot].byte_offset < cursor) continue;
                if (chosen == segment_end) {
                    chosen = slot;
                    continue;
                }
                if (matches[slot].byte_offset != matches[chosen].byte_offset) {
                    if (matches[slot].byte_offset < matches[chosen].byte_offset) chosen = slot;
                    continue;
                }
                if (policy == sz_substrings_leftmost_longest_k &&
                    matches[slot].byte_length != matches[chosen].byte_length) {
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

/**
 *  @brief Gathers the surviving matches into their scanned slots, order preserved, and publishes
 *      how many survived, which is what every later boundary is read against.
 *  @param[in] keep_offsets The scanned keep flags, one longer than @p count so the last
 *      has a successor.
 *
 *  The scan overwrote the flags it summed, so survival is read back out of it: a match was kept
 *  exactly when the scan steps across it.
 */
static __global__ void sz_substrings_compact_simt_kernel_(sz_substrings_match_t const *matches,
                                                          sz_substrings_report_t *report, sz_size_t const *keep_offsets,
                                                          sz_substrings_match_t *survivors) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    if (index == 0) report->matches_stored = keep_offsets[count];
    for (; index < count; index += stride)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
static __global__ void sz_substrings_haystack_offsets_simt_kernel_(sz_size_t const *chunk_offsets,
                                                                   sz_size_t const *chunk_slots,
                                                                   sz_size_t const *keep_offsets,
                                                                   sz_substrings_report_t const *report,
                                                                   sz_size_t *haystack_offsets, sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    for (; index < count; index += stride) {
        sz_size_t const emitted_before = chunk_slots[chunk_offsets[index]];
        haystack_offsets[index] = keep_offsets ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
static __global__ void sz_substrings_counts_simt_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts,
                                                         sz_size_t counts_stride, sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < count; index += stride)
        counts[index * counts_stride] = haystack_offsets[index + 1] - haystack_offsets[index];
}

/**
 *  @brief Copies the surviving matches into the caller's array, clipped at a capacity
 *      only it knows.
 *
 *  The survivor count lives on the device, so a host-issued @c cudaMemcpyAsync cannot express the
 *  clip; one grid-strided kernel can, and @ref sz_substrings_stored_simt_kernel_ publishes what it
 *  stored once no block is left reading the report.
 */
static __global__ void sz_substrings_store_matches_simt_kernel_(sz_substrings_match_t const *reported,
                                                                sz_substrings_report_t const *report,
                                                                sz_substrings_match_t *matches,
                                                                sz_size_t matches_capacity) {
    sz_size_t const kept = report->shortfall ? 0 : report->matches_stored;
    sz_size_t const fitting = sz_min_of_two(kept, matches_capacity);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < fitting; index += stride) matches[index] = reported[index];
}

/** Publishes what the store kept, a launch of its own, since a block of the store still reading the
 *  report would otherwise see its shortfall change and copy nothing. */
static __global__ void sz_substrings_stored_simt_kernel_(sz_substrings_report_t *report, sz_size_t matches_capacity) {
    sz_size_t kept;
    if (blockIdx.x || threadIdx.x) return;
    kept = report->shortfall ? 0 : report->matches_stored;
    report->matches_stored = sz_min_of_two(kept, matches_capacity);
    if (kept > matches_capacity) report->shortfall = kept - matches_capacity;
}

#pragma endregion Cover Kernels

#pragma region Rewrite Kernels

/**
 *  @brief Writes where each match's preceding gap lands, and how long each haystack becomes.
 *
 *  One block per haystack, threads striding its match range. A rewrite is a tiling of gaps and
 *  replacements, and every boundary in that tiling follows from one running quantity: how far the
 *  output has drifted from the input by the time a match is reached. So that drift is all this
 *  stores - one scanned offset per match - and the copy kernel derives the rest from the match list
 *  it already has.
 *
 *  Offsets are relative to the haystack's own start, because the base is only known after the scan
 *  across haystacks that this kernel feeds.
 */
static __global__ void sz_substrings_rewrite_offsets_simt_kernel_(sz_sequence_t haystacks, sz_sequence_t replacements,
                                                                  sz_size_t const *haystack_offsets,
                                                                  sz_substrings_match_t const *matches,
                                                                  sz_substrings_report_t const *report,
                                                                  sz_size_t *gap_offsets, sz_size_t *output_sizes) {
    __shared__ sz_size_t shared[sz_substrings_threads_per_block_simt_k];
    __shared__ sz_size_t drift_carry;
    sz_size_t haystack_index;
    // The cover never ran, so there are no boundaries to tile the rewrite against.
    if (report->shortfall) return;
    for (haystack_index = blockIdx.x; haystack_index < haystacks.count; haystack_index += gridDim.x) {
        sz_size_t const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
        sz_size_t tile_first;
        if (threadIdx.x == 0) drift_carry = 0;
        __syncthreads();

        for (tile_first = first; tile_first < last; tile_first += blockDim.x) {
            sz_size_t const match_index = tile_first + threadIdx.x;
            sz_bool_t const owns = (sz_bool_t)(match_index < last);
            sz_size_t drift_here = 0, previous_end = 0, drift_before, drift_in_tile = 0;
            if (owns) {
                sz_size_t const needle = matches[match_index].needle_index;
                // Shrinking matches make this wrap, which is exactly right: only the prefix sums are ever
                // read, every one of them names a real offset, and modular arithmetic reproduces each.
                drift_here = sz_sequence_tape_length_simt_(replacements.handle, needle) -
                             matches[match_index].byte_length;
                previous_end = match_index == first
                                   ? 0
                                   : matches[match_index - 1].byte_offset + matches[match_index - 1].byte_length;
            }
            drift_before = sz_block_scan_simt_(drift_here, shared, &drift_in_tile);
            if (owns) gap_offsets[match_index] = previous_end + drift_carry + drift_before;
            __syncthreads();
            if (threadIdx.x == 0) drift_carry += drift_in_tile;
            __syncthreads();
        }

        // The scan's own aggregate is the haystack's total drift, so no second pass reduces what it knows.
        if (threadIdx.x == 0)
            output_sizes[haystack_index] = sz_sequence_tape_length_simt_(haystacks.handle, haystack_index) +
                                           drift_carry;
        __syncthreads(); // ! The next haystack resets the carry this one is still reading.
    }
}

/** Copies one stretch, clipped to the tile, with @p lane striding the surviving bytes. */
STRINGZILLA_DEVICE void sz_substrings_copy_clipped_simt_(sz_ptr_t output, sz_size_t tile_begin, sz_size_t tile_end,
                                                         sz_size_t output_offset, sz_cptr_t source, sz_size_t bytes,
                                                         unsigned lane) {
    sz_size_t const copy_begin = sz_max_of_two(output_offset, tile_begin);
    sz_size_t const copy_end = sz_min_of_two(output_offset + bytes, tile_end);
    sz_size_t position;
    for (position = copy_begin + lane; position < copy_end; position += 32)
        output[position] = source[position - output_offset];
}

/**
 *  @brief Copies the rewritten target, one fixed-width output tile per block, one warp per
 *      gap or replacement.
 *
 *  Tiling the output rather than the matches bounds how long any one block works: a corpus of
 *  one huge document with a single match and a corpus of a million tiny ones give every block
 *  the same slice. Within a block the warps take stretches in parallel, because a rewrite over
 *  prose has stretches of tens of bytes and striding a whole block across one of them would
 *  leave most lanes idle.
 */
static __global__ void sz_substrings_rewrite_copy_simt_kernel_(sz_sequence_t haystacks, sz_sequence_t replacements,
                                                               sz_size_t const *haystack_offsets,
                                                               sz_substrings_match_t const *matches,
                                                               sz_size_t const *gap_offsets,
                                                               sz_size_t const *output_offsets,
                                                               sz_substrings_report_t const *report, sz_ptr_t output) {
    sz_size_t const output_bytes_total = output_offsets[haystacks.count];
    sz_size_t const tile_count = sz_size_divide_round_up(output_bytes_total, sz_substrings_gpu_rewrite_tile_bytes_k);
    unsigned const warp_index = threadIdx.x / 32u, warps_per_block = blockDim.x / 32u, lane = threadIdx.x % 32u;
    sz_size_t tile_index;
    // A target too small for the whole rewrite stays untouched rather than holding a valid prefix.
    if (report->shortfall) return;

    for (tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
        sz_size_t const tile_begin = tile_index * sz_substrings_gpu_rewrite_tile_bytes_k;
        sz_size_t const tile_end = sz_min_of_two(tile_begin + sz_substrings_gpu_rewrite_tile_bytes_k,
                                                 output_bytes_total);
        sz_size_t haystack_index = sz_substrings_last_not_above_simt_(output_offsets, haystacks.count + 1, tile_begin);

        for (; haystack_index < haystacks.count && output_offsets[haystack_index] < tile_end; ++haystack_index) {
            sz_cptr_t const haystack = sz_sequence_tape_start_simt_(haystacks.handle, haystack_index);
            sz_size_t const haystack_length = sz_sequence_tape_length_simt_(haystacks.handle, haystack_index);
            sz_size_t const base = output_offsets[haystack_index];
            sz_size_t const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
            // Every match contributes a gap and a replacement; one more stretch closes the haystack.
            sz_size_t const stretches = last - first + 1;
            sz_size_t const wanted = tile_begin > base ? tile_begin - base : 0;
            sz_size_t const skip = first == last
                                       ? 0
                                       : sz_substrings_last_not_above_simt_(gap_offsets + first, last - first, wanted);
            // Past the last match the drift is whatever the whole haystack accumulated, which its rewritten
            // length already names - so the closing stretch needs no offset of its own.
            sz_size_t const total_drift = (output_offsets[haystack_index + 1] - base) - haystack_length;
            sz_size_t stretch;

            for (stretch = skip + warp_index; stretch < stretches; stretch += warps_per_block) {
                sz_size_t const match_index = first + stretch;
                sz_bool_t const closes = (sz_bool_t)(match_index == last);
                sz_size_t const previous_end = match_index == first ? 0
                                                                    : matches[match_index - 1].byte_offset +
                                                                          matches[match_index - 1].byte_length;
                sz_size_t const gap_source_end = closes ? haystack_length : matches[match_index].byte_offset;
                sz_size_t const gap_begin = base + (closes ? previous_end + total_drift : gap_offsets[match_index]);
                sz_size_t needle;

                sz_substrings_copy_clipped_simt_(output, tile_begin, tile_end, gap_begin, haystack + previous_end,
                                                 gap_source_end - previous_end, lane);
                if (closes) continue;
                needle = matches[match_index].needle_index;
                sz_substrings_copy_clipped_simt_(output, tile_begin, tile_end,
                                                 gap_begin + (gap_source_end - previous_end),
                                                 sz_sequence_tape_start_simt_(replacements.handle, needle),
                                                 sz_sequence_tape_length_simt_(replacements.handle, needle), lane);
            }
        }
    }
}

/**
 *  @brief Publishes the bytes the rewrite needs, and whether the target could hold them.
 *  @param[in] target_ceiling Bytes the caller's target holds.
 */
static __global__ void sz_substrings_target_simt_kernel_(sz_size_t const *rewritten_at, sz_size_t target_ceiling,
                                                         sz_substrings_report_t *report) {
    sz_size_t const rewritten = *rewritten_at;
    if (blockIdx.x || threadIdx.x) return;
    report->target_length = rewritten;
    if (!report->shortfall && rewritten > target_ceiling) report->shortfall = rewritten - target_ceiling;
}

#pragma endregion Rewrite Kernels

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBSTRINGS_SIMT_CUH_
