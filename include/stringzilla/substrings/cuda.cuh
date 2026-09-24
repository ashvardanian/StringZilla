/**
 *  @brief CUDA backend for multi-pattern search: one thread per haystack chunk, the automaton's hot head
 *      staged in shared memory, and the leftmost cover resolved after the walk rather than inside it.
 *  @file include/stringzilla/substrings/cuda.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzilla/substrings.h
 *
 *  The transition is the serial tier's, reached from the device through `--expt-relaxed-constexpr`, so both
 *  sides answer the same matches rather than similar ones.
 *
 *  Parallelism comes from @b chunks rather than from haystacks: a corpus of one long document and a corpus
 *  of a million short ones then fill the device the same way. A chunk reports every match @b ending inside
 *  it, priming itself from the `max_source_match_bytes - 1` bytes before its own start - clamped to its
 *  haystack, never earlier - so every match is found exactly once and no chunk reads a neighbour's text.
 *
 *  A cover is a property of the matches, not of the bytes, so it is resolved after the walk. Inside the walk
 *  it would cost every thread a ring wide enough for the longest match, and a second walk to find a safe
 *  place to start; both are gone.
 *
 *  No compute verb joins the stream. Every size a launch needs is either fixed when the engine is built -
 *  the chunk and match budgets - or derived on the device from one the host never sees, and what a round
 *  discovered reaches the caller through `sz_substrings_report_t` after the caller's own join. When a round
 *  outruns its match budget the writing walk, the cover, the compaction and the boundaries kernel each
 *  retire at their first instruction, so an output is left untouched rather than truncated.
 *
 *  Written in C, as every `.cuh` in this library is: the only constructs here a C compiler would not take
 *  are the `extern "C"` that lets a C dispatch unit link against it, and the kernels' launches, which go
 *  through @c cudaLaunchKernel rather than the @c <<< @c >>> the language reserves for C++.
 */
#ifndef STRINGZILLA_SUBSTRINGS_CUDA_CUH_
#define STRINGZILLA_SUBSTRINGS_CUDA_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/substrings/serial.h"

#if SZ_USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Shapes

/** Threads every kernel here launches with; occupancy is shared-memory-bound rather than thread-bound, so a
 *  modest fixed block keeps the launch geometry simple and the block scan one power of two. */
enum { sz_substrings_cuda_threads_per_block_k = 256 };

/** Blocks one launch covers the device with, per multiprocessor, when the work is grid-strided. */
enum { sz_substrings_cuda_blocks_per_multiprocessor_k = 8 };

/** Tiles a scan cuts its input into. The carry across tiles is serial by construction, so past this a
 *  wider grid only lengthens the one block that walks it. */
enum { sz_substrings_cuda_scan_tiles_max_k = 1024 };

/** Candidates one thread scans quadratically before a cover segment falls back to emitted order. */
enum { sz_substrings_cuda_cover_segment_limit_k = 4096 };

/** Output bytes one block of a rewrite's copy owns, so no block's work scales with one run's width. */
enum { sz_substrings_cuda_rewrite_tile_bytes_k = 4096 };

/** Whether the caller reads the emitted matches, or only the boundaries the sizing walk already scanned. */
typedef enum sz_substrings_cuda_matches_t {
    /** Counting under an overlapping policy: the scanned chunk slots are the whole answer. */
    sz_substrings_cuda_matches_unneeded_k = 0,
    /** Finding, rewriting, or any cover: the list has to exist before anything can read or thin it. */
    sz_substrings_cuda_matches_needed_k = 1,
} sz_substrings_cuda_matches_t;

/** What a chunk walk does at each match: size the output so the caller can scan it, or write it. */
typedef enum sz_substrings_cuda_pass_t {
    /** Store each chunk's match count, so a scan can hand every chunk a private output range. */
    sz_substrings_cuda_sizing_k = 0,
    /** Write each match at the offset that scan left behind. */
    sz_substrings_cuda_writing_k = 1,
    /** Count each match against its needle in the block's tally, for scoring. */
    sz_substrings_cuda_tallying_k = 2,
} sz_substrings_cuda_pass_t;

/** Bits of a tally slot index: 4096 slots of a key and a count each, 32 KB of static shared memory. */
enum { sz_substrings_cuda_tally_slot_bits_k = 12 };

/** Slots one block's tally holds. */
enum { sz_substrings_cuda_tally_slots_k = 1 << sz_substrings_cuda_tally_slot_bits_k };

/** Slots a hashed tally probes before spilling a needle to the block's overflow row. */
enum { sz_substrings_cuda_tally_probes_k = 16 };

/** How a tally maps a needle to a slot. */
typedef enum sz_substrings_cuda_tally_layout_t {
    /** The vocabulary fits the slots, so a needle's index is its slot. */
    sz_substrings_cuda_tally_direct_k = 0,
    /** A larger vocabulary, hashed into the slots with linear probing and an overflow row behind them. */
    sz_substrings_cuda_tally_hashed_k = 1,
} sz_substrings_cuda_tally_layout_t;

/** One block's per-needle counts for the haystack it is scoring. */
typedef struct sz_substrings_cuda_tally_t {
    /** How @c counts is indexed. */
    sz_substrings_cuda_tally_layout_t layout;
    /** In shared memory: each hashed slot's needle index plus one, zero while free; @c SZ_NULL when direct. */
    sz_u32_t *keys;
    /** In shared memory: each slot's occurrences, one slot per needle when direct. */
    sz_u32_t *counts;
    /** The block's own @b [needles] global row for needles no probe could seat, @c SZ_NULL when direct. */
    sz_u32_t *overflow;
    /** In shared memory: nonzero once anything reached @c overflow, so scoring scans it only then. */
    sz_u32_t *overflowed;
} sz_substrings_cuda_tally_t;

#pragma endregion Shapes

#pragma region Device Helpers

/** Four tape bytes as one load; the address is peeled to its own alignment by the caller. */
SZ_DEVICE_INLINE sz_u32_t sz_substrings_cuda_load_quad_(sz_u8_t const *pointer) {
    sz_u32_t loaded;
    asm("ld.global.u32 %0, [%1];" : "=r"(loaded) : "l"(pointer));
    return loaded;
}

/**
 *  @brief One byte's transition, staged-shared-memory-first.
 *
 *  The staged prefix resolves branch-free out of shared memory, and everything else - the hot tier beyond
 *  the prefix, and the whole cold tier - defers to @ref sz_substrings_step, the single transition definition
 *  every backend shares. A single cold lane still makes the whole warp pay that lane's failure-chase depth,
 *  which is the cost this staging exists to shrink.
 */
SZ_DEVICE_INLINE sz_u32_t sz_substrings_cuda_step_(sz_substrings_engine_t const *engine,
                                                   sz_u32_t const *staged_rows, sz_u32_t staged_count, sz_u32_t state,
                                                   sz_u8_t byte) {
    if (state < staged_count)
        return staged_rows[(sz_size_t)state * engine->classes_count + engine->byte_to_class[byte]];
    return sz_substrings_step(engine, state, byte);
}

/**
 *  @brief Cooperatively stages the class map and the head of the hot tier, once per block, rebinding the
 *         block's own copy of @p engine to the staged map.
 *
 *  The hot tier's out-degree ordering makes its head the best prefix to stage, and every step reads the map.
 */
SZ_DEVICE_INLINE void sz_substrings_cuda_stage_(sz_substrings_engine_t *engine, sz_u8_t *staged_classes,
                                                sz_u32_t *staged_rows, sz_u32_t staged_count) {
    sz_size_t const cells = (sz_size_t)staged_count * engine->classes_count;
    sz_size_t cell;
    for (cell = threadIdx.x; cell < SZ_U8_MAX + 1; cell += blockDim.x)
        staged_classes[cell] = engine->byte_to_class[cell];
    for (cell = threadIdx.x; cell < cells; cell += blockDim.x) staged_rows[cell] = engine->hot_rows[cell];
    engine->byte_to_class = staged_classes;
    __syncthreads();
}

/**
 *  @brief The block's exclusive prefix sum of @p value, with the block's own total left in @p total.
 *
 *  A Hillis-Steele scan over @p shared, which the caller sizes at one entry per thread. Thirty lines rather
 *  than a dependency: a block scan is the only collective this tier needs, and pulling a template library
 *  into a C tier for it would cost the property the tier exists for.
 */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_block_scan_(sz_size_t value, sz_size_t *shared, sz_size_t *total) {
    unsigned const lane = threadIdx.x;
    unsigned offset;
    sz_size_t inclusive;
    shared[lane] = value;
    __syncthreads();
    for (offset = 1; offset < blockDim.x; offset *= 2) {
        sz_size_t const addend = lane >= offset ? shared[lane - offset] : 0;
        __syncthreads();
        shared[lane] += addend;
        __syncthreads();
    }
    inclusive = shared[lane];
    *total = shared[blockDim.x - 1];
    __syncthreads(); // ! The caller reuses `shared` for the next tile.
    return inclusive - value;
}

/** How many chunks of @p chunk_bytes a haystack of @p length bytes needs - at least one, so even an empty
 *  haystack still gets a thread and still lands its own boundary. */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_chunks_for_(sz_size_t length, sz_size_t chunk_bytes) {
    return length == 0 ? 1 : (length + chunk_bytes - 1) / chunk_bytes;
}

/** Which haystack owns global chunk @p chunk_index, from the exclusive prefix sum of per-haystack counts. */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_haystack_of_(sz_size_t const *chunk_offsets, sz_size_t haystacks_count,
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
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_last_not_above_(sz_size_t const *ascending, sz_size_t count,
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
SZ_DEVICE_INLINE void sz_substrings_cuda_tally_(sz_substrings_cuda_tally_t const *tally, sz_u32_t needle) {
    sz_u32_t const key = needle + 1;
    // Volatile, so a probe re-reads a slot a rival may have seated since.
    sz_u32_t const volatile *const seated_keys = tally->keys;
    sz_u32_t slot, probe;
    if (tally->layout == sz_substrings_cuda_tally_direct_k) {
        atomicAdd(tally->counts + needle, 1u);
        return;
    }
    slot = (sz_u32_t)(((sz_u64_t)key * 0x9E3779B97F4A7C15ull) >> (64 - sz_substrings_cuda_tally_slot_bits_k));
    for (probe = 0; probe != sz_substrings_cuda_tally_probes_k; ++probe) {
        sz_u32_t seated = seated_keys[slot];
        if (seated == 0) seated = atomicCAS(tally->keys + slot, 0u, key);
        if (seated == 0 || seated == key) {
            atomicAdd(tally->counts + slot, 1u);
            return;
        }
        slot = (slot + 1) & (sz_substrings_cuda_tally_slots_k - 1);
    }
    atomicAdd(tally->overflow + needle, 1u);
    *tally->overflowed = 1;
}

#pragma endregion Device Helpers

#pragma region Scan Kernels

/** Reduces one block's own contiguous tile into @c tile_sums[blockIdx.x]. */
static __global__ void sz_substrings_cuda_scan_reduce_kernel_(sz_size_t const *values, sz_size_t count,
                                                              sz_size_t elements_per_tile, sz_size_t *tile_sums) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
    sz_size_t const begin = (sz_size_t)blockIdx.x * elements_per_tile;
    sz_size_t const end = sz_min_of_two(begin + elements_per_tile, count);
    sz_size_t running = 0, first;
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < end ? values[index] : 0;
        sz_size_t total = 0;
        sz_substrings_cuda_block_scan_(value, shared, &total);
        if (threadIdx.x == 0) running += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) tile_sums[blockIdx.x] = running;
}

/** Scans @p tile_sums in place, on one block, carrying a running offset across as many tiles as it takes. */
static __global__ void sz_substrings_cuda_scan_carry_kernel_(sz_size_t *tile_sums, sz_size_t count) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
    __shared__ sz_size_t carry;
    sz_size_t first;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (first = 0; first < count; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < count ? tile_sums[index] : 0;
        sz_size_t total = 0;
        sz_size_t const exclusive = sz_substrings_cuda_block_scan_(value, shared, &total);
        if (index < count) tile_sums[index] = carry + exclusive;
        __syncthreads();
        if (threadIdx.x == 0) carry += total;
        __syncthreads();
    }
}

/** Scans one block's own tile in place, seeded by the base the carry settled for it. */
static __global__ void sz_substrings_cuda_scan_apply_kernel_(sz_size_t *values, sz_size_t count,
                                                             sz_size_t elements_per_tile, sz_size_t const *tile_sums) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
    sz_size_t const begin = (sz_size_t)blockIdx.x * elements_per_tile;
    sz_size_t const end = sz_min_of_two(begin + elements_per_tile, count);
    sz_size_t running = tile_sums[blockIdx.x], first;
    for (first = begin; first < end; first += blockDim.x) {
        sz_size_t const index = first + threadIdx.x;
        sz_size_t const value = index < end ? values[index] : 0;
        sz_size_t total = 0;
        sz_size_t const exclusive = sz_substrings_cuda_block_scan_(value, shared, &total);
        if (index < end) values[index] = running + exclusive;
        running += total;
        __syncthreads();
    }
}

#pragma endregion Scan Kernels

#pragma region Walk Kernels

/** Sums the haystacks' lengths, so the host can size a chunk without reaching a device accessor itself. */
static __global__ void sz_substrings_cuda_total_bytes_kernel_(sz_sequence_t haystacks, sz_size_t *total) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t mine = 0, block_total = 0;
    for (; index < haystacks.count; index += stride) mine += haystacks.get_length(haystacks.handle, index);
    sz_substrings_cuda_block_scan_(mine, shared, &block_total);
    if (threadIdx.x == 0) atomicAdd((unsigned long long *)total, (unsigned long long)block_total);
}

/**
 *  @brief Derives this round's chunk width from the corpus the device just summed.
 *
 *  @param[in] chunk_budget Chunks the arena holds beyond one per haystack, which is what fixes the width.
 *  @param[in] floor_bytes Four times the longest match, so a warm-up never outgrows a quarter of a chunk.
 *
 *  There is no ceiling on the width, and that is what makes the budget a bound rather than a hope: a chunk
 *  holds at least @c total/budget bytes, so the corpus contributes at most @c budget chunks, and each
 *  haystack's own remainder contributes at most one more. A ceiling would let a large corpus outrun any
 *  fixed budget, which is the readback this inversion exists to remove.
 */
static __global__ void sz_substrings_cuda_chunk_bytes_kernel_(sz_size_t const *total_bytes, sz_size_t chunk_budget,
                                                              sz_size_t floor_bytes, sz_size_t *chunk_bytes) {
    sz_size_t const budget = chunk_budget ? chunk_budget : 1;
    sz_size_t const share = (*total_bytes + budget - 1) / budget;
    if (blockIdx.x || threadIdx.x) return;
    *chunk_bytes = sz_max_of_two(sz_max_of_two(share, floor_bytes), (sz_size_t)1);
}

/** Writes how many chunks each haystack is cut into, which the scan then turns into its chunk range. */
static __global__ void sz_substrings_cuda_chunk_counts_kernel_(sz_sequence_t haystacks, sz_size_t const *chunk_bytes,
                                                               sz_size_t *chunk_offsets) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < haystacks.count; index += stride)
        chunk_offsets[index] = sz_substrings_cuda_chunks_for_(haystacks.get_length(haystacks.handle, index),
                                                              *chunk_bytes);
}

/**
 *  @brief Reports every match ending at @p delta, writing them when @p pass asks, and returns how many.
 *
 *  The acceptance bit answers "does anything end here" without touching the counts array, which at scale
 *  costs nearly as much as the tape read itself.
 */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_emit_(sz_substrings_engine_t const *engine, sz_u32_t state,
                                                    sz_size_t walk_begin, sz_u32_t delta, sz_size_t haystack_index,
                                                    sz_substrings_cuda_pass_t pass, sz_substrings_match_t *matches_out,
                                                    sz_substrings_cuda_tally_t const *tally) {
    sz_size_t output_offset, found = 0, index;
    sz_u32_t output_count;
    if (!sz_substrings_accepts(engine, state)) return 0;
    output_count = engine->outputs_counts[state];
    output_offset = engine->outputs_offsets[state];
    for (index = 0; index != output_count; ++index) {
        sz_substrings_output_t const output = engine->outputs[output_offset + index];
        // `walk_begin` is clamped to the haystack's own start, so underflowing the walk and underflowing
        // the haystack are the same test.
        if (delta + 1 < output.folded_match_bytes) continue;
        if (pass == sz_substrings_cuda_writing_k) {
            sz_substrings_match_t match;
            match.haystack_index = haystack_index;
            match.needle_index = output.needle_index;
            match.byte_offset = walk_begin + delta + 1 - output.folded_match_bytes;
            match.byte_length = output.folded_match_bytes;
            matches_out[found] = match;
        }
        else if (pass == sz_substrings_cuda_tallying_k) sz_substrings_cuda_tally_(tally, output.needle_index);
        ++found;
    }
    return found;
}

/**
 *  @brief Walks one chunk of a byte-exact haystack, counting or writing every match ending inside it.
 *  @param[in] chunk_begin First byte of the chunk, relative to the haystack's own start.
 *  @return The number of matches the chunk holds.
 *
 *  The warm-up primes the state from before the chunk and reports nothing, so once it ends the emit test is
 *  gone from the loop rather than being re-asked on every byte.
 */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_walk_chunk_cased_(
    sz_substrings_engine_t const *engine, sz_u32_t const *staged_rows, sz_u32_t staged_count, sz_cptr_t haystack,
    sz_size_t length, sz_size_t chunk_begin, sz_size_t chunk_end, sz_size_t haystack_index,
    sz_substrings_cuda_pass_t pass, sz_substrings_match_t *matches_at_chunk, sz_substrings_cuda_tally_t const *tally) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1
                                                                    : 0;
    sz_size_t const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const walk_base = (sz_u8_t const *)haystack + walk_begin;
    // Every 64-bit quantity is resolved here, once; the per-byte loops below ride 32-bit deltas from it.
    sz_u32_t const walk_span = (sz_u32_t)(chunk_end - walk_begin);
    sz_u32_t const emit_from = (sz_u32_t)(chunk_begin - walk_begin);
    sz_size_t found = 0;
    sz_u32_t delta = 0, lane;
    sz_u32_t state = engine->root; // ? Fresh at `walk_begin`; no state crosses a haystack boundary.
    sz_unused_(length);

    for (; delta < emit_from; ++delta)
        state = sz_substrings_cuda_step_(engine, staged_rows, staged_count, state, walk_base[delta]);

    // Peeled to the load's own alignment, so the body pays one four-byte load per four transitions; the
    // transition chain stays strictly serial, only the tape reads widen.
    for (; delta < walk_span && (((sz_size_t)(walk_base + delta)) & 3u) != 0; ++delta) {
        state = sz_substrings_cuda_step_(engine, staged_rows, staged_count, state, walk_base[delta]);
        found += sz_substrings_cuda_emit_(engine, state, walk_begin, delta, haystack_index, pass,
                                          matches_at_chunk + found, tally);
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        sz_u32_t const quad = sz_substrings_cuda_load_quad_(walk_base + delta);
#pragma unroll
        for (lane = 0; lane != 4; ++lane) {
            state = sz_substrings_cuda_step_(engine, staged_rows, staged_count, state,
                                             (sz_u8_t)(quad >> (lane * 8)));
            found += sz_substrings_cuda_emit_(engine, state, walk_begin, delta + lane, haystack_index, pass,
                                              matches_at_chunk + found, tally);
        }
    }
    for (; delta < walk_span; ++delta) {
        state = sz_substrings_cuda_step_(engine, staged_rows, staged_count, state, walk_base[delta]);
        found += sz_substrings_cuda_emit_(engine, state, walk_begin, delta, haystack_index, pass,
                                          matches_at_chunk + found, tally);
    }
    return found;
}

/**
 *  @brief Walks one chunk as folded bytes, the case-insensitive twin of the walk above.
 *
 *  Folding makes a walk restart-safe only at a codepoint start, so the warm-up snaps back to one before it
 *  begins - three bytes at most, and always earlier, so the extra transitions only prime state further.
 *  Match ends are reported at the source codepoint's end, which keeps chunk ownership comparable against
 *  the unsnapped chunk bounds the planner handed out.
 */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_walk_chunk_uncased_(
    sz_substrings_engine_t const *engine, sz_u32_t const *staged_rows, sz_u32_t staged_count, sz_cptr_t haystack,
    sz_size_t length, sz_size_t chunk_begin, sz_size_t chunk_end, sz_size_t haystack_index,
    sz_substrings_cuda_pass_t pass, sz_substrings_match_t *matches_at_chunk, sz_substrings_cuda_tally_t const *tally) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1
                                                                    : 0;
    sz_size_t walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_substrings_folded_cursor_t cursor;
    sz_substrings_folded_byte_t step;
    sz_u32_t state = engine->root;
    sz_size_t folded = 0, last_break_folded_end = 0, found = 0;

    // A fold is only restartable on a lead byte, so the walk snaps outward to one - never inward, which
    // would drop a match whose needle began mid-codepoint.
    sz_u8_t const *const haystack_bytes = (sz_u8_t const *)haystack;
    while (walk_begin != 0 && haystack_bytes[walk_begin] >= 0x80 && (haystack_bytes[walk_begin] & 0xC0) == 0x80)
        --walk_begin;

    sz_substrings_folded_cursor_init(&cursor, haystack + walk_begin, length - walk_begin);
    while (sz_substrings_folded_cursor_next(&cursor, &step)) {
        sz_size_t output_offset, output_count, index, source_end;
        ++folded;
        if (step.malformed) {
            state = engine->root;
            continue;
        }
        state = sz_substrings_cuda_step_(engine, staged_rows, staged_count, state, step.byte);
        if (!step.rune_end) continue;
        if (step.breaks_boundary) last_break_folded_end = folded + step.trailing;

        source_end = walk_begin + step.codepoint_end;
        if (source_end > chunk_end) break; // ? Past this chunk's share; the next chunk owns these ends.
        if (!sz_substrings_accepts(engine, state)) continue;
        if (source_end <= chunk_begin) continue; // ? Still warming up, where this chunk reports nothing.

        output_count = engine->outputs_counts[state];
        output_offset = engine->outputs_offsets[state];
        for (index = 0; index != output_count; ++index) {
            sz_substrings_output_t const output = engine->outputs[output_offset + index];
            sz_size_t const folded_length = output.folded_match_bytes;
            sz_substrings_resolved_match_t resolved;
            if (folded < folded_length) continue;
            resolved = sz_substrings_folded_span(haystack + walk_begin, &step, folded, last_break_folded_end,
                                                 folded_length);
            if (resolved.repeats) continue;
            if (pass == sz_substrings_cuda_writing_k) {
                matches_at_chunk[found].haystack_index = haystack_index;
                matches_at_chunk[found].needle_index = output.needle_index;
                matches_at_chunk[found].byte_offset = walk_begin + resolved.source_offset;
                matches_at_chunk[found].byte_length = step.codepoint_end - resolved.source_offset;
            }
            else if (pass == sz_substrings_cuda_tallying_k) sz_substrings_cuda_tally_(tally, output.needle_index);
            ++found;
        }
    }
    return found;
}

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in whichever pass @p pass names.
 *
 *  Both passes share @p chunk_slots, because the host's in-place exclusive scan already makes them one
 *  allocation: sizing writes each chunk's match count into its slot, and writing reads the exclusive offset
 *  the scan left there. Every chunk owns a private, non-overlapping output range, so a write needs no atomic.
 */
static __global__ void sz_substrings_cuda_walk_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                                       sz_sequence_t haystacks, sz_size_t const *chunk_offsets,
                                                       sz_size_t const *chunk_bytes_at,
                                                       sz_substrings_report_t const *report, sz_size_t *chunk_slots,
                                                       sz_substrings_match_t *matches,
                                                       sz_substrings_cuda_pass_t pass) {
    extern __shared__ sz_u32_t sz_substrings_cuda_staged_[];
    __shared__ sz_u8_t staged_classes[SZ_U8_MAX + 1];
    sz_size_t const chunk_count = chunk_offsets[haystacks.count];
    sz_size_t const chunk_bytes = *chunk_bytes_at;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk_index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // The writing pass has nowhere to write once the sizing pass outran the budget, so it retires whole.
    if (pass == sz_substrings_cuda_writing_k && report->shortfall) return;
    sz_substrings_cuda_stage_(&engine, staged_classes, sz_substrings_cuda_staged_, staged_count);

    for (; chunk_index < chunk_count; chunk_index += stride) {
        sz_size_t const haystack_index = sz_substrings_cuda_haystack_of_(chunk_offsets, haystacks.count, chunk_index);
        sz_cptr_t const haystack = haystacks.get_start(haystacks.handle, haystack_index);
        sz_size_t const length = haystacks.get_length(haystacks.handle, haystack_index);
        sz_size_t const local_index = chunk_index - chunk_offsets[haystack_index];
        sz_size_t const chunk_begin = local_index * chunk_bytes;
        sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
        sz_substrings_match_t *const matches_at_chunk = pass == sz_substrings_cuda_writing_k
                                                            ? matches + chunk_slots[chunk_index]
                                                            : matches;
        sz_size_t found;
        // One vocabulary is byte-exact or folded for its whole lifetime, so every thread takes the same side
        // and the branch costs no divergence.
        if (chunk_begin >= chunk_end && length != 0) found = 0;
        else if (engine.case_sensitivity == sz_substrings_uncased_k)
            found = sz_substrings_cuda_walk_chunk_uncased_(&engine, sz_substrings_cuda_staged_, staged_count,
                                                           haystack, length, chunk_begin, chunk_end, haystack_index,
                                                           pass, matches_at_chunk, SZ_NULL);
        else
            found = sz_substrings_cuda_walk_chunk_cased_(&engine, sz_substrings_cuda_staged_, staged_count, haystack,
                                                         length, chunk_begin, chunk_end, haystack_index, pass,
                                                         matches_at_chunk, SZ_NULL);
        if (pass == sz_substrings_cuda_sizing_k) chunk_slots[chunk_index] = found;
    }
}

#pragma endregion Walk Kernels

#pragma region Scoring Kernels

/** Fixed-point scale of a BM25 sum: integer addition commutes, so the score never depends on thread order. */
#define SZ_SUBSTRINGS_CUDA_BM25_SCALE (4294967296.0)

/** Slots a block's tally takes: one per needle when the vocabulary fits, a hashed table otherwise. */
SZ_HELPER_AUTO sz_size_t sz_substrings_cuda_tally_slots_for_(sz_size_t needles_count) {
    return needles_count <= sz_substrings_cuda_tally_slots_k ? needles_count : sz_substrings_cuda_tally_slots_k;
}

/** Words of the acceptance bitmap, one bit per double-array slot. */
SZ_HELPER_AUTO sz_size_t sz_substrings_cuda_accepts_words_(sz_substrings_engine_t const *engine) {
    return sz_size_divide_round_up(engine->slots_count, 32);
}

/**
 *  @brief Scores one haystack per block: its threads walk contiguous chunks into one shared tally, then
 *         sum the tallied terms in fixed point.
 *  @param[in] scores_stride Entries from one haystack's score to the next, so one launch writes one column.
 *  @param[in] staged_accepts_words Acceptance words staged in shared memory, all of them or zero.
 *  @param[in] staged_count Hot rows staged after them, a prefix of the tier.
 *
 *  Dynamic shared memory holds the counts, the keys when hashed, then the staged acceptance bitmap and
 *  rows. A block rather than a grid per haystack keeps the tally in shared memory, at the price of one long
 *  document spreading across one block's threads only.
 */
static __global__ void sz_substrings_cuda_bm25_kernel_(sz_substrings_engine_t engine, sz_sequence_t haystacks,
                                                       sz_f32_t const *document_lengths,
                                                       sz_substrings_bm25_t parameters, sz_f32_t const *needle_weights,
                                                       sz_u32_t *overflow_rows, sz_f32_t *scores,
                                                       sz_size_t scores_stride, sz_u32_t staged_accepts_words,
                                                       sz_u32_t staged_count) {
    extern __shared__ sz_u32_t sz_substrings_cuda_scoring_[];
    __shared__ sz_u8_t staged_classes[SZ_U8_MAX + 1];
    __shared__ sz_u32_t overflowed;
    __shared__ unsigned long long block_sum;
    sz_size_t const needles_count = engine.needles_count;
    sz_size_t const table_slots = sz_substrings_cuda_tally_slots_for_(needles_count);
    sz_size_t const warm_up = sz_max_of_two((sz_size_t)engine.max_source_match_bytes, (sz_size_t)1);
    sz_substrings_cuda_tally_t tally;
    sz_u32_t *staged_accepts, *staged_rows;
    sz_size_t haystack_index, slot;
    tally.layout = needles_count <= sz_substrings_cuda_tally_slots_k ? sz_substrings_cuda_tally_direct_k
                                                                     : sz_substrings_cuda_tally_hashed_k;
    tally.counts = sz_substrings_cuda_scoring_;
    tally.keys = tally.layout == sz_substrings_cuda_tally_hashed_k ? tally.counts + table_slots : SZ_NULL;
    tally.overflowed = &overflowed;
    tally.overflow = overflow_rows ? overflow_rows + (sz_size_t)blockIdx.x * needles_count : SZ_NULL;
    staged_accepts = tally.counts + (tally.keys ? 2 : 1) * table_slots;
    staged_rows = staged_accepts + staged_accepts_words;

    for (slot = threadIdx.x; slot < table_slots; slot += blockDim.x) {
        tally.counts[slot] = 0;
        if (tally.keys) tally.keys[slot] = 0;
    }
    for (slot = threadIdx.x; slot < staged_accepts_words; slot += blockDim.x)
        staged_accepts[slot] = engine.accepts_words[slot];
    // The walks read acceptance through the automaton, so rebinding the by-value copy is the whole change.
    if (staged_accepts_words) engine.accepts_words = staged_accepts;
    if (threadIdx.x == 0) overflowed = 0;
    sz_substrings_cuda_stage_(&engine, staged_classes, staged_rows,
                              staged_count); // ! Ends in the barrier the zeroing needs.

    for (haystack_index = blockIdx.x; haystack_index < haystacks.count; haystack_index += gridDim.x) {
        sz_cptr_t const haystack = haystacks.get_start(haystacks.handle, haystack_index);
        sz_size_t const length = haystacks.get_length(haystacks.handle, haystack_index);
        // Never narrower than the longest match, past which a chunk re-walks more warm-up than it owns.
        sz_size_t const chunk_bytes = sz_max_of_two((length + blockDim.x - 1) / blockDim.x, warm_up);
        sz_size_t const chunk_begin = (sz_size_t)threadIdx.x * chunk_bytes;
        sz_f64_t const norm = sz_substrings_bm25_norm(
            &parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        sz_i64_t mine = 0;
        sz_size_t needle;
        if (threadIdx.x == 0) block_sum = 0;
        if (chunk_begin < length) {
            sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
            if (engine.case_sensitivity == sz_substrings_uncased_k)
                sz_substrings_cuda_walk_chunk_uncased_(&engine, staged_rows, staged_count, haystack, length,
                                                       chunk_begin, chunk_end, haystack_index,
                                                       sz_substrings_cuda_tallying_k, SZ_NULL, &tally);
            else
                sz_substrings_cuda_walk_chunk_cased_(&engine, staged_rows, staged_count, haystack, length,
                                                     chunk_begin, chunk_end, haystack_index,
                                                     sz_substrings_cuda_tallying_k, SZ_NULL, &tally);
        }
        __syncthreads();

        // Every slot and overflow entry is zeroed as it is read, so the next haystack starts from a clean tally.
        for (slot = threadIdx.x; slot < table_slots; slot += blockDim.x) {
            sz_u32_t const frequency = tally.counts[slot];
            if (!frequency) continue;
            needle = tally.keys ? (sz_size_t)tally.keys[slot] - 1 : slot;
            mine += __double2ll_rn(sz_substrings_bm25_term(&parameters, norm, needle_weights[needle], frequency) *
                                   SZ_SUBSTRINGS_CUDA_BM25_SCALE);
            tally.counts[slot] = 0;
            if (tally.keys) tally.keys[slot] = 0;
        }
        if (overflowed)
            for (needle = threadIdx.x; needle < needles_count; needle += blockDim.x) {
                sz_u32_t const frequency = tally.overflow[needle];
                if (!frequency) continue;
                mine += __double2ll_rn(sz_substrings_bm25_term(&parameters, norm, needle_weights[needle], frequency) *
                                       SZ_SUBSTRINGS_CUDA_BM25_SCALE);
                tally.overflow[needle] = 0;
            }
        // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
        if (mine) atomicAdd(&block_sum, (unsigned long long)mine);
        __syncthreads();
        if (threadIdx.x == 0) {
            scores[haystack_index * scores_stride] =
                (sz_f32_t)((sz_f64_t)(sz_i64_t)block_sum / SZ_SUBSTRINGS_CUDA_BM25_SCALE);
            overflowed = 0;
        }
        __syncthreads();
    }
}

#pragma endregion Scoring Kernels

#pragma region Cover Kernels

/** Whether the boundary before @p index is real: nothing still to come starts before the maximum end
 *  already reached. Only matches ending within one match's length of it can, which bounds the look-ahead. */
SZ_DEVICE_INLINE sz_bool_t sz_substrings_cuda_boundary_before_(sz_substrings_match_t const *matches, sz_size_t count,
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
 *  Within a haystack the walk emits in non-decreasing end order, so the running maximum end is simply the
 *  previous match's end, and a boundary sits where nothing still to come reaches back across it. Nothing
 *  before such a boundary can reach past it, so each segment resolves against a cursor of zero,
 *  independently of every other.
 *
 *  Segments are short in real text - a needle set drawn from a vocabulary leaves a median of one match
 *  between boundaries - so one thread takes a whole one. That is a measurement rather than a guarantee: a
 *  needle and its own suffixes over repetitive text make one segment of the whole document, and the greedy
 *  below is quadratic in a segment, so past @ref sz_substrings_cuda_cover_segment_limit_k candidates a
 *  segment falls back to accepting in emitted order - the same cover whenever starts ascend with ends, and
 *  a documented approximation when they do not. Without the cap one thread could hold the grid.
 */
static __global__ void sz_substrings_cuda_cover_kernel_(sz_substrings_match_t const *matches,
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
        if (!sz_substrings_cuda_boundary_before_(matches, count, longest, index)) continue;
        for (segment_end = index + 1; segment_end < count; ++segment_end)
            if (sz_substrings_cuda_boundary_before_(matches, count, longest, segment_end)) break;

        if (segment_end - index > sz_substrings_cuda_cover_segment_limit_k) {
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
 *  @brief Gathers the surviving matches into their scanned slots, order preserved.
 *  @param[in] keep_offsets The scanned keep flags, one longer than @p count so the last has a successor.
 *
 *  The scan overwrote the flags it summed, so survival is read back out of it: a match was kept exactly
 *  when the scan steps across it.
 */
static __global__ void sz_substrings_cuda_compact_kernel_(sz_substrings_match_t const *matches,
                                                          sz_substrings_report_t const *report,
                                                          sz_size_t const *keep_offsets,
                                                          sz_substrings_match_t *survivors) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    for (; index < count; index += stride)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/**
 *  @brief Publishes what the sizing walk found, which is the one place a round learns whether it fit.
 *  @param[in] emitted_at The last entry of the scanned chunk slots, which every trailing zero carries.
 *  @param[in] emitting Whether the round reads the matches themselves, since a count that never does cannot
 *             overrun a match budget however many matches the corpus holds.
 */
static __global__ void sz_substrings_cuda_sized_kernel_(sz_size_t const *emitted_at, sz_size_t matches_budget,
                                                        sz_bool_t emitting, sz_substrings_report_t *report) {
    sz_size_t const emitted = *emitted_at;
    if (blockIdx.x || threadIdx.x) return;
    report->matches_emitted = emitted;
    report->matches_stored = emitted;
    report->tape_bytes = 0;
    report->shortfall = emitting && emitted > matches_budget ? emitted - matches_budget : 0;
}

/** Publishes how many matches the cover kept, which is what every later boundary is read against. */
static __global__ void sz_substrings_cuda_covered_kernel_(sz_size_t const *kept_at, sz_substrings_report_t *report) {
    sz_size_t const kept = *kept_at;
    if (blockIdx.x || threadIdx.x) return;
    if (!report->shortfall) report->matches_stored = kept;
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
static __global__ void sz_substrings_cuda_haystack_offsets_kernel_(sz_size_t const *chunk_offsets,
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
static __global__ void sz_substrings_cuda_counts_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts,
                                                         sz_size_t counts_stride, sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < count; index += stride)
        counts[index * counts_stride] = haystack_offsets[index + 1] - haystack_offsets[index];
}

/**
 *  @brief Copies the surviving matches into the caller's array, clipped at a capacity only it knows.
 *
 *  The survivor count lives on the device, so a host-issued @c cudaMemcpyAsync cannot express the clip; one
 *  grid-strided kernel can, and it publishes what it stored in the same launch.
 */
static __global__ void sz_substrings_cuda_store_matches_kernel_(sz_substrings_match_t const *reported,
                                                                sz_substrings_report_t *report,
                                                                sz_substrings_match_t *matches,
                                                                sz_size_t matches_capacity) {
    sz_size_t const kept = report->shortfall ? 0 : report->matches_stored;
    sz_size_t const fitting = sz_min_of_two(kept, matches_capacity);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < fitting; index += stride) matches[index] = reported[index];
    if (blockIdx.x || threadIdx.x) return;
    report->matches_stored = fitting;
    if (kept > matches_capacity) report->shortfall = kept - matches_capacity;
}

#pragma endregion Cover Kernels

#pragma region Rewrite Kernels

/**
 *  @brief Writes where each match's preceding gap lands, and how long each haystack becomes.
 *
 *  One block per haystack, threads striding its match range. A rewrite is a tiling of gaps and
 *  replacements, and every boundary in that tiling follows from one running quantity: how far the output
 *  has drifted from the input by the time a match is reached. So that drift is all this stores - one
 *  scanned offset per match - and the copy kernel derives the rest from the match list it already has.
 *
 *  Offsets are relative to the haystack's own start, because the base is only known after the scan across
 *  haystacks that this kernel feeds.
 */
static __global__ void sz_substrings_cuda_rewrite_offsets_kernel_(sz_sequence_t haystacks, sz_sequence_t replacements,
                                                                  sz_size_t const *haystack_offsets,
                                                                  sz_substrings_match_t const *matches,
                                                                  sz_substrings_report_t const *report,
                                                                  sz_size_t *gap_offsets, sz_size_t *output_sizes) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
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
                drift_here = replacements.get_length(replacements.handle, needle) - matches[match_index].byte_length;
                previous_end = match_index == first
                                   ? 0
                                   : matches[match_index - 1].byte_offset + matches[match_index - 1].byte_length;
            }
            drift_before = sz_substrings_cuda_block_scan_(drift_here, shared, &drift_in_tile);
            if (owns) gap_offsets[match_index] = previous_end + drift_carry + drift_before;
            __syncthreads();
            if (threadIdx.x == 0) drift_carry += drift_in_tile;
            __syncthreads();
        }

        // The scan's own aggregate is the haystack's total drift, so no second pass reduces what it knows.
        if (threadIdx.x == 0)
            output_sizes[haystack_index] = haystacks.get_length(haystacks.handle, haystack_index) + drift_carry;
        __syncthreads(); // ! The next haystack resets the carry this one is still reading.
    }
}

/** Copies one stretch, clipped to the tile, with @p lane striding the surviving bytes. */
SZ_DEVICE_INLINE void sz_substrings_cuda_copy_clipped_(sz_ptr_t output, sz_size_t tile_begin, sz_size_t tile_end,
                                                       sz_size_t output_offset, sz_cptr_t source, sz_size_t bytes,
                                                       unsigned lane) {
    sz_size_t const copy_begin = sz_max_of_two(output_offset, tile_begin);
    sz_size_t const copy_end = sz_min_of_two(output_offset + bytes, tile_end);
    sz_size_t position;
    for (position = copy_begin + lane; position < copy_end; position += 32)
        output[position] = source[position - output_offset];
}

/**
 *  @brief Copies the rewritten tape, one fixed-width output tile per block, one warp per gap or replacement.
 *
 *  Tiling the output rather than the matches bounds how long any one block works: a corpus of one huge
 *  document with a single match and a corpus of a million tiny ones give every block the same slice. Within
 *  a block the warps take stretches in parallel, because a rewrite over prose has stretches of tens of bytes
 *  and striding a whole block across one of them would leave most lanes idle.
 */
static __global__ void sz_substrings_cuda_rewrite_copy_kernel_(sz_sequence_t haystacks, sz_sequence_t replacements,
                                                               sz_size_t const *haystack_offsets,
                                                               sz_substrings_match_t const *matches,
                                                               sz_size_t const *gap_offsets,
                                                               sz_size_t const *output_offsets,
                                                               sz_substrings_report_t const *report,
                                                               sz_ptr_t output) {
    sz_size_t const output_bytes_total = output_offsets[haystacks.count];
    sz_size_t const tile_count = (output_bytes_total + sz_substrings_cuda_rewrite_tile_bytes_k - 1) /
                                 sz_substrings_cuda_rewrite_tile_bytes_k;
    unsigned const warp_index = threadIdx.x / 32u, warps_per_block = blockDim.x / 32u, lane = threadIdx.x % 32u;
    sz_size_t tile_index;
    // A tape that cannot hold the whole rewrite is left untouched rather than holding a valid prefix of one.
    if (report->shortfall) return;

    for (tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
        sz_size_t const tile_begin = tile_index * sz_substrings_cuda_rewrite_tile_bytes_k;
        sz_size_t const tile_end = sz_min_of_two(tile_begin + sz_substrings_cuda_rewrite_tile_bytes_k,
                                                 output_bytes_total);
        sz_size_t haystack_index = sz_substrings_cuda_last_not_above_(output_offsets, haystacks.count + 1, tile_begin);

        for (; haystack_index < haystacks.count && output_offsets[haystack_index] < tile_end; ++haystack_index) {
            sz_cptr_t const haystack = haystacks.get_start(haystacks.handle, haystack_index);
            sz_size_t const haystack_length = haystacks.get_length(haystacks.handle, haystack_index);
            sz_size_t const base = output_offsets[haystack_index];
            sz_size_t const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
            // Every match contributes a gap and a replacement; one more stretch closes the haystack.
            sz_size_t const stretches = last - first + 1;
            sz_size_t const wanted = tile_begin > base ? tile_begin - base : 0;
            sz_size_t const skip = first == last
                                       ? 0
                                       : sz_substrings_cuda_last_not_above_(gap_offsets + first, last - first, wanted);
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

                sz_substrings_cuda_copy_clipped_(output, tile_begin, tile_end, gap_begin, haystack + previous_end,
                                                 gap_source_end - previous_end, lane);
                if (closes) continue;
                needle = matches[match_index].needle_index;
                sz_substrings_cuda_copy_clipped_(output, tile_begin, tile_end,
                                                 gap_begin + (gap_source_end - previous_end),
                                                 replacements.get_start(replacements.handle, needle),
                                                 replacements.get_length(replacements.handle, needle), lane);
            }
        }
    }
}

/**
 *  @brief Publishes the bytes the rewrite needs, and whether the tape could hold them.
 *  @param[in] tape_ceiling The lower of the caller's capacity and the engine's tape budget.
 */
static __global__ void sz_substrings_cuda_tape_kernel_(sz_size_t const *rewritten_at, sz_size_t tape_ceiling,
                                                       sz_substrings_report_t *report) {
    sz_size_t const rewritten = *rewritten_at;
    if (blockIdx.x || threadIdx.x) return;
    report->tape_bytes = rewritten;
    if (!report->shortfall && rewritten > tape_ceiling) report->shortfall = rewritten - tape_ceiling;
}

#pragma endregion Rewrite Kernels

#pragma region Host Plumbing

/** The device the caller's stream belongs to, which is not always device zero. */
SZ_API_COMPTIME int sz_substrings_cuda_device_(void) {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

/** Multiprocessors on the current device, or one when the driver will not say. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_multiprocessors_(void) {
    int multiprocessors = 0;
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, sz_substrings_cuda_device_()) !=
        cudaSuccess)
        return 1;
    return (sz_size_t)sz_max_of_two(multiprocessors, 1);
}

/** Blocks of @p kernel this device holds resident per multiprocessor at @p shared_bytes of dynamic shared. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_resident_blocks_(void const *kernel, sz_size_t shared_bytes) {
    int blocks_per_multiprocessor = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_multiprocessor, kernel, sz_substrings_cuda_threads_per_block_k, shared_bytes) != cudaSuccess)
        return 1;
    return (sz_size_t)sz_max_of_two(blocks_per_multiprocessor, 1);
}

/** Threads the walk keeps resident across the whole device, which is what a chunk width is derived from. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_resident_threads_(void const *kernel, sz_size_t shared_bytes) {
    return sz_substrings_cuda_multiprocessors_() * sz_substrings_cuda_resident_blocks_(kernel, shared_bytes) *
           sz_substrings_cuda_threads_per_block_k;
}

/**
 *  @brief Grid for a grid-strided kernel over @p items, covering the device without exceeding the work.
 *
 *  Capped at what @p kernel actually keeps resident rather than at a fixed blocks-per-multiprocessor guess,
 *  since a kernel's residency moves with its registers and its dynamic shared memory.
 */
SZ_API_COMPTIME unsigned sz_substrings_cuda_grid_for_(void const *kernel, sz_size_t shared_bytes, sz_size_t items) {
    sz_size_t blocks = (items + sz_substrings_cuda_threads_per_block_k - 1) / sz_substrings_cuda_threads_per_block_k;
    sz_size_t const covering = sz_substrings_cuda_multiprocessors_() *
                               sz_substrings_cuda_resident_blocks_(kernel, shared_bytes);
    if (blocks == 0) blocks = 1;
    return (unsigned)sz_min_of_two(blocks, covering);
}

/** Grid for the flat helper kernels, none of which takes dynamic shared memory. */
SZ_API_COMPTIME unsigned sz_substrings_cuda_grid_(sz_size_t items) {
    sz_size_t blocks = (items + sz_substrings_cuda_threads_per_block_k - 1) / sz_substrings_cuda_threads_per_block_k;
    sz_size_t const covering = sz_substrings_cuda_multiprocessors_() * sz_substrings_cuda_blocks_per_multiprocessor_k;
    if (blocks == 0) blocks = 1;
    return (unsigned)sz_min_of_two(blocks, covering);
}

/** Launches @p kernel over @p blocks blocks of the tier's fixed block size, on @p stream. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_launch_(void const *kernel, unsigned blocks, void **arguments,
                                                       sz_size_t shared_bytes, void *stream) {
    dim3 grid, block;
    grid.x = blocks, grid.y = 1, grid.z = 1;
    block.x = sz_substrings_cuda_threads_per_block_k, block.y = 1, block.z = 1;
    return cudaLaunchKernel(kernel, grid, block, arguments, shared_bytes, (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Tiles the tier's fixed block size covers @p count elements in. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_tiles_(sz_size_t count) {
    return (count + sz_substrings_cuda_threads_per_block_k - 1) / sz_substrings_cuda_threads_per_block_k;
}

/** Entries a scan over @p count elements needs for its tile totals, which no corpus size can outgrow. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_scan_scratch_(sz_size_t count) {
    return sz_min_of_two(sz_substrings_cuda_tiles_(count), (sz_size_t)sz_substrings_cuda_scan_tiles_max_k);
}

/**
 *  @brief Turns @p values into its own exclusive prefix sum, leaving the grand total in @c values[count].
 *  @param[in] tile_sums Scratch of one entry per tile the scan cuts @p values into.
 *
 *  Three launches - scan each tile, carry the tile totals across on one block, add each tile's base back -
 *  which is the shape a block scan composes into without a device-wide collective.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_scan_(sz_size_t *values, sz_size_t count, sz_size_t *tile_sums,
                                                     void *stream) {
    sz_size_t const tiles = sz_min_of_two(sz_substrings_cuda_tiles_(count),
                                          (sz_size_t)sz_substrings_cuda_scan_tiles_max_k);
    sz_size_t counted = count, tiles_counted = tiles;
    sz_size_t elements_per_tile;
    void *arguments[4];
    sz_status_t status;
    if (!count) return sz_success_k;
    elements_per_tile = (count + tiles - 1) / tiles;

    arguments[0] = &values, arguments[1] = &counted, arguments[2] = &elements_per_tile, arguments[3] = &tile_sums;
    status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_scan_reduce_kernel_, (unsigned)tiles,
                                        arguments, 0, stream);
    if (status != sz_success_k) return status;

    arguments[0] = &tile_sums, arguments[1] = &tiles_counted;
    status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_scan_carry_kernel_, 1, arguments, 0, stream);
    if (status != sz_success_k) return status;

    arguments[0] = &values, arguments[1] = &counted, arguments[2] = &elements_per_tile, arguments[3] = &tile_sums;
    return sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_scan_apply_kernel_, (unsigned)tiles, arguments,
                                      0, stream);
}

/** How many hot rows a kernel may stage in shared memory. */
typedef enum sz_substrings_cuda_staging_t {
    /** All of the hot tier or none of it: the walk pays no bounds test for a prefix most steps miss. */
    sz_substrings_cuda_stage_whole_k = 0,
    /** The longest prefix that fits beside the reservation, for a kernel that shares its memory anyway. */
    sz_substrings_cuda_stage_prefix_k = 1,
} sz_substrings_cuda_staging_t;

/**
 *  @brief Hot rows one block of @p kernel stages beside @p reserved_bytes of its own dynamic shared memory.
 *
 *  Rows are staged only while they displace no resident block: the walk is latency-bound, so warps are
 *  worth more than rows. The ceiling is the block's default, since raising it needs `cudaFuncSetAttribute`
 *  per kernel, and a launch asking for more than the default is rejected outright.
 */
SZ_API_COMPTIME sz_u32_t sz_substrings_cuda_staged_rows_(sz_substrings_engine_t const *engine, void const *kernel,
                                                         sz_size_t reserved_bytes,
                                                         sz_substrings_cuda_staging_t staging) {
    sz_size_t const row_bytes = engine->classes_count * sizeof(sz_u32_t);
    sz_size_t const unstaged_blocks = sz_substrings_cuda_resident_blocks_(kernel, reserved_bytes);
    sz_size_t fitting, low, high;
    int ceiling = 0;
    if (!engine->hot_count) return 0;
    if (cudaDeviceGetAttribute(&ceiling, cudaDevAttrMaxSharedMemoryPerBlock, sz_substrings_cuda_device_()) !=
        cudaSuccess)
        return 0;
    if (reserved_bytes >= (sz_size_t)ceiling) return 0;
    fitting = sz_min_of_two(((sz_size_t)ceiling - reserved_bytes) / row_bytes, (sz_size_t)engine->hot_count);
    if (staging == sz_substrings_cuda_stage_whole_k) {
        if (fitting < engine->hot_count) return 0;
        return sz_substrings_cuda_resident_blocks_(kernel, reserved_bytes + fitting * row_bytes) < unstaged_blocks
                   ? 0
                   : engine->hot_count;
    }
    // The longest prefix keeping the block count, found by bisection since residency falls monotonically.
    low = 0, high = fitting;
    while (low < high) {
        sz_size_t const middle = low + sz_size_divide_round_up(high - low, 2);
        if (sz_substrings_cuda_resident_blocks_(kernel, reserved_bytes + middle * row_bytes) < unstaged_blocks)
            high = middle - 1;
        else low = middle;
    }
    return (sz_u32_t)low;
}

/** Bytes a chunk never falls below: four times the longest match, capping its warm-up at a quarter of it. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_chunk_floor_(sz_substrings_engine_t const *engine) {
    return sz_max_of_two(4 * (sz_size_t)engine->max_source_match_bytes, (sz_size_t)1);
}

/** Devices the runtime answers for, which is what decides whether the device table is filled at all. */
SZ_API_COMPTIME int sz_substrings_cuda_devices_(void) {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess ? devices : 0;
}

/** Hot rows this tier's walk stages, which fixes both its shared memory and its residency. */
SZ_API_COMPTIME sz_u32_t sz_substrings_cuda_walk_rows_(sz_substrings_engine_t const *engine) {
    return sz_substrings_cuda_staged_rows_(engine, (void const *)sz_substrings_cuda_walk_kernel_, 0,
                                           sz_substrings_cuda_stage_whole_k);
}

/**
 *  @brief Global tally rows a hashed BM25 launch may spill into, which is zero for a direct one.
 *
 *  Occupancy is measured without the shared memory the launch will actually reserve, so the count is an
 *  upper bound on the blocks any later round can run - the one property an arena sized once needs.
 */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_bm25_rows_(sz_substrings_engine_t const *engine) {
    if (engine->needles_count <= (sz_u32_t)sz_substrings_cuda_tally_slots_k) return 0;
    return sz_substrings_cuda_multiprocessors_() *
           sz_substrings_cuda_resident_blocks_((void const *)sz_substrings_cuda_bm25_kernel_, 0);
}

/**
 *  @brief The tier-private head of the arena, which growth carries across and no kernel ever reads.
 *
 *  Host-readable because the arena is unified: the same allocator writes the vocabulary the host builder
 *  fills in place, so a block only the device could address would already have failed construction.
 */
typedef struct sz_substrings_cuda_head_t {
    /** The stream @c _init_gpu bound, and the only one a round enqueues on. */
    void *stream;
} sz_substrings_cuda_head_t;

/**
 *  @brief Byte offsets of the one arena every device round runs out of.
 *
 *  Everything past the boundaries is sized by the engine's budgets rather than by what a round discovers, so
 *  no launch waits on a count to reach the host first. Those two budget-sized halves are the whole reason
 *  the three readbacks that used to feed host allocations and grid dimensions are gone.
 */
typedef struct sz_substrings_cuda_arena_t {
    /** Offset of the tier-private head, which is always zero and is never memset by a round. */
    sz_size_t head;
    /** Offset of the round's report, which is the only thing a caller reads after its own join. */
    sz_size_t report;
    /** Offset of the corpus byte counter the first launch sums into. */
    sz_size_t corpus_bytes;
    /** Offset of this round's chunk width, derived on the device from the counter above. */
    sz_size_t chunk_bytes;
    /** Offset of the @b [haystacks + 1] exclusive chunk boundaries, per haystack. */
    sz_size_t chunk_offsets;
    /** Offset of the @b [haystacks + 1] boundaries of the reported matches, per haystack. */
    sz_size_t haystack_offsets;
    /** Offset of the @b [slots_count] per-chunk counts, which the scan turns into output offsets. */
    sz_size_t chunk_slots;
    /** Offset of the scratch the three-launch scan carries tile totals through. */
    sz_size_t tile_sums;
    /** Offset of the @b [matches_budget] emitted matches, before any cover thins them. */
    sz_size_t emitted;
    /** Offset of the cover's survivors, equal to @c emitted when no cover runs. */
    sz_size_t reported;
    /** Offset of the @b [matches_budget + 1] scanned keep flags, unallocated without a cover. */
    sz_size_t keep_offsets;
    /** Offset of the @b [matches_budget] rewrite drifts, unallocated without a cover. */
    sz_size_t gap_offsets;
    /** Offset of the hashed BM25 tally rows, unallocated for a vocabulary a block's own table holds. */
    sz_size_t overflow_rows;
    /** Entries of @c chunk_slots, which bounds the chunk count by construction rather than by hope. */
    sz_size_t slots_count;
    /** Rows at @c overflow_rows, which caps how many blocks a hashed BM25 launch may run. */
    sz_size_t overflow_count;
    /** Bytes the whole arena takes. */
    sz_size_t total;
} sz_substrings_cuda_arena_t;

/** Lays the device arena out for one round over @p haystacks_count texts. */
SZ_API_COMPTIME sz_substrings_cuda_arena_t sz_substrings_cuda_arena_(sz_substrings_engine_t const *engine,
                                                                     sz_size_t haystacks_count) {
    sz_size_t const boundaries = haystacks_count + 1;
    sz_size_t const matches = engine->matches_budget;
    sz_size_t const match_bytes = matches * sizeof(sz_substrings_match_t);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_substrings_cuda_arena_t arena;
    // A chunk is at least the corpus over the budget wide, so the corpus contributes at most `chunk_budget`
    // chunks and each haystack's own remainder at most one more.
    arena.slots_count = engine->chunk_budget + haystacks_count + 1;
    arena.overflow_count = sz_substrings_cuda_bm25_rows_(engine);
    arena.head = 0;
    arena.report = sizeof(sz_substrings_cuda_head_t);
    arena.corpus_bytes = arena.report + sizeof(sz_substrings_report_t);
    arena.chunk_bytes = arena.corpus_bytes + sizeof(sz_size_t);
    arena.chunk_offsets = arena.chunk_bytes + sizeof(sz_size_t);
    arena.haystack_offsets = arena.chunk_offsets + boundaries * sizeof(sz_size_t);
    arena.chunk_slots = arena.haystack_offsets + boundaries * sizeof(sz_size_t);
    arena.tile_sums = arena.chunk_slots + arena.slots_count * sizeof(sz_size_t);
    arena.emitted = arena.tile_sums + sz_substrings_cuda_scan_tiles_max_k * sizeof(sz_size_t);
    arena.reported = arena.emitted + match_bytes;
    arena.keep_offsets = arena.reported + (covering ? match_bytes : 0);
    arena.gap_offsets = arena.keep_offsets + (covering ? (matches + 1) * sizeof(sz_size_t) : 0);
    arena.overflow_rows = arena.gap_offsets + (covering ? matches * sizeof(sz_size_t) : 0);
    arena.total = arena.overflow_rows + arena.overflow_count * engine->needles_count * sizeof(sz_u32_t);
    return arena;
}

/** What one round's launches read out of the arena, every pointer of it device-resident. */
typedef struct sz_substrings_cuda_round_t {
    /** The corpus byte counter, summed by the first launch and read by no host code. */
    sz_size_t *corpus_bytes;
    /** This round's chunk width, derived on the device. */
    sz_size_t *chunk_bytes;
    /** The @b [haystacks + 1] exclusive chunk boundaries. */
    sz_size_t *chunk_offsets;
    /** The @b [haystacks + 1] boundaries of the reported matches, which a verb may redirect to its output. */
    sz_size_t *haystack_offsets;
    /** One slot per chunk: its match count from the sizing pass, then its exclusive output offset. */
    sz_size_t *chunk_slots;
    /** Scratch the three-launch scan carries tile totals through. */
    sz_size_t *tile_sums;
    /** Every emitted match, before any cover thins them. */
    sz_substrings_match_t *emitted;
    /** The matches a cover kept, which is @c emitted itself when no cover ran. */
    sz_substrings_match_t *reported;
    /** The scanned keep flags, or @c SZ_NULL when every match is reported. */
    sz_size_t *keep_offsets;
    /** One rewrite drift per reported match, or @c SZ_NULL when no rewrite can run. */
    sz_size_t *gap_offsets;
    /** Entries of @c chunk_slots, so the scan's grand total is its last one. */
    sz_size_t slots_count;
} sz_substrings_cuda_round_t;

/** Binds one round's pointers onto the engine's arena, which a compute verb does before it launches. */
SZ_API_COMPTIME void sz_substrings_cuda_round_bind_(sz_substrings_engine_t const *engine, sz_size_t haystacks_count,
                                                    sz_substrings_cuda_round_t *round) {
    sz_substrings_cuda_arena_t const arena = sz_substrings_cuda_arena_(engine, haystacks_count);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    round->corpus_bytes = (sz_size_t *)(block + arena.corpus_bytes);
    round->chunk_bytes = (sz_size_t *)(block + arena.chunk_bytes);
    round->chunk_offsets = (sz_size_t *)(block + arena.chunk_offsets);
    round->haystack_offsets = (sz_size_t *)(block + arena.haystack_offsets);
    round->chunk_slots = (sz_size_t *)(block + arena.chunk_slots);
    round->tile_sums = (sz_size_t *)(block + arena.tile_sums);
    round->emitted = (sz_substrings_match_t *)(block + arena.emitted);
    round->reported = (sz_substrings_match_t *)(block + arena.reported);
    round->keep_offsets = covering ? (sz_size_t *)(block + arena.keep_offsets) : SZ_NULL;
    round->gap_offsets = covering ? (sz_size_t *)(block + arena.gap_offsets) : SZ_NULL;
    round->slots_count = arena.slots_count;
}

/** The stream every round of this engine enqueues on, which lives in the arena's tier-private head. */
SZ_API_COMPTIME void *sz_substrings_cuda_stream_(sz_substrings_engine_t const *engine) {
    return ((sz_substrings_cuda_head_t *)engine->scratch)->stream;
}

/** Grows the engine's arena to hold one round over @p haystacks_count texts, and never shrinks it. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_arena_reserve_(sz_substrings_engine_t *engine,
                                                              sz_size_t haystacks_count) {
    sz_substrings_cuda_arena_t const arena = sz_substrings_cuda_arena_(engine, haystacks_count);
    sz_memory_allocator_t *const alloc = &engine->alloc;
    sz_substrings_cuda_head_t head;
    void *block;
    head.stream = SZ_NULL;
    if (engine->scratch && engine->scratch_bytes >= arena.total) return sz_success_k;
    if (engine->scratch) head = *(sz_substrings_cuda_head_t *)engine->scratch;
    block = alloc->allocate(arena.total, alloc->handle);
    if (!block) return sz_bad_alloc_k;
    if (!sz_memory_reaches_device(block)) {
        alloc->free(block, arena.total, alloc->handle);
        return sz_device_memory_mismatch_k;
    }
    if (engine->scratch) alloc->free(engine->scratch, engine->scratch_bytes, alloc->handle);
    *(sz_substrings_cuda_head_t *)block = head;
    engine->scratch = block, engine->scratch_bytes = arena.total;
    engine->report = (sz_substrings_report_t *)((sz_ptr_t)block + arena.report);
    return sz_success_k;
}

/** Zeroes everything a round reads before it writes: the report, the counters and the boundaries. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_arena_clear_(sz_substrings_engine_t const *engine,
                                                            sz_size_t haystacks_count, sz_bool_t covering,
                                                            void *stream) {
    sz_substrings_cuda_arena_t const arena = sz_substrings_cuda_arena_(engine, haystacks_count);
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    // The head of the arena only, so no round pays a memset proportional to a budget it did not spend.
    if (cudaMemsetAsync(block + arena.report, 0, arena.tile_sums - arena.report, (cudaStream_t)stream) != cudaSuccess)
        return sz_device_code_mismatch_k;
    if (!covering) return sz_success_k;
    // The cover writes only the flags below the emitted count, and the scan reads every one of them.
    return cudaMemsetAsync(block + arena.keep_offsets, 0, (engine->matches_budget + 1) * sizeof(sz_size_t),
                           (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Zeroes the terminator a scan reads as its own last element, so the total lands in it. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_clear_terminator_(sz_size_t *values, sz_size_t count, void *stream) {
    return cudaMemsetAsync(values + count, 0, sizeof(sz_size_t), (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/**
 *  @brief Walks every chunk and settles the cover, leaving @p round holding what the verbs read.
 *
 *  Every size this needs is either fixed at construction or derived on the device from one the host never
 *  sees, so the whole pass enqueues without a single join: the corpus total feeds a chunk width the device
 *  computes, the chunk count is bounded by the engine's own budget, and the emitted count reaches the later
 *  launches through @c engine->report rather than through a copy back.
 *
 *  @param[in] wanted Whether the caller reads the matches, since an overlapping count answers from the
 *             boundaries its sizing walk already scanned and never touches the match arena at all.
 *  @param[out] haystack_offsets Where the per-haystack boundaries land, or @c SZ_NULL to leave them in the
 *              arena; @ref sz_substrings_find points this straight at its own output.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_walk_(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_substrings_cuda_matches_t wanted,
                                                     sz_size_t *haystack_offsets, sz_substrings_cuda_round_t *round,
                                                     void *stream) {
    sz_u32_t staged_rows = sz_substrings_cuda_walk_rows_(engine);
    sz_size_t const staged_bytes = (sz_size_t)staged_rows * engine->classes_count * sizeof(sz_u32_t);
    sz_size_t boundaries = haystacks->count + 1;
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    // A cover is decided between matches, so counting one costs what finding one costs; only an
    // overlapping count can answer from the boundaries its sizing walk already scanned.
    sz_bool_t emitting = (sz_bool_t)(covering || wanted == sz_substrings_cuda_matches_needed_k);
    sz_size_t chunk_floor = sz_substrings_cuda_chunk_floor_(engine);
    sz_size_t chunk_budget = engine->chunk_budget, matches_budget = engine->matches_budget;
    sz_size_t longest = engine->max_source_match_bytes;
    sz_substrings_overlap_policy_t overlap_policy = engine->overlap_policy;
    sz_substrings_report_t *report;
    sz_sequence_t launched_haystacks = *haystacks;
    sz_substrings_engine_t launched_engine;
    sz_substrings_cuda_pass_t walk_pass = sz_substrings_cuda_sizing_k;
    sz_size_t *emitted_at, *kept_at;
    void *arguments[9];
    sz_status_t status = sz_substrings_cuda_arena_reserve_(engine, haystacks->count);
    if (status != sz_success_k) return status;
    sz_substrings_cuda_round_bind_(engine, haystacks->count, round);
    if (haystack_offsets) round->haystack_offsets = haystack_offsets;
    report = engine->report, launched_engine = *engine;
    emitted_at = round->chunk_slots + round->slots_count - 1;
    kept_at = round->keep_offsets ? round->keep_offsets + matches_budget : SZ_NULL;
    status = sz_substrings_cuda_arena_clear_(engine, haystacks->count, covering, stream);

    // The corpus total and the width it implies, both device-side, which is what removes the first readback.
    if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &round->corpus_bytes;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_total_bytes_kernel_,
                                            sz_substrings_cuda_grid_(haystacks->count), arguments, 0, stream);
    }
    if (status == sz_success_k) {
        arguments[0] = &round->corpus_bytes, arguments[1] = &chunk_budget, arguments[2] = &chunk_floor;
        arguments[3] = &round->chunk_bytes;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_chunk_bytes_kernel_, 1, arguments, 0,
                                            stream);
    }

    // Round one: how many chunks each haystack owns, scanned into the range that haystack's chunks take.
    if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &round->chunk_bytes, arguments[2] = &round->chunk_offsets;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_chunk_counts_kernel_,
                                            sz_substrings_cuda_grid_(haystacks->count), arguments, 0, stream);
    }
    if (status == sz_success_k)
        status = sz_substrings_cuda_scan_(round->chunk_offsets, boundaries, round->tile_sums, stream);

    // Round two: the sizing walk, launched against the chunk budget rather than a discovered chunk count.
    if (status == sz_success_k) {
        arguments[0] = &launched_engine, arguments[1] = &staged_rows, arguments[2] = &launched_haystacks;
        arguments[3] = &round->chunk_offsets, arguments[4] = &round->chunk_bytes, arguments[5] = &report;
        arguments[6] = &round->chunk_slots, arguments[7] = &round->emitted, arguments[8] = &walk_pass;
        status = sz_substrings_cuda_launch_(
            (void const *)sz_substrings_cuda_walk_kernel_,
            sz_substrings_cuda_grid_for_((void const *)sz_substrings_cuda_walk_kernel_, staged_bytes,
                                         round->slots_count),
            arguments, staged_bytes, stream);
    }
    if (status == sz_success_k)
        status = sz_substrings_cuda_scan_(round->chunk_slots, round->slots_count, round->tile_sums, stream);
    if (status == sz_success_k) {
        arguments[0] = &emitted_at, arguments[1] = &matches_budget, arguments[2] = &emitting, arguments[3] = &report;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_sized_kernel_, 1, arguments, 0, stream);
    }

    // Round three: the writing walk, the cover over what it wrote, and the per-haystack boundaries both
    // feed. Each retires at its first instruction when the sizing walk outran the budget.
    if (status == sz_success_k && emitting) {
        walk_pass = sz_substrings_cuda_writing_k;
        arguments[0] = &launched_engine, arguments[1] = &staged_rows, arguments[2] = &launched_haystacks;
        arguments[3] = &round->chunk_offsets, arguments[4] = &round->chunk_bytes, arguments[5] = &report;
        arguments[6] = &round->chunk_slots, arguments[7] = &round->emitted, arguments[8] = &walk_pass;
        status = sz_substrings_cuda_launch_(
            (void const *)sz_substrings_cuda_walk_kernel_,
            sz_substrings_cuda_grid_for_((void const *)sz_substrings_cuda_walk_kernel_, staged_bytes,
                                         round->slots_count),
            arguments, staged_bytes, stream);
    }
    if (status == sz_success_k && covering) {
        arguments[0] = &round->emitted, arguments[1] = &report, arguments[2] = &longest;
        arguments[3] = &overlap_policy, arguments[4] = &round->keep_offsets;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_cover_kernel_,
                                            sz_substrings_cuda_grid_(matches_budget), arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_substrings_cuda_scan_(round->keep_offsets, matches_budget + 1, round->tile_sums, stream);
        if (status == sz_success_k) {
            arguments[0] = &round->emitted, arguments[1] = &report;
            arguments[2] = &round->keep_offsets, arguments[3] = &round->reported;
            status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_compact_kernel_,
                                                sz_substrings_cuda_grid_(matches_budget), arguments, 0, stream);
        }
        if (status == sz_success_k) {
            arguments[0] = &kept_at, arguments[1] = &report;
            status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_covered_kernel_, 1, arguments, 0,
                                                stream);
        }
    }
    if (status == sz_success_k) {
        sz_size_t *keep_offsets = round->keep_offsets;
        arguments[0] = &round->chunk_offsets, arguments[1] = &round->chunk_slots, arguments[2] = &keep_offsets;
        arguments[3] = &report, arguments[4] = &round->haystack_offsets, arguments[5] = &boundaries;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_haystack_offsets_kernel_,
                                            sz_substrings_cuda_grid_(boundaries), arguments, 0, stream);
    }
    return status;
}

/** Whether every argument the device verbs read or write is one a kernel can address. */
SZ_API_COMPTIME sz_bool_t sz_substrings_cuda_resident_(sz_substrings_engine_t const *engine,
                                                       sz_sequence_t const *haystacks) {
    // An array the kernel dereferences, rather than the owning handle it never touches, so a caller
    // holding a borrowed view of a resident engine is not refused for a null owner.
    if (!sz_memory_reaches_device(engine->base)) return sz_false_k;
    // The handle is checked, never the accessors: those are the device's to call, so the host must not,
    // and a pointer is all this side can inspect. That the texts they answer are device-reachable is the
    // caller's word, which `_init_gpu` makes easy to keep by handing back a unified allocator.
    return sz_memory_reaches_device(haystacks->handle);
}

#pragma endregion Host Plumbing

#pragma region Construction

/** Matches one round may emit when a caller names no budget, which is 32 MB of match arena. */
enum { sz_substrings_cuda_matches_budget_default_k = 1u << 20 };

SZ_API_COMPTIME sz_status_t sz_substrings_engine_init_cuda(sz_sequence_t const *needles,
                                                           sz_substrings_case_sensitivity_t case_sensitivity,
                                                           sz_substrings_overlap_policy_t overlap_policy,
                                                           sz_size_t hot_states, sz_size_t matches_budget,
                                                           sz_memory_allocator_t *alloc,
                                                           void *stream, sz_substrings_engine_t *engine) {
    sz_memory_allocator_t unified;
    sz_size_t staged_bytes;
    sz_status_t status;
    if (!alloc) {
        sz_memory_allocator_init_unified(&unified, SZ_NULL);
        alloc = &unified;
    }
    if (!matches_budget) matches_budget = (sz_size_t)sz_substrings_cuda_matches_budget_default_k;
    status = sz_substrings_engine_compile_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                       (sz_capability_t)sz_caps_cuda_k, alloc, engine);
    if (status != sz_success_k) return status;
    // The host builder writes the block in place, so a device-only allocation cannot serve as the vocabulary.
    if (!sz_memory_reaches_device(engine->memory)) {
        sz_substrings_engine_free_(engine);
        return sz_device_memory_mismatch_k;
    }
    staged_bytes = (sz_size_t)sz_substrings_cuda_walk_rows_(engine) * engine->classes_count * sizeof(sz_u32_t);
    engine->chunk_budget = sz_substrings_cuda_resident_threads_((void const *)sz_substrings_cuda_walk_kernel_,
                                                                staged_bytes);
    status = sz_substrings_cuda_arena_reserve_(engine, 0);
    if (status != sz_success_k) {
        sz_substrings_engine_free_(engine);
        return status;
    }
    ((sz_substrings_cuda_head_t *)engine->scratch)->stream = stream;
    return sz_success_k;
}

#pragma endregion Construction

#pragma region CUDA Backends

SZ_API_COMPTIME sz_status_t sz_substrings_counts_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_size_t *counts, sz_size_t counts_stride) {
    sz_substrings_cuda_round_t round;
    sz_size_t haystacks_count = haystacks->count;
    void *stream = sz_substrings_cuda_stream_(engine);
    void *arguments[4];
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(counts)) return sz_device_memory_mismatch_k;

    status = sz_substrings_cuda_walk_(engine, haystacks, sz_substrings_cuda_matches_unneeded_k, SZ_NULL, &round,
                                      stream);
    if (status != sz_success_k) return status;

    arguments[0] = &round.haystack_offsets, arguments[1] = &counts, arguments[2] = &counts_stride;
    arguments[3] = &haystacks_count;
    return sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_counts_kernel_,
                                      sz_substrings_cuda_grid_(haystacks_count), arguments, 0, stream);
}

SZ_API_COMPTIME sz_status_t sz_substrings_find_cuda(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                    sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                    sz_size_t *matches_offsets) {
    sz_substrings_cuda_round_t round;
    sz_substrings_report_t *report;
    void *stream = sz_substrings_cuda_stream_(engine);
    void *arguments[4];
    sz_status_t status;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(matches_offsets)) return sz_device_memory_mismatch_k;
    if (matches_capacity && !sz_memory_reaches_device(matches)) return sz_device_memory_mismatch_k;
    // The offsets kernel retires when the matches did not fit, so the caller's array is zeroed rather than
    // left holding whatever it held before.
    if (cudaMemsetAsync(matches_offsets, 0, (haystacks->count + 1) * sizeof(sz_size_t), (cudaStream_t)stream) !=
        cudaSuccess)
        return sz_device_code_mismatch_k;

    // The boundaries land straight in the caller's array, which is the same shape a rewrite already takes.
    status = sz_substrings_cuda_walk_(engine, haystacks, sz_substrings_cuda_matches_needed_k, matches_offsets, &round,
                                      stream);
    if (status != sz_success_k) return status;

    // The survivor count lives on the device, so the clip at the capacity is a kernel rather than a copy.
    report = engine->report;
    arguments[0] = &round.reported, arguments[1] = &report, arguments[2] = &matches, arguments[3] = &matches_capacity;
    return sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_store_matches_kernel_,
                                      sz_substrings_cuda_grid_(matches_capacity), arguments, 0, stream);
}

SZ_API_COMPTIME sz_status_t sz_substrings_replace_cuda(sz_substrings_engine_t *engine,
                                                       sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements, sz_ptr_t tape,
                                                       sz_size_t tape_capacity, sz_size_t *offsets) {
    sz_substrings_cuda_round_t round;
    sz_substrings_report_t *report;
    sz_sequence_t launched_haystacks, launched_replacements;
    sz_size_t boundaries = haystacks->count + 1;
    sz_size_t tape_ceiling = tape_capacity;
    sz_size_t *rewritten_at;
    void *stream = sz_substrings_cuda_stream_(engine);
    void *arguments[8];
    sz_status_t status;
    // A substitution over matches that share bytes is not a function, so there is no cover to apply.
    if (engine->overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != engine->needles_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(replacements->handle)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(offsets)) return sz_device_memory_mismatch_k;
    if (tape_capacity && !sz_memory_reaches_device(tape)) return sz_device_memory_mismatch_k;

    status = sz_substrings_cuda_walk_(engine, haystacks, sz_substrings_cuda_matches_needed_k, SZ_NULL, &round, stream);
    if (status != sz_success_k) return status;
    // The offsets kernel retires when the matches did not fit, so the caller's array is zeroed rather than
    // left holding whatever it held before.
    if (cudaMemsetAsync(offsets, 0, boundaries * sizeof(sz_size_t), (cudaStream_t)stream) != cudaSuccess)
        return sz_device_code_mismatch_k;

    report = engine->report;
    rewritten_at = offsets + haystacks->count;
    launched_haystacks = *haystacks, launched_replacements = *replacements;

    // One block per haystack, so the drift scan a rewrite needs stays inside one block's carry.
    arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
    arguments[2] = &round.haystack_offsets, arguments[3] = &round.reported, arguments[4] = &report;
    arguments[5] = &round.gap_offsets, arguments[6] = &offsets;
    status = sz_substrings_cuda_launch_(
        (void const *)sz_substrings_cuda_rewrite_offsets_kernel_,
        sz_substrings_cuda_grid_(haystacks->count * sz_substrings_cuda_threads_per_block_k), arguments, 0, stream);
    if (status == sz_success_k) status = sz_substrings_cuda_scan_(offsets, boundaries, round.tile_sums, stream);
    if (status == sz_success_k) {
        arguments[0] = &rewritten_at, arguments[1] = &tape_ceiling, arguments[2] = &report;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_tape_kernel_, 1, arguments, 0, stream);
    }
    // The copy's grid comes from the caller's own capacity, which is the last host number a rewrite needs.
    if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
        arguments[2] = &round.haystack_offsets, arguments[3] = &round.reported, arguments[4] = &round.gap_offsets;
        arguments[5] = &offsets, arguments[6] = &report, arguments[7] = &tape;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_rewrite_copy_kernel_,
                                            sz_substrings_cuda_grid_(tape_ceiling), arguments, 0, stream);
    }
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_engine_t *engine,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                           sz_size_t scores_stride) {
    void const *const kernel = (void const *)sz_substrings_cuda_bm25_kernel_;
    sz_substrings_cuda_arena_t const arena = sz_substrings_cuda_arena_(engine, haystacks->count);
    sz_size_t const needles_count = engine->needles_count;
    sz_size_t const table_slots = sz_substrings_cuda_tally_slots_for_(needles_count);
    sz_substrings_cuda_tally_layout_t const layout = needles_count <= sz_substrings_cuda_tally_slots_k
                                                         ? sz_substrings_cuda_tally_direct_k
                                                         : sz_substrings_cuda_tally_hashed_k;
    sz_size_t const table_bytes = (layout == sz_substrings_cuda_tally_hashed_k ? 2 : 1) * table_slots *
                                  sizeof(sz_u32_t);
    sz_size_t const accepts_bytes = sz_substrings_cuda_accepts_words_(engine) * sizeof(sz_u32_t);
    sz_substrings_engine_t engine_copy;
    sz_sequence_t haystacks_copy = *haystacks;
    sz_substrings_bm25_t parameters_copy;
    sz_u32_t *overflow_rows = SZ_NULL;
    sz_size_t reserved_bytes = table_bytes, shared_bytes, blocks;
    sz_u32_t staged_accepts_words = 0, staged_count;
    void *stream = sz_substrings_cuda_stream_(engine);
    int ceiling = 0;
    void *arguments[10];
    sz_status_t status = sz_substrings_bm25_check(parameters, needle_weights);
    if (status != sz_success_k) return status;
    if (!scores_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(engine, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(scores)) return sz_device_memory_mismatch_k;
    if (needles_count && !sz_memory_reaches_device(needle_weights)) return sz_device_memory_mismatch_k;
    if (document_lengths && !sz_memory_reaches_device(document_lengths)) return sz_device_memory_mismatch_k;
    status = sz_substrings_cuda_arena_reserve_(engine, haystacks->count);
    if (status != sz_success_k) return status;
    engine_copy = *engine;

    // The acceptance bitmap is read on every step, so it is staged whenever it fits without costing a block.
    cudaDeviceGetAttribute(&ceiling, cudaDevAttrMaxSharedMemoryPerBlock, sz_substrings_cuda_device_());
    if (table_bytes + accepts_bytes <= (sz_size_t)ceiling &&
        sz_substrings_cuda_resident_blocks_(kernel, table_bytes + accepts_bytes) >=
            sz_substrings_cuda_resident_blocks_(kernel, table_bytes))
        staged_accepts_words = (sz_u32_t)(accepts_bytes / sizeof(sz_u32_t)), reserved_bytes += accepts_bytes;
    staged_count = sz_substrings_cuda_staged_rows_(engine, kernel, reserved_bytes,
                                                   sz_substrings_cuda_stage_prefix_k);
    shared_bytes = reserved_bytes + (sz_size_t)staged_count * engine->classes_count * sizeof(sz_u32_t);
    blocks = sz_min_of_two(haystacks->count, sz_substrings_cuda_multiprocessors_() *
                                                 sz_substrings_cuda_resident_blocks_(kernel, shared_bytes));

    // A hashed tally spills into one arena row per block, so the block count is capped by the rows the
    // arena was sized for rather than by what the device happens to have free at this moment.
    if (layout == sz_substrings_cuda_tally_hashed_k) {
        sz_size_t const overflow_bytes = arena.overflow_count * needles_count * sizeof(sz_u32_t);
        blocks = sz_max_of_two(sz_min_of_two(blocks, arena.overflow_count), (sz_size_t)1);
        overflow_rows = (sz_u32_t *)((sz_ptr_t)engine->scratch + arena.overflow_rows);
        if (cudaMemsetAsync(overflow_rows, 0, overflow_bytes, (cudaStream_t)stream) != cudaSuccess)
            return sz_device_code_mismatch_k;
    }
    if (cudaMemsetAsync(engine->report, 0, sizeof(sz_substrings_report_t), (cudaStream_t)stream) != cudaSuccess)
        return sz_device_code_mismatch_k;

    parameters_copy = *parameters;
    arguments[0] = &engine_copy, arguments[1] = &haystacks_copy, arguments[2] = &document_lengths;
    arguments[3] = &parameters_copy, arguments[4] = &needle_weights, arguments[5] = &overflow_rows;
    arguments[6] = &scores, arguments[7] = &scores_stride, arguments[8] = &staged_accepts_words;
    arguments[9] = &staged_count;
    return sz_substrings_cuda_launch_(kernel, (unsigned)blocks, arguments, shared_bytes, stream);
}

#pragma endregion CUDA Backends

#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_SUBSTRINGS_CUDA_CUH_
