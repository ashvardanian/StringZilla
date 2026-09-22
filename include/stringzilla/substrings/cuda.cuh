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

/** Largest chunk, so a single long document still spreads across the device rather than across one warp. */
enum { sz_substrings_cuda_chunk_bytes_max_k = 65536 };

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
    /** In shared memory: each hashed slot's needle index plus one, zero while free; unread when direct. */
    sz_u32_t *keys;
    /** In shared memory: each slot's occurrences. */
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
SZ_DEVICE_INLINE sz_u32_t sz_substrings_cuda_step_(sz_substrings_automaton_t const *automaton,
                                                   sz_u32_t const *staged_rows, sz_u32_t staged_count,
                                                   sz_u32_t state, sz_u8_t byte) {
    if (state < staged_count) return staged_rows[(sz_size_t)state * (SZ_U8_MAX + 1) + byte];
    return sz_substrings_step(automaton, state, byte);
}

/** Cooperatively fills @p staged_rows from the head of the hot tier, once per block. The hot tier's
 *  out-degree ordering makes its head the best prefix to stage. */
SZ_DEVICE_INLINE void sz_substrings_cuda_stage_(sz_substrings_automaton_t const *automaton, sz_u32_t *staged_rows,
                                                sz_u32_t staged_count) {
    sz_size_t const cells = (sz_size_t)staged_count * (SZ_U8_MAX + 1);
    sz_size_t cell;
    for (cell = threadIdx.x; cell < cells; cell += blockDim.x) staged_rows[cell] = automaton->hot_rows[cell];
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
    sz_u32_t slot, probe;
    if (tally->layout == sz_substrings_cuda_tally_direct_k) {
        atomicAdd(tally->counts + needle, 1u);
        return;
    }
    slot = (sz_u32_t)(((sz_u64_t)key * 0x9E3779B97F4A7C15ull) >> (64 - sz_substrings_cuda_tally_slot_bits_k));
    for (probe = 0; probe != sz_substrings_cuda_tally_probes_k; ++probe) {
        sz_u32_t const seated = atomicCAS(tally->keys + slot, 0u, key);
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
                                                             sz_size_t elements_per_tile,
                                                             sz_size_t const *tile_sums) {
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

/** Writes how many chunks each haystack is cut into, which the scan then turns into its chunk range. */
static __global__ void sz_substrings_cuda_chunk_counts_kernel_(sz_sequence_t haystacks, sz_size_t chunk_bytes,
                                                               sz_size_t *chunk_offsets) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < haystacks.count; index += stride)
        chunk_offsets[index] =
            sz_substrings_cuda_chunks_for_(haystacks.get_length(haystacks.handle, index), chunk_bytes);
}

/**
 *  @brief Reports every match ending at @p delta, writing them when @p pass asks, and returns how many.
 *
 *  The acceptance bit answers "does anything end here" without touching the counts array, which at scale
 *  costs nearly as much as the tape read itself.
 */
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_emit_(sz_substrings_automaton_t const *automaton, sz_u32_t state,
                                                    sz_size_t walk_begin, sz_u32_t delta,
                                                    sz_size_t haystack_index, sz_substrings_cuda_pass_t pass,
                                                    sz_substrings_match_t *matches_out,
                                                    sz_substrings_cuda_tally_t const *tally) {
    sz_size_t output_offset, found = 0, index;
    sz_u32_t output_count;
    if (!sz_substrings_accepts(automaton, state)) return 0;
    output_count = automaton->outputs_counts[state];
    output_offset = automaton->outputs_offsets[state];
    for (index = 0; index != output_count; ++index) {
        sz_substrings_output_t const output = automaton->outputs[output_offset + index];
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
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_walk_chunk_cased_(sz_substrings_automaton_t const *automaton,
                                                                sz_u32_t const *staged_rows, sz_u32_t staged_count,
                                                                sz_cptr_t haystack, sz_size_t length,
                                                                sz_size_t chunk_begin, sz_size_t chunk_end,
                                                                sz_size_t haystack_index,
                                                                sz_substrings_cuda_pass_t pass,
                                                                sz_substrings_match_t *matches_at_chunk,
                                                                sz_substrings_cuda_tally_t const *tally) {
    sz_size_t const warm_up = automaton->max_source_match_bytes > 0
                                  ? (sz_size_t)automaton->max_source_match_bytes - 1
                                  : 0;
    sz_size_t const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const walk_base = (sz_u8_t const *)haystack + walk_begin;
    // Every 64-bit quantity is resolved here, once; the per-byte loops below ride 32-bit deltas from it.
    sz_u32_t const walk_span = (sz_u32_t)(chunk_end - walk_begin);
    sz_u32_t const emit_from = (sz_u32_t)(chunk_begin - walk_begin);
    sz_size_t found = 0;
    sz_u32_t delta = 0, lane;
    sz_u32_t state = automaton->root; // ? Fresh at `walk_begin`; no state crosses a haystack boundary.
    sz_unused_(length);

    for (; delta < emit_from; ++delta)
        state = sz_substrings_cuda_step_(automaton, staged_rows, staged_count, state, walk_base[delta]);

    // Peeled to the load's own alignment, so the body pays one four-byte load per four transitions; the
    // transition chain stays strictly serial, only the tape reads widen.
    for (; delta < walk_span && (((sz_size_t)(walk_base + delta)) & 3u) != 0; ++delta) {
        state = sz_substrings_cuda_step_(automaton, staged_rows, staged_count, state, walk_base[delta]);
        found += sz_substrings_cuda_emit_(automaton, state, walk_begin, delta, haystack_index, pass,
                                          matches_at_chunk + found, tally);
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        sz_u32_t const quad = sz_substrings_cuda_load_quad_(walk_base + delta);
#pragma unroll
        for (lane = 0; lane != 4; ++lane) {
            state = sz_substrings_cuda_step_(automaton, staged_rows, staged_count, state,
                                             (sz_u8_t)(quad >> (lane * 8)));
            found += sz_substrings_cuda_emit_(automaton, state, walk_begin, delta + lane, haystack_index, pass,
                                              matches_at_chunk + found, tally);
        }
    }
    for (; delta < walk_span; ++delta) {
        state = sz_substrings_cuda_step_(automaton, staged_rows, staged_count, state, walk_base[delta]);
        found += sz_substrings_cuda_emit_(automaton, state, walk_begin, delta, haystack_index, pass,
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
SZ_DEVICE_INLINE sz_size_t sz_substrings_cuda_walk_chunk_uncased_(sz_substrings_automaton_t const *automaton,
                                                                  sz_u32_t const *staged_rows, sz_u32_t staged_count,
                                                                  sz_cptr_t haystack, sz_size_t length,
                                                                  sz_size_t chunk_begin, sz_size_t chunk_end,
                                                                  sz_size_t haystack_index,
                                                                  sz_substrings_cuda_pass_t pass,
                                                                  sz_substrings_match_t *matches_at_chunk,
                                                                  sz_substrings_cuda_tally_t const *tally) {
    sz_size_t const warm_up = automaton->max_source_match_bytes > 0
                                  ? (sz_size_t)automaton->max_source_match_bytes - 1
                                  : 0;
    sz_size_t walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_substrings_folded_cursor_t cursor;
    sz_substrings_folded_byte_t step;
    sz_u32_t state = automaton->root;
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
            state = automaton->root;
            continue;
        }
        state = sz_substrings_cuda_step_(automaton, staged_rows, staged_count, state, step.byte);
        if (!step.rune_end) continue;
        if (step.breaks_boundary) last_break_folded_end = folded + step.trailing;

        source_end = walk_begin + step.codepoint_end;
        if (source_end > chunk_end) break;   // ? Past this chunk's share; the next chunk owns these ends.
        if (!sz_substrings_accepts(automaton, state)) continue;
        if (source_end <= chunk_begin) continue; // ? Still warming up, where this chunk reports nothing.

        output_count = automaton->outputs_counts[state];
        output_offset = automaton->outputs_offsets[state];
        for (index = 0; index != output_count; ++index) {
            sz_substrings_output_t const output = automaton->outputs[output_offset + index];
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
static __global__ void sz_substrings_cuda_walk_kernel_(sz_substrings_automaton_t automaton, sz_u32_t staged_count,
                                                       sz_sequence_t haystacks, sz_size_t const *chunk_offsets,
                                                       sz_size_t chunk_bytes, sz_size_t chunk_count,
                                                       sz_size_t *chunk_slots, sz_substrings_match_t *matches,
                                                       sz_substrings_cuda_pass_t pass) {
    extern __shared__ sz_u32_t sz_substrings_cuda_staged_[];
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk_index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_substrings_cuda_stage_(&automaton, sz_substrings_cuda_staged_, staged_count);

    for (; chunk_index < chunk_count; chunk_index += stride) {
        sz_size_t const haystack_index =
            sz_substrings_cuda_haystack_of_(chunk_offsets, haystacks.count, chunk_index);
        sz_cptr_t const haystack = haystacks.get_start(haystacks.handle, haystack_index);
        sz_size_t const length = haystacks.get_length(haystacks.handle, haystack_index);
        sz_size_t const local_index = chunk_index - chunk_offsets[haystack_index];
        sz_size_t const chunk_begin = local_index * chunk_bytes;
        sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
        sz_substrings_match_t *const matches_at_chunk =
            pass == sz_substrings_cuda_writing_k ? matches + chunk_slots[chunk_index] : matches;
        sz_size_t found;
        // One vocabulary is byte-exact or folded for its whole lifetime, so every thread takes the same side
        // and the branch costs no divergence.
        if (chunk_begin >= chunk_end && length != 0) found = 0;
        else if (automaton.case_sensitivity == sz_substrings_uncased_k)
            found = sz_substrings_cuda_walk_chunk_uncased_(&automaton, sz_substrings_cuda_staged_, staged_count,
                                                           haystack, length, chunk_begin, chunk_end, haystack_index,
                                                           pass, matches_at_chunk, SZ_NULL);
        else
            found = sz_substrings_cuda_walk_chunk_cased_(&automaton, sz_substrings_cuda_staged_, staged_count,
                                                         haystack, length, chunk_begin, chunk_end, haystack_index,
                                                         pass, matches_at_chunk, SZ_NULL);
        if (pass == sz_substrings_cuda_sizing_k) chunk_slots[chunk_index] = found;
    }
}

#pragma endregion Walk Kernels

#pragma region Scoring Kernels

/** Fixed-point scale of a BM25 sum: integer addition commutes, so the score never depends on thread order. */
#define SZ_SUBSTRINGS_CUDA_BM25_SCALE (4294967296.0)

/**
 *  @brief Scores one haystack per block: its threads walk contiguous chunks into one shared tally, then
 *         sum the tallied terms in fixed point.
 *
 *  A block rather than a grid per haystack keeps the tally in shared memory, at the price of one long
 *  document spreading across one block's threads only.
 */
static __global__ void sz_substrings_cuda_bm25_kernel_(sz_substrings_automaton_t automaton, sz_sequence_t haystacks,
                                                       sz_f32_t const *document_lengths,
                                                       sz_substrings_bm25_t parameters,
                                                       sz_f32_t const *needle_weights, sz_u32_t *overflow_rows,
                                                       sz_f32_t *scores) {
    __shared__ sz_u32_t keys[sz_substrings_cuda_tally_slots_k];
    __shared__ sz_u32_t counts[sz_substrings_cuda_tally_slots_k];
    __shared__ sz_u32_t overflowed;
    __shared__ unsigned long long block_sum;
    sz_size_t const needles_count = automaton.needles_count;
    sz_size_t const warm_up = sz_max_of_two((sz_size_t)automaton.max_source_match_bytes, (sz_size_t)1);
    sz_substrings_cuda_tally_t tally;
    sz_size_t haystack_index, slot;
    tally.layout = needles_count <= sz_substrings_cuda_tally_slots_k ? sz_substrings_cuda_tally_direct_k
                                                                     : sz_substrings_cuda_tally_hashed_k;
    tally.keys = keys, tally.counts = counts, tally.overflowed = &overflowed;
    tally.overflow = overflow_rows ? overflow_rows + (sz_size_t)blockIdx.x * needles_count : SZ_NULL;
    for (slot = threadIdx.x; slot < sz_substrings_cuda_tally_slots_k; slot += blockDim.x)
        keys[slot] = 0, counts[slot] = 0;
    if (threadIdx.x == 0) overflowed = 0;
    __syncthreads();

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
            if (automaton.case_sensitivity == sz_substrings_uncased_k)
                sz_substrings_cuda_walk_chunk_uncased_(&automaton, SZ_NULL, 0, haystack, length, chunk_begin,
                                                       chunk_end, haystack_index, sz_substrings_cuda_tallying_k,
                                                       SZ_NULL, &tally);
            else
                sz_substrings_cuda_walk_chunk_cased_(&automaton, SZ_NULL, 0, haystack, length, chunk_begin,
                                                     chunk_end, haystack_index, sz_substrings_cuda_tallying_k,
                                                     SZ_NULL, &tally);
        }
        __syncthreads();

        // Every slot and overflow entry is zeroed as it is read, so the next haystack starts from a clean tally.
        for (slot = threadIdx.x; slot < sz_substrings_cuda_tally_slots_k; slot += blockDim.x) {
            sz_u32_t const frequency = counts[slot];
            if (!frequency) continue;
            needle = tally.layout == sz_substrings_cuda_tally_direct_k ? slot : (sz_size_t)keys[slot] - 1;
            mine += __double2ll_rn(sz_substrings_bm25_term(&parameters, norm, needle_weights[needle], frequency) *
                                   SZ_SUBSTRINGS_CUDA_BM25_SCALE);
            keys[slot] = 0, counts[slot] = 0;
        }
        if (overflowed)
            for (needle = threadIdx.x; needle < needles_count; needle += blockDim.x) {
                sz_u32_t const frequency = tally.overflow[needle];
                if (!frequency) continue;
                mine += __double2ll_rn(sz_substrings_bm25_term(&parameters, norm, needle_weights[needle],
                                                               frequency) *
                                       SZ_SUBSTRINGS_CUDA_BM25_SCALE);
                tally.overflow[needle] = 0;
            }
        // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
        if (mine) atomicAdd(&block_sum, (unsigned long long)mine);
        __syncthreads();
        if (threadIdx.x == 0) {
            scores[haystack_index] = (sz_f32_t)((sz_f64_t)(sz_i64_t)block_sum / SZ_SUBSTRINGS_CUDA_BM25_SCALE);
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
static __global__ void sz_substrings_cuda_cover_kernel_(sz_substrings_match_t const *matches, sz_size_t count,
                                                        sz_size_t longest, sz_substrings_overlap_policy_t policy,
                                                        sz_size_t *keep) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
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
static __global__ void sz_substrings_cuda_compact_kernel_(sz_substrings_match_t const *matches, sz_size_t count,
                                                          sz_size_t const *keep_offsets,
                                                          sz_substrings_match_t *survivors) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < count; index += stride)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
static __global__ void sz_substrings_cuda_haystack_offsets_kernel_(sz_size_t const *chunk_offsets,
                                                                   sz_size_t const *chunk_slots,
                                                                   sz_size_t const *keep_offsets,
                                                                   sz_size_t *haystack_offsets, sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < count; index += stride) {
        sz_size_t const emitted_before = chunk_slots[chunk_offsets[index]];
        haystack_offsets[index] = keep_offsets ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
static __global__ void sz_substrings_cuda_counts_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts,
                                                         sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < count; index += stride) counts[index] = haystack_offsets[index + 1] - haystack_offsets[index];
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
static __global__ void sz_substrings_cuda_rewrite_offsets_kernel_(sz_sequence_t haystacks,
                                                                  sz_sequence_t replacements,
                                                                  sz_size_t const *haystack_offsets,
                                                                  sz_substrings_match_t const *matches,
                                                                  sz_size_t *gap_offsets, sz_size_t *output_sizes) {
    __shared__ sz_size_t shared[sz_substrings_cuda_threads_per_block_k];
    __shared__ sz_size_t drift_carry;
    sz_size_t haystack_index;
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
                previous_end = match_index == first ? 0
                                                    : matches[match_index - 1].byte_offset +
                                                          matches[match_index - 1].byte_length;
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
                                                               sz_size_t const *output_offsets, sz_ptr_t output) {
    sz_size_t const output_bytes_total = output_offsets[haystacks.count];
    sz_size_t const tile_count = (output_bytes_total + sz_substrings_cuda_rewrite_tile_bytes_k - 1) /
                                 sz_substrings_cuda_rewrite_tile_bytes_k;
    unsigned const warp_index = threadIdx.x / 32u, warps_per_block = blockDim.x / 32u, lane = threadIdx.x % 32u;
    sz_size_t tile_index;

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
                sz_size_t const previous_end = match_index == first
                                                   ? 0
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
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                               sz_substrings_cuda_device_()) != cudaSuccess)
        return 1;
    return (sz_size_t)sz_max_of_two(multiprocessors, 1);
}

/** Blocks of @p kernel this device holds resident per multiprocessor at @p shared_bytes of dynamic shared. */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_resident_blocks_(void const *kernel, sz_size_t shared_bytes) {
    int blocks_per_multiprocessor = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, kernel,
                                                      sz_substrings_cuda_threads_per_block_k,
                                                      shared_bytes) != cudaSuccess)
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
    sz_size_t const covering = sz_substrings_cuda_multiprocessors_() *
                               sz_substrings_cuda_blocks_per_multiprocessor_k;
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
    return sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_scan_apply_kernel_, (unsigned)tiles,
                                      arguments, 0, stream);
}

/** Reads one device-resident @c sz_size_t back, whether the block is managed or plain device memory. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_read_(sz_size_t const *device_value, sz_size_t *host_value,
                                                     void *stream) {
    if (cudaMemcpyAsync(host_value, device_value, sizeof(sz_size_t), cudaMemcpyDeviceToHost,
                        (cudaStream_t)stream) != cudaSuccess)
        return sz_device_code_mismatch_k;
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/**
 *  @brief Hot rows one block stages: all of them, or none.
 *
 *  A partial prefix pays the copy and a per-byte bounds test while most transitions miss it, and its shared
 *  memory comes out of residency, so the tier is staged only when all of it fits without displacing a block.
 */
SZ_API_COMPTIME sz_u32_t sz_substrings_cuda_staged_rows_(sz_substrings_automaton_t const *automaton,
                                                         void const *kernel) {
    sz_size_t const whole_bytes = (sz_size_t)automaton->hot_count * (SZ_U8_MAX + 1) * sizeof(sz_u32_t);
    sz_size_t const unstaged_blocks = sz_substrings_cuda_resident_blocks_(kernel, 0);
    int ceiling = 0;
    if (!automaton->hot_count) return 0;
    // The block's default ceiling, not the opt-in one: raising that needs `cudaFuncSetAttribute` per kernel,
    // and a launch asking for more than the default is rejected outright.
    if (cudaDeviceGetAttribute(&ceiling, cudaDevAttrMaxSharedMemoryPerBlock,
                               sz_substrings_cuda_device_()) != cudaSuccess)
        return 0;
    if (whole_bytes > (sz_size_t)ceiling) return 0;
    // Free only while it displaces nothing: the walk is latency-bound, so warps are worth more than a row.
    if (sz_substrings_cuda_resident_blocks_(kernel, whole_bytes) < unstaged_blocks) return 0;
    return automaton->hot_count;
}

/**
 *  @brief Chunk width that fills the device: the corpus split across a full wave, clamped to the tier's own
 *         bounds and never under the warm-up a chunk has to pay before it reports anything.
 */
SZ_API_COMPTIME sz_size_t sz_substrings_cuda_chunk_bytes_(sz_substrings_automaton_t const *automaton,
                                                          sz_size_t total_bytes, sz_size_t resident_threads) {
    // Four times the longest match caps the warm-up at a quarter of a chunk, which is the only floor a
    // chunk needs; anything wider than the ceiling stops one long document from spreading.
    sz_size_t const floor = sz_max_of_two(4 * (sz_size_t)automaton->max_source_match_bytes, (sz_size_t)1);
    sz_size_t const threads = sz_max_of_two(resident_threads, (sz_size_t)1);
    sz_size_t const share = (total_bytes + threads - 1) / threads;
    // The floor is applied last, so a vocabulary whose longest match outruns the ceiling still warms up.
    return sz_max_of_two(sz_min_of_two(share, (sz_size_t)sz_substrings_cuda_chunk_bytes_max_k), floor);
}

/**
 *  @brief What the one shared device-side pass leaves behind for the verbs to read.
 *
 *  Two blocks, because the second cannot be sized until the first has been walked: the plan is sized by the
 *  haystack and chunk counts, the matches by what the sizing pass found.
 */
typedef struct sz_substrings_cuda_pass_state_t {
    /** The @b [haystacks + 1] exclusive chunk boundaries, per haystack. */
    sz_size_t *chunk_offsets;
    /** The @b [haystacks + 1] boundaries of the reported matches, per haystack. */
    sz_size_t *haystack_offsets;
    /** One slot per chunk: its match count from the sizing pass, then its exclusive output offset. */
    sz_size_t *chunk_slots;
    /** Scratch the three-launch scan carries tile totals through. */
    sz_size_t *tile_sums;
    /** Every emitted match, before any cover thins them. */
    sz_substrings_match_t *emitted;
    /** The matches a cover kept, which is @c emitted itself when no cover ran. */
    sz_substrings_match_t *reported;
    /** The scanned keep flags, @b [emitted + 1], or @c SZ_NULL when every match is reported. */
    sz_size_t *keep_offsets;
    /** Chunks the corpus was cut into. */
    sz_size_t chunk_count;
    /** Bytes per chunk. */
    sz_size_t chunk_bytes;
    /** Matches the walk emitted. */
    sz_size_t emitted_count;
    /** Matches surviving the cover, which every verb reports against. */
    sz_size_t reported_count;
    /** The plan block: boundaries, slots and scan scratch. */
    void *plan_memory;
    /** Bytes of the plan block. */
    sz_size_t plan_bytes;
    /** The match block: emitted matches, survivors and keep flags. */
    void *match_memory;
    /** Bytes of the match block. */
    sz_size_t match_bytes;
} sz_substrings_cuda_pass_state_t;

/** Hands both of the pass's blocks back to its allocator. */
SZ_API_COMPTIME void sz_substrings_cuda_pass_free_(sz_substrings_cuda_pass_state_t *pass,
                                                   sz_memory_allocator_t *alloc) {
    if (pass->plan_memory) alloc->free(pass->plan_memory, pass->plan_bytes, alloc->handle);
    if (pass->match_memory) alloc->free(pass->match_memory, pass->match_bytes, alloc->handle);
    pass->plan_memory = SZ_NULL, pass->plan_bytes = 0;
    pass->match_memory = SZ_NULL, pass->match_bytes = 0;
}

/** Sums the corpus through a device-side @p counter, which the host cannot read off a device-bound sequence. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_total_bytes_(sz_sequence_t *haystacks, sz_size_t *counter,
                                                            void *stream, sz_size_t *total_bytes) {
    void *arguments[2];
    sz_status_t status;
    if (cudaMemsetAsync(counter, 0, sizeof(sz_size_t), (cudaStream_t)stream) != cudaSuccess)
        return sz_device_code_mismatch_k;
    arguments[0] = haystacks, arguments[1] = &counter;
    status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_total_bytes_kernel_,
                                        sz_substrings_cuda_grid_(haystacks->count), arguments, 0, stream);
    if (status == sz_success_k) status = sz_substrings_cuda_read_(counter, total_bytes, stream);
    return status;
}

/** Zeroes the terminator a scan reads as its own last element, so the total lands in it. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_clear_terminator_(sz_size_t *values, sz_size_t count, void *stream) {
    return cudaMemsetAsync(values + count, 0, sizeof(sz_size_t), (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/**
 *  @brief Walks every chunk and settles the cover, leaving @p pass holding what the verbs read.
 *
 *  Three device-side sizes have to reach the host before the next allocation can be made - the corpus's
 *  byte count, its chunk count, and the emitted match count - so the pass is three rounds rather than one
 *  launch. Each round is a whole grid's worth of work, so the joins cost a launch apiece.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_pass_(sz_substrings_automaton_t const *automaton,
                                                     sz_sequence_t const *haystacks,
                                                     sz_substrings_overlap_policy_t overlap_policy,
                                                     sz_substrings_cuda_matches_t wanted,
                                                     sz_memory_allocator_t *alloc, void *stream,
                                                     sz_substrings_cuda_pass_state_t *pass) {
    sz_u32_t staged_rows = sz_substrings_cuda_staged_rows_(automaton,
                                                           (void const *)sz_substrings_cuda_walk_kernel_);
    sz_size_t const staged_bytes = (sz_size_t)staged_rows * (SZ_U8_MAX + 1) * sizeof(sz_u32_t);
    sz_size_t boundaries = haystacks->count + 1;
    sz_size_t const boundaries_bytes = boundaries * sizeof(sz_size_t);
    // The corpus counter, the chunk boundaries and their scan scratch: everything sized before the first walk.
    sz_size_t const sizing_bytes = sizeof(sz_size_t) + boundaries_bytes +
                                   sz_substrings_cuda_scan_scratch_(boundaries) * sizeof(sz_size_t);
    sz_bool_t const covering = (sz_bool_t)(overlap_policy != sz_substrings_overlapping_k);
    // A cover is decided between matches, so counting one costs what finding one costs; only an
    // overlapping count can answer from the boundaries its sizing walk already scanned.
    sz_bool_t const emitting = (sz_bool_t)(covering || wanted == sz_substrings_cuda_matches_needed_k);
    sz_sequence_t launched_haystacks = *haystacks;
    sz_substrings_automaton_t launched_automaton = *automaton;
    sz_substrings_cuda_pass_t walk_pass = sz_substrings_cuda_sizing_k;
    sz_size_t total_bytes = 0, slots_bytes, scan_tiles, emitted_bytes, keep_bytes, keep_tiles;
    sz_ptr_t sizing, plan, matches_block;
    void *arguments[9];
    sz_status_t status;

    pass->plan_memory = SZ_NULL, pass->plan_bytes = 0;
    pass->match_memory = SZ_NULL, pass->match_bytes = 0;
    pass->keep_offsets = SZ_NULL, pass->emitted = SZ_NULL, pass->reported = SZ_NULL;
    pass->emitted_count = 0, pass->reported_count = 0, pass->chunk_count = 0;

    // The residency of the allocator's scratch is checked on its first block, before anything is launched.
    sizing = (sz_ptr_t)alloc->allocate(sizing_bytes, alloc->handle);
    if (!sizing) return sz_bad_alloc_k;
    if (!sz_memory_reaches_device(sizing)) {
        alloc->free(sizing, sizing_bytes, alloc->handle);
        return sz_device_memory_mismatch_k;
    }
    pass->chunk_offsets = (sz_size_t *)(sizing + sizeof(sz_size_t));
    pass->tile_sums = pass->chunk_offsets + boundaries;
    status = sz_substrings_cuda_total_bytes_(&launched_haystacks, (sz_size_t *)sizing, stream, &total_bytes);
    pass->chunk_bytes = sz_substrings_cuda_chunk_bytes_(
        automaton, total_bytes,
        sz_substrings_cuda_resident_threads_((void const *)sz_substrings_cuda_walk_kernel_, staged_bytes));

    // Round one: how many chunks each haystack owns, scanned into the range that haystack's chunks take.
    // The plan block is sized against the chunk count, which this round is what discovers.
    if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &pass->chunk_bytes, arguments[2] = &pass->chunk_offsets;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_chunk_counts_kernel_,
                                            sz_substrings_cuda_grid_(haystacks->count), arguments, 0, stream);
    }
    if (status == sz_success_k)
        status = sz_substrings_cuda_clear_terminator_(pass->chunk_offsets, haystacks->count, stream);
    if (status == sz_success_k)
        status = sz_substrings_cuda_scan_(pass->chunk_offsets, boundaries, pass->tile_sums, stream);
    if (status == sz_success_k)
        status = sz_substrings_cuda_read_(pass->chunk_offsets + haystacks->count, &pass->chunk_count, stream);

    if (status == sz_success_k) {
        // One block now that the chunk count is known, with the boundaries carried across rather than
        // recomputed: the chunk counts kernel and its scan are a whole grid's work apiece.
        slots_bytes = (pass->chunk_count + 1) * sizeof(sz_size_t);
        scan_tiles = sz_substrings_cuda_scan_scratch_(sz_max_of_two(pass->chunk_count + 1, boundaries));
        pass->plan_bytes = boundaries_bytes * 2 + slots_bytes + scan_tiles * sizeof(sz_size_t);
        plan = (sz_ptr_t)alloc->allocate(pass->plan_bytes, alloc->handle);
        if (!plan) status = sz_bad_alloc_k;
        else if (cudaMemcpyAsync(plan, pass->chunk_offsets, boundaries_bytes, cudaMemcpyDeviceToDevice,
                                 (cudaStream_t)stream) != cudaSuccess) {
            alloc->free(plan, pass->plan_bytes, alloc->handle);
            status = sz_device_code_mismatch_k;
        }
        else {
            pass->plan_memory = plan;
            pass->chunk_offsets = (sz_size_t *)plan;
            pass->haystack_offsets = (sz_size_t *)(plan + boundaries_bytes);
            pass->chunk_slots = (sz_size_t *)(plan + boundaries_bytes * 2);
            pass->tile_sums = (sz_size_t *)(plan + boundaries_bytes * 2 + slots_bytes);
        }
    }
    alloc->free(sizing, sizing_bytes, alloc->handle);
    if (status != sz_success_k) {
        pass->plan_bytes = pass->plan_memory ? pass->plan_bytes : 0;
        sz_substrings_cuda_pass_free_(pass, alloc);
        return status;
    }

    // Round two: the sizing walk, so every chunk can then write into an output range of its own.
    arguments[0] = &launched_automaton, arguments[1] = &staged_rows, arguments[2] = &launched_haystacks;
    arguments[3] = &pass->chunk_offsets, arguments[4] = &pass->chunk_bytes, arguments[5] = &pass->chunk_count;
    arguments[6] = &pass->chunk_slots, arguments[7] = &pass->emitted, arguments[8] = &walk_pass;
    status = sz_substrings_cuda_launch_(
        (void const *)sz_substrings_cuda_walk_kernel_,
        sz_substrings_cuda_grid_for_((void const *)sz_substrings_cuda_walk_kernel_, staged_bytes,
                                     pass->chunk_count),
        arguments, staged_bytes, stream);
    if (status == sz_success_k)
        status = sz_substrings_cuda_clear_terminator_(pass->chunk_slots, pass->chunk_count, stream);
    if (status == sz_success_k)
        status = sz_substrings_cuda_scan_(pass->chunk_slots, pass->chunk_count + 1, pass->tile_sums, stream);
    if (status == sz_success_k)
        status = sz_substrings_cuda_read_(pass->chunk_slots + pass->chunk_count, &pass->emitted_count, stream);
    if (status != sz_success_k) {
        sz_substrings_cuda_pass_free_(pass, alloc);
        return status;
    }

    // Round three: the writing walk, the cover over what it wrote, and the per-haystack boundaries both
    // feed. Skipped whole when nobody reads the matches, which saves a second pass over the corpus and an
    // allocation that scales with the match count rather than with the corpus.
    emitted_bytes = emitting ? pass->emitted_count * sizeof(sz_substrings_match_t) : 0;
    keep_bytes = covering && pass->emitted_count ? (pass->emitted_count + 1) * sizeof(sz_size_t) : 0;
    keep_tiles = keep_bytes ? sz_substrings_cuda_scan_scratch_(pass->emitted_count + 1) : 0;
    pass->match_bytes = emitted_bytes + (covering ? emitted_bytes : 0) + keep_bytes +
                        keep_tiles * sizeof(sz_size_t);
    matches_block = pass->match_bytes ? (sz_ptr_t)alloc->allocate(pass->match_bytes, alloc->handle)
                                      : (sz_ptr_t)SZ_NULL;
    if (pass->match_bytes && !matches_block) {
        pass->match_bytes = 0;
        sz_substrings_cuda_pass_free_(pass, alloc);
        return sz_bad_alloc_k;
    }
    pass->match_memory = matches_block;
    pass->emitted = (sz_substrings_match_t *)matches_block;
    pass->reported = covering && pass->emitted_count ? (sz_substrings_match_t *)(matches_block + emitted_bytes)
                                                     : pass->emitted;
    pass->keep_offsets = keep_bytes ? (sz_size_t *)(matches_block + emitted_bytes * 2) : SZ_NULL;
    pass->reported_count = pass->emitted_count;

    if (emitting && pass->emitted_count) {
        walk_pass = sz_substrings_cuda_writing_k;
        arguments[7] = &pass->emitted, arguments[8] = &walk_pass;
        status = sz_substrings_cuda_launch_(
            (void const *)sz_substrings_cuda_walk_kernel_,
            sz_substrings_cuda_grid_for_((void const *)sz_substrings_cuda_walk_kernel_, staged_bytes,
                                         pass->chunk_count),
            arguments, staged_bytes, stream);
    }
    if (status == sz_success_k && keep_bytes) {
        sz_size_t longest = automaton->max_source_match_bytes;
        sz_size_t *keep_tile_sums = pass->keep_offsets + pass->emitted_count + 1;
        arguments[0] = &pass->emitted, arguments[1] = &pass->emitted_count, arguments[2] = &longest;
        arguments[3] = &overlap_policy, arguments[4] = &pass->keep_offsets;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_cover_kernel_,
                                            sz_substrings_cuda_grid_(pass->emitted_count), arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_substrings_cuda_clear_terminator_(pass->keep_offsets, pass->emitted_count, stream);
        if (status == sz_success_k)
            status = sz_substrings_cuda_scan_(pass->keep_offsets, pass->emitted_count + 1, keep_tile_sums, stream);
        if (status == sz_success_k) {
            arguments[0] = &pass->emitted, arguments[1] = &pass->emitted_count;
            arguments[2] = &pass->keep_offsets, arguments[3] = &pass->reported;
            status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_compact_kernel_,
                                                sz_substrings_cuda_grid_(pass->emitted_count), arguments, 0, stream);
        }
        if (status == sz_success_k)
            status = sz_substrings_cuda_read_(pass->keep_offsets + pass->emitted_count, &pass->reported_count,
                                              stream);
    }
    if (status == sz_success_k) {
        sz_size_t *keep_offsets = pass->keep_offsets;
        arguments[0] = &pass->chunk_offsets, arguments[1] = &pass->chunk_slots, arguments[2] = &keep_offsets;
        arguments[3] = &pass->haystack_offsets, arguments[4] = &boundaries;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_haystack_offsets_kernel_,
                                            sz_substrings_cuda_grid_(boundaries), arguments, 0, stream);
    }
    if (status == sz_success_k)
        status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                           : sz_device_code_mismatch_k;
    if (status != sz_success_k) sz_substrings_cuda_pass_free_(pass, alloc);
    return status;
}

/** Whether every argument the strict verbs read or write is one a kernel can address. */
SZ_API_COMPTIME sz_bool_t sz_substrings_cuda_resident_(sz_substrings_automaton_t const *automaton,
                                                       sz_sequence_t const *haystacks) {
    // An array the kernel dereferences, rather than the owning handle it never touches, so a caller
    // holding a borrowed view of a resident automaton is not refused for a null owner.
    if (!sz_memory_reaches_device(automaton->base)) return sz_false_k;
    // The handle is checked, never the accessors: those are the device's to call, so the host must not,
    // and a pointer is all this side can inspect. That the texts they answer are device-reachable is the
    // caller's word, which is what the staging verbs exist to make unnecessary.
    return sz_memory_reaches_device(haystacks->handle);
}

#pragma endregion Host Plumbing

#pragma region CUDA Backends

SZ_API_COMPTIME sz_status_t sz_substrings_counts_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                sz_sequence_t const *haystacks,
                                                                sz_substrings_overlap_policy_t overlap_policy,
                                                                sz_memory_allocator_t *alloc, sz_size_t *counts,
                                                                void *stream) {
    sz_substrings_cuda_pass_state_t pass;
    sz_size_t boundaries = haystacks->count;
    void *arguments[3];
    sz_status_t status;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(automaton, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(counts)) return sz_device_memory_mismatch_k;

    status = sz_substrings_cuda_pass_(automaton, haystacks, overlap_policy,
                                     sz_substrings_cuda_matches_unneeded_k, alloc, stream, &pass);
    if (status != sz_success_k) return status;

    arguments[0] = &pass.haystack_offsets, arguments[1] = &counts, arguments[2] = &boundaries;
    status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_counts_kernel_,
                                        sz_substrings_cuda_grid_(boundaries), arguments, 0, stream);
    if (status == sz_success_k)
        status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                           : sz_device_code_mismatch_k;
    sz_substrings_cuda_pass_free_(&pass, alloc);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_find_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                              sz_sequence_t const *haystacks,
                                                              sz_substrings_overlap_policy_t overlap_policy,
                                                              sz_memory_allocator_t *alloc,
                                                              sz_substrings_match_t *matches,
                                                              sz_size_t matches_capacity, sz_size_t *matches_found,
                                                              void *stream) {
    sz_substrings_cuda_pass_state_t pass;
    sz_status_t status;
    if (!haystacks->count) {
        *matches_found = 0;
        return sz_success_k;
    }
    if (!sz_substrings_cuda_resident_(automaton, haystacks)) return sz_device_memory_mismatch_k;
    if (matches_capacity && !sz_memory_reaches_device(matches)) return sz_device_memory_mismatch_k;

    status = sz_substrings_cuda_pass_(automaton, haystacks, overlap_policy,
                                     sz_substrings_cuda_matches_needed_k, alloc, stream, &pass);
    if (status != sz_success_k) return status;

    // The prefix is copied whatever the total, so the refusal below carries the same contract the CPU
    // backend keeps: as many matches as fit, and the true count beside them.
    {
        sz_size_t const fitting = sz_min_of_two(pass.reported_count, matches_capacity);
        *matches_found = pass.reported_count;
        if (fitting && cudaMemcpyAsync(matches, pass.reported, fitting * sizeof(sz_substrings_match_t),
                                       cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess)
            status = sz_device_code_mismatch_k;
    }
    if (status == sz_success_k)
        status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                           : sz_device_code_mismatch_k;
    if (status == sz_success_k && pass.reported_count > matches_capacity) status = sz_unexpected_dimensions_k;
    sz_substrings_cuda_pass_free_(&pass, alloc);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_replace_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                 sz_sequence_t const *haystacks,
                                                                 sz_sequence_t const *replacements,
                                                                 sz_substrings_overlap_policy_t overlap_policy,
                                                                 sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                                 sz_size_t tape_capacity, sz_size_t *offsets,
                                                                 void *stream) {
    sz_substrings_cuda_pass_state_t pass;
    sz_sequence_t launched_haystacks, launched_replacements;
    sz_size_t boundaries = haystacks->count + 1;
    sz_size_t gap_bytes, scan_tiles, rewritten = 0;
    sz_ptr_t rewrite_block;
    sz_size_t *gap_offsets, *tile_sums;
    void *arguments[7];
    sz_status_t status;
    // A substitution over matches that share bytes is not a function, so there is no cover to apply.
    if (overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != automaton->needles_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(automaton, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(replacements->handle)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(offsets)) return sz_device_memory_mismatch_k;
    if (tape_capacity && !sz_memory_reaches_device(tape)) return sz_device_memory_mismatch_k;

    status = sz_substrings_cuda_pass_(automaton, haystacks, overlap_policy,
                                     sz_substrings_cuda_matches_needed_k, alloc, stream, &pass);
    if (status != sz_success_k) return status;

    launched_haystacks = *haystacks, launched_replacements = *replacements;
    gap_bytes = pass.reported_count * sizeof(sz_size_t);
    scan_tiles = sz_substrings_cuda_scan_scratch_(boundaries);
    rewrite_block = (sz_ptr_t)alloc->allocate(gap_bytes + scan_tiles * sizeof(sz_size_t) + sizeof(sz_size_t),
                                              alloc->handle);
    if (!rewrite_block) {
        sz_substrings_cuda_pass_free_(&pass, alloc);
        return sz_bad_alloc_k;
    }
    gap_offsets = (sz_size_t *)rewrite_block;
    tile_sums = (sz_size_t *)(rewrite_block + gap_bytes);

    // One block per haystack, so the drift scan a rewrite needs stays inside one block's carry.
    arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
    arguments[2] = &pass.haystack_offsets, arguments[3] = &pass.reported;
    arguments[4] = &gap_offsets, arguments[5] = &offsets;
    status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_rewrite_offsets_kernel_,
                                        sz_substrings_cuda_grid_(haystacks->count *
                                                                 sz_substrings_cuda_threads_per_block_k),
                                        arguments, 0, stream);
    if (status == sz_success_k) status = sz_substrings_cuda_clear_terminator_(offsets, haystacks->count, stream);
    if (status == sz_success_k) status = sz_substrings_cuda_scan_(offsets, boundaries, tile_sums, stream);
    if (status == sz_success_k) status = sz_substrings_cuda_read_(offsets + haystacks->count, &rewritten, stream);

    if (status == sz_success_k && (!tape || rewritten > tape_capacity))
        status = rewritten > tape_capacity ? sz_unexpected_dimensions_k : sz_success_k;
    else if (status == sz_success_k) {
        arguments[0] = &launched_haystacks, arguments[1] = &launched_replacements;
        arguments[2] = &pass.haystack_offsets, arguments[3] = &pass.reported;
        arguments[4] = &gap_offsets, arguments[5] = &offsets, arguments[6] = &tape;
        status = sz_substrings_cuda_launch_((void const *)sz_substrings_cuda_rewrite_copy_kernel_,
                                            sz_substrings_cuda_grid_(rewritten), arguments, 0, stream);
        if (status == sz_success_k)
            status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                                : sz_device_code_mismatch_k;
    }

    alloc->free(rewrite_block, gap_bytes + scan_tiles * sizeof(sz_size_t) + sizeof(sz_size_t), alloc->handle);
    sz_substrings_cuda_pass_free_(&pass, alloc);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_scheduled_cuda(sz_substrings_automaton_t const *automaton,
                                                                     sz_sequence_t const *haystacks,
                                                                     sz_f32_t const *document_lengths,
                                                                     sz_substrings_bm25_t const *parameters,
                                                                     sz_f32_t const *needle_weights,
                                                                     sz_memory_allocator_t *alloc, sz_f32_t *scores,
                                                                     void *stream) {
    void const *const kernel = (void const *)sz_substrings_cuda_bm25_kernel_;
    sz_size_t const needles_count = automaton->needles_count;
    sz_substrings_automaton_t automaton_copy = *automaton;
    sz_sequence_t haystacks_copy = *haystacks;
    sz_substrings_bm25_t parameters_copy;
    sz_u32_t *overflow_rows = SZ_NULL;
    sz_size_t overflow_bytes = 0;
    unsigned blocks;
    void *arguments[7];
    sz_status_t status = sz_substrings_bm25_check(parameters, needle_weights);
    if (status != sz_success_k) return status;
    if (!haystacks->count) return sz_success_k;
    if (!sz_substrings_cuda_resident_(automaton, haystacks)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(scores)) return sz_device_memory_mismatch_k;
    if (needles_count && !sz_memory_reaches_device(needle_weights)) return sz_device_memory_mismatch_k;
    if (document_lengths && !sz_memory_reaches_device(document_lengths)) return sz_device_memory_mismatch_k;

    // A hashed tally spills into one global row per block, so its grid stops at one block per multiprocessor
    // to keep that scratch at `multiprocessors × needles` counters however resident the kernel could be.
    if (needles_count <= sz_substrings_cuda_tally_slots_k)
        blocks = (unsigned)sz_min_of_two(haystacks->count, sz_substrings_cuda_multiprocessors_() *
                                                               sz_substrings_cuda_resident_blocks_(kernel, 0));
    else {
        blocks = (unsigned)sz_min_of_two(haystacks->count, sz_substrings_cuda_multiprocessors_());
        overflow_bytes = (sz_size_t)blocks * needles_count * sizeof(sz_u32_t);
        overflow_rows = (sz_u32_t *)alloc->allocate(overflow_bytes, alloc->handle);
        if (!overflow_rows) return sz_bad_alloc_k;
        if (!sz_memory_reaches_device(overflow_rows)) {
            alloc->free(overflow_rows, overflow_bytes, alloc->handle);
            return sz_device_memory_mismatch_k;
        }
        if (cudaMemsetAsync(overflow_rows, 0, overflow_bytes, (cudaStream_t)stream) != cudaSuccess)
            status = sz_device_code_mismatch_k;
    }

    parameters_copy = *parameters;
    arguments[0] = &automaton_copy, arguments[1] = &haystacks_copy, arguments[2] = &document_lengths;
    arguments[3] = &parameters_copy, arguments[4] = &needle_weights, arguments[5] = &overflow_rows;
    arguments[6] = &scores;
    if (status == sz_success_k) status = sz_substrings_cuda_launch_(kernel, blocks, arguments, 0, stream);
    if (status == sz_success_k)
        status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                           : sz_device_code_mismatch_k;
    if (overflow_rows) alloc->free(overflow_rows, overflow_bytes, alloc->handle);
    return status;
}

#pragma endregion CUDA Backends

#pragma region Staging

/** One sequence staged for a kernel: its texts in plain device memory, the views over them in unified memory. */
typedef struct sz_substrings_cuda_staged_t {
    /** The views a kernel reads, which the host fills. */
    sz_string_view_t *views;
    /** Bytes of the views. */
    sz_size_t views_bytes;
    /** The texts, which only the device reads. */
    sz_ptr_t texts;
    /** The sequence bound to the device's own accessors over those views. */
    sz_sequence_t sequence;
} sz_substrings_cuda_staged_t;

/** Returns a staged sequence's views to @p staging and its texts to the device. */
SZ_API_COMPTIME void sz_substrings_cuda_staged_free_(sz_substrings_cuda_staged_t *staged,
                                                     sz_memory_allocator_t *staging) {
    if (staged->views) staging->free(staged->views, staged->views_bytes, staging->handle);
    if (staged->texts) cudaFree(staged->texts);
    staged->views = SZ_NULL, staged->views_bytes = 0, staged->texts = SZ_NULL;
}

/**
 *  @brief Copies @p source to the device through a host buffer from @p alloc, and binds the device's
 *         accessors over the copy.
 *
 *  The texts are only ever read by the device, so they cross once into plain device memory; only the views,
 *  which the host fills, come from @p staging. Bound through @ref sz_sequence_from_string_views_cuda, so the
 *  staged round reaches a kernel exactly as a caller's own device-resident sequence would.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_stage_sequence_(sz_sequence_t const *source,
                                                               sz_memory_allocator_t *alloc,
                                                               sz_memory_allocator_t *staging,
                                                               sz_substrings_cuda_staged_t *staged) {
    sz_size_t texts_bytes = 0, flat_bytes, written = 0, index;
    sz_ptr_t flat;
    sz_status_t status;
    for (index = 0; index != source->count; ++index) texts_bytes += source->get_length(source->handle, index);
    // Never a zero-byte request, which an allocator may answer with the null that means a refusal.
    flat_bytes = texts_bytes ? texts_bytes : 1;
    staged->views_bytes = source->count * sizeof(sz_string_view_t);
    staged->texts = SZ_NULL;

    staged->views = (sz_string_view_t *)staging->allocate(staged->views_bytes, staging->handle);
    flat = (sz_ptr_t)alloc->allocate(flat_bytes, alloc->handle);
    if (!staged->views || !flat || cudaMalloc((void **)&staged->texts, flat_bytes) != cudaSuccess) {
        if (flat) alloc->free(flat, flat_bytes, alloc->handle);
        staged->texts = SZ_NULL;
        sz_substrings_cuda_staged_free_(staged, staging);
        return sz_bad_alloc_k;
    }

    for (index = 0; index != source->count; ++index) {
        sz_size_t const length = source->get_length(source->handle, index);
        sz_copy(flat + written, source->get_start(source->handle, index), length);
        staged->views[index].start = staged->texts + written, staged->views[index].length = length;
        written += length;
    }
    status = cudaMemcpy(staged->texts, flat, texts_bytes, cudaMemcpyDefault) == cudaSuccess
                 ? sz_success_k
                 : sz_device_code_mismatch_k;
    alloc->free(flat, flat_bytes, alloc->handle);
    if (status == sz_success_k)
        status = sz_sequence_from_string_views_cuda(staged->views, source->count, &staged->sequence);
    if (status != sz_success_k) sz_substrings_cuda_staged_free_(staged, staging);
    return status;
}

/**
 *  @brief Copies @p source's block into plain device memory, rebasing every view onto the copy.
 *
 *  Every pointer the automaton carries is an offset into the one block it owns, so restaging it is one copy
 *  plus one shift per view rather than a rebuild. The copy is released with @c cudaFree.
 */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_stage_automaton_(sz_substrings_automaton_t const *source,
                                                                sz_substrings_automaton_t *staged) {
    sz_ptr_t const origin = (sz_ptr_t)source->memory;
    sz_ptr_t block = SZ_NULL;
    if (cudaMalloc((void **)&block, source->memory_bytes) != cudaSuccess) return sz_bad_alloc_k;
    if (cudaMemcpy(block, origin, source->memory_bytes, cudaMemcpyDefault) != cudaSuccess) {
        cudaFree(block);
        return sz_device_code_mismatch_k;
    }
    *staged = *source;
    staged->hot_rows = (sz_u32_t const *)(block + ((sz_ptr_t)source->hot_rows - origin));
    staged->base = (sz_u32_t const *)(block + ((sz_ptr_t)source->base - origin));
    staged->check = (sz_u32_t const *)(block + ((sz_ptr_t)source->check - origin));
    staged->fail = (sz_u32_t const *)(block + ((sz_ptr_t)source->fail - origin));
    staged->accepts_words = (sz_u32_t const *)(block + ((sz_ptr_t)source->accepts_words - origin));
    staged->outputs = (sz_substrings_output_t const *)(block + ((sz_ptr_t)source->outputs - origin));
    staged->outputs_counts = (sz_u32_t const *)(block + ((sz_ptr_t)source->outputs_counts - origin));
    staged->outputs_offsets = (sz_size_t const *)(block + ((sz_ptr_t)source->outputs_offsets - origin));
    staged->memory = block;
    return sz_success_k;
}

/** Stages the automaton and the haystacks both, flattening the texts through @p alloc. */
SZ_API_COMPTIME sz_status_t sz_substrings_cuda_stage_inputs_(sz_substrings_automaton_t const *automaton,
                                                             sz_sequence_t const *haystacks,
                                                             sz_memory_allocator_t *alloc,
                                                             sz_memory_allocator_t *staging,
                                                             sz_substrings_automaton_t *staged_automaton,
                                                             sz_substrings_cuda_staged_t *staged_haystacks) {
    sz_status_t status = sz_substrings_cuda_stage_automaton_(automaton, staged_automaton);
    if (status != sz_success_k) return status;
    status = sz_substrings_cuda_stage_sequence_(haystacks, alloc, staging, staged_haystacks);
    if (status != sz_success_k) cudaFree(staged_automaton->memory);
    return status;
}

#pragma endregion Staging

#pragma region Staging Backends

SZ_API_COMPTIME sz_status_t sz_substrings_counts_cuda(sz_substrings_automaton_t const *automaton,
                                                      sz_sequence_t const *haystacks,
                                                      sz_substrings_overlap_policy_t overlap_policy,
                                                      sz_memory_allocator_t *alloc, sz_size_t *counts) {
    sz_memory_allocator_t staging;
    sz_substrings_automaton_t staged_automaton;
    sz_substrings_cuda_staged_t staged_haystacks;
    sz_size_t *staged_counts;
    sz_status_t status;
    // The one probe is the strict verb's own: it refuses whatever the device cannot reach, and only then is
    // there anything to stage. A caller already holding its data on the device pays nothing for the attempt.
    status = sz_substrings_counts_scheduled_cuda(automaton, haystacks, overlap_policy, alloc, counts, SZ_NULL);
    if (status != sz_device_memory_mismatch_k) return status;

    sz_memory_allocator_init_unified(&staging);
    status = sz_substrings_cuda_stage_inputs_(automaton, haystacks, alloc, &staging, &staged_automaton,
                                              &staged_haystacks);
    if (status != sz_success_k) return status;
    staged_counts = (sz_size_t *)staging.allocate(haystacks->count * sizeof(sz_size_t), staging.handle);
    if (!staged_counts) {
        sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
        cudaFree(staged_automaton.memory);
        return sz_bad_alloc_k;
    }

    status = sz_substrings_counts_scheduled_cuda(&staged_automaton, &staged_haystacks.sequence, overlap_policy,
                                                 &staging, staged_counts, SZ_NULL);
    if (status == sz_success_k) {
        sz_size_t index;
        for (index = 0; index != haystacks->count; ++index) counts[index] = staged_counts[index];
    }
    staging.free(staged_counts, haystacks->count * sizeof(sz_size_t), staging.handle);
    sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
    cudaFree(staged_automaton.memory);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_find_cuda(sz_substrings_automaton_t const *automaton,
                                                    sz_sequence_t const *haystacks,
                                                    sz_substrings_overlap_policy_t overlap_policy,
                                                    sz_memory_allocator_t *alloc, sz_substrings_match_t *matches,
                                                    sz_size_t matches_capacity, sz_size_t *matches_found) {
    sz_memory_allocator_t staging;
    sz_substrings_automaton_t staged_automaton;
    sz_substrings_cuda_staged_t staged_haystacks;
    sz_size_t const staged_bytes = matches_capacity * sizeof(sz_substrings_match_t);
    sz_substrings_match_t *staged_matches;
    sz_status_t status;
    status = sz_substrings_find_scheduled_cuda(automaton, haystacks, overlap_policy, alloc, matches,
                                               matches_capacity, matches_found, SZ_NULL);
    if (status != sz_device_memory_mismatch_k) return status;

    sz_memory_allocator_init_unified(&staging);
    status = sz_substrings_cuda_stage_inputs_(automaton, haystacks, alloc, &staging, &staged_automaton,
                                              &staged_haystacks);
    if (status != sz_success_k) return status;
    staged_matches = staged_bytes ? (sz_substrings_match_t *)staging.allocate(staged_bytes, staging.handle)
                                  : SZ_NULL;
    if (staged_bytes && !staged_matches) {
        sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
        cudaFree(staged_automaton.memory);
        return sz_bad_alloc_k;
    }

    status = sz_substrings_find_scheduled_cuda(&staged_automaton, &staged_haystacks.sequence, overlap_policy,
                                               &staging, staged_matches, matches_capacity, matches_found, SZ_NULL);
    if (status == sz_success_k || status == sz_unexpected_dimensions_k) {
        sz_size_t const fitting = sz_min_of_two(*matches_found, matches_capacity);
        sz_size_t index;
        for (index = 0; index != fitting; ++index) matches[index] = staged_matches[index];
    }
    if (staged_matches) staging.free(staged_matches, staged_bytes, staging.handle);
    sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
    cudaFree(staged_automaton.memory);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_replace_cuda(sz_substrings_automaton_t const *automaton,
                                                       sz_sequence_t const *haystacks,
                                                       sz_sequence_t const *replacements,
                                                       sz_substrings_overlap_policy_t overlap_policy,
                                                       sz_memory_allocator_t *alloc, sz_ptr_t tape,
                                                       sz_size_t tape_capacity, sz_size_t *offsets) {
    sz_memory_allocator_t staging;
    sz_substrings_automaton_t staged_automaton;
    sz_substrings_cuda_staged_t staged_haystacks, staged_replacements;
    sz_size_t const offsets_bytes = (haystacks->count + 1) * sizeof(sz_size_t);
    sz_size_t *staged_offsets;
    sz_ptr_t staged_tape;
    sz_status_t status;
    status = sz_substrings_replace_scheduled_cuda(automaton, haystacks, replacements, overlap_policy, alloc, tape,
                                                  tape_capacity, offsets, SZ_NULL);
    if (status != sz_device_memory_mismatch_k) return status;

    sz_memory_allocator_init_unified(&staging);
    status = sz_substrings_cuda_stage_inputs_(automaton, haystacks, alloc, &staging, &staged_automaton,
                                              &staged_haystacks);
    if (status != sz_success_k) return status;
    status = sz_substrings_cuda_stage_sequence_(replacements, alloc, &staging, &staged_replacements);
    if (status != sz_success_k) {
        sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
        cudaFree(staged_automaton.memory);
        return status;
    }
    staged_offsets = (sz_size_t *)staging.allocate(offsets_bytes, staging.handle);
    staged_tape = tape_capacity ? (sz_ptr_t)staging.allocate(tape_capacity, staging.handle) : SZ_NULL;
    if (!staged_offsets || (tape_capacity && !staged_tape)) {
        if (staged_offsets) staging.free(staged_offsets, offsets_bytes, staging.handle);
        if (staged_tape) staging.free(staged_tape, tape_capacity, staging.handle);
        sz_substrings_cuda_staged_free_(&staged_replacements, &staging);
        sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
        cudaFree(staged_automaton.memory);
        return sz_bad_alloc_k;
    }

    status = sz_substrings_replace_scheduled_cuda(&staged_automaton, &staged_haystacks.sequence,
                                                  &staged_replacements.sequence, overlap_policy, &staging,
                                                  tape ? staged_tape : SZ_NULL, tape_capacity, staged_offsets,
                                                  SZ_NULL);
    // The boundaries are filled whether or not the tape held the result, which is what sizes the next call.
    if (status == sz_success_k || status == sz_unexpected_dimensions_k) {
        sz_size_t index;
        for (index = 0; index != haystacks->count + 1; ++index) offsets[index] = staged_offsets[index];
    }
    if (status == sz_success_k && tape) sz_copy(tape, staged_tape, offsets[haystacks->count]);

    staging.free(staged_offsets, offsets_bytes, staging.handle);
    if (staged_tape) staging.free(staged_tape, tape_capacity, staging.handle);
    sz_substrings_cuda_staged_free_(&staged_replacements, &staging);
    sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
    cudaFree(staged_automaton.memory);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_cuda(sz_substrings_automaton_t const *automaton,
                                                           sz_sequence_t const *haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights,
                                                           sz_memory_allocator_t *alloc, sz_f32_t *scores) {
    sz_memory_allocator_t staging;
    sz_substrings_automaton_t staged_automaton;
    sz_substrings_cuda_staged_t staged_haystacks;
    sz_size_t const count = haystacks->count, needles_count = automaton->needles_count;
    // One block holds the scores, the lengths when given, and the weights, in that order.
    sz_size_t const staged_floats = count * (document_lengths ? 2 : 1) + needles_count;
    sz_size_t const staged_bytes = staged_floats * sizeof(sz_f32_t);
    sz_f32_t *staged_scores;
    sz_size_t index;
    sz_status_t status = sz_substrings_bm25_scores_scheduled_cuda(automaton, haystacks, document_lengths, parameters,
                                                                  needle_weights, alloc, scores, SZ_NULL);
    if (status != sz_device_memory_mismatch_k) return status;

    sz_memory_allocator_init_unified(&staging);
    status = sz_substrings_cuda_stage_inputs_(automaton, haystacks, alloc, &staging, &staged_automaton,
                                              &staged_haystacks);
    if (status != sz_success_k) return status;
    staged_scores = (sz_f32_t *)staging.allocate(staged_bytes, staging.handle);
    if (!staged_scores) {
        sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
        cudaFree(staged_automaton.memory);
        return sz_bad_alloc_k;
    }
    {
        sz_f32_t *const staged_lengths = document_lengths ? staged_scores + count : SZ_NULL;
        sz_f32_t *const staged_weights = staged_scores + count * (document_lengths ? 2 : 1);
        for (index = 0; index != needles_count; ++index) staged_weights[index] = needle_weights[index];
        if (staged_lengths)
            for (index = 0; index != count; ++index) staged_lengths[index] = document_lengths[index];
        status = sz_substrings_bm25_scores_scheduled_cuda(&staged_automaton, &staged_haystacks.sequence,
                                                          staged_lengths, parameters, staged_weights, &staging,
                                                          staged_scores, SZ_NULL);
    }
    if (status == sz_success_k)
        for (index = 0; index != count; ++index) scores[index] = staged_scores[index];
    staging.free(staged_scores, staged_bytes, staging.handle);
    sz_substrings_cuda_staged_free_(&staged_haystacks, &staging);
    cudaFree(staged_automaton.memory);
    return status;
}

#pragma endregion Staging Backends


#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_SUBSTRINGS_CUDA_CUH_
