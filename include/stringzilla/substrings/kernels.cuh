/**
 *  @file include/stringzilla/substrings/kernels.cuh
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief StringZilla's own launch bodies over the walks of <tt>substrings/simt.cuh</tt>, which products
 *      never include, as only <tt>substrings/cuda.cuh</tt> and <tt>substrings/rocm.cuh</tt> launch them.
 */
#ifndef STRINGZILLA_SUBSTRINGS_KERNELS_CUH_
#define STRINGZILLA_SUBSTRINGS_KERNELS_CUH_

#include "stringzilla/substrings/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

/*  StringZilla's own kernels and the block bodies they share, which read @c blockIdx, declare
 *  @c __shared__ and drive the multi-launch rounds each vendor's host side sizes; the walks of
 *  <tt>substrings/simt.cuh</tt> do the walking. */
#pragma region Kernels

/**
 *  @brief Scans one ticket's own tile in place, seeded by the totals the tiles before it publish in
 *      @p states, so a whole scan is one launch.
 *  @param[in] live How many leading values the device wrote, scanned with one more slot for their
 *      total, or @c STRINGZILLA_NULL to scan all @p count.
 *  @param[in] states One zeroed word per tile, which this scan publishes its tile totals through.
 *  @param[out] stale The other buffer of words, which the next scan takes, zeroed by every launched
 *      block, live or not.
 *  @return Whether this block scanned the last tile, whose values are final once it returns.
 *
 *  The ticket slot starts zeroed and the last tile zeroes it again, so every scan of a round reuses
 *  the one slot the round's clear zeroed. Only blocks with values to scan take a ticket, so the
 *  last ticket is the slot's last touch however few of the launched blocks are live.
 */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_scan_tile_simt_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                                           sz_size_t elements_per_tile, sz_size_t *chain,
                                                           sz_u64_t *states, sz_u64_t *stale) {
    __shared__ sz_size_t shared[sz_substrings_threads_per_block_simt_k];
    // A live count past the buffer means its producer outran it and wrote none of it.
    sz_size_t const counted = !live ? count : *live < count ? *live + 1 : 0;
    sz_size_t const tiles = sz_size_divide_round_up(counted, elements_per_tile);
    sz_size_t tile, begin, end, running = 0, first;
    for (first = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; first < sz_chain_tiles_max_k;
         first += (sz_size_t)gridDim.x * blockDim.x)
        stale[first] = 0;
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
    running = sz_chain_look_back_simt_(states, tile, running);
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

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_simt_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                    sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale);
}

/**
 *  @brief Scans the sizing walk's chunk slots, then publishes from the last tile what the walk
 *      found, which is the one place a round learns whether it fit.
 *  @param[in] chunks_count How many chunks the round holds, whose slots alone the walk wrote, the
 *      slot after them receiving the total.
 *  @param[in] emitting Whether the round reads the matches themselves, since a count that never
 *      does cannot overrun a match budget however many matches the corpus holds.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_sized_simt_kernel_(sz_size_t *chunk_slots, sz_size_t count, sz_size_t const *chunks_count,
                                          sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                          sz_u64_t *stale, sz_size_t matches_budget, sz_bool_t emitting,
                                          sz_substrings_report_t *report) {
    sz_size_t emitted;
    if (!sz_substrings_scan_tile_simt_(chunk_slots, count, chunks_count, elements_per_tile, chain, states, stale) ||
        threadIdx.x != 0)
        return;
    emitted = chunk_slots[*chunks_count];
    report->matches_emitted = emitted;
    report->matches_stored = emitted;
    report->target_length = 0;
    report->shortfall = emitting && emitted > matches_budget ? emitted - matches_budget : 0;
}

/** Writes how many chunks each haystack is cut into, which the scan then turns into
 *  its chunk range. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_chunk_counts_simt_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                            sz_size_t *chunk_offsets) {
    sz_size_t const chunk_bytes = sz_substrings_round_chunk_bytes_simt_(&engine, haystacks);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < haystacks.count; index += stride)
        chunk_offsets[index] = sz_substrings_chunks_for_simt_(sz_strings_length_simt_(haystacks, index), chunk_bytes);
}

/**
 *  @brief Counts every chunk's matches into its slot, one thread per chunk, keeping them for
 *      @ref sz_substrings_gather_simt_kernel_ when @p kept is given, or tallies them into @p pairs.
 *  @param[in] pass Sizing, tallying with @p pairs, or marking, where each chunk stops at its first
 *      match and sets the bit of its haystack.
 *  @param[out] kept Three words per kept match, a slot's count apart, the first
 *      @c sz_substrings_kept_matches_simt_k of each chunk, or @c STRINGZILLA_NULL for a round that
 *      reads no matches.
 *  @param[out] spills The rest, @p spills_capacity at most, counted in @p spilled.
 *  @param[out] pairs A scoring round's table, which takes every match instead of the chunk's slot,
 *      or @c STRINGZILLA_NULL.
 */
STRINGZILLA_DEVICE void sz_substrings_walk_chunks_simt_(
    sz_substrings_engine_t const *engine, sz_case_sensitivity_t case_sensitivity, sz_substrings_gpu_pass_t pass,
    sz_u32_t staged_count, sz_strings_view_t haystacks, sz_size_t const *chunk_offsets, sz_size_t *chunk_slots,
    sz_size_t slots_count, sz_u32_t *kept, sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
    sz_size_t spills_capacity, sz_substrings_pairs_simt_t const *pairs) {
    extern __shared__ sz_u32_t sz_substrings_staged_simt_[];
    __shared__ sz_u8_t staged_classes[STRINGZILLA_U8_MAX + 1];
    sz_size_t const chunk_count = chunk_offsets[haystacks.count];
    sz_size_t const chunk_bytes = sz_substrings_round_chunk_bytes_simt_(engine, haystacks);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk_index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_substrings_staged_simt_t staged;
    sz_substrings_sink_simt_t sink;
    sz_substrings_stage_simt_(engine, staged_classes, sz_substrings_staged_simt_, staged_count);
    staged.classes = staged_classes, staged.rows = sz_substrings_staged_simt_, staged.rows_count = staged_count;
    staged.accepts_words = engine->accepts_words;
    sink.tally = STRINGZILLA_NULL, sink.pairs = pairs, sink.stride = slots_count;
    sink.spills = spills, sink.spilled = spilled, sink.spills_capacity = spills_capacity;

    for (; chunk_index < chunk_count; chunk_index += stride) {
        sz_size_t const haystack_index = sz_substrings_haystack_of_simt_(chunk_offsets, haystacks.count, chunk_index);
        sz_cptr_t const haystack = sz_strings_start_simt_(haystacks, haystack_index);
        sz_size_t const length = sz_strings_length_simt_(haystacks, haystack_index);
        sz_size_t const chunk_begin = (chunk_index - chunk_offsets[haystack_index]) * chunk_bytes;
        sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
        sz_size_t found;
        sink.kept = kept ? kept + chunk_index : STRINGZILLA_NULL, sink.chunk_index = chunk_index;
        sink.haystack_index = haystack_index;
        if (chunk_begin >= chunk_end && length != 0) found = 0;
        else if (case_sensitivity == sz_uncased_k)
            found = sz_substrings_walk_chunk_uncased_simt_(engine, &staged, haystack, length, chunk_begin, chunk_end,
                                                           pass, &sink);
        else if (pass == sz_substrings_gpu_marking_k) {
            sz_u32_t carried = sz_substrings_prime_simt_(engine, &staged, haystack, chunk_begin);
            found = sz_substrings_first_chunk_simt_(engine, &staged, haystack, length, chunk_begin, chunk_end,
                                                    &carried);
        }
        else
            found = sz_substrings_walk_chunk_cased_simt_(engine, &staged, haystack, length, chunk_begin, chunk_end,
                                                         pass, &sink);
        if (!pairs) chunk_slots[chunk_index] = found;
    }
}

/*  A byte-exact and a folded walk are compiled apart, so the folding cursor's registers never cut
 *  the residency of a walk that does not fold, and so are a walk that only counts and one that also
 *  keeps, so a count never pays for the registers keeping takes, and one that stops at a chunk's
 *  first match, so neither pays for that test. */

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_cased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                          sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                          sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_uncased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_cased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                          sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                          sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_uncased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_cased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                          sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                          sz_size_t *chunk_slots, sz_size_t slots_count, sz_u32_t *kept,
                                          sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
                                          sz_size_t spills_capacity) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, slots_count, kept, spills, spilled, spills_capacity,
                                    STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_uncased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots, sz_size_t slots_count, sz_u32_t *kept,
                                            sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
                                            sz_size_t spills_capacity) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, slots_count, kept, spills, spilled, spills_capacity,
                                    STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_cased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                           sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                           sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_uncased_simt_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                             sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                             sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs);
}

/**
 *  @brief Writes every match the sizing walk kept or spilled at the offset the scan left in its
 *      chunk's slot, so a round walks its haystacks once.
 *
 *  Every chunk owns a private, non-overlapping output range, so a write needs no atomic.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_gather_simt_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                      sz_size_t const *chunk_offsets, sz_substrings_report_t const *report,
                                      sz_size_t const *chunk_slots, sz_size_t slots_count, sz_u32_t const *kept,
                                      sz_substrings_spill_simt_t const *spills, sz_size_t const *spilled,
                                      sz_match_t *matches) {
    sz_size_t const chunk_count = chunk_offsets[haystacks.count];
    sz_size_t const chunk_bytes = sz_substrings_round_chunk_bytes_simt_(&engine, haystacks);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const first = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t index, slot;
    // The walk outran the budget, so there is nowhere to write.
    if (report->shortfall) return;
    for (index = first; index < chunk_count; index += stride) {
        sz_size_t const offset = chunk_slots[index];
        sz_size_t const found = sz_min_of_two(chunk_slots[index + 1] - offset,
                                              (sz_size_t)sz_substrings_kept_matches_simt_k);
        if (!found) continue;
        sz_size_t const haystack_index = sz_substrings_haystack_of_simt_(chunk_offsets, haystacks.count, index);
        sz_size_t const chunk_begin = (index - chunk_offsets[haystack_index]) * chunk_bytes;
        for (slot = 0; slot != found; ++slot) {
            sz_u32_t const *const entry = kept + slot * 3 * slots_count + index;
            sz_match_t match;
            match.haystack_index = haystack_index;
            match.needle_index = entry[0];
            // A match starts at most its length before its chunk, so 32 offset bits restore it.
            match.byte_offset = chunk_begin +
                                (sz_size_t)(sz_i64_t)(sz_i32_t)(entry[slots_count] - (sz_u32_t)chunk_begin);
            match.byte_length = entry[2 * slots_count];
            matches[offset + slot] = match;
        }
    }
    // With no shortfall every spill fit, as spills never outnumber the budget the pool is sized by.
    for (index = first; index < *spilled; index += stride) {
        sz_substrings_spill_simt_t const spill = spills[index];
        sz_match_t match;
        match.haystack_index = sz_substrings_haystack_of_simt_(chunk_offsets, haystacks.count, spill.chunk_index);
        match.needle_index = spill.needle_index;
        match.byte_offset = spill.byte_offset;
        match.byte_length = spill.byte_length;
        matches[chunk_slots[spill.chunk_index] + spill.slot] = match;
    }
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
 *  @param[in] tallying Whether the block counts into its own tally, which it zeroes only then.
 *
 *  Dynamic shared memory holds the counts, the keys when hashed, then the staged acceptance words
 *  and hot rows, in that order.
 */
STRINGZILLA_DEVICE sz_substrings_bm25_block_simt_t
sz_substrings_bm25_prepare_simt_(sz_substrings_engine_t const *engine, sz_u32_t *overflow_rows,
                                 sz_u32_t staged_accepts_words, sz_u32_t staged_count, sz_bool_t tallying) {
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

    for (slot = threadIdx.x; tallying && slot < table_slots; slot += blockDim.x) {
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

/** Adds the term of every seated pair into its haystack's fixed-point word of @p sums, zeroing the
 *  table behind it, so the next scoring round finds it clear. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_pairs_simt_kernel_(sz_substrings_pairs_simt_t pairs, sz_strings_view_t haystacks,
                                          sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                          sz_f32_t const *needle_weights, unsigned long long *sums) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t first;
    sz_u32_t lane;
    // Most slots are free, so each thread loads several keys before it tests any.
    for (first = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; first < pairs.capacity;
         first += sz_substrings_pairs_unroll_simt_k * stride) {
        sz_u64_t keys[sz_substrings_pairs_unroll_simt_k];
#pragma unroll
        for (lane = 0; lane != sz_substrings_pairs_unroll_simt_k; ++lane)
            keys[lane] = first + lane * stride < pairs.capacity ? pairs.keys[first + lane * stride] : 0;
#pragma unroll
        for (lane = 0; lane != sz_substrings_pairs_unroll_simt_k; ++lane)
            if (keys[lane])
                sz_substrings_bm25_pair_term_simt_(&pairs, first + lane * stride, keys[lane], haystacks,
                                                   document_lengths, &parameters, needle_weights, sums);
    }
}

/**
 *  @brief Writes the score of every haystack the round's pairs table held whole, and says whether
 *      any haystack the caller walks missed it: every @p group th from @p first on.
 *
 *  Every thread of the block calls it, and it ends in a barrier, so the whole block agrees.
 */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_bm25_finish_simt_(sz_strings_view_t haystacks, sz_f32_t *scores,
                                                             sz_size_t scores_stride, unsigned long long const *sums,
                                                             sz_size_t const *overflowed, sz_size_t first,
                                                             sz_size_t group) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t haystack_index;
    int missed = 0;
    for (haystack_index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; haystack_index < haystacks.count;
         haystack_index += stride)
        if (!overflowed[haystack_index])
            scores[haystack_index * scores_stride] = sz_substrings_bm25_fixed_score_simt_(sums[haystack_index]);
    // The caller's haystacks checked a thread each, rather than a block one by one.
    for (haystack_index = first + threadIdx.x * group; haystack_index < haystacks.count;
         haystack_index += (sz_size_t)blockDim.x * group)
        missed |= overflowed[haystack_index] != 0;
    return (sz_bool_t)(__syncthreads_or(missed) != 0);
}

/**
 *  @brief Writes the score of every haystack the round's pairs table held whole, and scores each of
 *      the rest with a block of its own: its threads walk contiguous chunks into one shared tally,
 *      then sum the tallied terms in fixed point.
 *  @param[in] sums Each haystack's fixed-point sum of the table's terms.
 *  @param[in] overflowed Each haystack's flag, set where one of its needles found no slot in the
 *      table, so its sum is partial.
 *  @param[in] scores_stride Entries from one haystack's score to the next, so one launch
 *      writes one column.
 *
 *  A block whose haystacks the table held whole stages nothing. Where a batch holds too few
 *  haystacks to fill the device, Hopper's tier walks each missed one with a cluster of blocks.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_simt_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                    sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                    sz_f32_t const *needle_weights, sz_u32_t *overflow_rows, sz_f32_t *scores,
                                    sz_size_t scores_stride, sz_u32_t staged_accepts_words, sz_u32_t staged_count,
                                    unsigned long long const *sums, sz_size_t const *overflowed) {
    sz_size_t const table_slots = sz_substrings_tally_slots_for_simt_(engine.needles_count);
    sz_size_t haystack_index;
    if (!sz_substrings_bm25_finish_simt_(haystacks, scores, scores_stride, sums, overflowed, blockIdx.x, gridDim.x))
        return;
    sz_substrings_bm25_block_simt_t const block = sz_substrings_bm25_prepare_simt_(
        &engine, overflow_rows, staged_accepts_words, staged_count, sz_true_k);
    for (haystack_index = blockIdx.x; haystack_index < haystacks.count; haystack_index += gridDim.x) {
        if (!overflowed[haystack_index]) continue;
        sz_cptr_t const haystack = sz_strings_start_simt_(haystacks, haystack_index);
        sz_size_t const length = sz_strings_length_simt_(haystacks, haystack_index);
        sz_f64_t const norm = sz_substrings_bm25_norm(
            &parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index] : (sz_f64_t)length);
        sz_substrings_bm25_walk_simt_(&engine, &block.staged, haystack, length, threadIdx.x, blockDim.x, &block.tally);
        __syncthreads();
        sz_substrings_bm25_score_simt_(&block.tally, table_slots, engine.needles_count, &parameters, norm,
                                       needle_weights, block.sum, scores + haystack_index * scores_stride);
    }
}

/**
 *  @brief Links every match to its successor and marks each haystack's first chosen one in @p keep.
 *
 *  Within a haystack the walk emits matches in non-decreasing end order and each starts at most
 *  @p longest bytes before its end, so a successor's search window is bounded by the vocabulary,
 *  never by the text.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_link_simt_kernel_(sz_match_t const *matches, sz_substrings_report_t const *report, sz_size_t longest,
                                    sz_substrings_overlap_policy_t policy, sz_size_t *jumps, sz_size_t *keep) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Nothing was written, so there is nothing to cover; the report already names what did not fit.
    if (report->shortfall) return;
    for (; index < count; index += stride) {
        sz_match_t const *match = &matches[index];
        jumps[index] = index + 1 < count && matches[index + 1].haystack_index == match->haystack_index
                           ? sz_substrings_successor_simt_(matches, count, longest, policy, index + 1,
                                                           match->byte_offset + match->byte_length)
                           : count;
        if (index == 0 || matches[index - 1].haystack_index != match->haystack_index)
            keep[sz_substrings_successor_simt_(matches, count, longest, policy, index, 0)] = 1;
    }
}

/** One doubling round: every marked match marks the matches its first hops reach, and jumps as far
 *  as all of them next round. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_double_simt_kernel_(sz_substrings_report_t const *report, sz_size_t const *jumps, sz_size_t *doubled,
                                      sz_size_t *keep) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    for (; index < count; index += stride) {
        sz_size_t jump = jumps[index];
        // A mark set in this round only reaches matches that are on the chain anyway.
        sz_bool_t const marked = (sz_bool_t)(keep[index] != 0);
        for (sz_size_t hop = 1; hop != sz_substrings_round_hops_simt_k && jump != count; ++hop) {
            if (marked) keep[jump] = 1;
            jump = jumps[jump];
        }
        doubled[index] = jump;
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
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_compact_simt_kernel_(sz_match_t const *matches, sz_substrings_report_t *report,
                                       sz_size_t const *keep_offsets, sz_match_t *survivors) {
    sz_size_t const count = report->matches_emitted;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    if (index == 0) report->matches_stored = keep_offsets[count];
    for (; index < count; index += stride)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/** Maps each haystack's match range onto the boundaries its reported matches occupy. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_haystack_offsets_simt_kernel_(sz_size_t const *chunk_offsets, sz_size_t const *chunk_slots,
                                                sz_size_t const *keep_offsets, sz_substrings_report_t const *report,
                                                sz_size_t *haystack_offsets, sz_size_t count) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (report->shortfall) return;
    for (; index < count; index += stride) {
        sz_size_t const emitted_before = chunk_slots[chunk_offsets[index]];
        haystack_offsets[index] = keep_offsets ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** Marks which haystacks own a match, a lane each, as a gap between its two boundaries, every
 *  group of 32 lanes storing what @p ballot gathers as four bytes of @p contained. */
STRINGZILLA_DEVICE void sz_substrings_contains_simt_(sz_size_t const *haystack_offsets, sz_u1x8_t *contained,
                                                     sz_size_t count, sz_u32_t (*ballot)(int)) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x, lanes = sz_size_divide_round_up(count, 32) * 32;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < lanes; index += stride) {
        sz_u32_t const bits = ballot(index < count && haystack_offsets[index + 1] > haystack_offsets[index]);
        if (index % 32 == 0) sz_u1x8_store_simt_(contained, count, index, bits);
    }
}

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_counts_simt_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts, sz_size_t counts_stride,
                                      sz_size_t count) {
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
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_store_matches_simt_kernel_(sz_match_t const *reported, sz_substrings_report_t const *report,
                                             sz_match_t *matches, sz_size_t matches_capacity) {
    sz_size_t const kept = report->shortfall ? 0 : report->matches_stored;
    sz_size_t const fitting = sz_min_of_two(kept, matches_capacity);
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; index < fitting; index += stride) matches[index] = reported[index];
}

/** Publishes what the store kept, a launch of its own, since a block of the store still reading the
 *  report would otherwise see its shortfall change and copy nothing. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_stored_simt_kernel_(sz_substrings_report_t *report, sz_size_t matches_capacity) {
    sz_size_t kept;
    if (blockIdx.x || threadIdx.x) return;
    kept = report->shortfall ? 0 : report->matches_stored;
    report->matches_stored = sz_min_of_two(kept, matches_capacity);
    if (kept > matches_capacity) report->shortfall = kept - matches_capacity;
}

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
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_offsets_simt_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                               sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                               sz_substrings_report_t const *report, sz_size_t *gap_offsets,
                                               sz_size_t *output_sizes) {
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
                // Shrinking matches make this wrap, which is right: only the prefix sums are ever
                // read, each names a real offset, and modular arithmetic reproduces every one.
                drift_here = sz_strings_length_simt_(replacements, needle) - matches[match_index].byte_length;
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

        // The scan's own aggregate is the haystack's total drift, so no second pass reduces it.
        if (threadIdx.x == 0)
            output_sizes[haystack_index] = sz_strings_length_simt_(haystacks, haystack_index) + drift_carry;
        __syncthreads(); // ! The next haystack resets the carry this one is still reading.
    }
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
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_copy_simt_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                            sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                            sz_size_t const *gap_offsets, sz_size_t const *output_offsets,
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
            sz_cptr_t const haystack = sz_strings_start_simt_(haystacks, haystack_index);
            sz_size_t const haystack_length = sz_strings_length_simt_(haystacks, haystack_index);
            sz_size_t const base = output_offsets[haystack_index];
            sz_size_t const first = haystack_offsets[haystack_index], last = haystack_offsets[haystack_index + 1];
            // Every match adds a gap and a replacement; one more stretch closes the haystack.
            sz_size_t const stretches = last - first + 1;
            sz_size_t const wanted = tile_begin > base ? tile_begin - base : 0;
            sz_size_t const skip = first == last
                                       ? 0
                                       : sz_substrings_last_not_above_simt_(gap_offsets + first, last - first, wanted);
            // Past the last match the drift is whatever the whole haystack accumulated, which its
            // rewritten length already names - so the closing stretch needs no offset of its own.
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
                sz_substrings_copy_clipped_simt_(
                    output, tile_begin, tile_end, gap_begin + (gap_source_end - previous_end),
                    sz_strings_start_simt_(replacements, needle), sz_strings_length_simt_(replacements, needle), lane);
            }
        }
    }
}

/**
 *  @brief Publishes the bytes the rewrite needs, and whether the target could hold them.
 *  @param[in] target_ceiling Bytes the caller's target holds.
 */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_target_simt_kernel_(sz_size_t const *rewritten_at, sz_size_t target_ceiling,
                                      sz_substrings_report_t *report) {
    sz_size_t const rewritten = *rewritten_at;
    if (blockIdx.x || threadIdx.x) return;
    report->target_length = rewritten;
    if (!report->shortfall && rewritten > target_ceiling) report->shortfall = rewritten - target_ceiling;
}

#pragma endregion Kernels

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBSTRINGS_KERNELS_CUH_
