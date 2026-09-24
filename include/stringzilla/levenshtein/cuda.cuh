/**
 *  @brief CUDA backend for Levenshtein distances: one candidate per thread, streaming Myers' bit-parallel
 *      recurrence against a batch of queries prepared once into match masks the whole grid shares.
 *  @file include/stringzilla/levenshtein/cuda.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzilla/levenshtein.h
 *
 *  The step is the serial tier's, reached from the device through `--expt-relaxed-constexpr`, so the distances
 *  are the same integers rather than merely close ones. Myers is add-with-carry and bitwise operations over
 *  @c u64 words, all of which the device runs at its integer rate; there is nothing here for the DPX three-way
 *  maxima to accelerate, since those serve the score-based recurrences of Needleman-Wunsch and Smith-Waterman
 *  rather than this one. That is why this family has one GPU tier and not a ladder of them.
 *
 *  A candidate's recurrence is a dependency chain, so it stays on one thread and the parallelism comes from the
 *  candidates on @c blockIdx.x and from the queries on @c blockIdx.y. The verticals live in the thread's own
 *  registers or local memory, @c words of them, while the match masks are read-only and shared: every thread
 *  indexes the same @c classes × stride plane by the class of the byte it is stepping, so the rows stay hot in
 *  cache instead of being rebuilt per candidate.
 *
 *  A batch is bucketed by rung key at @ref sz_levenshtein_engine_init_cuda, so one launch carries only queries
 *  that share an entry point, and the occupancy walk each of those entry points needs is paid there rather than
 *  once per round. The byte planes are built by a kernel from the queries staged into device-reachable memory;
 *  the rune planes are built on the host, an init being allowed to join where a round is not.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_CUDA_CUH_
#define STRINGZILLA_LEVENSHTEIN_CUDA_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/levenshtein/serial.h"

#if SZ_USE_CUDA

#pragma region Myers Threaded

/** Query words one thread keeps verticals for, on the byte rung and the rune rung alike. */
enum { sz_levenshtein_cuda_thread_words_max_k = 16 };

/** Query words one lane of the warped rung keeps verticals for; a warp carries thirty-two times as many. */
enum { sz_levenshtein_cuda_warp_words_per_lane_max_k = 8 };

/** Lanes a warped candidate is spread across, which is a warp. */
enum { sz_levenshtein_cuda_warp_lanes_k = 32 };

/** Words one lane of a warped candidate owns, which is what selects the rung's entry point. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_cuda_warp_words_per_lane_(sz_size_t words) {
    return sz_size_divide_round_up(words, sz_levenshtein_cuda_warp_lanes_k);
}

/** Query words the kernels reach together: a warp's thirty-two lanes at their widest. A query past this
 *  is refused rather than silently truncated. */
enum {
    sz_levenshtein_cuda_words_max_k =
        sz_levenshtein_cuda_warp_lanes_k * sz_levenshtein_cuda_warp_words_per_lane_max_k
};

/** Query words from which the warped rung is launched rather than the threaded one. Swept on XLSum lines, the
 *  two cross between eight words and sixteen: at sixteen the warp leads by 1.5x in corpus order and 1.8x sorted
 *  by length, at eight it trails. The crossing moves with the longest candidate, not with the query, since a
 *  thread rung's warp costs the longest of its thirty-two candidates while a warped candidate costs its own. */
enum { sz_levenshtein_cuda_warp_words_min_k = 16 };

/** Threads a block runs when the device cannot be asked; the register budget differs per word count, so the
 *  launcher takes what the occupancy calculator answers for the entry point it is about to launch instead. */
enum { sz_levenshtein_cuda_candidates_per_block_k = 128 };

/** Widest block the launcher considers: past this a block schedules too coarsely for what residency returns. */
enum { sz_levenshtein_cuda_candidates_per_block_max_k = 256 };

/** Rows one launch's query axis spans, past which a bucket is cut into several launches of the same kernel. */
enum { sz_levenshtein_cuda_grid_rows_max_k = 65535 };

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief One candidate's Myers sweep at a compile-time @p words, which is what keeps the verticals in registers.
 *
 *  A word count the compiler cannot see makes @c verticals a dynamically indexed array, and the only place it can
 *  live is local memory - a 256-byte stack frame and four local accesses per word-step, which caps the kernel near
 *  thirty percent of the device's integer issue rate. Reached from entry points that each pass a literal, the same
 *  body is worth 2.2x at a one-word query and 3.3x at sixteen.
 *
 *  @param[in] order The launch's own slice of the batch's bucketing, one query index per @c blockIdx.y row.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                 sz_sequence_t candidates, sz_size_t *distances,
                                                 sz_size_t distances_stride, sz_size_t words) {
    sz_size_t const candidate = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidates.count) return;

    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;
    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
    sz_size_t const length = candidates.get_length(candidates.handle, candidate);
    sz_u8_t const *const byte_to_class = query.byte_to_class;

    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_cuda_thread_words_max_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);

    // One byte per step is one uncoalesced sector per step, and at a one-word query that load is most of the
    // round. Eight bytes arrive in one, so the cost amortizes over eight steps; the head walks to an aligned
    // boundary and the tail finishes whatever the last chunk leaves.
    sz_size_t position = 0;
    sz_size_t const head = (sz_size_t)(-(sz_ssize_t)(sz_size_t)text) & 7;
    sz_size_t const aligned_head = head < length ? head : length;
    for (; position != aligned_head; ++position)
        sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, byte_to_class[(sz_u8_t)text[position]]);
    for (; position + 8 <= length; position += 8) {
        sz_u64_t const octet = *(sz_u64_t const *)(text + position);
        for (sz_size_t byte = 0; byte != 8; ++byte)
            sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query,
                                             byte_to_class[(sz_u8_t)(octet >> (byte * 8))]);
    }
    for (; position != length; ++position)
        sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, byte_to_class[(sz_u8_t)text[position]]);
    row[candidate] = sz_levenshtein_u64x1_score_serial(&state, candidate);
}

/*  One entry point per word count, each passing its own literal, so every query length gets its own register
 *  budget - thirty-eight registers at one word against ninety-six at sixteen, which one shared kernel would have
 *  to spend on every launch. `sz_levenshtein_cuda_entry_point_` picks between them.
 */
static __global__ void sz_levenshtein_u64x1_distances_cuda_w1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w9_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w10_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w11_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w12_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w13_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w14_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w15_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w16_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_(engine, order, candidates, distances, distances_stride, 16);
}

/*  One entry point per word count, addressed by it. `cudaLaunchKernel` takes the host-side symbol of a
 *  `__global__`, so the table is what the `<<< >>>` operator would have selected, spelled as data. */
static void const *const sz_levenshtein_cuda_entry_points_[sz_levenshtein_cuda_thread_words_max_k] = {
    (void const *)sz_levenshtein_u64x1_distances_cuda_w1_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w2_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w3_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w4_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w5_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w6_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w7_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w8_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w9_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w10_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w11_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w12_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w13_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w14_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w15_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w16_,
};

/**
 *  @brief Threads a sweep gives a block on this device, for the entry point it is about to launch.
 *
 *  Register pressure moves the residency ceiling from one word count to the next, so the block landing the most
 *  warps per multiprocessor is the device's answer and not a constant's. The walk stops at
 *  @c sz_levenshtein_cuda_candidates_per_block_max_k rather than at what the entry point allows, since past it a
 *  block schedules too coarsely for what its residency returns; ties below it go to the wider block.
 */
static sz_size_t sz_levenshtein_cuda_per_block_(void const *entry_point) {
    cudaFuncAttributes attributes;
    sz_size_t per_block = sz_levenshtein_cuda_candidates_per_block_k;
    sz_size_t most_warps = 0, ceiling, candidate;
    if (cudaFuncGetAttributes(&attributes, entry_point) != cudaSuccess) return per_block;

    ceiling = (sz_size_t)attributes.maxThreadsPerBlock;
    if (ceiling > sz_levenshtein_cuda_candidates_per_block_max_k)
        ceiling = sz_levenshtein_cuda_candidates_per_block_max_k;
    for (candidate = 64; candidate <= ceiling; candidate *= 2) {
        int resident_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks, entry_point, (int)candidate, 0) !=
            cudaSuccess)
            continue;
        sz_size_t const warps = (sz_size_t)resident_blocks * (candidate / 32);
        if (warps >= most_warps && warps != 0) most_warps = warps, per_block = candidate;
    }
    return per_block;
}

#pragma endregion Myers Threaded

#pragma region Myers Warped

/**
 *  @brief One candidate's Myers sweep across a whole warp, @p words_per_lane words to a lane, skewed by one
 *      character per lane so the carry between words walks one lane per step.
 *
 *  A lane owns @p words_per_lane consecutive words and advances the character @p words_per_lane steps behind
 *  the lane below it, so the carry out of the lane below's top word was finalized one step earlier and arrives
 *  through a single @c __shfl_up_sync. Unskewed, that same carry is a warp-wide prefix, and the lookahead
 *  resolving it costs a dozen shuffles per character against roughly twenty useful word operations. The skew
 *  costs a lane of fill and a lane of drain, so a warp runs @c length+live_lanes-1 steps for a candidate of
 *  @c length characters, and the thirty-two bytes one step reads are thirty-two consecutive ones.
 *
 *  Words past the query's last are stepped rather than branched around: the recurrence only ever carries
 *  upward, so whatever such a word holds never reaches a live one, and the distance below reads none of them.
 *
 *  @param[in] words_per_lane Exactly @c ceil(query_words/32); a literal, which keeps the verticals in registers.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_warp_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                      sz_sequence_t candidates, sz_size_t *distances,
                                                      sz_size_t distances_stride, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_size_t const candidate = (sz_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    // Warp uniform, so the shuffles below still see a whole warp.
    if (candidate >= candidates.count) return;

    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;
    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
    sz_size_t const length = candidates.get_length(candidates.handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = (words + words_per_lane - 1) / words_per_lane;
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_cuda_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_cuda_warp_words_per_lane_max_k];
#pragma unroll
    for (sz_size_t word = 0; word != words_per_lane; ++word) positive[word] = ~(sz_u64_t)0, negative[word] = 0;

    // What a lane hands the lane above: Myers' horizontal positive in bit zero, horizontal negative in bit one.
    // The addition's sixty-fifth bit needs no room of its own, being the horizontal negative wherever the
    // vertical positive's top bit is set and zero wherever it is not.
    unsigned carry = 0;
    sz_size_t const steps = length + live_lanes - 1;
    for (sz_size_t step = 0; step != steps; ++step) {
        unsigned const received = __shfl_up_sync(0xFFFFFFFFu, carry, 1);
        sz_ssize_t const position = (sz_ssize_t)step - (sz_ssize_t)lane;
        if (lane >= live_lanes || position < 0 || position >= (sz_ssize_t)length) continue;

        sz_u64_t const *const masks = lane_masks + (sz_size_t)query.byte_to_class[(sz_u8_t)text[position]] *
                                                       query.stride;
        // The row above the query's first word is one edit higher than the cell left of it, which is lane
        // zero's carry at every character rather than anything a shuffle brings.
        sz_u64_t positive_carry = lane == 0 ? 1 : (sz_u64_t)(received & 1u);
        sz_u64_t negative_carry = lane == 0 ? 0 : (sz_u64_t)((received >> 1) & 1u);
#pragma unroll
        for (sz_size_t word = 0; word != words_per_lane; ++word) {
            sz_u64_t const equality = masks[word];
            sz_u64_t const vertical_carry = equality | negative[word];
            sz_u64_t const matched = equality | negative_carry;
            sz_u64_t const diagonal = (((matched & positive[word]) + positive[word]) ^ positive[word]) | matched;
            sz_u64_t horizontal_positive = negative[word] | ~(diagonal | positive[word]);
            sz_u64_t horizontal_negative = positive[word] & diagonal;
            sz_u64_t const next_positive_carry = horizontal_positive >> 63;
            sz_u64_t const next_negative_carry = horizontal_negative >> 63;
            horizontal_positive = (horizontal_positive << 1) | positive_carry;
            horizontal_negative = (horizontal_negative << 1) | negative_carry;
            positive_carry = next_positive_carry, negative_carry = next_negative_carry;
            positive[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
            negative[word] = horizontal_positive & vertical_carry;
        }
        carry = (unsigned)positive_carry | ((unsigned)negative_carry << 1);
    }

    // The verticals are the query column's own deltas and the row above the query costs one edit per candidate
    // character, so their sum over the query's bits is the distance and no lane has to track a running score
    // through the sweep. A dead word contributes nothing, and the last word drops the bits past the last symbol.
    sz_u64_t const last_symbol_bit = sz_levenshtein_last_symbol_bit_(query.length);
    sz_u64_t const last_word_live = last_symbol_bit | (last_symbol_bit - 1);
    int deltas = 0;
#pragma unroll
    for (sz_size_t word = 0; word != words_per_lane; ++word) {
        sz_size_t const index = first_word + word;
        sz_u64_t const live = index + 1 < words ? ~(sz_u64_t)0 : index + 1 == words ? last_word_live : 0;
        deltas += __popcll(positive[word] & live) - __popcll(negative[word] & live);
    }
#pragma unroll
    for (unsigned offset = 16; offset != 0; offset >>= 1) deltas += __shfl_down_sync(0xFFFFFFFFu, deltas, offset);
    if (lane == 0) row[candidate] = (sz_size_t)((sz_ssize_t)length + deltas);
}

/*  One entry point per words-per-lane, each passing its own literal, for the reason the threaded rung states:
 *  a count the compiler cannot see spills the verticals to local memory. `K` rounds up by one word rather than
 *  to the next power of two, since the remainder costs only idle lanes - linear leaves worst-case lane
 *  utilization at seven eighths where doubling drops it to just over half above each boundary.
 */
static __global__ void sz_levenshtein_u64x32_distances_cuda_k1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x32_distances_cuda_k8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_(engine, order, candidates, distances, distances_stride, 8);
}

/*  One entry point per words-per-lane, addressed by it, beside the threaded rung's own table. */
static void const *const sz_levenshtein_cuda_entry_points_warp_[sz_levenshtein_cuda_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_u64x32_distances_cuda_k1_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k2_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k3_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k4_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k5_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k6_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k7_,
    (void const *)sz_levenshtein_u64x32_distances_cuda_k8_,
};

#pragma endregion Myers Warped

#pragma region Myers Byte Lanes

/** Query symbols one byte lane of a Myers word holds, which is the longest query the byte rung takes. */
enum { sz_levenshtein_cuda_byte_lanes_symbols_max_k = 8 };

/** Query symbols one sixteen-bit lane holds, which is the longest query either narrow rung takes. */
enum { sz_levenshtein_cuda_short_lanes_symbols_max_k = 16 };

/** Candidates one thread advances together at its widest: four byte lanes to a thirty-two-bit register. A
 *  sixty-four-bit register holds eight, and it costs two instructions per operation to do it, so the wider
 *  packing buys no arithmetic and spends the thread-level parallelism that hides this rung's loads. */
enum { sz_levenshtein_cuda_lanes_per_thread_max_k = 4 };

/** Text bytes one refill hands a lane, which is the aligned word its cursor sits in. */
enum { sz_levenshtein_cuda_bytes_per_refill_k = 8 };

/** Times over the device's resident threads a batch must reach before a narrow rung is launched at all. Below
 *  it neither rung's grid fills the device and the narrower one loses the latency it cannot hide; swept on
 *  XLSum words, the two cross between two and four million candidates on a 188-multiprocessor device, at both
 *  lane widths rather than at a count that scales with the lanes. */
enum { sz_levenshtein_cuda_lanes_waves_min_k = 12 };

/**
 *  @brief Advances every lane one symbol through its own Myers word, all of them packed in one register.
 *
 *  The device has no per-lane add, so the one carry that would cross a lane boundary is blocked the standard
 *  way: the low bits of every lane sum on their own, where a carry cannot leave the lane, and each lane's top
 *  bit is folded back in by exclusive or.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_lanes_step_(sz_u32_t *positive, sz_u32_t *negative, sz_u32_t equality,
                                                      sz_u32_t lane_low_bits, sz_u32_t lane_high_bits) {
    sz_u32_t const vertical_positive = *positive, vertical_negative = *negative;
    sz_u32_t const vertical_carry = equality | vertical_negative;
    sz_u32_t const low_sum = (equality & vertical_positive & ~lane_high_bits) + (vertical_positive & ~lane_high_bits);
    sz_u32_t const high_carry = vertical_positive & ~equality & lane_high_bits;
    sz_u32_t const diagonal = (low_sum ^ high_carry ^ vertical_positive) | equality;
    sz_u32_t const horizontal_positive = vertical_negative | ~(diagonal | vertical_positive);
    sz_u32_t const horizontal_negative = vertical_positive & diagonal;
    // Doubling inside a lane is the word's own shift, and the row above the query enters as a one at every
    // lane's lowest bit, which is exactly where the shift's cross-lane bit would otherwise land.
    sz_u32_t const shifted_positive = (horizontal_positive << 1) | lane_low_bits;
    sz_u32_t const shifted_negative = (horizontal_negative << 1) & ~lane_low_bits;
    *positive = shifted_negative | ~(vertical_carry | shifted_positive);
    *negative = shifted_positive & vertical_carry;
}

/** One lane's distance, read off the verticals at the position that lane's text ends: the row above the query
 *  costs one edit per candidate symbol and the query column's own deltas carry the rest. */
SZ_DEVICE_INLINE sz_size_t sz_levenshtein_cuda_lane_distance_(sz_u32_t positive, sz_u32_t negative,
                                                              sz_size_t lane_shift, sz_u32_t live, sz_size_t length) {
    sz_u32_t const lane_positive = (positive >> lane_shift) & live;
    sz_u32_t const lane_negative = (negative >> lane_shift) & live;
    return (sz_size_t)((sz_ssize_t)length + __popc(lane_positive) - __popc(lane_negative));
}

/**
 *  @brief Sweeps as many candidates as one register holds lanes, a whole Myers word to each lane.
 *
 *  A query of at most @p lane_bits symbols needs only that many bits of a Myers word, so four candidates ride
 *  where the threaded rung advances one and the recurrence's fifteen operations serve all of them. The lanes
 *  share nothing but the arithmetic: each keeps its own cursor and takes its own text eight bytes at a time
 *  from the aligned word the cursor sits in, so the loads stay as scattered as the threaded rung's.
 *
 *  A lane's distance is read off the verticals at the position its text ends, the way the warped rung reads
 *  its own, so no lane carries a running score and a lane past its text keeps stepping whatever it holds.
 *
 *  @param[in] lane_bits Bits one lane spans: 8 or 16, and a literal, which is what folds the lane masks.
 *  @param[in] lanes Lanes the register carries - a literal too, which is what keeps the lane arrays in registers.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_lanes_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                       sz_sequence_t candidates, sz_size_t *distances,
                                                       sz_size_t distances_stride, sz_size_t lane_bits,
                                                       sz_size_t lanes) {
    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;

    // At these query lengths a class's whole mask fits one lane, so the class map and the mask rows collapse
    // into one table the block builds once and every lane of every thread in it reads. Folding the two lookups
    // into one is worth 1.22x on XLSum words and 1.73x where the candidates are long enough to be step bound.
    __shared__ sz_u32_t byte_to_mask[sz_levenshtein_byte_classes_k];
    for (sz_size_t entry = threadIdx.x; entry < sz_levenshtein_byte_classes_k; entry += blockDim.x)
        byte_to_mask[entry] = (sz_u32_t)query.masks[(sz_size_t)query.byte_to_class[entry] * query.stride];
    __syncthreads();

    // A block takes one contiguous range, and a lane strides through it by the block's width, so the thirty-two
    // views one instruction reads are thirty-two consecutive ones and the texts behind them lie together too.
    sz_size_t const lane_stride = blockDim.x;
    sz_size_t const first_candidate = (sz_size_t)blockIdx.x * blockDim.x * lanes + threadIdx.x;
    if (first_candidate >= candidates.count) return;

    // A lane past the batch repeats the first candidate's text, whose address is one the device may read,
    // and takes length zero, which retires it before the first step.
    sz_cptr_t texts[sz_levenshtein_cuda_lanes_per_thread_max_k];
    sz_size_t lengths[sz_levenshtein_cuda_lanes_per_thread_max_k];
    sz_u64_t words[sz_levenshtein_cuda_lanes_per_thread_max_k];
    sz_u64_t octets[sz_levenshtein_cuda_lanes_per_thread_max_k];
    sz_u32_t unread = 0;
#pragma unroll
    for (sz_size_t lane = 0; lane != lanes; ++lane) {
        sz_size_t const candidate = first_candidate + lane * lane_stride;
        sz_size_t const index = candidate < candidates.count ? candidate : first_candidate;
        sz_size_t const length = candidate < candidates.count ? candidates.get_length(candidates.handle, index) : 0;
        sz_cptr_t const text = candidates.get_start(candidates.handle, index);
        sz_size_t const head = (sz_size_t)text & 7;
        sz_u64_t const *const aligned = (sz_u64_t const *)(text - head);
        texts[lane] = text, lengths[lane] = length, octets[lane] = 0;
        words[lane] = length ? aligned[0] : 0;
        if (length) unread |= (sz_u32_t)1 << lane;
        else if (candidate < candidates.count) row[candidate] = query.length;
    }

    sz_u32_t const live = ((sz_u32_t)1 << query.length) - 1;
    sz_u32_t const lane_low_bits = lane_bits == 8 ? 0x01010101u : 0x00010001u;
    sz_u32_t const lane_high_bits = lane_bits == 8 ? 0x80808080u : 0x80008000u;
    sz_u32_t positive = ~(sz_u32_t)0, negative = 0;
    // The earliest position an unread lane's text ends at, so a step costs one compare rather than a rescan.
    sz_size_t next_end = SZ_SIZE_MAX;
#pragma unroll
    for (sz_size_t lane = 0; lane != lanes; ++lane)
        if (unread & ((sz_u32_t)1 << lane)) next_end = sz_min_of_two(next_end, lengths[lane]);

    for (sz_size_t position = 0; unread != 0; position += sz_levenshtein_cuda_bytes_per_refill_k) {
        // A lane's next eight bytes span two aligned words, and the second is read only once it holds a byte
        // of the text: an aligned word that holds one lies in a page the device may read all eight bytes of.
#pragma unroll
        for (sz_size_t lane = 0; lane != lanes; ++lane) {
            sz_size_t const head = (sz_size_t)texts[lane] & 7;
            sz_u64_t const *const aligned = (sz_u64_t const *)(texts[lane] - head);
            sz_size_t const shift = head * 8;
            sz_u64_t const following = position + sz_levenshtein_cuda_bytes_per_refill_k < head + lengths[lane]
                                           ? aligned[position / sz_levenshtein_cuda_bytes_per_refill_k + 1]
                                           : 0;
            octets[lane] = shift ? (words[lane] >> shift) | (following << (64 - shift)) : words[lane];
            words[lane] = following;
        }
#pragma unroll
        for (sz_size_t slot = 0; slot != sz_levenshtein_cuda_bytes_per_refill_k; ++slot) {
            sz_u32_t equality = 0;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                sz_u32_t const byte = (sz_u32_t)(octets[lane] >> (slot * 8)) & 0xFF;
                equality |= byte_to_mask[byte] << (lane * lane_bits);
            }
            sz_levenshtein_cuda_lanes_step_(&positive, &negative, equality, lane_low_bits, lane_high_bits);
            if (position + slot + 1 != next_end) continue;
            next_end = SZ_SIZE_MAX;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                if ((unread & ((sz_u32_t)1 << lane)) == 0) continue;
                if (lengths[lane] != position + slot + 1) {
                    next_end = sz_min_of_two(next_end, lengths[lane]);
                    continue;
                }
                row[first_candidate + lane * lane_stride] = sz_levenshtein_cuda_lane_distance_(
                    positive, negative, lane * lane_bits, live, lengths[lane]);
                unread &= ~((sz_u32_t)1 << lane);
            }
            if (unread == 0) break;
        }
    }
}

/*  One entry point per lane width, each passing its own literals, for the reason the threaded rung states: a
 *  width the compiler cannot see turns the lane arrays into local memory. */
static __global__ void sz_levenshtein_u8x4_distances_cuda_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                           sz_sequence_t candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride) {
    sz_levenshtein_cuda_lanes_sweep_(engine, order, candidates, distances, distances_stride, 8, 4);
}

static __global__ void sz_levenshtein_u16x2_distances_cuda_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                            sz_sequence_t candidates, sz_size_t *distances,
                                                            sz_size_t distances_stride) {
    sz_levenshtein_cuda_lanes_sweep_(engine, order, candidates, distances, distances_stride, 16, 2);
}

/** Candidates one thread takes, which is the lanes a query of @p length symbols leaves in one register. */
static sz_size_t sz_levenshtein_cuda_lanes_per_thread_(sz_size_t length) {
    return length <= sz_levenshtein_cuda_byte_lanes_symbols_max_k ? 4 : 2;
}

/** Candidates a batch must carry before a narrow rung is launched at all, which the device's own residency
 *  scales. @c SZ_SIZE_MAX where the device cannot be asked, so the threaded rung keeps every batch. */
static sz_size_t sz_levenshtein_cuda_lanes_candidates_min_(void) {
    int device = 0, multiprocessors = 0, threads_per_multiprocessor = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return SZ_SIZE_MAX;
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return SZ_SIZE_MAX;
    if (cudaDeviceGetAttribute(&threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device) !=
        cudaSuccess)
        return SZ_SIZE_MAX;
    return (sz_size_t)multiprocessors * (sz_size_t)threads_per_multiprocessor * sz_levenshtein_cuda_lanes_waves_min_k;
}

#pragma endregion Myers Byte Lanes

#pragma region Myers UTF 8

/**
 *  @brief One candidate's Myers sweep over runes at a compile-time @p words, the byte sweep with a rune cursor.
 *
 *  The byte sweep's octet load has no counterpart here: a rune spans one to four bytes, so its width is known
 *  only once the lead byte is read, and the loop takes one rune per step. The class comes from the query's
 *  page table - two dependent loads, and no scratch of its own per candidate - rather than a byte map.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_sweep_utf8_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                      sz_sequence_t candidates, sz_size_t *distances,
                                                      sz_size_t distances_stride, sz_size_t words) {
    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;
    // The grid is what stays resident rather than what the corpus needs, so a block strides through many
    // candidates and its launch and setup are paid once instead of once per candidate.
    sz_size_t const grid_size = (sz_size_t)gridDim.x * blockDim.x;
    for (sz_size_t candidate = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; candidate < candidates.count;
         candidate += grid_size) {
        sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
        sz_size_t const length = candidates.get_length(candidates.handle, candidate);

        sz_levenshtein_u64x1_state_serial_t state;
        sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_cuda_thread_words_max_k];
        sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);

        sz_size_t position = 0;
        while (position < length) {
            sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
            sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, sz_levenshtein_utf8_class(&query, rune));
        }
        row[candidate] = sz_levenshtein_u64x1_score_serial(&state, candidate);
    }
}

/*  One rune entry point per word count, each passing its own literal, for the reason the byte tier states: a
 *  word count the compiler cannot see spills the verticals to local memory.
 */
static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w9_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w10_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w11_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w12_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w13_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w14_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w15_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_cuda_w16_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_sweep_utf8_(engine, order, candidates, distances, distances_stride, 16);
}

/*  One rune entry point per word count, addressed by it, beside the byte tier's own table. */
static void const *const sz_levenshtein_cuda_entry_points_utf8_[sz_levenshtein_cuda_thread_words_max_k] = {
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w1_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w2_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w3_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w4_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w5_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w6_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w7_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w8_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w9_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w10_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w11_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w12_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w13_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w14_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w15_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_cuda_w16_,
};

#pragma endregion Myers UTF 8

#pragma region Myers UTF 8 Warped

/**
 *  @brief One candidate's Myers sweep over runes across a whole warp, @p words_per_lane words to a lane, the
 *      byte warped sweep with the rune's class riding the chain the carry already rides.
 *
 *  Runes are variable-width, so no lane can address the rune at @c step-lane without decoding everything below
 *  it - and nothing here wants the rune, only its class. Lane zero decodes one rune per step and hands its class
 *  up through a second @c __shfl_up_sync, so the class walks one lane per step, which is the lag the skew already
 *  imposes; every rune is decoded once per candidate and its width is never inverted.
 *
 *  The chain carries the class plus one, leaving zero to mark a step whose rune is past the candidate's end -
 *  the liveness the byte sweep reads off its own position instead, and what a lane forwards through the fill and
 *  the drain. The warp runs while a live lane still holds a rune, so a candidate costs its runes, not its bytes.
 *
 *  @param[in] words_per_lane Exactly @c ceil(query_words/32); a literal, which keeps the verticals in registers.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_warp_sweep_utf8_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                           sz_sequence_t candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_size_t const candidate = (sz_size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    // Warp uniform, so the shuffles below still see a whole warp.
    if (candidate >= candidates.count) return;

    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;
    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
    sz_size_t const length = candidates.get_length(candidates.handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = (words + words_per_lane - 1) / words_per_lane;
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_cuda_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_cuda_warp_words_per_lane_max_k];
#pragma unroll
    for (sz_size_t word = 0; word != words_per_lane; ++word) positive[word] = ~(sz_u64_t)0, negative[word] = 0;

    // Two values walk one lane per step: the carry the byte rung packs, and the class of the rune being stepped.
    // Only lane zero's rune count is ever read, it being the lane that both decodes and writes the distance.
    unsigned carry = 0;
    sz_u32_t held = 0;
    sz_size_t cursor = 0, runes = 0;
    unsigned running = 1;
    while (running) {
        sz_u32_t decoded = 0;
        if (lane == 0 && cursor < length) {
            sz_rune_t const rune = sz_utf8_next_rune_(text, length, &cursor);
            decoded = sz_levenshtein_utf8_class(&query, rune) + 1, ++runes;
        }
        sz_u32_t const inherited = __shfl_up_sync(0xFFFFFFFFu, held, 1);
        unsigned const received = __shfl_up_sync(0xFFFFFFFFu, carry, 1);
        held = lane == 0 ? decoded : inherited;
        unsigned const live = held != 0 && lane < live_lanes;
        running = (unsigned)__any_sync(0xFFFFFFFFu, (int)live);
        if (!live) continue;

        sz_u64_t const *const masks = lane_masks + (sz_size_t)(held - 1) * query.stride;
        // The row above the query's first word is one edit higher than the cell left of it, which is lane
        // zero's carry at every rune rather than anything a shuffle brings.
        sz_u64_t positive_carry = lane == 0 ? 1 : (sz_u64_t)(received & 1u);
        sz_u64_t negative_carry = lane == 0 ? 0 : (sz_u64_t)((received >> 1) & 1u);
#pragma unroll
        for (sz_size_t word = 0; word != words_per_lane; ++word) {
            sz_u64_t const equality = masks[word];
            sz_u64_t const vertical_carry = equality | negative[word];
            sz_u64_t const matched = equality | negative_carry;
            sz_u64_t const diagonal = (((matched & positive[word]) + positive[word]) ^ positive[word]) | matched;
            sz_u64_t horizontal_positive = negative[word] | ~(diagonal | positive[word]);
            sz_u64_t horizontal_negative = positive[word] & diagonal;
            sz_u64_t const next_positive_carry = horizontal_positive >> 63;
            sz_u64_t const next_negative_carry = horizontal_negative >> 63;
            horizontal_positive = (horizontal_positive << 1) | positive_carry;
            horizontal_negative = (horizontal_negative << 1) | negative_carry;
            positive_carry = next_positive_carry, negative_carry = next_negative_carry;
            positive[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
            negative[word] = horizontal_positive & vertical_carry;
        }
        carry = (unsigned)positive_carry | ((unsigned)negative_carry << 1);
    }

    // The verticals are the query column's own deltas and the row above the query costs one edit per candidate
    // rune, so their sum over the query's bits is the distance and no lane has to track a running score through
    // the sweep. A dead word contributes nothing, and the last word drops the bits past the last symbol.
    sz_u64_t const last_symbol_bit = sz_levenshtein_last_symbol_bit_(query.length);
    sz_u64_t const last_word_live = last_symbol_bit | (last_symbol_bit - 1);
    int deltas = 0;
#pragma unroll
    for (sz_size_t word = 0; word != words_per_lane; ++word) {
        sz_size_t const index = first_word + word;
        sz_u64_t const live = index + 1 < words ? ~(sz_u64_t)0 : index + 1 == words ? last_word_live : 0;
        deltas += __popcll(positive[word] & live) - __popcll(negative[word] & live);
    }
#pragma unroll
    for (unsigned offset = 16; offset != 0; offset >>= 1) deltas += __shfl_down_sync(0xFFFFFFFFu, deltas, offset);
    if (lane == 0) row[candidate] = (sz_size_t)((sz_ssize_t)runes + deltas);
}

/*  One warped rune entry point per words-per-lane, each passing its own literal, for the reason the threaded
 *  rung states: a count the compiler cannot see spills the verticals to local memory.
 */
static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_cuda_k8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates,
    sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_cuda_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 8);
}

/*  One warped rune entry point per words-per-lane, addressed by it, beside the threaded rune tier's own table. */
static void const *const sz_levenshtein_cuda_entry_points_utf8_warp_[sz_levenshtein_cuda_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k1_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k2_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k3_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k4_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k5_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k6_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k7_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_cuda_k8_,
};

#pragma endregion Myers UTF 8 Warped

#pragma region Myers Engine

/** What a device round needs that the batch already fixed: its rung buckets and the geometry each of them takes. */
typedef struct sz_levenshtein_cuda_head_t {
    void *stream;                  /**< The @c cudaStream_t every round of this engine is scheduled on. */
    sz_size_t buckets;             /**< Rung keys the batch spans: @c words_max plus the two narrow ones. */
    sz_size_t candidates_min;      /**< Candidates before a narrow rung is launched, asked of the device once. */
    sz_size_t narrow_per_block[2]; /**< Threads the byte-lane and the short-lane entry points take. */
    sz_size_t *bucket_offsets;     /**< The @b [buckets+1] first position of each key inside @c order. */
    sz_size_t *per_block;          /**< The @b [buckets] threads that key's own wide entry point takes. */
    sz_u32_t *order;               /**< The @b [count] query indices, the keys' runs back to back. */
} sz_levenshtein_cuda_head_t;

/** The rung key a query of @p length symbols launches from: the two narrow lanes, then one key per word. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_cuda_bucket_(sz_size_t length) {
    if (length <= sz_levenshtein_cuda_byte_lanes_symbols_max_k) return 0;
    if (length <= sz_levenshtein_cuda_short_lanes_symbols_max_k) return 1;
    return 1 + sz_levenshtein_query_words(length);
}

/** Rung keys a batch whose widest query spans @p words words can reach. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_cuda_buckets_(sz_size_t words) { return words + 2; }

/** Query words the wide rung of @p bucket steps, the two narrow keys sharing the one-word entry points. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_cuda_bucket_words_(sz_size_t bucket) { return bucket <= 1 ? 1 : bucket - 1; }

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning @p buckets rung keys. */
SZ_HELPER_AUTO sz_size_t sz_levenshtein_cuda_head_bytes_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_cuda_head_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/**
 *  @brief The entry point @p bucket 's wide rung reaches: one candidate per thread, or one per warp past the crossing.
 *
 *  A narrow query's verticals fit one thread's registers, and nothing beats keeping the recurrence there. A wider
 *  one spreads them across a warp, where the skew turns the carry between words into one shuffle.
 */
static void const *sz_levenshtein_cuda_entry_point_(sz_levenshtein_symbol_t symbol, sz_size_t bucket) {
    sz_size_t const words = sz_levenshtein_cuda_bucket_words_(bucket);
    if (words < sz_levenshtein_cuda_warp_words_min_k)
        return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_cuda_entry_points_[words - 1]
                                                : sz_levenshtein_cuda_entry_points_utf8_[words - 1];
    sz_size_t const per_lane = sz_levenshtein_cuda_warp_words_per_lane_(words);
    return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_cuda_entry_points_warp_[per_lane - 1]
                                            : sz_levenshtein_cuda_entry_points_utf8_warp_[per_lane - 1];
}

/**
 *  @brief Moves a host-filled block to the device in one migration, on the stream that is about to read it.
 *
 *  Managed pages sit wherever their last toucher left them, so a kernel reading a table the host has just filled
 *  faults it in a page at a time - 1,732 microseconds against 225 for the bulk move at the widest query this
 *  ladder admits. Memory the driver does not manage is not migratable and reports as much, which is not an error.
 */
static void sz_levenshtein_cuda_reach_device_(void const *table, sz_size_t bytes, cudaStream_t stream) {
    cudaMemLocation where;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    where.type = cudaMemLocationTypeDevice;
    where.id = device;
    [[maybe_unused]] cudaError_t const moved = cudaMemPrefetchAsync(table, bytes, where, 0, stream);
}

/** Points the head's tables into the block, buckets the batch by rung key, and asks the device its geometry once. */
static void sz_levenshtein_cuda_bind_head_(sz_levenshtein_engine_t *engine, sz_size_t buckets_bound, void *stream) {
    sz_levenshtein_cuda_head_t *const head = (sz_levenshtein_cuda_head_t *)engine->memory;
    sz_size_t *const tables = (sz_size_t *)((sz_ptr_t)engine->memory + sizeof(sz_levenshtein_cuda_head_t));
    sz_size_t cursors[sz_levenshtein_cuda_words_max_k + 2];
    head->stream = stream;
    head->buckets = sz_levenshtein_cuda_buckets_(sz_levenshtein_engine_words_max_(engine));
    head->bucket_offsets = tables;
    head->per_block = tables + buckets_bound + 1;
    head->order = (sz_u32_t *)(tables + 2 * buckets_bound + 1);

    // A counting sort by rung key, so one launch only ever carries queries that share an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) head->bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        ++head->bucket_offsets[sz_levenshtein_cuda_bucket_(engine->lengths[index]) + 1];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        head->bucket_offsets[bucket] += head->bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = head->bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        head->order[cursors[sz_levenshtein_cuda_bucket_(engine->lengths[index])]++] = (sz_u32_t)index;

    // Every driver round trip a round would otherwise pay: the residency floor once, and one occupancy walk
    // per entry point the batch can reach, which the buckets fixed here and no later call can widen.
    head->candidates_min = sz_levenshtein_cuda_lanes_candidates_min_();
    head->narrow_per_block[0] = sz_levenshtein_cuda_per_block_((void const *)sz_levenshtein_u8x4_distances_cuda_);
    head->narrow_per_block[1] = sz_levenshtein_cuda_per_block_((void const *)sz_levenshtein_u16x2_distances_cuda_);
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket)
        head->per_block[bucket] = head->bucket_offsets[bucket] == head->bucket_offsets[bucket + 1]
                                      ? 0
                                      : sz_levenshtein_cuda_per_block_(
                                            sz_levenshtein_cuda_entry_point_(engine->symbol, bucket));
}

/** Threads the mask builder runs: one per byte value, so a thread owns one class flag for the whole build. */
enum { sz_levenshtein_cuda_masks_threads_k = sz_levenshtein_byte_classes_k };

/**
 *  @brief Builds one byte query's match masks and class map per grid row, on the device itself.
 *
 *  The passes are @ref sz_levenshtein_query_prepare 's: flag the byte values the query holds, hand them dense
 *  classes in byte order, give the row past them to every byte the query lacks, then set one bit per position
 *  in its class's row. One block per query, so the ranks walk shared memory rather than the grid, and the planes
 *  arrive zeroed from the memset the stream runs ahead of this launch.
 *
 *  @param[in] first The batch index of grid row zero, since the query axis is cut at the grid's own ceiling.
 */
static __global__ void sz_levenshtein_cuda_masks_kernel_(sz_levenshtein_engine_t engine, sz_string_view_t const *texts,
                                                         sz_size_t first) {
    __shared__ sz_u32_t ranks[sz_levenshtein_byte_classes_k];
    __shared__ sz_u32_t classes;
    sz_size_t const query = first + (sz_size_t)blockIdx.y;
    sz_cptr_t const text = texts[query].start;
    sz_size_t const length = texts[query].length;
    sz_size_t const stride = sz_levenshtein_query_stride(length);
    sz_u64_t *const masks = (sz_u64_t *)engine.masks + engine.masks_offsets[query];
    sz_u8_t *const byte_to_class = (sz_u8_t *)engine.symbol_to_class + query * sz_levenshtein_byte_classes_k;

    // The block is exactly as wide as the byte values, so a thread owns one of them throughout.
    ranks[threadIdx.x] = 0;
    __syncthreads();
    for (sz_size_t position = threadIdx.x; position < length; position += blockDim.x)
        ranks[(sz_u8_t)text[position]] = 1;
    __syncthreads();
    // One thread ranks the flags: the walk is 256 shared-memory steps, and every pass below needs all of them.
    if (threadIdx.x == 0) {
        sz_u32_t distinct = 0;
        for (sz_size_t byte = 0; byte != sz_levenshtein_byte_classes_k; ++byte)
            if (ranks[byte]) ranks[byte] = ++distinct;
        classes = distinct < sz_levenshtein_byte_classes_k ? distinct + 1 : sz_levenshtein_byte_classes_k;
    }
    __syncthreads();
    byte_to_class[threadIdx.x] = ranks[threadIdx.x] ? (sz_u8_t)(ranks[threadIdx.x] - 1) : (sz_u8_t)(classes - 1);
    // A thread owns one word of every class's row, so no two of them ever fold into the same entry.
    for (sz_size_t word = threadIdx.x; word * 64 < length; word += blockDim.x) {
        sz_size_t const start = word * 64, end = sz_min_of_two(start + 64, length);
        for (sz_size_t position = start; position != end; ++position)
            masks[(sz_size_t)(ranks[(sz_u8_t)text[position]] - 1) * stride + word] |= (sz_u64_t)1 << (position & 63);
    }
}

/**
 *  @brief Stages the batch's query texts where the device reads them and builds every byte plane there.
 *  @note Joins @p stream, which is what lets the staging be released; only an init is allowed to.
 */
static sz_status_t sz_levenshtein_cuda_build_masks_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                    cudaStream_t stream) {
    sz_size_t texts_bytes = 0;
    for (sz_size_t index = 0; index != queries->count; ++index)
        texts_bytes += queries->get_length(queries->handle, index);
    sz_size_t const views_bytes = queries->count * sizeof(sz_string_view_t);
    sz_size_t const staged_bytes = views_bytes + texts_bytes;
    sz_ptr_t const staged = (sz_ptr_t)engine->alloc.allocate(staged_bytes, engine->alloc.handle);
    if (!staged) return sz_bad_alloc_k;
    sz_string_view_t *const views = (sz_string_view_t *)staged;
    sz_ptr_t const arena = staged + views_bytes;
    for (sz_size_t index = 0, written = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        for (sz_size_t byte = 0; byte != bytes; ++byte) arena[written + byte] = text[byte];
        views[index].start = arena + written, views[index].length = bytes;
        written += bytes;
    }

    sz_size_t const masks_bytes = engine->masks_offsets[engine->count] * sizeof(sz_u64_t);
    cudaError_t launched = cudaMemsetAsync((void *)engine->masks, 0, masks_bytes, stream);
    sz_levenshtein_cuda_reach_device_(engine->memory, engine->memory_bytes, stream);
    sz_levenshtein_cuda_reach_device_(staged, staged_bytes, stream);
    for (sz_size_t first = 0; first < queries->count && launched == cudaSuccess;
         first += sz_levenshtein_cuda_grid_rows_max_k) {
        sz_levenshtein_engine_t launch_engine = *engine;
        sz_string_view_t const *launch_views = views;
        sz_size_t launch_first = first;
        void *arguments[3];
        arguments[0] = &launch_engine, arguments[1] = &launch_views, arguments[2] = &launch_first;
        dim3 grid, block;
        grid.x = 1, grid.z = 1;
        grid.y = (unsigned)sz_min_of_two(queries->count - first, (sz_size_t)sz_levenshtein_cuda_grid_rows_max_k);
        block.x = sz_levenshtein_cuda_masks_threads_k, block.y = 1, block.z = 1;
        launched = cudaLaunchKernel((void const *)sz_levenshtein_cuda_masks_kernel_, grid, block, arguments, 0,
                                    stream);
    }
    // The staging is the host's, so it outlives the builder only as long as the join below takes.
    cudaError_t const drained = launched == cudaSuccess ? cudaStreamSynchronize(stream) : launched;
    engine->alloc.free(staged, staged_bytes, engine->alloc.handle);
    return drained == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_engine_init_cuda(sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol,
                                                            sz_memory_allocator_t *alloc, void *stream,
                                                            sz_levenshtein_engine_t *engine) {
    cudaStream_t const on = (cudaStream_t)stream;
    sz_memory_allocator_t unified;
    if (alloc) unified = *alloc;
    else sz_memory_allocator_init_unified(&unified, SZ_NULL);
    if (queries->count == 0) return sz_unexpected_dimensions_k;

    // Every query seeds its score from its own last word, so an empty one has no word to read it off, and the
    // rung ceiling is checked twice: on the bytes here, which bound the runes, and on the symbols once measured.
    sz_size_t longest = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (bytes == 0) return sz_unexpected_dimensions_k;
        longest = sz_max_of_two(longest, bytes);
    }
    if (symbol == sz_levenshtein_bytes_k && longest > sz_levenshtein_cuda_words_max_k * 64)
        return sz_unexpected_dimensions_k;

    sz_size_t const buckets_bound = sz_levenshtein_cuda_buckets_(sz_size_divide_round_up(longest, 64));
    sz_size_t const head_bytes = sz_levenshtein_cuda_head_bytes_(queries->count, buckets_bound);
    sz_status_t status = sz_levenshtein_engine_build_(queries, symbol, head_bytes, &unified, engine);
    if (status != sz_success_k) return status;
    if (sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_cuda_words_max_k) {
        sz_levenshtein_engine_free_(engine);
        return sz_unexpected_dimensions_k;
    }
    engine->capability = sz_cap_cuda_k;
    sz_levenshtein_cuda_bind_head_(engine, buckets_bound, stream);

    // The rune planes are the host's: a page table is a scan with no counterpart here, and an init may join
    // where a round may not, so the one crossing is a bulk migration rather than a fault per page.
    if (symbol == sz_levenshtein_runes_k) {
        sz_levenshtein_engine_fill_(engine, queries);
        sz_levenshtein_cuda_reach_device_(engine->memory, engine->memory_bytes, on);
        return sz_success_k;
    }
    status = sz_levenshtein_cuda_build_masks_(engine, queries, on);
    if (status != sz_success_k) sz_levenshtein_engine_free_(engine);
    return status;
}

/** Launches whichever rung's entry point the caller chose, over a grid of candidate tiles by prepared queries. */
static cudaError_t sz_levenshtein_cuda_distances_(void const *entry_point, sz_size_t blocks, sz_size_t queries,
                                                  sz_size_t per_block, cudaStream_t stream,
                                                  sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                  sz_sequence_t candidates, sz_size_t *distances,
                                                  sz_size_t distances_stride) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)queries, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[5];
    arguments[0] = &engine, arguments[1] = &order, arguments[2] = &candidates, arguments[3] = &distances;
    arguments[4] = &distances_stride;
    return cudaLaunchKernel(entry_point, grid, block, arguments, 0, stream);
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (candidates->count == 0) return sz_success_k;

    // The handle is checked, never the accessors: those are the device's to call, so the host must not, and a
    // pointer is all this side can inspect. That the texts they answer are device-reachable is the caller's word.
    if (!sz_memory_reaches_device(distances)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(candidates->handle)) return sz_device_memory_mismatch_k;

    sz_levenshtein_cuda_head_t const *const head = (sz_levenshtein_cuda_head_t const *)engine->memory;
    cudaStream_t const on = (cudaStream_t)head->stream;
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        sz_size_t const first = head->bucket_offsets[bucket], last = head->bucket_offsets[bucket + 1];
        if (first == last) continue;

        // A query short enough to leave a word's bits unused shares that word between candidates, which pays
        // once the batch is wide enough to keep the narrower grid resident; the lanes read a byte map, so a
        // rune batch keeps the threaded rung whatever its queries are worth.
        sz_bool_t const narrow = bucket <= 1 && engine->symbol == sz_levenshtein_bytes_k &&
                                         candidates->count >= head->candidates_min
                                     ? sz_true_k
                                     : sz_false_k;
        void const *entry_point = sz_levenshtein_cuda_entry_point_(engine->symbol, bucket);
        sz_size_t per_block = head->per_block[bucket];
        sz_size_t candidates_per_block = per_block;
        if (narrow) {
            entry_point = bucket == 0 ? (void const *)sz_levenshtein_u8x4_distances_cuda_
                                      : (void const *)sz_levenshtein_u16x2_distances_cuda_;
            per_block = head->narrow_per_block[bucket];
            candidates_per_block = per_block * sz_levenshtein_cuda_lanes_per_thread_(
                                                   bucket == 0 ? sz_levenshtein_cuda_byte_lanes_symbols_max_k
                                                               : sz_levenshtein_cuda_short_lanes_symbols_max_k);
        }
        else if (sz_levenshtein_cuda_bucket_words_(bucket) >= sz_levenshtein_cuda_warp_words_min_k)
            candidates_per_block = per_block / sz_levenshtein_cuda_warp_lanes_k;
        sz_size_t const blocks = sz_size_divide_round_up(candidates->count, candidates_per_block);

        for (sz_size_t row = first; row < last; row += sz_levenshtein_cuda_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(last - row, (sz_size_t)sz_levenshtein_cuda_grid_rows_max_k);
            cudaError_t const launched = sz_levenshtein_cuda_distances_(entry_point, blocks, rows, per_block, on,
                                                                        *engine, head->order + row, *candidates,
                                                                        distances, distances_stride);
            if (launched != cudaSuccess) return sz_device_code_mismatch_k;
        }
    }
    return sz_success_k;
}

#pragma endregion Myers Engine

#pragma region Tiled

/**
 *  @brief Tile geometry of the device-spanning wavefront: one warp owns a 128-wide tile-column, one lane
 *      owns a 4x4 register micro-tile, so a tile is a 63-step anti-diagonal sweep across 32 micro-rows.
 */
enum {
    sz_levenshtein_cuda_tile_side_k = 128,
    sz_levenshtein_cuda_micro_side_k = 4,
    sz_levenshtein_cuda_lanes_k = 32,
    sz_levenshtein_cuda_micro_rows_k = sz_levenshtein_cuda_tile_side_k / sz_levenshtein_cuda_micro_side_k,
    sz_levenshtein_cuda_tiled_warps_per_block_k = 8,
    sz_levenshtein_cuda_tiled_threads_per_block_k = sz_levenshtein_cuda_tiled_warps_per_block_k *
                                                    sz_levenshtein_cuda_lanes_k,
};

/*  Longest text the wavefront indexes. Lengths and cells are `sz_u32_t` inside the kernel, and the last tile
 *  of each axis is padded up to 128 columns, so the ceiling leaves that padding room rather than letting a
 *  length near the word's top wrap a column index. */
#define SZ_LEVENSHTEIN_CUDA_TILED_LENGTH_MAX (0xFFFFFF00u)

/** Bytes standing in for a character past the end of a text. The two differ, so a padded row never matches
 *  a padded column, and a padded cell never reaches an in-bounds one. */
enum {
    sz_levenshtein_cuda_past_query_k = 0xFE,
    sz_levenshtein_cuda_past_target_k = 0xFF,
};

/*  Tile-rows a pair needs before the wavefront is worth reaching for. Its makespan is tile-rows plus
 *  tile-columns tile latencies, so tile-rows is how many tile-columns it ever runs at once: under three of
 *  them the wavefront is a serial chain down the long axis and one Myers lane's bit-parallel sweep is
 *  quicker, while from three up it is measured ahead at every long-axis length. */
enum { sz_levenshtein_cuda_tiled_rows_min_k = 3 };

/** Selects what a micro-tile march does with a finished cell: nothing, or the corner test that publishes the
 *  pair's distance. */
typedef enum {
    sz_levenshtein_cuda_march_fast_k = 0,
    sz_levenshtein_cuda_march_checked_k = 1,
} sz_levenshtein_cuda_march_t;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

/** Three-way unsigned minimum, one instruction on Hopper and Blackwell. */
SZ_DEVICE_INLINE sz_u32_t sz_levenshtein_cuda_min3_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
    return __vimin3_u32(first, second, third);
}

#else

/** Three-way unsigned minimum as a pair of comparisons, for targets without the fused form. */
SZ_DEVICE_INLINE sz_u32_t sz_levenshtein_cuda_min3_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
    sz_u32_t const smaller = first < second ? first : second;
    return smaller < third ? smaller : third;
}

#endif

/**
 *  @brief Publishes that this tile-column finished @p tile_row, releasing the frontier writes before it.
 *
 *  The release is spelled in PTX because the library's C tiers cannot reach @c cuda::atomic_ref, and because
 *  the alternative - a @c __threadfence next to a @c volatile store - orders every prior access of the thread
 *  where one location's release is all the protocol needs, and costs a membar this form does not. Only the
 *  lane that wrote the frontier calls it, so the release orders its own stores and nothing has to argue about
 *  what a warp barrier carries across lanes.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_publish_(sz_u32_t *counter, sz_u32_t tile_row) {
    sz_u32_t const published = tile_row + 1u;
    asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(counter), "r"(published) : "memory");
}

/**
 *  @brief Spins until the left tile-column published past @p tile_row, acquiring its frontier writes.
 *
 *  Every lane of the warp polls, not just one: the warp's loads of a single address collapse into one
 *  transaction, so the traffic is a lane-0 poll's, and each lane's own acquire orders each lane's own reads.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_await_(sz_u32_t const *counter, sz_u32_t tile_row) {
    sz_u32_t observed = 0;
    do {
        asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(observed) : "l"(counter) : "memory");
    } while (observed <= tile_row);
}

/**
 *  @brief Marches one 128x128 tile as an anti-diagonal wavefront of 4x4 register micro-tiles.
 *
 *  Lane @e l owns micro-column @e l and enters at wavefront step @e l, so 32 micro-rows across 32 lanes take
 *  @c micro_rows_k+lanes_k-1 steps. Lane 0 reads its left column and diagonal corner from the staged
 *  @p shared_left; every other lane receives the left neighbour's right column and top-right corner from
 *  @c __shfl_up_sync, which is itself the warp-wide rendezvous ordering the two. @p carry_top enters holding
 *  the row above the tile and leaves holding the tile's bottom row.
 *
 *  @param[in] march Whether finished cells need the corner test, which only a partial or corner tile does.
 *  @param[in] lane_index This thread's lane, which is also its micro-column.
 *  @param[in] tile_first_row Zero-based DP row the tile's first micro-row cell sits under.
 *  @param[in] tile_first_column Zero-based DP column the tile's first micro-column cell sits beside.
 *  @param[in] shorter_length Length of the text along the row axis, for the corner test.
 *  @param[in] longer_length Length of the text along the column axis, for the corner test.
 *  @param[in] shared_query The tile's 128 query characters, staged once per tile-row.
 *  @param[in] shared_left The tile's 128 incoming left-boundary cells, staged once per tile-row.
 *  @param[in] tile_corner The cell diagonally above the tile's top-left cell.
 *  @param[in] target_chars This lane's four target characters, constant down the whole tile-column.
 *  @param[inout] carry_top The four cells directly above this lane's micro-tile.
 *  @param[out] row_frontier One cell per DP row; the rightmost lane publishes the tile's right column into it.
 *  @param[out] distance Written once, by the thread that computes the matrix corner.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_march_tile_(                           //
    sz_levenshtein_cuda_march_t march, unsigned lane_index,                      //
    sz_u32_t tile_first_row, sz_u32_t tile_first_column,                         //
    sz_u32_t shorter_length, sz_u32_t longer_length,                             //
    char const *shared_query, sz_u32_t const *shared_left, sz_u32_t tile_corner, //
    char const *target_chars, sz_u32_t *carry_top, sz_u32_t *row_frontier, sz_size_t *distance) {

    sz_u32_t previous_right_edge[sz_levenshtein_cuda_micro_side_k];
    sz_u32_t previous_topright = 0;
    unsigned const wavefront_steps = sz_levenshtein_cuda_micro_rows_k + sz_levenshtein_cuda_lanes_k - 1;
    unsigned element, wavefront_step, micro_row_cell, micro_column_cell;
#pragma unroll
    for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element) previous_right_edge[element] = 0;

    for (wavefront_step = 0; wavefront_step != wavefront_steps; ++wavefront_step) {
        // Underflows for a lane the wavefront has not reached yet, which the unsigned bound below rejects.
        unsigned const micro_row = wavefront_step - lane_index;
        sz_u32_t shuffled_right_edge[sz_levenshtein_cuda_micro_side_k];
        sz_u32_t shuffled_topright;
#pragma unroll
        for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
            shuffled_right_edge[element] = __shfl_up_sync(0xFFFFFFFFu, previous_right_edge[element], 1);
        shuffled_topright = __shfl_up_sync(0xFFFFFFFFu, previous_topright, 1);
        if (micro_row >= sz_levenshtein_cuda_micro_rows_k) continue;

        sz_u32_t const micro_first_row = tile_first_row + micro_row * sz_levenshtein_cuda_micro_side_k;
        sz_u32_t left_column[sz_levenshtein_cuda_micro_side_k], diagonal_corner;
        if (lane_index == 0) {
#pragma unroll
            for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
                left_column[element] = shared_left[micro_row * sz_levenshtein_cuda_micro_side_k + element];
            diagonal_corner = micro_row == 0 ? tile_corner
                                             : shared_left[micro_row * sz_levenshtein_cuda_micro_side_k - 1];
        }
        else {
#pragma unroll
            for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
                left_column[element] = shuffled_right_edge[element];
            diagonal_corner = shuffled_topright;
        }

        // The top-right corner the lane to the right needs is the last cell of this micro-tile's incoming top
        // row, which the march below overwrites, so it is captured first.
        sz_u32_t const topright_for_next_lane = carry_top[sz_levenshtein_cuda_micro_side_k - 1];
        sz_u32_t above_row[sz_levenshtein_cuda_micro_side_k + 1];
        sz_u32_t right_edge[sz_levenshtein_cuda_micro_side_k];
        above_row[0] = diagonal_corner;
#pragma unroll
        for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
            above_row[element + 1] = carry_top[element];

#pragma unroll
        for (micro_row_cell = 1; micro_row_cell <= sz_levenshtein_cuda_micro_side_k; ++micro_row_cell) {
            sz_u32_t current_row[sz_levenshtein_cuda_micro_side_k + 1];
            char const query_char = shared_query[micro_row * sz_levenshtein_cuda_micro_side_k + micro_row_cell - 1];
            current_row[0] = left_column[micro_row_cell - 1];
#pragma unroll
            for (micro_column_cell = 1; micro_column_cell <= sz_levenshtein_cuda_micro_side_k; ++micro_column_cell) {
                sz_u32_t const substitution = query_char == target_chars[micro_column_cell - 1] ? 0u : 1u;
                sz_u32_t const cell = sz_levenshtein_cuda_min3_(above_row[micro_column_cell - 1] + substitution,
                                                                above_row[micro_column_cell] + 1u,
                                                                current_row[micro_column_cell - 1] + 1u);
                current_row[micro_column_cell] = cell;
                if (march == sz_levenshtein_cuda_march_checked_k) {
                    sz_u32_t const matrix_row = micro_first_row + micro_row_cell;
                    sz_u32_t const matrix_column = tile_first_column + lane_index * sz_levenshtein_cuda_micro_side_k +
                                                   micro_column_cell;
                    if (matrix_row == shorter_length && matrix_column == longer_length) *distance = (sz_size_t)cell;
                }
            }
            right_edge[micro_row_cell - 1] = current_row[sz_levenshtein_cuda_micro_side_k];
#pragma unroll
            for (element = 0; element <= sz_levenshtein_cuda_micro_side_k; ++element)
                above_row[element] = current_row[element];
        }

#pragma unroll
        for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
            carry_top[element] = above_row[element + 1], previous_right_edge[element] = right_edge[element];
        previous_topright = topright_for_next_lane;
        // The rightmost micro-column is the tile's right column, which is the next tile-column's left one.
        if (lane_index == sz_levenshtein_cuda_lanes_k - 1)
#pragma unroll
            for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element)
                row_frontier[micro_first_row + element + 1] = right_edge[element];
    }
}

/**
 *  @brief One pair's Levenshtein distance as an anti-diagonal wavefront spanning the whole device.
 *
 *  A warp owns a 128-wide tile-column and marches it top to bottom, taking the tile-column a whole grid
 *  further on when it runs out, so the launched grid can be capped at what stays co-resident. Every tile's
 *  top edge is free - it stays in the marching warp's registers - and only the left edge crosses warps,
 *  through @p row_frontier under the @p progress counters. Tile-column @e c reads the band tile-column
 *  @e c-1 released and overwrites it with its own right column; the counters serialize that into one read
 *  and one write per band per column, which is also what lets a warp reuse the band for a later
 *  tile-column. There is no grid barrier and no cooperative launch.
 *
 *  @param[in] shorter_text Text along the row axis, device-reachable, @p shorter_length bytes.
 *  @param[in] shorter_length Its length, at most @c SZ_LEVENSHTEIN_CUDA_TILED_LENGTH_MAX.
 *  @param[in] longer_text Text along the column axis, device-reachable, @p longer_length bytes.
 *  @param[in] longer_length Its length, at most @c SZ_LEVENSHTEIN_CUDA_TILED_LENGTH_MAX.
 *  @param[out] row_frontier Scratch of @c round_up(shorter_length,128)+1 cells, needing no seeding.
 *  @param[out] progress One counter per tile-column, zeroed before the launch.
 *  @param[out] distance The pair's distance, written once by the thread owning the matrix corner.
 */
static __global__ __launch_bounds__(sz_levenshtein_cuda_tiled_threads_per_block_k) void //
    sz_levenshtein_cuda_tiled_kernel_(                                                  //
        sz_cptr_t shorter_text, sz_u32_t shorter_length,                                //
        sz_cptr_t longer_text, sz_u32_t longer_length,                                  //
        sz_u32_t *row_frontier, sz_u32_t *progress, sz_size_t *distance) {

    // Each warp stages its tile's query window and its incoming left boundary once per tile-row, so the
    // wavefront's scattered lane-0 boundary reads come off-chip once instead of once per micro-tile.
    __shared__ char shared_query[sz_levenshtein_cuda_tiled_warps_per_block_k][sz_levenshtein_cuda_tile_side_k];
    __shared__ sz_u32_t shared_left[sz_levenshtein_cuda_tiled_warps_per_block_k][sz_levenshtein_cuda_tile_side_k];

    unsigned const warp_in_block = threadIdx.x >> 5;
    unsigned const lane_index = threadIdx.x & 31u;
    sz_u32_t const tile_grid_rows = (shorter_length + sz_levenshtein_cuda_tile_side_k - 1u) /
                                    sz_levenshtein_cuda_tile_side_k;
    sz_u32_t const tile_grid_columns = (longer_length + sz_levenshtein_cuda_tile_side_k - 1u) /
                                       sz_levenshtein_cuda_tile_side_k;
    sz_u32_t const warps_in_grid = (gridDim.x * blockDim.x) >> 5;
    sz_u32_t const corner_tile_row = (shorter_length - 1u) / sz_levenshtein_cuda_tile_side_k;
    sz_u32_t const corner_tile_column = (longer_length - 1u) / sz_levenshtein_cuda_tile_side_k;
    sz_u32_t const first_tile_column = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    sz_u32_t tile_column, tile_row;
    unsigned element, stage_row;

    for (tile_column = first_tile_column; tile_column < tile_grid_columns; tile_column += warps_in_grid) {
        sz_u32_t const tile_first_column = tile_column * sz_levenshtein_cuda_tile_side_k;
        char target_chars[sz_levenshtein_cuda_micro_side_k];
        sz_u32_t carry_top[sz_levenshtein_cuda_micro_side_k];
        // The row above the whole matrix costs one deletion per column, and this lane's target characters
        // never change down the column.
#pragma unroll
        for (element = 0; element != sz_levenshtein_cuda_micro_side_k; ++element) {
            sz_u32_t const target_index = tile_first_column + lane_index * sz_levenshtein_cuda_micro_side_k + element;
            target_chars[element] = target_index < longer_length ? longer_text[target_index]
                                                                 : (char)sz_levenshtein_cuda_past_target_k;
            carry_top[element] = target_index + 1u;
        }
        // The leftmost column of the whole matrix costs one insertion per row, so the first tile-column
        // synthesizes its left boundary instead of reading a seeded one, and nothing has to seed it.
        sz_u32_t tile_corner = tile_first_column;

        for (tile_row = 0; tile_row != tile_grid_rows; ++tile_row) {
            sz_u32_t const tile_first_row = tile_row * sz_levenshtein_cuda_tile_side_k;
            sz_u32_t tile_bottom_left;
            sz_levenshtein_cuda_march_t march;

            if (tile_column != 0) sz_levenshtein_cuda_await_(progress + (tile_column - 1u), tile_row);
            // Holds this tile-row's staging behind the previous one's reads of the same shared window.
            __syncwarp();
            if (tile_column == 0)
                for (stage_row = lane_index; stage_row < sz_levenshtein_cuda_tile_side_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = tile_first_row + 1u + stage_row;
            else
                for (stage_row = lane_index; stage_row < sz_levenshtein_cuda_tile_side_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = row_frontier[tile_first_row + 1u + stage_row];
            for (stage_row = lane_index; stage_row < sz_levenshtein_cuda_tile_side_k; stage_row += 32u)
                shared_query[warp_in_block][stage_row] = tile_first_row + stage_row < shorter_length
                                                             ? shorter_text[tile_first_row + stage_row]
                                                             : (char)sz_levenshtein_cuda_past_query_k;
            __syncwarp();

            // The tile's bottom-left cell is the diagonal corner of the tile below, and the march is about to
            // overwrite the frontier band it sits in.
            tile_bottom_left = shared_left[warp_in_block][sz_levenshtein_cuda_tile_side_k - 1];
            // Exactly one tile of the whole matrix holds the corner cell, and only that tile pays the
            // per-cell test; keeping the guarded store out of every other tile's hot loop is what lets the
            // march stay in registers.
            march = tile_row == corner_tile_row && tile_column == corner_tile_column
                        ? sz_levenshtein_cuda_march_checked_k
                        : sz_levenshtein_cuda_march_fast_k;
            if (march == sz_levenshtein_cuda_march_fast_k)
                sz_levenshtein_cuda_march_tile_(sz_levenshtein_cuda_march_fast_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            else
                sz_levenshtein_cuda_march_tile_(sz_levenshtein_cuda_march_checked_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            tile_corner = tile_bottom_left;
            if (lane_index == sz_levenshtein_cuda_lanes_k - 1)
                sz_levenshtein_cuda_publish_(progress + tile_column, tile_row);
        }
    }
}

/** Blocks of the wavefront the device holds at once; every launched block has to be resident, since a block
 *  spins on a tile-column another block owns. */
static cudaError_t sz_levenshtein_cuda_tiled_resident_blocks_(sz_size_t *blocks) {
    int device = 0, multiprocessors = 0, blocks_per_multiprocessor = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    error = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess) return error;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                          (void const *)sz_levenshtein_cuda_tiled_kernel_,
                                                          sz_levenshtein_cuda_tiled_threads_per_block_k, 0);
    if (error != cudaSuccess) return error;
    if (blocks_per_multiprocessor < 1) blocks_per_multiprocessor = 1;
    *blocks = (sz_size_t)blocks_per_multiprocessor * (sz_size_t)multiprocessors;
    return cudaSuccess;
}

/**
 *  @brief One pair's Levenshtein distance through the tiled wavefront, on texts the device already reaches.
 *  @param[in] a First text, device-reachable.
 *  @param[in] a_length Its length in bytes.
 *  @param[in] b Second text, device-reachable.
 *  @param[in] b_length Its length in bytes.
 *  @param[in] alloc Hands back the frontier scratch, which has to be memory the device reaches.
 *  @param[out] distance Host-readable slot receiving the distance.
 *  @param[in] stream The @c cudaStream_t to schedule on, or @c SZ_NULL for the default one.
 *  @retval sz_unexpected_dimensions_k when either text is longer than the kernel indexes.
 *  @retval sz_device_memory_mismatch_k when a text or the scratch is not memory the device reaches.
 *  @sa sz_levenshtein_distances_cuda, which scores a whole batch through the bit-parallel rungs instead.
 */
SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_tiled_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                               sz_size_t *distance, void *stream) {
    if (a_length > SZ_LEVENSHTEIN_CUDA_TILED_LENGTH_MAX || b_length > SZ_LEVENSHTEIN_CUDA_TILED_LENGTH_MAX)
        return sz_unexpected_dimensions_k;

    // The recurrence is symmetric, and putting the shorter text on the row axis keeps the frontier small and
    // makes the parallel axis the long one.
    sz_cptr_t const shorter_text = a_length <= b_length ? a : b;
    sz_cptr_t const longer_text = a_length <= b_length ? b : a;
    sz_u32_t const shorter_length = (sz_u32_t)(a_length <= b_length ? a_length : b_length);
    sz_u32_t const longer_length = (sz_u32_t)(a_length <= b_length ? b_length : a_length);
    if (shorter_length == 0) {
        *distance = longer_length;
        return sz_success_k;
    }
    if (!sz_memory_reaches_device(shorter_text) || !sz_memory_reaches_device(longer_text))
        return sz_device_memory_mismatch_k;

    sz_size_t const tile_grid_columns = (longer_length + sz_levenshtein_cuda_tile_side_k - 1u) /
                                        sz_levenshtein_cuda_tile_side_k;
    sz_size_t const row_frontier_cells = (sz_size_t)((shorter_length + sz_levenshtein_cuda_tile_side_k - 1u) /
                                                     sz_levenshtein_cuda_tile_side_k) *
                                             sz_levenshtein_cuda_tile_side_k +
                                         1u;
    // The result leads the block, so the wider alignment is the allocator's rather than this arithmetic's.
    sz_size_t const scratch_bytes = sizeof(sz_size_t) + (row_frontier_cells + tile_grid_columns) * sizeof(sz_u32_t);
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    if (!sz_memory_reaches_device(scratch)) {
        alloc->free(scratch, scratch_bytes, alloc->handle);
        return sz_device_memory_mismatch_k;
    }

    sz_size_t *const device_distance = (sz_size_t *)scratch;
    sz_u32_t *const row_frontier = (sz_u32_t *)(device_distance + 1);
    sz_u32_t *const progress = row_frontier + row_frontier_cells;

    cudaStream_t const on = (cudaStream_t)stream;
    sz_size_t resident_blocks = 0;
    if (cudaMemsetAsync(progress, 0, tile_grid_columns * sizeof(sz_u32_t), on) != cudaSuccess ||
        sz_levenshtein_cuda_tiled_resident_blocks_(&resident_blocks) != cudaSuccess) {
        alloc->free(scratch, scratch_bytes, alloc->handle);
        return sz_device_code_mismatch_k;
    }

    // A warp waits on the tile-column to its left, so a block that never gets scheduled is a block its
    // neighbour spins on forever; the grid is capped at what the occupancy answer says stays resident.
    sz_size_t const wanted_blocks = (tile_grid_columns + sz_levenshtein_cuda_tiled_warps_per_block_k - 1) /
                                    sz_levenshtein_cuda_tiled_warps_per_block_k;
    dim3 grid, block;
    grid.x = (unsigned)(wanted_blocks < resident_blocks ? wanted_blocks : resident_blocks), grid.y = 1, grid.z = 1;
    block.x = sz_levenshtein_cuda_tiled_threads_per_block_k, block.y = 1, block.z = 1;

    sz_cptr_t launch_shorter_text = shorter_text, launch_longer_text = longer_text;
    sz_u32_t launch_shorter_length = shorter_length, launch_longer_length = longer_length;
    sz_u32_t *launch_row_frontier = row_frontier, *launch_progress = progress;
    sz_size_t *launch_distance = device_distance;
    void *arguments[7];
    arguments[0] = &launch_shorter_text, arguments[1] = &launch_shorter_length;
    arguments[2] = &launch_longer_text, arguments[3] = &launch_longer_length;
    arguments[4] = &launch_row_frontier, arguments[5] = &launch_progress, arguments[6] = &launch_distance;
    cudaError_t const launched = cudaLaunchKernel((void const *)sz_levenshtein_cuda_tiled_kernel_, grid, block,
                                                  arguments, 0, on);
    if (launched != cudaSuccess) {
        alloc->free(scratch, scratch_bytes, alloc->handle);
        return sz_device_code_mismatch_k;
    }

    sz_size_t computed = 0;
    cudaError_t const copied = cudaMemcpyAsync(&computed, device_distance, sizeof(sz_size_t), cudaMemcpyDeviceToHost,
                                               on);
    // Only this stream is waited on, so the caller's other work on the device keeps running.
    cudaError_t const finished = cudaStreamSynchronize(on);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    if (copied != cudaSuccess || finished != cudaSuccess) return sz_device_code_mismatch_k;
    *distance = computed;
    return sz_success_k;
}

#pragma endregion Tiled

#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_LEVENSHTEIN_CUDA_CUH_

