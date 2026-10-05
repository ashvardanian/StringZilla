/**
 *  @file include/stringzilla/levenshtein/simt.cuh
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief CUDA backend for Levenshtein distances: a candidate per thread or per warp at a time,
 *      streaming Myers' bit-parallel recurrence against a batch of queries prepared once into match
 *      masks the whole grid shares.
 *
 *  The step is the serial tier's, reached from the device through `--expt-relaxed-constexpr`, so
 *  the distances are the same integers rather than merely close ones. Myers is add-with-carry and
 *  bitwise operations over @c u64 words, all of which the device runs at its integer rate; there is
 *  nothing here for the DPX three-way maxima to accelerate, since those serve the score-based
 *  recurrences of Needleman-Wunsch and Smith-Waterman rather than this one. That is why this family
 *  has one GPU tier and not a ladder of them.
 *
 *  A candidate's recurrence is a dependency chain, so it stays on one thread and the parallelism
 *  comes from the candidates on @c blockIdx.x and from the queries on @c blockIdx.y. The verticals
 *  live in the thread's own registers or local memory, @c words of them, while the match masks are
 *  read-only and shared: every thread indexes the same @c classes × stride plane by the class of
 *  the byte it is stepping, so the rows stay hot in cache instead of being rebuilt per candidate.
 *
 *  A block owns a tile of candidates rather than one each per thread, and a thread or warp done
 *  with one takes the next one of the tile, so lengths that differ by orders of magnitude cost a
 *  warp about their sum over its lanes rather than thirty-two times the longest. On Blackwell a
 *  block done with its tile takes over a block not yet started, the grid is as wide as the batch,
 *  and a tile holds one round; elsewhere it holds several. Neither keeps state on the device.
 *
 *  A batch is bucketed by rung key when each vendor's engine init runs, so one launch carries only
 *  queries that share an entry point, and the occupancy walk each of those entry points needs is
 *  paid there rather than once per round. The byte planes are built by a kernel from the queries
 *  staged into device-reachable memory; the rune planes are built on the host, an init being
 *  allowed to join where a round is not.
 *
 *  @sa include/stringzilla/levenshtein.h
 */
#ifndef STRINGZILLA_LEVENSHTEIN_SIMT_CUH_
#define STRINGZILLA_LEVENSHTEIN_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/levenshtein/serial.h"

#if STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM

#pragma region Myers Threaded

/** Query words one thread keeps verticals for, on the byte rung and the rune rung alike. */
enum { sz_levenshtein_thread_words_max_simt_k = 16 };

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief One block's tile of candidates, Myers-swept at a compile-time @p words, which is what
 *      keeps the verticals in registers.
 *
 *  A word count the compiler cannot see makes @c verticals a dynamically indexed array, and the
 *  only place it can live is local memory - a 256-byte stack frame and four local accesses per
 *  word-step, which caps the kernel near thirty percent of the device's integer issue rate. Reached
 *  from entry points that each pass a literal, the same body is worth 2.2x at a one-word query and
 *  3.3x at sixteen.
 *
 *  A thread draws candidates from its block's queue one after another, and the sweep is one flat
 *  loop of eight steps per turn, which is where a thread past its text draws the next one. Nested
 *  loops would hold every lane at the inner loop's end until the warp's longest candidate is done.
 *
 *  @param[in] order The launch's own slice of the batch's bucketing, one query index per
 *      @c blockIdx.y row.
 */
STRINGZILLA_DEVICE void sz_levenshtein_sweep_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                   sz_sequence_t candidates, sz_size_t *distances,
                                                   sz_size_t distances_stride, sz_size_t words) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index, drawn_row;
    sz_tile_queue_open_simt_(&queue, tile_size, candidates.count, blockDim.x);
    if (!sz_tile_queue_first_simt_(&queue, tile_size, candidates.count, threadIdx.x, &row_index, &candidate)) return;

    sz_size_t query_index = order[row_index];
    sz_levenshtein_query_t query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *row = distances + query_index * distances_stride;
    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_thread_words_max_simt_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);

    // One byte per step is one uncoalesced sector per step, and at a one-word query that load is most of the
    // round. A text's eight bytes span two aligned words, and the second is read only once it holds
    // a byte of the text: an aligned word that holds one lies in a page the device may read all
    // eight bytes of.
    sz_cptr_t text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
    sz_size_t left = sz_sequence_tape_length_simt_(candidates.handle, candidate);
    sz_size_t *slot_out = row + candidate;
    sz_u32_t shift = (sz_u32_t)((sz_size_t)text & 7) * 8;
    sz_u64_t const *cursor = (sz_u64_t const *)(text - shift / 8);
    sz_u64_t word = left ? cursor[0] : 0;
    for (;;) {
        if (!left) {
            *slot_out = sz_levenshtein_u64x1_score_serial(&state, candidate);
            if (!sz_tile_queue_draw_simt_(&queue, tile_size, candidates.count, &drawn_row, &candidate)) break;
            if (drawn_row != row_index) {
                row_index = drawn_row, query_index = order[row_index];
                query = sz_levenshtein_engine_row_(&engine, query_index);
                row = distances + query_index * distances_stride;
            }
            text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
            left = sz_sequence_tape_length_simt_(candidates.handle, candidate);
            slot_out = row + candidate;
            shift = (sz_u32_t)((sz_size_t)text & 7) * 8;
            cursor = (sz_u64_t const *)(text - shift / 8);
            word = left ? cursor[0] : 0;
            sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
            continue;
        }
        sz_u64_t const following = left + shift / 8 > 8 ? cursor[1] : 0;
        sz_u64_t const octet = shift ? (word >> shift) | (following << (64 - shift)) : word;
        word = following, ++cursor;
        // A whole octet steps unguarded; only the last one of a text tests its slots.
        if (left >= 8) {
#pragma unroll
            for (sz_size_t slot = 0; slot != 8; ++slot)
                sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query,
                                                 query.byte_to_class[(sz_u8_t)(octet >> (slot * 8))]);
            left -= 8;
            continue;
        }
        for (sz_size_t slot = 0; slot != left; ++slot)
            sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query,
                                             query.byte_to_class[(sz_u8_t)(octet >> (slot * 8))]);
        left = 0;
    }
}

/*  One entry point per word count, each passing its own literal, so every query length gets its
 *  own register budget - thirty-eight registers at one word against ninety-six at sixteen, which
 *  one shared kernel would have to spend on every launch, and each vendor's launcher picks one. */
static __global__ void sz_levenshtein_distances_w1_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_distances_w2_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_distances_w3_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_distances_w4_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_distances_w5_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_distances_w6_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_distances_w7_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_distances_w8_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_distances_w9_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_distances_w10_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_distances_w11_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_distances_w12_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_distances_w13_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_distances_w14_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_distances_w15_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_distances_w16_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, 16);
}

#pragma endregion Myers Threaded

#pragma region Myers Warped

/**
 *  @brief One candidate's Myers sweep across a whole warp, @p words_per_lane words to a lane,
 *      skewed by one character per lane so the carry between words walks one lane per step.
 *
 *  A lane owns @p words_per_lane consecutive words and advances the character @p words_per_lane
 *  steps behind the lane below it, so the carry out of the lane below's top word was finalized one
 *  step earlier and arrives through a single @c sz_shuffle_up_simt_. Unskewed, that same carry is a
 *  warp-wide prefix, and the lookahead resolving it costs a dozen shuffles per character against
 *  roughly twenty useful word operations. The skew costs a lane of fill and a lane of drain, so a
 *  warp runs @c length+live_lanes-1 steps for a candidate of @c length characters, and the
 *  thirty-two bytes one step reads are thirty-two consecutive ones.
 *
 *  Words past the query's last are stepped rather than branched around: the recurrence only ever
 *  carries upward, so whatever such a word holds never reaches a live one, and the distance below
 *  reads none of them.
 *
 *  @param[in] words_per_lane Exactly @c ceil(query_words/32); a literal, which keeps the
 *      verticals in registers.
 */
STRINGZILLA_DEVICE void sz_levenshtein_warp_candidate_simt_(sz_levenshtein_query_t const *query_pointer,
                                                            sz_sequence_t const *candidates, sz_size_t candidate,
                                                            sz_size_t *row, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_levenshtein_query_t const query = *query_pointer;
    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates->handle, candidate);
    sz_size_t const length = sz_sequence_tape_length_simt_(candidates->handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = sz_size_divide_round_up(words, words_per_lane);
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_gpu_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_gpu_warp_words_per_lane_max_k];
#pragma unroll
    for (sz_size_t word = 0; word != words_per_lane; ++word) positive[word] = ~(sz_u64_t)0, negative[word] = 0;

    // What a lane hands the lane above: Myers' horizontal positive in bit zero, horizontal negative in bit one.
    // The addition's sixty-fifth bit needs no room of its own, being the horizontal negative wherever the
    // vertical positive's top bit is set and zero wherever it is not.
    unsigned carry = 0;
    sz_size_t const steps = length + live_lanes - 1;
    for (sz_size_t step = 0; step != steps; ++step) {
        unsigned const received = sz_shuffle_up_simt_(carry, 1);
        sz_ssize_t const position = (sz_ssize_t)step - (sz_ssize_t)lane;
        if (lane >= live_lanes || position < 0 || position >= (sz_ssize_t)length) continue;

        sz_u64_t const *const masks = lane_masks +
                                      (sz_size_t)query.byte_to_class[(sz_u8_t)text[position]] * query.stride;
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
    for (unsigned offset = 16; offset != 0; offset >>= 1) deltas += sz_shuffle_down_simt_(deltas, offset);
    if (lane == 0) row[candidate] = (sz_size_t)((sz_ssize_t)length + deltas);
}

/** Lane zero's draw from the block's queue, which every lane of the warp then holds. */
STRINGZILLA_DEVICE int sz_levenshtein_warp_draw_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                      sz_u32_t *row_index, sz_size_t *candidate) {
    int drawn = 0;
    sz_u32_t drawn_row = 0;
    sz_size_t drawn_candidate = 0;
    if ((threadIdx.x & 31u) == 0)
        drawn = sz_tile_queue_draw_simt_(queue, tile_size, count, &drawn_row, &drawn_candidate);
    *row_index = sz_lanes_broadcast_simt_(drawn_row);
    *candidate = (sz_size_t)sz_lanes_broadcast_simt_((sz_u32_t)drawn_candidate) |
                 ((sz_size_t)sz_lanes_broadcast_simt_((sz_u32_t)(drawn_candidate >> 32)) << 32);
    return (int)sz_lanes_broadcast_simt_((sz_u32_t)drawn);
}

/** Opens the block's queue, a warp to a seat, and takes each warp's first candidate. */
STRINGZILLA_DEVICE int sz_levenshtein_warp_first_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                       sz_u32_t *row_index, sz_size_t *candidate) {
    int drawn = 0;
    sz_u32_t drawn_row = 0;
    sz_size_t drawn_candidate = 0;
    sz_tile_queue_open_simt_(queue, tile_size, count, blockDim.x >> 5);
    if ((threadIdx.x & 31u) == 0)
        drawn = sz_tile_queue_first_simt_(queue, tile_size, count, threadIdx.x >> 5, &drawn_row, &drawn_candidate);
    *row_index = sz_lanes_broadcast_simt_(drawn_row);
    *candidate = (sz_size_t)sz_lanes_broadcast_simt_((sz_u32_t)drawn_candidate) |
                 ((sz_size_t)sz_lanes_broadcast_simt_((sz_u32_t)(drawn_candidate >> 32)) << 32);
    return (int)sz_lanes_broadcast_simt_((sz_u32_t)drawn);
}

/** One block's tile of candidates, a whole warp to each, a warp done with one taking the next. */
STRINGZILLA_DEVICE void sz_levenshtein_warp_sweep_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                        sz_sequence_t candidates, sz_size_t *distances,
                                                        sz_size_t distances_stride, sz_size_t words_per_lane) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index;
    int drawn = sz_levenshtein_warp_first_simt_(&queue, tile_size, candidates.count, &row_index, &candidate);
    // Warp uniform, so the shuffles inside still see a whole warp.
    for (; drawn; drawn = sz_levenshtein_warp_draw_simt_(&queue, tile_size, candidates.count, &row_index, &candidate)) {
        sz_size_t const query_index = order[row_index];
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
        sz_levenshtein_warp_candidate_simt_(&query, &candidates, candidate, distances + query_index * distances_stride,
                                            words_per_lane);
    }
}

/*  One entry point per words-per-lane, each passing its own literal, for the reason the threaded
 *  rung states: a count the compiler cannot see spills the verticals to local memory. @c K rounds
 *  up by one word rather than to the next power of two, since the remainder costs only idle lanes -
 *  linear leaves worst-case lane utilization at seven eighths where doubling drops it to just over
 *  half above each boundary. */
static __global__ void sz_levenshtein_distances_k1_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_distances_k2_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_distances_k3_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_distances_k4_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_distances_k5_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_distances_k6_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_distances_k7_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_distances_k8_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 8);
}

#pragma endregion Myers Warped

#pragma region Myers Byte Lanes

/** Candidates one thread advances together at its widest: four byte lanes to a thirty-two-bit
 *  register. A sixty-four-bit register holds eight, and it costs two instructions per operation to
 *  do it, so the wider packing buys no arithmetic and spends the thread-level parallelism that
 *  hides this rung's loads. */
enum { sz_levenshtein_lanes_per_thread_max_simt_k = 4 };

/** Text bytes one refill hands a lane, which is the aligned word its cursor sits in. */
enum { sz_levenshtein_bytes_per_refill_simt_k = 8 };

/**
 *  @brief Advances every lane one symbol through its own Myers word, all of them packed
 *      in one register.
 *
 *  The device has no per-lane add, so the one carry that would cross a lane boundary is blocked the
 *  standard way: the low bits of every lane sum on their own, where a carry cannot leave the lane,
 *  and each lane's top bit is folded back in by exclusive or.
 */
STRINGZILLA_DEVICE void sz_levenshtein_lanes_step_simt_(sz_u32_t *positive, sz_u32_t *negative, sz_u32_t equality,
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

/** One lane's distance, read off the verticals at the position that lane's text ends: the row
 *  above the query costs one edit per candidate symbol and the query column's own deltas
 *  carry the rest. */
STRINGZILLA_DEVICE sz_size_t sz_levenshtein_lane_distance_simt_(sz_u32_t positive, sz_u32_t negative,
                                                                sz_size_t lane_shift, sz_u32_t live, sz_size_t length) {
    sz_u32_t const lane_positive = (positive >> lane_shift) & live;
    sz_u32_t const lane_negative = (negative >> lane_shift) & live;
    return (sz_size_t)((sz_ssize_t)length + __popc(lane_positive) - __popc(lane_negative));
}

/**
 *  @brief Sweeps as many candidates as one register holds lanes, a whole Myers word to each lane.
 *
 *  A query of at most @p lane_bits symbols needs only that many bits of a Myers word, so four
 *  candidates ride where the threaded rung advances one and the recurrence's fifteen operations
 *  serve all of them. The lanes share nothing but the arithmetic: each keeps its own cursor and
 *  takes its own text eight bytes at a time from the aligned word the cursor sits in, so the loads
 *  stay as scattered as the threaded rung's.
 *
 *  A lane's distance is read off the verticals at the position its text ends, the way the warped
 *  rung reads its own, so no lane carries a running score and a lane past its text keeps stepping
 *  whatever it holds.
 *
 *  @param[in] lane_bits Bits one lane spans: 8 or 16, and a literal, which is what folds
 *      the lane masks.
 *  @param[in] lanes Lanes the register carries - a literal too, which is what keeps the lane
 *      arrays in registers.
 */
STRINGZILLA_DEVICE void sz_levenshtein_lanes_sweep_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                         sz_sequence_t candidates, sz_size_t *distances,
                                                         sz_size_t distances_stride, sz_size_t lane_bits,
                                                         sz_size_t lanes) {
    sz_size_t const query_index = order[blockIdx.y];
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *const row = distances + query_index * distances_stride;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t const tile_first = (sz_size_t)blockIdx.x * tile_size;
    sz_size_t const tile_end = sz_min_of_two(tile_first + tile_size, candidates.count);

    // At these query lengths a class's whole mask fits one lane, so the class map and the mask rows collapse
    // into one table the block builds once and every lane of every thread in it reads. Folding the two lookups
    // into one is worth 1.22x on XLSum words and 1.73x where the candidates are long enough to be step bound.
    __shared__ sz_u32_t byte_to_mask[sz_levenshtein_byte_classes_k];
    __shared__ sz_u32_t taken;
    for (sz_size_t entry = threadIdx.x; entry < sz_levenshtein_byte_classes_k; entry += blockDim.x)
        byte_to_mask[entry] = (sz_u32_t)query.masks[(sz_size_t)query.byte_to_class[entry] * query.stride];
    if (threadIdx.x == 0) taken = 0;
    __syncthreads();

    // A lane done with its text takes the next candidate of the block's tile at the following
    // refill, its bits reset to a fresh column, so a thread's lanes stay busy however their lengths
    // differ. A lane's cursor is the aligned word its next eight bytes start in, and its count is
    // the bytes it has left.
    sz_u64_t const *cursors[sz_levenshtein_lanes_per_thread_max_simt_k];
    sz_size_t lefts[sz_levenshtein_lanes_per_thread_max_simt_k];
    sz_u64_t words[sz_levenshtein_lanes_per_thread_max_simt_k], octets[sz_levenshtein_lanes_per_thread_max_simt_k];
    sz_u32_t shifts[sz_levenshtein_lanes_per_thread_max_simt_k], held[sz_levenshtein_lanes_per_thread_max_simt_k];
    sz_u32_t const lane_mask = lane_bits == 8 ? 0xFFu : 0xFFFFu;
    sz_u32_t const live = ((sz_u32_t)1 << query.length) - 1;
    sz_u32_t const lane_low_bits = lane_bits == 8 ? 0x01010101u : 0x00010001u;
    sz_u32_t const lane_high_bits = lane_bits == 8 ? 0x80808080u : 0x80008000u;
    sz_u32_t positive = ~(sz_u32_t)0, negative = 0, unread = 0, refill = ((sz_u32_t)1 << lanes) - 1;
#pragma unroll
    for (sz_size_t lane = 0; lane != lanes; ++lane)
        lefts[lane] = 0, shifts[lane] = 0, words[lane] = 0, octets[lane] = 0;

    for (;;) {
        if (refill) {
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                if ((refill & ((sz_u32_t)1 << lane)) == 0) continue;
                // An empty candidate's distance is the query's length, and it takes no lane at all.
                for (;;) {
                    sz_size_t const candidate = tile_first + atomicAdd(&taken, 1u);
                    if (candidate >= tile_end) break;
                    sz_size_t const length = sz_sequence_tape_length_simt_(candidates.handle, candidate);
                    if (!length) {
                        row[candidate] = query.length;
                        continue;
                    }
                    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
                    sz_size_t const head = (sz_size_t)text & 7;
                    cursors[lane] = (sz_u64_t const *)(text - head), shifts[lane] = (sz_u32_t)(head * 8);
                    words[lane] = cursors[lane][0], lefts[lane] = length;
                    held[lane] = (sz_u32_t)(candidate - tile_first), unread |= (sz_u32_t)1 << lane;
                    positive |= lane_mask << (lane * lane_bits), negative &= ~(lane_mask << (lane * lane_bits));
                    break;
                }
            }
            refill = 0;
            if (!unread) break;
        }

        // A lane's next eight bytes span two aligned words, and the second is read only once it holds a byte
        // of the text: an aligned word that holds one lies in a page the device may read all eight bytes of.
        // A lane holding no text steps whatever its last octet was, which no distance reads.
        sz_u32_t ending = 0;
#pragma unroll
        for (sz_size_t lane = 0; lane != lanes; ++lane) {
            if ((unread & ((sz_u32_t)1 << lane)) == 0) continue;
            sz_u64_t const following = lefts[lane] + shifts[lane] / 8 > sz_levenshtein_bytes_per_refill_simt_k
                                           ? cursors[lane][1]
                                           : 0;
            octets[lane] = shifts[lane] ? (words[lane] >> shifts[lane]) | (following << (64 - shifts[lane]))
                                        : words[lane];
            words[lane] = following, ++cursors[lane];
            if (lefts[lane] <= sz_levenshtein_bytes_per_refill_simt_k) ending |= (sz_u32_t)1 << (lefts[lane] - 1);
        }
#pragma unroll
        for (sz_size_t slot = 0; slot != sz_levenshtein_bytes_per_refill_simt_k; ++slot) {
            sz_u32_t equality = 0;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                sz_u32_t const byte = (sz_u32_t)(octets[lane] >> (slot * 8)) & 0xFF;
                equality |= byte_to_mask[byte] << (lane * lane_bits);
            }
            sz_levenshtein_lanes_step_simt_(&positive, &negative, equality, lane_low_bits, lane_high_bits);
            if ((ending & ((sz_u32_t)1 << slot)) == 0) continue;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                if ((unread & ((sz_u32_t)1 << lane)) == 0 || lefts[lane] != slot + 1) continue;
                sz_size_t const candidate = tile_first + held[lane];
                row[candidate] = sz_levenshtein_lane_distance_simt_(
                    positive, negative, lane * lane_bits, live,
                    sz_sequence_tape_length_simt_(candidates.handle, candidate));
                unread &= ~((sz_u32_t)1 << lane), refill |= (sz_u32_t)1 << lane;
            }
        }
#pragma unroll
        for (sz_size_t lane = 0; lane != lanes; ++lane) lefts[lane] -= sz_min_of_two(lefts[lane], (sz_size_t)8);
    }
}

/*  One entry point per lane width, each passing its own literals, for the reason the threaded rung
 *  states: a width the compiler cannot see turns the lane arrays into local memory. */
static __global__ void sz_levenshtein_distances_u8x4_simt_kernel_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                                  sz_sequence_t candidates, sz_size_t *distances,
                                                                  sz_size_t distances_stride) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, 8, 4);
}

static __global__ void sz_levenshtein_distances_u16x2_simt_kernel_(sz_levenshtein_engine_t engine,
                                                                   sz_u32_t const *order, sz_sequence_t candidates,
                                                                   sz_size_t *distances, sz_size_t distances_stride) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, 16, 2);
}

#pragma endregion Myers Byte Lanes

#pragma region Myers UTF 8

/**
 *  @brief One block's tile of candidates, Myers-swept over runes at a compile-time @p words, the
 *      byte sweep with a rune cursor.
 *
 *  The byte sweep's octet load has no counterpart here: a rune spans one to four bytes, so its
 *  width is known only once the lead byte is read, and the loop takes one rune per step. The class
 *  comes from the query's page table - two dependent loads, and no scratch of its own per candidate
 *  - rather than a byte map.
 *
 *  A thread past its text draws the next candidate inside one flat loop, as the byte sweep does.
 */
STRINGZILLA_DEVICE void sz_levenshtein_sweep_utf8_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                        sz_sequence_t candidates, sz_size_t *distances,
                                                        sz_size_t distances_stride, sz_size_t words) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index, drawn_row;
    sz_tile_queue_open_simt_(&queue, tile_size, candidates.count, blockDim.x);
    if (!sz_tile_queue_first_simt_(&queue, tile_size, candidates.count, threadIdx.x, &row_index, &candidate)) return;

    sz_size_t query_index = order[row_index];
    sz_levenshtein_query_t query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t *row = distances + query_index * distances_stride;
    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_thread_words_max_simt_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    sz_cptr_t text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
    sz_size_t length = sz_sequence_tape_length_simt_(candidates.handle, candidate);
    sz_size_t position = 0;
    for (;;) {
        if (position < length) {
            sz_rune_t const rune = sz_utf8_next_rune_(text, length, &position);
            sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, sz_levenshtein_utf8_class(&query, rune));
            continue;
        }
        row[candidate] = sz_levenshtein_u64x1_score_serial(&state, candidate);
        if (!sz_tile_queue_draw_simt_(&queue, tile_size, candidates.count, &drawn_row, &candidate)) break;
        if (drawn_row != row_index) {
            row_index = drawn_row, query_index = order[row_index];
            query = sz_levenshtein_engine_row_(&engine, query_index);
            row = distances + query_index * distances_stride;
        }
        text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
        length = sz_sequence_tape_length_simt_(candidates.handle, candidate);
        position = 0;
        sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    }
}

/*  One rune entry point per word count, each passing its own literal, for the reason the byte tier
 *  states: a word count the compiler cannot see spills the verticals to local memory. */
static __global__ void sz_levenshtein_distances_utf8_w1_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_distances_utf8_w2_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_distances_utf8_w3_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_distances_utf8_w4_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_distances_utf8_w5_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_distances_utf8_w6_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_distances_utf8_w7_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_distances_utf8_w8_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_distances_utf8_w9_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_distances_utf8_w10_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_distances_utf8_w11_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_distances_utf8_w12_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_distances_utf8_w13_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_distances_utf8_w14_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_distances_utf8_w15_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_distances_utf8_w16_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 16);
}

#pragma endregion Myers UTF 8

#pragma region Myers UTF 8 Warped

/**
 *  @brief One candidate's Myers sweep over runes across a whole warp, @p words_per_lane words to a
 *      lane, the byte warped sweep with the rune's class riding the chain the carry already rides.
 *
 *  Runes are variable-width, so no lane can address the rune at @c step-lane without decoding
 *  everything below it - and nothing here wants the rune, only its class. Lane zero decodes one
 *  rune per step and hands its class up through a second @c sz_shuffle_up_simt_, so the class
 *  walks one lane per step, which is the lag the skew already imposes; every rune is decoded once
 *  per candidate and its width is never inverted.
 *
 *  The chain carries the class plus one, leaving zero to mark a step whose rune is past the
 *  candidate's end - the liveness the byte sweep reads off its own position instead, and what a
 *  lane forwards through the fill and the drain. The warp runs while a live lane still holds a
 *  rune, so a candidate costs its runes, not its bytes.
 *
 *  @param[in] words_per_lane Exactly @c ceil(query_words/32); a literal, which keeps the
 *      verticals in registers.
 */
STRINGZILLA_DEVICE void sz_levenshtein_warp_candidate_utf8_simt_(sz_levenshtein_query_t const *query_pointer,
                                                                 sz_sequence_t const *candidates, sz_size_t candidate,
                                                                 sz_size_t *row, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_levenshtein_query_t const query = *query_pointer;
    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates->handle, candidate);
    sz_size_t const length = sz_sequence_tape_length_simt_(candidates->handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = sz_size_divide_round_up(words, words_per_lane);
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_gpu_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_gpu_warp_words_per_lane_max_k];
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
        sz_u32_t const inherited = sz_shuffle_up_simt_(held, 1);
        unsigned const received = sz_shuffle_up_simt_(carry, 1);
        held = lane == 0 ? decoded : inherited;
        unsigned const live = held != 0 && lane < live_lanes;
        running = (unsigned)sz_lanes_any_simt_((int)live);
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
    for (unsigned offset = 16; offset != 0; offset >>= 1) deltas += sz_shuffle_down_simt_(deltas, offset);
    if (lane == 0) row[candidate] = (sz_size_t)((sz_ssize_t)runes + deltas);
}

/** One block's tile of rune candidates, a warp to each, a warp done with one taking the next. */
STRINGZILLA_DEVICE void sz_levenshtein_warp_sweep_utf8_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                             sz_sequence_t candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_size_t words_per_lane) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index;
    int drawn = sz_levenshtein_warp_first_simt_(&queue, tile_size, candidates.count, &row_index, &candidate);
    // Warp uniform, so the shuffles inside still see a whole warp.
    for (; drawn; drawn = sz_levenshtein_warp_draw_simt_(&queue, tile_size, candidates.count, &row_index, &candidate)) {
        sz_size_t const query_index = order[row_index];
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
        sz_levenshtein_warp_candidate_utf8_simt_(&query, &candidates, candidate,
                                                 distances + query_index * distances_stride, words_per_lane);
    }
}

/*  One warped rune entry point per words-per-lane, each passing its own literal, for the reason the
 *  threaded rung states: a count the compiler cannot see spills the verticals to local memory. */
static __global__ void sz_levenshtein_distances_utf8_k1_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_distances_utf8_k2_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_distances_utf8_k3_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_distances_utf8_k4_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_distances_utf8_k5_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_distances_utf8_k6_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_distances_utf8_k7_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_distances_utf8_k8_simt_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 8);
}

#pragma endregion Myers UTF 8 Warped

#pragma region Myers Engine

/**
 *  @brief Builds one byte query's match masks and class map per grid row, on the device itself.
 *
 *  The passes are @ref sz_levenshtein_query_prepare 's: flag the byte values the query holds, hand
 *  them dense classes in byte order, give the row past them to every byte the query lacks, then
 *  set one bit per position in its class's row. One block per query, so the ranks walk shared
 *  memory rather than the grid, and the planes arrive zeroed from the memset the stream runs ahead
 *  of this launch.
 *
 *  @param[in] first The batch index of grid row zero, since the query axis is cut at the
 *      grid's own ceiling.
 */
static __global__ void sz_levenshtein_masks_simt_kernel_(sz_levenshtein_engine_t engine, sz_string_view_t const *texts,
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

#pragma endregion Myers Engine

#pragma region Tiled

/** Tile geometry of the device-spanning wavefront: one warp owns a 128-wide tile-column,
 *  one lane owns a 4x4 register micro-tile, so a tile is a 63-step anti-diagonal sweep
 *  across 32 micro-rows. */
enum {
    sz_levenshtein_tile_side_simt_k = sz_levenshtein_tile_side_k,
    sz_levenshtein_micro_side_simt_k = 4,
    sz_levenshtein_lanes_simt_k = 32,
    sz_levenshtein_micro_rows_simt_k = sz_levenshtein_tile_side_simt_k / sz_levenshtein_micro_side_simt_k,
    sz_levenshtein_tiled_warps_per_block_simt_k = 8,
#if defined(__HIP__)
    // A 64-wide wavefront marches two tile-columns in lockstep, one spinning on the other forever.
    sz_levenshtein_tiled_warp_stride_simt_k = 64,
#else
    sz_levenshtein_tiled_warp_stride_simt_k = sz_levenshtein_lanes_simt_k,
#endif
    sz_levenshtein_tiled_threads_per_block_simt_k = sz_levenshtein_tiled_warps_per_block_simt_k *
        sz_levenshtein_tiled_warp_stride_simt_k,
};

/** Bytes below 2³² the wavefront leaves unindexed. Lengths and cells are @c sz_u32_t inside the
 *  kernel, and the last tile of each axis is padded up to 128 columns, so a text is at most 2³²
 *  less this padding room rather than a length near the word's top wrapping a column index. */
enum { sz_levenshtein_tiled_padding_simt_k = 256 };

/** Bytes standing in for a character past the end of a text. The two differ, so a padded row never
 *  matches a padded column, and a padded cell never reaches an in-bounds one. */
enum {
    sz_levenshtein_past_query_simt_k = 0xFE,
    sz_levenshtein_past_target_simt_k = 0xFF,
};

/** Selects what a micro-tile march does with a finished cell: nothing, or the corner test that
 *  publishes the pair's distance. */
typedef enum {
    sz_levenshtein_march_fast_simt_k = 0,
    sz_levenshtein_march_checked_simt_k = 1,
} sz_levenshtein_march_simt_t;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

/** Three-way unsigned minimum, one instruction on Hopper and Blackwell. */
STRINGZILLA_DEVICE sz_u32_t sz_levenshtein_min3_simt_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
    return __vimin3_u32(first, second, third);
}

#else

/** Three-way unsigned minimum as a pair of comparisons, for targets without the fused form. */
STRINGZILLA_DEVICE sz_u32_t sz_levenshtein_min3_simt_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
    sz_u32_t const smaller = first < second ? first : second;
    return smaller < third ? smaller : third;
}

#endif

/**
 *  @brief Publishes that this tile-column finished @p tile_row, releasing the frontier
 *      writes before it.
 *
 *  The release is spelled in PTX, or as HIP's builtin, because the library's C tiers cannot reach
 *  @c cuda::atomic_ref, and because the alternative - a @c __threadfence next to a @c volatile
 *  store - orders every prior access of the thread where one location's release is all the
 *  protocol needs, and costs a membar this form does not. Only the lane that wrote the frontier
 *  calls it, so the release orders its own stores and nothing has to argue about what a warp
 *  barrier carries across lanes.
 */
STRINGZILLA_DEVICE void sz_levenshtein_publish_simt_(sz_u32_t *counter, sz_u32_t tile_row) {
    sz_u32_t const published = tile_row + 1u;
#if defined(__HIP__)
    __hip_atomic_store(counter, published, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(counter), "r"(published) : "memory");
#endif
}

/**
 *  @brief Spins until the left tile-column published past @p tile_row, acquiring
 *      its frontier writes.
 *
 *  Every lane of the warp polls, not just one: the warp's loads of a single address collapse into
 *  one transaction, so the traffic is a lane-0 poll's, and each lane's own acquire orders each
 *  lane's own reads.
 */
STRINGZILLA_DEVICE void sz_levenshtein_await_simt_(sz_u32_t const *counter, sz_u32_t tile_row) {
    sz_u32_t observed = 0;
    do {
#if defined(__HIP__)
        observed = __hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else
        asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(observed) : "l"(counter) : "memory");
#endif
    } while (observed <= tile_row);
}

/**
 *  @brief Marches one 128x128 tile as an anti-diagonal wavefront of 4x4 register micro-tiles.
 *
 *  Lane @e l owns micro-column @e l and enters at wavefront step @e l, so 32 micro-rows across
 *  32 lanes take @c micro_rows_k+lanes_k-1 steps. Lane 0 reads its left column and diagonal
 *  corner from the staged @p shared_left; every other lane receives the left neighbour's right
 *  column and top-right corner from @c sz_shuffle_up_simt_, which is itself the warp-wide
 *  rendezvous ordering the two. @p carry_top enters holding the row above the tile and leaves
 *  holding the tile's bottom row.
 *
 *  @param[in] march Whether finished cells need the corner test, which only a partial or
 *      corner tile does.
 *  @param[in] lane_index This thread's lane, which is also its micro-column.
 *  @param[in] tile_first_row Zero-based DP row the tile's first micro-row cell sits under.
 *  @param[in] tile_first_column Zero-based DP column the tile's first micro-column
 *      cell sits beside.
 *  @param[in] shorter_length Length of the text along the row axis, for the corner test.
 *  @param[in] longer_length Length of the text along the column axis, for the corner test.
 *  @param[in] shared_query The tile's 128 query characters, staged once per tile-row.
 *  @param[in] shared_left The tile's 128 incoming left-boundary cells, staged once per tile-row.
 *  @param[in] tile_corner The cell diagonally above the tile's top-left cell.
 *  @param[in] target_chars This lane's four target characters, constant down the whole tile-column.
 *  @param[inout] carry_top The four cells directly above this lane's micro-tile.
 *  @param[out] row_frontier One cell per DP row; the rightmost lane publishes the tile's right
 *      column into it.
 *  @param[out] distance Written once, by the thread that computes the matrix corner.
 */
STRINGZILLA_DEVICE void sz_levenshtein_march_tile_simt_(                         //
    sz_levenshtein_march_simt_t march, unsigned lane_index,                      //
    sz_u32_t tile_first_row, sz_u32_t tile_first_column,                         //
    sz_u32_t shorter_length, sz_u32_t longer_length,                             //
    char const *shared_query, sz_u32_t const *shared_left, sz_u32_t tile_corner, //
    char const *target_chars, sz_u32_t *carry_top, sz_u32_t *row_frontier, sz_size_t *distance) {

    sz_u32_t previous_right_edge[sz_levenshtein_micro_side_simt_k];
    sz_u32_t previous_topright = 0;
    unsigned const wavefront_steps = sz_levenshtein_micro_rows_simt_k + sz_levenshtein_lanes_simt_k - 1;
    unsigned element, wavefront_step, micro_row_cell, micro_column_cell;
#pragma unroll
    for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element) previous_right_edge[element] = 0;

    for (wavefront_step = 0; wavefront_step != wavefront_steps; ++wavefront_step) {
        // Underflows for a lane the wavefront has not reached yet, which the unsigned bound below rejects.
        unsigned const micro_row = wavefront_step - lane_index;
        sz_u32_t shuffled_right_edge[sz_levenshtein_micro_side_simt_k];
        sz_u32_t shuffled_topright;
#pragma unroll
        for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
            shuffled_right_edge[element] = sz_shuffle_up_simt_(previous_right_edge[element], 1);
        shuffled_topright = sz_shuffle_up_simt_(previous_topright, 1);
        if (micro_row >= sz_levenshtein_micro_rows_simt_k) continue;

        sz_u32_t const micro_first_row = tile_first_row + micro_row * sz_levenshtein_micro_side_simt_k;
        sz_u32_t left_column[sz_levenshtein_micro_side_simt_k], diagonal_corner;
        if (lane_index == 0) {
#pragma unroll
            for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
                left_column[element] = shared_left[micro_row * sz_levenshtein_micro_side_simt_k + element];
            diagonal_corner = micro_row == 0 ? tile_corner
                                             : shared_left[micro_row * sz_levenshtein_micro_side_simt_k - 1];
        }
        else {
#pragma unroll
            for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
                left_column[element] = shuffled_right_edge[element];
            diagonal_corner = shuffled_topright;
        }

        // The top-right corner the lane to the right needs is the last cell of this micro-tile's incoming top
        // row, which the march below overwrites, so it is captured first.
        sz_u32_t const topright_for_next_lane = carry_top[sz_levenshtein_micro_side_simt_k - 1];
        sz_u32_t above_row[sz_levenshtein_micro_side_simt_k + 1];
        sz_u32_t right_edge[sz_levenshtein_micro_side_simt_k];
        above_row[0] = diagonal_corner;
#pragma unroll
        for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
            above_row[element + 1] = carry_top[element];

#pragma unroll
        for (micro_row_cell = 1; micro_row_cell <= sz_levenshtein_micro_side_simt_k; ++micro_row_cell) {
            sz_u32_t current_row[sz_levenshtein_micro_side_simt_k + 1];
            char const query_char = shared_query[micro_row * sz_levenshtein_micro_side_simt_k + micro_row_cell - 1];
            current_row[0] = left_column[micro_row_cell - 1];
#pragma unroll
            for (micro_column_cell = 1; micro_column_cell <= sz_levenshtein_micro_side_simt_k; ++micro_column_cell) {
                sz_u32_t const substitution = query_char == target_chars[micro_column_cell - 1] ? 0u : 1u;
                sz_u32_t const cell = sz_levenshtein_min3_simt_(above_row[micro_column_cell - 1] + substitution,
                                                                above_row[micro_column_cell] + 1u,
                                                                current_row[micro_column_cell - 1] + 1u);
                current_row[micro_column_cell] = cell;
                if (march == sz_levenshtein_march_checked_simt_k) {
                    sz_u32_t const matrix_row = micro_first_row + micro_row_cell;
                    sz_u32_t const matrix_column = tile_first_column + lane_index * sz_levenshtein_micro_side_simt_k +
                                                   micro_column_cell;
                    if (matrix_row == shorter_length && matrix_column == longer_length) *distance = (sz_size_t)cell;
                }
            }
            right_edge[micro_row_cell - 1] = current_row[sz_levenshtein_micro_side_simt_k];
#pragma unroll
            for (element = 0; element <= sz_levenshtein_micro_side_simt_k; ++element)
                above_row[element] = current_row[element];
        }

#pragma unroll
        for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
            carry_top[element] = above_row[element + 1], previous_right_edge[element] = right_edge[element];
        previous_topright = topright_for_next_lane;
        // The rightmost micro-column is the tile's right column, which is the next tile-column's left one.
        if (lane_index == sz_levenshtein_lanes_simt_k - 1)
#pragma unroll
            for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element)
                row_frontier[micro_first_row + element + 1] = right_edge[element];
    }
}

/**
 *  @brief One pair's Levenshtein distance as an anti-diagonal wavefront spanning the whole device.
 *
 *  A warp owns a 128-wide tile-column and marches it top to bottom, taking the tile-column a
 *  whole grid further on when it runs out, so the launched grid can be capped at what stays
 *  co-resident. Every tile's top edge is free - it stays in the marching warp's registers - and
 *  only the left edge crosses warps, through @p row_frontier under the @p progress counters.
 *  Tile-column @e c reads the band tile-column @e c-1 released and overwrites it with its own
 *  right column; the counters serialize that into one read and one write per band per column,
 *  which is also what lets a warp reuse the band for a later tile-column. Multiple blocks per
 *  pair require a cooperative launch so waiting blocks cannot exclude their producers.
 *
 *  @param[in] shorter_text Text along the row axis, device-reachable, @p shorter_length bytes.
 *  @param[in] shorter_length Its length, at most 2³² less @ref sz_levenshtein_tiled_padding_simt_k.
 *  @param[in] longer_text Text along the column axis, device-reachable, @p longer_length bytes.
 *  @param[in] longer_length Its length, at most 2³² less @ref sz_levenshtein_tiled_padding_simt_k.
 *  @param[out] row_frontier Scratch of `round_up(shorter_length, 128) + 1` unseeded cells.
 *  @param[out] progress One counter per tile-column, zeroed before the launch.
 *  @param[out] distance The pair's distance, written once by the thread owning the matrix corner.
 */
STRINGZILLA_DEVICE void sz_levenshtein_tiled_simt_(sz_cptr_t shorter_text, sz_u32_t shorter_length,
                                                   sz_cptr_t longer_text, sz_u32_t longer_length,
                                                   sz_u32_t *row_frontier, sz_u32_t *progress, sz_size_t *distance) {

    // Each warp stages its tile's query window and its incoming left boundary once per tile-row, so the
    // wavefront's scattered lane-0 boundary reads come off-chip once instead of once per micro-tile.
    __shared__ char shared_query[sz_levenshtein_tiled_warps_per_block_simt_k][sz_levenshtein_tile_side_simt_k];
    __shared__ sz_u32_t shared_left[sz_levenshtein_tiled_warps_per_block_simt_k][sz_levenshtein_tile_side_simt_k];

    unsigned const warp_in_block = threadIdx.x / sz_levenshtein_tiled_warp_stride_simt_k;
    unsigned const lane_index = threadIdx.x % sz_levenshtein_tiled_warp_stride_simt_k;
    sz_u32_t const tile_grid_rows = sz_u32_divide_round_up(shorter_length, sz_levenshtein_tile_side_simt_k);
    sz_u32_t const tile_grid_columns = sz_u32_divide_round_up(longer_length, sz_levenshtein_tile_side_simt_k);
    sz_u32_t const warps_in_grid = gridDim.x * sz_levenshtein_tiled_warps_per_block_simt_k;
    sz_u32_t const corner_tile_row = (shorter_length - 1u) / sz_levenshtein_tile_side_simt_k;
    sz_u32_t const corner_tile_column = (longer_length - 1u) / sz_levenshtein_tile_side_simt_k;
    sz_u32_t const first_tile_column = blockIdx.x * sz_levenshtein_tiled_warps_per_block_simt_k + warp_in_block;
    sz_u32_t tile_column, tile_row;
    unsigned element, stage_row;
    if (lane_index >= sz_levenshtein_lanes_simt_k) return;

    for (tile_column = first_tile_column; tile_column < tile_grid_columns; tile_column += warps_in_grid) {
        sz_u32_t const tile_first_column = tile_column * sz_levenshtein_tile_side_simt_k;
        char target_chars[sz_levenshtein_micro_side_simt_k];
        sz_u32_t carry_top[sz_levenshtein_micro_side_simt_k];
        // The row above the whole matrix costs one deletion per column, and this lane's target characters
        // never change down the column.
#pragma unroll
        for (element = 0; element != sz_levenshtein_micro_side_simt_k; ++element) {
            sz_u32_t const target_index = tile_first_column + lane_index * sz_levenshtein_micro_side_simt_k + element;
            target_chars[element] = target_index < longer_length ? longer_text[target_index]
                                                                 : (char)sz_levenshtein_past_target_simt_k;
            carry_top[element] = target_index + 1u;
        }
        // The leftmost column of the whole matrix costs one insertion per row, so the first tile-column
        // synthesizes its left boundary instead of reading a seeded one, and nothing has to seed it.
        sz_u32_t tile_corner = tile_first_column;

        for (tile_row = 0; tile_row != tile_grid_rows; ++tile_row) {
            sz_u32_t const tile_first_row = tile_row * sz_levenshtein_tile_side_simt_k;
            sz_u32_t tile_bottom_left;
            sz_levenshtein_march_simt_t march;

            if (tile_column != 0) sz_levenshtein_await_simt_(progress + (tile_column - 1u), tile_row);
            // Holds this tile-row's staging behind the previous one's reads of the same shared window.
            __syncwarp();
            if (tile_column == 0)
                for (stage_row = lane_index; stage_row < sz_levenshtein_tile_side_simt_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = tile_first_row + 1u + stage_row;
            else
                for (stage_row = lane_index; stage_row < sz_levenshtein_tile_side_simt_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = row_frontier[tile_first_row + 1u + stage_row];
            for (stage_row = lane_index; stage_row < sz_levenshtein_tile_side_simt_k; stage_row += 32u)
                shared_query[warp_in_block][stage_row] = tile_first_row + stage_row < shorter_length
                                                             ? shorter_text[tile_first_row + stage_row]
                                                             : (char)sz_levenshtein_past_query_simt_k;
            __syncwarp();

            // The tile's bottom-left cell is the diagonal corner of the tile below, and the march is about to
            // overwrite the frontier band it sits in.
            tile_bottom_left = shared_left[warp_in_block][sz_levenshtein_tile_side_simt_k - 1];
            // Exactly one tile of the whole matrix holds the corner cell, and only that tile pays the
            // per-cell test; keeping the guarded store out of every other tile's hot loop is what lets the
            // march stay in registers.
            march = tile_row == corner_tile_row && tile_column == corner_tile_column
                        ? sz_levenshtein_march_checked_simt_k
                        : sz_levenshtein_march_fast_simt_k;
            if (march == sz_levenshtein_march_fast_simt_k)
                sz_levenshtein_march_tile_simt_(sz_levenshtein_march_fast_simt_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            else
                sz_levenshtein_march_tile_simt_(sz_levenshtein_march_checked_simt_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            tile_corner = tile_bottom_left;
            if (lane_index == sz_levenshtein_lanes_simt_k - 1)
                sz_levenshtein_publish_simt_(progress + tile_column, tile_row);
        }
    }
}

/** Scratch budget per scoring call; an oversized pair runs alone. */
enum { sz_levenshtein_workspace_bytes_simt_k = 64 * 1024 * 1024 };

/** Workspace and pair capacity for one cross-product of length buckets. */
typedef struct sz_levenshtein_long_layout_simt_t {
    sz_size_t scratch_stride, frontier_cells, progress_cells, pairs;
    sz_bool_t tiled;
} sz_levenshtein_long_layout_simt_t;

static sz_levenshtein_long_layout_simt_t sz_levenshtein_long_layout_simt_(sz_levenshtein_symbol_t symbol,
                                                                          sz_levenshtein_length_bucket_t queries,
                                                                          sz_levenshtein_length_bucket_t candidates,
                                                                          sz_size_t resident) {
    sz_levenshtein_long_layout_simt_t layout = {0, 0, 0, 0, sz_false_k};
    layout.tiled = symbol == sz_levenshtein_bytes_k && queries.length_max > sz_levenshtein_gpu_words_max_k * 64 &&
                           candidates.length_max > sz_levenshtein_gpu_words_max_k * 64 &&
                           candidates.length_max <= 0xFFFFFF00u
                       ? sz_true_k
                       : sz_false_k;
    if (layout.tiled) {
        sz_size_t const shorter = sz_min_of_two(queries.length_max, candidates.length_max);
        sz_size_t const longer = sz_max_of_two(queries.length_max, candidates.length_max);
        layout.frontier_cells =
            sz_size_divide_round_up(shorter, sz_levenshtein_tile_side_simt_k) * sz_levenshtein_tile_side_simt_k + 1;
        layout.progress_cells = sz_size_divide_round_up(longer, sz_levenshtein_tile_side_simt_k);
        layout.scratch_stride = (layout.frontier_cells + layout.progress_cells) * sizeof(sz_u32_t);
    }
    else
        layout.scratch_stride = sz_levenshtein_query_words(queries.length_max) *
                                sizeof(sz_levenshtein_u64x1_vertical_serial_t);
    layout.scratch_stride = (layout.scratch_stride + 7) & ~(sz_size_t)7;
    sz_size_t const capacity = layout.tiled ? resident * 4 : resident * 128 * 4;
    layout.pairs = sz_min_of_two(queries.count * candidates.count, capacity);
    layout.pairs = sz_min_of_two(layout.pairs, (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k);
    if (layout.scratch_stride)
        layout.pairs = sz_min_of_two(
            layout.pairs, sz_max_of_two((sz_size_t)1, sz_levenshtein_workspace_bytes_simt_k / layout.scratch_stride));
    return layout;
}

/** Candidate index ranges passed by value to the device counting sort. */
typedef struct sz_levenshtein_candidate_buckets_simt_t {
    sz_size_t offsets[sz_levenshtein_length_buckets_k];
} sz_levenshtein_candidate_buckets_simt_t;

static __global__ void sz_levenshtein_order_simt_kernel_(sz_sequence_t candidates,
                                                         sz_levenshtein_candidate_buckets_simt_t buckets,
                                                         sz_size_t *cursors, sz_size_t *order) {
    for (sz_size_t index = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; index < candidates.count;
         index += (sz_size_t)gridDim.x * blockDim.x) {
        sz_size_t const bucket = sz_levenshtein_length_bucket_(sz_sequence_tape_length_simt_(candidates.handle, index));
        sz_size_t const position = (sz_size_t)atomicAdd((unsigned long long *)(cursors + bucket), 1ull);
        order[buckets.offsets[bucket] + position] = index;
    }
}

/** One slice of the bucketed cross-product, retaining original output indices. */
typedef struct sz_levenshtein_long_arguments_simt_t {
    sz_size_t query_first, candidate_first, candidate_count, pair_count, distances_stride;
    sz_size_t scratch_stride, frontier_cells;
    sz_u32_t const *query_order;
    sz_size_t const *candidate_order;
    sz_size_t const *query_offsets;
    sz_cptr_t query_text;
} sz_levenshtein_long_arguments_simt_t;

static __global__ void sz_levenshtein_long_simt_kernel_(sz_levenshtein_engine_t engine, sz_sequence_t candidates,
                                                        sz_size_t *distances, sz_ptr_t workspace,
                                                        sz_levenshtein_long_arguments_simt_t batch) {
    sz_size_t const pair = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= batch.pair_count) return;
    sz_size_t const query_index = batch.query_order[batch.query_first + pair / batch.candidate_count];
    sz_size_t const position = batch.candidate_first + pair % batch.candidate_count;
    sz_size_t const candidate = batch.candidate_order ? batch.candidate_order[position] : position;
    sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
    sz_size_t const length = sz_sequence_tape_length_simt_(candidates.handle, candidate);
    sz_size_t score = query.length;
    if (!words) {
        if (engine.symbol == sz_levenshtein_bytes_k) score = length;
        else
            for (sz_size_t cursor = 0; cursor < length; ++score) sz_utf8_next_rune_(text, length, &cursor);
    }
    else {
        sz_levenshtein_u64x1_vertical_serial_t *const verticals =
            (sz_levenshtein_u64x1_vertical_serial_t *)(workspace + pair * batch.scratch_stride);
        sz_levenshtein_u64x1_state_serial_t state;
        sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
        for (sz_size_t cursor = 0; cursor < length;) {
            sz_u32_t const class_id = engine.symbol == sz_levenshtein_bytes_k
                                          ? query.byte_to_class[(sz_u8_t)text[cursor++]]
                                          : sz_levenshtein_utf8_class(&query,
                                                                      sz_utf8_next_rune_(text, length, &cursor));
            sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, class_id);
        }
        score = sz_levenshtein_u64x1_score_serial(&state, 0);
    }
    distances[query_index * batch.distances_stride + candidate] = score;
}

static __global__
__launch_bounds__(sz_levenshtein_tiled_threads_per_block_simt_k) void sz_levenshtein_tiled_batch_simt_kernel_(
    sz_levenshtein_engine_t engine, sz_sequence_t candidates, sz_size_t *distances, sz_ptr_t workspace,
    sz_levenshtein_long_arguments_simt_t batch) {
    sz_size_t const pair = blockIdx.y;
    sz_size_t const query_index = batch.query_order[batch.query_first + pair / batch.candidate_count];
    sz_size_t const position = batch.candidate_first + pair % batch.candidate_count;
    sz_size_t const candidate = batch.candidate_order ? batch.candidate_order[position] : position;
    sz_size_t const query_length = engine.lengths[query_index];
    sz_size_t const length = sz_sequence_tape_length_simt_(candidates.handle, candidate);
    sz_cptr_t const query = batch.query_text + batch.query_offsets[query_index];
    sz_cptr_t const text = sz_sequence_tape_start_simt_(candidates.handle, candidate);
    sz_bool_t const query_shorter = query_length <= length ? sz_true_k : sz_false_k;
    sz_u32_t *const frontier = (sz_u32_t *)(workspace + pair * batch.scratch_stride);
    sz_levenshtein_tiled_simt_(query_shorter ? query : text, (sz_u32_t)(query_shorter ? query_length : length),
                               query_shorter ? text : query, (sz_u32_t)(query_shorter ? length : query_length),
                               frontier, frontier + batch.frontier_cells,
                               distances + query_index * batch.distances_stride + candidate);
}

#pragma endregion Tiled

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM
#endif // STRINGZILLA_LEVENSHTEIN_SIMT_CUH_
