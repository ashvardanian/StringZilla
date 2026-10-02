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
 *  A batch is bucketed by rung key at @ref sz_levenshtein_engine_init_simt_scoped_, so one launch
 *  carries only queries that share an entry point, and the occupancy walk each of those entry
 *  points needs is paid there rather than once per round. The byte planes are built by a kernel
 *  from the queries staged into device-reachable memory; the rune planes are built on the host, an
 *  init being allowed to join where a round is not.
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
enum { sz_levenshtein_simt_thread_words_max_k = 16 };

/** Lanes a warped candidate is spread across, which is a warp, and the query words one lane keeps
 *  verticals for, so a warp at its widest reaches the family's word limit. */
enum {
    sz_levenshtein_simt_warp_lanes_k = 32,
    sz_levenshtein_simt_warp_words_per_lane_max_k = sz_levenshtein_simt_words_max_k / sz_levenshtein_simt_warp_lanes_k,
};

/** Words one lane of a warped candidate owns, which is what selects the rung's entry point. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_warp_words_per_lane_(sz_size_t words) {
    return sz_size_divide_round_up(words, sz_levenshtein_simt_warp_lanes_k);
}

/** Query words from which the warped rung is launched rather than the threaded one. Swept on XLSum
 *  lines, the two cross between eight words and sixteen: at sixteen the warp leads by 1.5x in
 *  corpus order and 1.8x sorted by length, at eight it trails. The crossing moves with the longest
 *  candidate, not with the query, since a thread rung's warp costs the longest of its thirty-two
 *  candidates while a warped candidate costs its own. */
enum { sz_levenshtein_simt_warp_words_min_k = 16 };

/** Threads a block runs when the device cannot be asked; the register budget differs per word
 *  count, so the launcher takes what the occupancy calculator answers for the entry point it is
 *  about to launch instead. */
enum { sz_levenshtein_simt_candidates_per_block_k = 128 };

/** Widest block the launcher considers: past this a block schedules too coarsely for
 *  what residency returns. */
enum { sz_levenshtein_simt_candidates_per_block_max_k = 256 };

/** Rows one launch's query axis spans, past which a bucket is cut into several launches of
 *  the same kernel. */
enum { sz_levenshtein_simt_grid_rows_max_k = 65535 };

/** Rounds of candidates one block's tile holds where its blocks cannot take over unstarted ones.
 *  Measured on one wave of XLSum lines, four rounds at a quarter of the blocks beat one, two and
 *  eight, the balance being worth more than the residency it costs. */
enum { sz_levenshtein_simt_tile_rounds_k = 4 };

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
STRINGZILLA_DEVICE void sz_levenshtein_simt_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
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
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_simt_thread_words_max_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);

    // One byte per step is one uncoalesced sector per step, and at a one-word query that load is most of the
    // round. A text's eight bytes span two aligned words, and the second is read only once it holds
    // a byte of the text: an aligned word that holds one lies in a page the device may read all
    // eight bytes of.
    sz_cptr_t text = candidates.get_start(candidates.handle, candidate);
    sz_size_t left = candidates.get_length(candidates.handle, candidate);
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
            text = candidates.get_start(candidates.handle, candidate);
            left = candidates.get_length(candidates.handle, candidate);
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
 *  one shared kernel would have to spend on every launch. @c sz_levenshtein_simt_entry_point_
 *  picks between them. */
static __global__ void sz_levenshtein_u64x1_distances_simt_w1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w9_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w10_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w11_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w12_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w13_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w14_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w15_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_u64x1_distances_simt_w16_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_(engine, order, candidates, distances, distances_stride, 16);
}

/*  One entry point per word count, addressed by it. @c cudaLaunchKernel takes the host-side
 *  symbol of a @c __global__, so the table is what the `<<< >>>` operator would have selected,
 *  spelled as data. */
static void const *const sz_levenshtein_simt_entry_points_[sz_levenshtein_simt_thread_words_max_k] = {
    (void const *)sz_levenshtein_u64x1_distances_simt_w1_,  (void const *)sz_levenshtein_u64x1_distances_simt_w2_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w3_,  (void const *)sz_levenshtein_u64x1_distances_simt_w4_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w5_,  (void const *)sz_levenshtein_u64x1_distances_simt_w6_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w7_,  (void const *)sz_levenshtein_u64x1_distances_simt_w8_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w9_,  (void const *)sz_levenshtein_u64x1_distances_simt_w10_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w11_, (void const *)sz_levenshtein_u64x1_distances_simt_w12_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w13_, (void const *)sz_levenshtein_u64x1_distances_simt_w14_,
    (void const *)sz_levenshtein_u64x1_distances_simt_w15_, (void const *)sz_levenshtein_u64x1_distances_simt_w16_,
};

/**
 *  @brief Threads a sweep gives a block on this device, for the entry point it is about to launch.
 *
 *  Register pressure moves the residency ceiling from one word count to the next, so the block
 *  landing the most warps per multiprocessor is the device's answer and not a constant's. The walk
 *  stops at @c sz_levenshtein_simt_candidates_per_block_max_k rather than at what the entry point
 *  allows, since past it a block schedules too coarsely for what its residency returns; ties below
 *  it go to the wider block.
 */
static sz_size_t sz_levenshtein_simt_per_block_(void const *entry_point) {
    return sz_device_block_size_(entry_point, 0, sz_levenshtein_simt_candidates_per_block_max_k,
                                 sz_levenshtein_simt_candidates_per_block_k);
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_warp_candidate_(sz_levenshtein_query_t const *query_pointer,
                                                            sz_sequence_t const *candidates, sz_size_t candidate,
                                                            sz_size_t *row, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_levenshtein_query_t const query = *query_pointer;
    sz_cptr_t const text = candidates->get_start(candidates->handle, candidate);
    sz_size_t const length = candidates->get_length(candidates->handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = sz_size_divide_round_up(words, words_per_lane);
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_simt_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_simt_warp_words_per_lane_max_k];
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
STRINGZILLA_DEVICE int sz_levenshtein_simt_warp_draw_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
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
STRINGZILLA_DEVICE int sz_levenshtein_simt_warp_first_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_warp_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                        sz_sequence_t candidates, sz_size_t *distances,
                                                        sz_size_t distances_stride, sz_size_t words_per_lane) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index;
    int drawn = sz_levenshtein_simt_warp_first_(&queue, tile_size, candidates.count, &row_index, &candidate);
    // Warp uniform, so the shuffles inside still see a whole warp.
    for (; drawn; drawn = sz_levenshtein_simt_warp_draw_(&queue, tile_size, candidates.count, &row_index, &candidate)) {
        sz_size_t const query_index = order[row_index];
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
        sz_levenshtein_simt_warp_candidate_(&query, &candidates, candidate, distances + query_index * distances_stride,
                                            words_per_lane);
    }
}

/*  One entry point per words-per-lane, each passing its own literal, for the reason the threaded
 *  rung states: a count the compiler cannot see spills the verticals to local memory. @c K rounds
 *  up by one word rather than to the next power of two, since the remainder costs only idle lanes -
 *  linear leaves worst-case lane utilization at seven eighths where doubling drops it to just over
 *  half above each boundary. */
static __global__ void sz_levenshtein_u64x32_distances_simt_k1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x32_distances_simt_k8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_(engine, order, candidates, distances, distances_stride, 8);
}

/*  One entry point per words-per-lane, addressed by it, beside the threaded rung's own table. */
static void const *const sz_levenshtein_simt_entry_points_warp_[sz_levenshtein_simt_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_u64x32_distances_simt_k1_, (void const *)sz_levenshtein_u64x32_distances_simt_k2_,
    (void const *)sz_levenshtein_u64x32_distances_simt_k3_, (void const *)sz_levenshtein_u64x32_distances_simt_k4_,
    (void const *)sz_levenshtein_u64x32_distances_simt_k5_, (void const *)sz_levenshtein_u64x32_distances_simt_k6_,
    (void const *)sz_levenshtein_u64x32_distances_simt_k7_, (void const *)sz_levenshtein_u64x32_distances_simt_k8_,
};

#pragma endregion Myers Warped

#pragma region Myers Byte Lanes

/** Query symbols one byte lane of a Myers word holds, which is the longest query the byte rung
 *  takes, and one sixteen-bit lane holds, which is the longest query either narrow rung takes. */
enum {
    sz_levenshtein_simt_byte_lanes_symbols_max_k = 8,
    sz_levenshtein_simt_short_lanes_symbols_max_k = 16,
};

/** Candidates one thread advances together at its widest: four byte lanes to a thirty-two-bit
 *  register. A sixty-four-bit register holds eight, and it costs two instructions per operation to
 *  do it, so the wider packing buys no arithmetic and spends the thread-level parallelism that
 *  hides this rung's loads. */
enum { sz_levenshtein_simt_lanes_per_thread_max_k = 4 };

/** Text bytes one refill hands a lane, which is the aligned word its cursor sits in. */
enum { sz_levenshtein_simt_bytes_per_refill_k = 8 };

/**
 *  @brief Advances every lane one symbol through its own Myers word, all of them packed
 *      in one register.
 *
 *  The device has no per-lane add, so the one carry that would cross a lane boundary is blocked the
 *  standard way: the low bits of every lane sum on their own, where a carry cannot leave the lane,
 *  and each lane's top bit is folded back in by exclusive or.
 */
STRINGZILLA_DEVICE void sz_levenshtein_simt_lanes_step_(sz_u32_t *positive, sz_u32_t *negative, sz_u32_t equality,
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
STRINGZILLA_DEVICE sz_size_t sz_levenshtein_simt_lane_distance_(sz_u32_t positive, sz_u32_t negative,
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_lanes_sweep_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
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
    sz_u64_t const *cursors[sz_levenshtein_simt_lanes_per_thread_max_k];
    sz_size_t lefts[sz_levenshtein_simt_lanes_per_thread_max_k];
    sz_u64_t words[sz_levenshtein_simt_lanes_per_thread_max_k], octets[sz_levenshtein_simt_lanes_per_thread_max_k];
    sz_u32_t shifts[sz_levenshtein_simt_lanes_per_thread_max_k], held[sz_levenshtein_simt_lanes_per_thread_max_k];
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
                    sz_size_t const length = candidates.get_length(candidates.handle, candidate);
                    if (!length) {
                        row[candidate] = query.length;
                        continue;
                    }
                    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
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
            sz_u64_t const following = lefts[lane] + shifts[lane] / 8 > sz_levenshtein_simt_bytes_per_refill_k
                                           ? cursors[lane][1]
                                           : 0;
            octets[lane] = shifts[lane] ? (words[lane] >> shifts[lane]) | (following << (64 - shifts[lane]))
                                        : words[lane];
            words[lane] = following, ++cursors[lane];
            if (lefts[lane] <= sz_levenshtein_simt_bytes_per_refill_k) ending |= (sz_u32_t)1 << (lefts[lane] - 1);
        }
#pragma unroll
        for (sz_size_t slot = 0; slot != sz_levenshtein_simt_bytes_per_refill_k; ++slot) {
            sz_u32_t equality = 0;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                sz_u32_t const byte = (sz_u32_t)(octets[lane] >> (slot * 8)) & 0xFF;
                equality |= byte_to_mask[byte] << (lane * lane_bits);
            }
            sz_levenshtein_simt_lanes_step_(&positive, &negative, equality, lane_low_bits, lane_high_bits);
            if ((ending & ((sz_u32_t)1 << slot)) == 0) continue;
#pragma unroll
            for (sz_size_t lane = 0; lane != lanes; ++lane) {
                if ((unread & ((sz_u32_t)1 << lane)) == 0 || lefts[lane] != slot + 1) continue;
                sz_size_t const candidate = tile_first + held[lane];
                row[candidate] = sz_levenshtein_simt_lane_distance_(
                    positive, negative, lane * lane_bits, live, candidates.get_length(candidates.handle, candidate));
                unread &= ~((sz_u32_t)1 << lane), refill |= (sz_u32_t)1 << lane;
            }
        }
#pragma unroll
        for (sz_size_t lane = 0; lane != lanes; ++lane) lefts[lane] -= sz_min_of_two(lefts[lane], (sz_size_t)8);
    }
}

/*  One entry point per lane width, each passing its own literals, for the reason the threaded rung
 *  states: a width the compiler cannot see turns the lane arrays into local memory. */
static __global__ void sz_levenshtein_u8x4_distances_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                           sz_sequence_t candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride) {
    sz_levenshtein_simt_lanes_sweep_(engine, order, candidates, distances, distances_stride, 8, 4);
}

static __global__ void sz_levenshtein_u16x2_distances_simt_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                            sz_sequence_t candidates, sz_size_t *distances,
                                                            sz_size_t distances_stride) {
    sz_levenshtein_simt_lanes_sweep_(engine, order, candidates, distances, distances_stride, 16, 2);
}

/** Candidates one thread takes, which is the lanes a query of @p length symbols leaves
 *  in one register. */
static sz_size_t sz_levenshtein_simt_lanes_per_thread_(sz_size_t length) {
    return length <= sz_levenshtein_simt_byte_lanes_symbols_max_k ? 4 : 2;
}

/** Candidates a batch must carry before a narrow rung is launched at all, which the device's own
 *  residency scales. @c STRINGZILLA_SIZE_MAX where the device cannot be asked, so the threaded rung
 *  keeps every batch. */
static sz_size_t sz_levenshtein_simt_lanes_candidates_min_(void) {
    sz_size_t const resident_threads = sz_device_multiprocessors_() * sz_device_threads_per_multiprocessor_();
    return resident_threads ? resident_threads * sz_levenshtein_simt_lanes_waves_min_k : STRINGZILLA_SIZE_MAX;
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_sweep_utf8_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
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
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_simt_thread_words_max_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    sz_cptr_t text = candidates.get_start(candidates.handle, candidate);
    sz_size_t length = candidates.get_length(candidates.handle, candidate);
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
        text = candidates.get_start(candidates.handle, candidate);
        length = candidates.get_length(candidates.handle, candidate);
        position = 0;
        sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    }
}

/*  One rune entry point per word count, each passing its own literal, for the reason the byte tier
 *  states: a word count the compiler cannot see spills the verticals to local memory. */
static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 8);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w9_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 9);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w10_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 10);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w11_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 11);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w12_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 12);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w13_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 13);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w14_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 14);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w15_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 15);
}

static __global__ void sz_levenshtein_u64x1_distances_utf8_simt_w16_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_sweep_utf8_(engine, order, candidates, distances, distances_stride, 16);
}

/*  One rune entry point per word count, addressed by it, beside the byte tier's own table. */
static void const *const sz_levenshtein_simt_entry_points_utf8_[sz_levenshtein_simt_thread_words_max_k] = {
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w1_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w2_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w3_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w4_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w5_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w6_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w7_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w8_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w9_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w10_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w11_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w12_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w13_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w14_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w15_,
    (void const *)sz_levenshtein_u64x1_distances_utf8_simt_w16_,
};

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
STRINGZILLA_DEVICE void sz_levenshtein_simt_warp_candidate_utf8_(sz_levenshtein_query_t const *query_pointer,
                                                                 sz_sequence_t const *candidates, sz_size_t candidate,
                                                                 sz_size_t *row, sz_size_t words_per_lane) {
    unsigned const lane = threadIdx.x & 31u;
    sz_levenshtein_query_t const query = *query_pointer;
    sz_cptr_t const text = candidates->get_start(candidates->handle, candidate);
    sz_size_t const length = candidates->get_length(candidates->handle, candidate);
    sz_size_t const words = sz_levenshtein_query_words(query.length);
    sz_size_t const live_lanes = sz_size_divide_round_up(words, words_per_lane);
    sz_size_t const first_word = (sz_size_t)lane * words_per_lane;
    sz_u64_t const *const lane_masks = query.masks + first_word;

    sz_u64_t positive[sz_levenshtein_simt_warp_words_per_lane_max_k];
    sz_u64_t negative[sz_levenshtein_simt_warp_words_per_lane_max_k];
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_warp_sweep_utf8_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                             sz_sequence_t candidates, sz_size_t *distances,
                                                             sz_size_t distances_stride, sz_size_t words_per_lane) {
    __shared__ sz_tile_queue_t queue;
    sz_size_t const tile_size = sz_size_divide_round_up(candidates.count, gridDim.x);
    sz_size_t candidate;
    sz_u32_t row_index;
    int drawn = sz_levenshtein_simt_warp_first_(&queue, tile_size, candidates.count, &row_index, &candidate);
    // Warp uniform, so the shuffles inside still see a whole warp.
    for (; drawn; drawn = sz_levenshtein_simt_warp_draw_(&queue, tile_size, candidates.count, &row_index, &candidate)) {
        sz_size_t const query_index = order[row_index];
        sz_levenshtein_query_t const query = sz_levenshtein_engine_row_(&engine, query_index);
        sz_levenshtein_simt_warp_candidate_utf8_(&query, &candidates, candidate,
                                                 distances + query_index * distances_stride, words_per_lane);
    }
}

/*  One warped rune entry point per words-per-lane, each passing its own literal, for the reason the
 *  threaded rung states: a count the compiler cannot see spills the verticals to local memory. */
static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k1_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 1);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k2_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 2);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k3_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 3);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k4_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 4);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k5_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 5);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k6_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 6);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k7_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 7);
}

static __global__ void sz_levenshtein_u64x32_distances_utf8_simt_k8_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_simt_warp_sweep_utf8_(engine, order, candidates, distances, distances_stride, 8);
}

/*  One warped rune entry point per words-per-lane, addressed by it, beside the threaded rune
 *  tier's own table. */
static void const *const sz_levenshtein_simt_entry_points_utf8_warp_[sz_levenshtein_simt_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k1_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k2_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k3_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k4_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k5_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k6_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k7_,
    (void const *)sz_levenshtein_u64x32_distances_utf8_simt_k8_,
};

#pragma endregion Myers UTF 8 Warped

#pragma region Myers Engine

/** What a device round needs that the batch already fixed: its rung buckets and the geometry each
 *  of them takes. */
typedef struct sz_levenshtein_simt_head_t {

    /** Rung keys the batch spans: @c words_max plus the two narrow ones. */
    sz_size_t buckets;

    /** Candidates before a narrow rung is launched, asked of the device once. */
    sz_size_t candidates_min;

    /** Whether the wide rungs' blocks take over unstarted ones, so a tile holds one round. */
    sz_bool_t steals;

    /** Threads the byte-lane and the short-lane entry points take. */
    sz_size_t narrow_per_block[2];

    /** The @b [buckets+1] first position of each key inside @c order. */
    sz_size_t *bucket_offsets;

    /** The @b [buckets] threads that key's own wide entry point takes. */
    sz_size_t *per_block;

    /** The @b [count] query indices, the keys' runs back to back. */
    sz_u32_t *order;
} sz_levenshtein_simt_head_t;

/** The rung key a query of @p length symbols launches from: the two narrow lanes, then one
 *  key per word. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_bucket_(sz_size_t length) {
    if (length <= sz_levenshtein_simt_byte_lanes_symbols_max_k) return 0;
    if (length <= sz_levenshtein_simt_short_lanes_symbols_max_k) return 1;
    return 1 + sz_levenshtein_query_words(length);
}

/** Rung keys a batch whose widest query spans @p words words can reach. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_buckets_(sz_size_t words) { return words + 2; }

/** Query words the wide rung of @p bucket steps, the two narrow keys sharing the
 *  one-word entry points. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_bucket_words_(sz_size_t bucket) {
    return bucket <= 1 ? 1 : bucket - 1;
}

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning
 *  @p buckets rung keys. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_head_bytes_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_simt_head_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/**
 *  @brief The entry point @p bucket 's wide rung reaches: one candidate per thread, or one per warp
 *      past the crossing.
 *
 *  A narrow query's verticals fit one thread's registers, and nothing beats keeping the recurrence
 *  there. A wider one spreads them across a warp, where the skew turns the carry between words
 *  into one shuffle.
 */
static void const *sz_levenshtein_simt_entry_point_(sz_levenshtein_symbol_t symbol, sz_size_t bucket) {
    sz_size_t const words = sz_levenshtein_simt_bucket_words_(bucket);
    if (words < sz_levenshtein_simt_warp_words_min_k)
        return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_simt_entry_points_[words - 1]
                                                : sz_levenshtein_simt_entry_points_utf8_[words - 1];
    sz_size_t const per_lane = sz_levenshtein_simt_warp_words_per_lane_(words);
    return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_simt_entry_points_warp_[per_lane - 1]
                                            : sz_levenshtein_simt_entry_points_utf8_warp_[per_lane - 1];
}

/** Points the head's tables into the block, buckets the batch by rung key, and asks the device
 *  its geometry once. */
static void sz_levenshtein_simt_bind_head_(sz_levenshtein_engine_t *engine, sz_size_t buckets_bound) {
    sz_levenshtein_simt_head_t *const head = (sz_levenshtein_simt_head_t *)engine->memory;
    sz_size_t *const tables = (sz_size_t *)((sz_ptr_t)engine->memory + sizeof(sz_levenshtein_simt_head_t));
    sz_size_t cursors[sz_levenshtein_simt_words_max_k + 2];
    head->buckets = sz_levenshtein_simt_buckets_(sz_levenshtein_engine_words_max_(engine));
    head->bucket_offsets = tables;
    head->per_block = tables + buckets_bound + 1;
    head->order = (sz_u32_t *)(tables + 2 * buckets_bound + 1);

    // A counting sort by rung key, so one launch only ever carries queries that share an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) head->bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        ++head->bucket_offsets[sz_levenshtein_simt_bucket_(engine->lengths[index]) + 1];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        head->bucket_offsets[bucket] += head->bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = head->bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        head->order[cursors[sz_levenshtein_simt_bucket_(engine->lengths[index])]++] = (sz_u32_t)index;

    // Every driver round trip a round would otherwise pay: the residency floor once, and one occupancy walk
    // per entry point the batch can reach, which the buckets fixed here and no later call can widen.
    head->candidates_min = sz_levenshtein_simt_lanes_candidates_min_();
    head->steals = sz_device_kernel_steals_((void const *)sz_levenshtein_u64x1_distances_simt_w1_);
    head->narrow_per_block[0] = sz_levenshtein_simt_per_block_((void const *)sz_levenshtein_u8x4_distances_simt_);
    head->narrow_per_block[1] = sz_levenshtein_simt_per_block_((void const *)sz_levenshtein_u16x2_distances_simt_);
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket)
        head->per_block[bucket] = head->bucket_offsets[bucket] == head->bucket_offsets[bucket + 1]
                                      ? 0
                                      : sz_levenshtein_simt_per_block_(
                                            sz_levenshtein_simt_entry_point_(engine->symbol, bucket));
}

/** Threads the mask builder runs: one per byte value, so a thread owns one class flag for
 *  the whole build. */
enum { sz_levenshtein_simt_masks_threads_k = sz_levenshtein_byte_classes_k };

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
static __global__ void sz_levenshtein_simt_masks_kernel_(sz_levenshtein_engine_t engine, sz_string_view_t const *texts,
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
 *  @brief Stages the batch's query texts where the device reads them and builds every
 *      byte plane there.
 *  @note Joins @p stream, which is what lets the staging be released; only an init is allowed to.
 */
static sz_status_t sz_levenshtein_simt_build_masks_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                    void *stream) {
    sz_size_t texts_bytes = 0;
    for (sz_size_t index = 0; index != queries->count; ++index)
        texts_bytes += queries->get_length(queries->handle, index);
    sz_size_t const views_bytes = queries->count * sizeof(sz_string_view_t);
    sz_size_t const staged_bytes = views_bytes + texts_bytes;
    sz_ptr_t const staged = (sz_ptr_t)engine->allocator.allocate(staged_bytes, engine->allocator.handle);
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
    sz_status_t status = sz_device_memset_((void *)engine->masks, 0, masks_bytes, stream);
    sz_device_prefetch_(engine->memory, engine->memory_bytes, stream);
    sz_device_prefetch_(staged, staged_bytes, stream);
    for (sz_size_t first = 0; first < queries->count && status == sz_success_k;
         first += sz_levenshtein_simt_grid_rows_max_k) {
        sz_levenshtein_engine_t launch_engine = *engine;
        sz_string_view_t const *launch_views = views;
        sz_size_t launch_first = first;
        void *arguments[3];
        arguments[0] = &launch_engine, arguments[1] = &launch_views, arguments[2] = &launch_first;
        dim3 grid, block;
        grid.x = 1, grid.z = 1;
        grid.y = (unsigned)sz_min_of_two(queries->count - first, (sz_size_t)sz_levenshtein_simt_grid_rows_max_k);
        block.x = sz_levenshtein_simt_masks_threads_k, block.y = 1, block.z = 1;
        status = sz_device_launch_((void const *)sz_levenshtein_simt_masks_kernel_, grid, block, arguments, 0, stream);
    }
    // The staging is the host's, so it outlives the builder only as long as the join below takes.
    if (status == sz_success_k) status = sz_device_synchronize_(stream);
    engine->allocator.free(staged, staged_bytes, engine->allocator.handle);
    return status;
}

/** Prepares @p queries on the device the caller already made current, which is every step of
 *  @ref sz_levenshtein_engine_init_simt_scoped_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_simt_(sz_levenshtein_engine_t *engine,
                                                                sz_sequence_t const *queries,
                                                                sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                                sz_memory_allocator_t *allocator, void *stream) {
    sz_memory_allocator_t unified;
    if (!sz_device_multiprocessors_()) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_memory_allocator_init_unified_(&unified, ordinal);
    if (queries->count == 0) return sz_unexpected_dimensions_k;

    // Every query seeds its score from its own last word, so an empty one has no word to read it off, and the
    // rung ceiling is checked twice: on the bytes here, which bound the runes, and on the symbols once measured.
    sz_size_t longest = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (bytes == 0) return sz_unexpected_dimensions_k;
        longest = sz_max_of_two(longest, bytes);
    }
    if (symbol == sz_levenshtein_bytes_k && longest > sz_levenshtein_simt_words_max_k * 64)
        return sz_unexpected_dimensions_k;

    sz_size_t const buckets_bound = sz_levenshtein_simt_buckets_(sz_size_divide_round_up(longest, 64));
    sz_size_t const head_bytes = sz_levenshtein_simt_head_bytes_(queries->count, buckets_bound);
    sz_status_t status = sz_levenshtein_engine_build_(queries, symbol, head_bytes, &unified, engine);
    if (status != sz_success_k) return status;
    if (sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_simt_words_max_k) {
        sz_levenshtein_engine_free_(engine);
        return sz_unexpected_dimensions_k;
    }
    engine->capability = STRINGZILLA_ARCH_ROCM_ ? sz_cap_rocm_k : sz_cap_cuda_k, engine->ordinal = ordinal;
    sz_levenshtein_simt_bind_head_(engine, buckets_bound);

    // The rune planes are the host's: a page table is a scan with no counterpart here, and an init may join
    // where a round may not, so the one crossing is a bulk migration rather than a fault per page.
    if (symbol == sz_levenshtein_runes_k) {
        sz_levenshtein_engine_fill_(engine, queries);
        sz_device_prefetch_(engine->memory, engine->memory_bytes, stream);
        return sz_success_k;
    }
    status = sz_levenshtein_simt_build_masks_(engine, queries, stream);
    if (status != sz_success_k) sz_levenshtein_engine_free_(engine);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_simt_scoped_(sz_levenshtein_engine_t *engine,
                                                                       sz_sequence_t const *queries,
                                                                       sz_levenshtein_symbol_t symbol,
                                                                       sz_size_t ordinal,
                                                                       sz_memory_allocator_t *allocator, void *stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_(ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_engine_init_simt_(engine, queries, symbol, ordinal, allocator, stream);
    sz_device_leave_(caller);
    return status;
}

/** Launches whichever rung's entry point the caller chose, over a grid of candidate tiles
 *  by prepared queries. */
static sz_status_t sz_levenshtein_simt_distances_(void const *entry_point, sz_size_t blocks, sz_size_t queries,
                                                  sz_size_t per_block, void *stream, sz_levenshtein_engine_t engine,
                                                  sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
                                                  sz_size_t distances_stride) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)queries, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[5];
    arguments[0] = &engine, arguments[1] = &order, arguments[2] = &candidates, arguments[3] = &distances;
    arguments[4] = &distances_stride;
    return sz_device_launch_(entry_point, grid, block, arguments, 0, stream);
}

/** Enqueues one round on the device the caller already made current, which is every step of
 *  @ref sz_levenshtein_distances_simt_scoped_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_simt_(sz_levenshtein_engine_t *engine,
                                                              sz_sequence_t const *candidates, sz_size_t *distances,
                                                              sz_size_t distances_stride, void *stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (candidates->count == 0) return sz_success_k;

    // The handle is checked, never the accessors: those are the device's to call, so the host must not, and a
    // pointer is all this side can inspect. That the texts they answer are device-reachable is the caller's word.
    if (!sz_memory_reaches_device_(distances)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device_(candidates->handle)) return sz_device_memory_mismatch_k;

    sz_levenshtein_simt_head_t const *const head = (sz_levenshtein_simt_head_t const *)engine->memory;
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
        void const *entry_point = sz_levenshtein_simt_entry_point_(engine->symbol, bucket);
        sz_size_t per_block = head->per_block[bucket];
        // A block that takes over unstarted ones needs no more than one round of its own, and the
        // grid is then as wide as the batch; elsewhere a tile holds several rounds for its block's
        // threads to share.
        sz_size_t const rounds = head->steals && !narrow ? 1 : sz_levenshtein_simt_tile_rounds_k;
        sz_size_t candidates_per_block = per_block * rounds;
        if (narrow) {
            entry_point = bucket == 0 ? (void const *)sz_levenshtein_u8x4_distances_simt_
                                      : (void const *)sz_levenshtein_u16x2_distances_simt_;
            per_block = head->narrow_per_block[bucket];
            candidates_per_block = per_block * rounds *
                                   sz_levenshtein_simt_lanes_per_thread_(
                                       bucket == 0 ? sz_levenshtein_simt_byte_lanes_symbols_max_k
                                                   : sz_levenshtein_simt_short_lanes_symbols_max_k);
        }
        else if (sz_levenshtein_simt_bucket_words_(bucket) >= sz_levenshtein_simt_warp_words_min_k)
            candidates_per_block = per_block / sz_levenshtein_simt_warp_lanes_k * rounds;
        sz_size_t const blocks = sz_size_divide_round_up(candidates->count, candidates_per_block);

        for (sz_size_t row = first; row < last; row += sz_levenshtein_simt_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(last - row, (sz_size_t)sz_levenshtein_simt_grid_rows_max_k);
            sz_status_t const launched = sz_levenshtein_simt_distances_(entry_point, blocks, rows, per_block, stream,
                                                                        *engine, head->order + row, *candidates,
                                                                        distances, distances_stride);
            if (launched != sz_success_k) return launched;
        }
    }
    return sz_success_k;
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_simt_scoped_(sz_levenshtein_engine_t *engine,
                                                                     sz_sequence_t const *candidates,
                                                                     sz_size_t *distances, sz_size_t distances_stride,
                                                                     void *stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_(engine->ordinal, stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_distances_simt_(engine, candidates, distances, distances_stride, stream);
    sz_device_leave_(caller);
    return status;
}

#pragma endregion Myers Engine

#pragma region Tiled

/** Tile geometry of the device-spanning wavefront: one warp owns a 128-wide tile-column,
 *  one lane owns a 4x4 register micro-tile, so a tile is a 63-step anti-diagonal sweep
 *  across 32 micro-rows. */
enum {
    sz_levenshtein_simt_tile_side_k = sz_levenshtein_tile_side_k,
    sz_levenshtein_simt_micro_side_k = 4,
    sz_levenshtein_simt_lanes_k = 32,
    sz_levenshtein_simt_micro_rows_k = sz_levenshtein_simt_tile_side_k / sz_levenshtein_simt_micro_side_k,
    sz_levenshtein_simt_tiled_warps_per_block_k = 8,
#if STRINGZILLA_ARCH_ROCM_
    // A 64-wide wavefront would march two tile-columns in lockstep, one spinning on the other for good.
    sz_levenshtein_simt_tiled_warp_stride_k = 64,
#else
    sz_levenshtein_simt_tiled_warp_stride_k = sz_levenshtein_simt_lanes_k,
#endif
    sz_levenshtein_simt_tiled_threads_per_block_k = sz_levenshtein_simt_tiled_warps_per_block_k *
                                                    sz_levenshtein_simt_tiled_warp_stride_k,
};

/** Longest text the wavefront indexes. Lengths and cells are @c sz_u32_t inside the kernel, and the
 *  last tile of each axis is padded up to 128 columns, so the ceiling leaves that padding room
 *  rather than letting a length near the word's top wrap a column index. */
#define STRINGZILLA_LEVENSHTEIN_SIMT_TILED_LENGTH_MAX (0xFFFFFF00u)

/** Bytes standing in for a character past the end of a text. The two differ, so a padded row never
 *  matches a padded column, and a padded cell never reaches an in-bounds one. */
enum {
    sz_levenshtein_simt_past_query_k = 0xFE,
    sz_levenshtein_simt_past_target_k = 0xFF,
};

/*  Tile-rows a pair needs before the wavefront is worth reaching for. Its makespan is
 *  tile-rows plus tile-columns tile latencies, so tile-rows is how many tile-columns it ever
 *  runs at once: under three of them the wavefront is a serial chain down the long axis and
 *  one Myers lane's bit-parallel sweep is quicker, while from three up it is measured ahead at
 *  every long-axis length. */
enum { sz_levenshtein_simt_tiled_rows_min_k = 3 };

/** Selects what a micro-tile march does with a finished cell: nothing, or the corner test that
 *  publishes the pair's distance. */
typedef enum {
    sz_levenshtein_simt_march_fast_k = 0,
    sz_levenshtein_simt_march_checked_k = 1,
} sz_levenshtein_simt_march_t;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

/** Three-way unsigned minimum, one instruction on Hopper and Blackwell. */
STRINGZILLA_DEVICE sz_u32_t sz_levenshtein_simt_min3_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
    return __vimin3_u32(first, second, third);
}

#else

/** Three-way unsigned minimum as a pair of comparisons, for targets without the fused form. */
STRINGZILLA_DEVICE sz_u32_t sz_levenshtein_simt_min3_(sz_u32_t first, sz_u32_t second, sz_u32_t third) {
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_publish_(sz_u32_t *counter, sz_u32_t tile_row) {
    sz_u32_t const published = tile_row + 1u;
#if STRINGZILLA_ARCH_ROCM_
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_await_(sz_u32_t const *counter, sz_u32_t tile_row) {
    sz_u32_t observed = 0;
    do {
#if STRINGZILLA_ARCH_ROCM_
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
STRINGZILLA_DEVICE void sz_levenshtein_simt_march_tile_(                         //
    sz_levenshtein_simt_march_t march, unsigned lane_index,                      //
    sz_u32_t tile_first_row, sz_u32_t tile_first_column,                         //
    sz_u32_t shorter_length, sz_u32_t longer_length,                             //
    char const *shared_query, sz_u32_t const *shared_left, sz_u32_t tile_corner, //
    char const *target_chars, sz_u32_t *carry_top, sz_u32_t *row_frontier, sz_size_t *distance) {

    sz_u32_t previous_right_edge[sz_levenshtein_simt_micro_side_k];
    sz_u32_t previous_topright = 0;
    unsigned const wavefront_steps = sz_levenshtein_simt_micro_rows_k + sz_levenshtein_simt_lanes_k - 1;
    unsigned element, wavefront_step, micro_row_cell, micro_column_cell;
#pragma unroll
    for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element) previous_right_edge[element] = 0;

    for (wavefront_step = 0; wavefront_step != wavefront_steps; ++wavefront_step) {
        // Underflows for a lane the wavefront has not reached yet, which the unsigned bound below rejects.
        unsigned const micro_row = wavefront_step - lane_index;
        sz_u32_t shuffled_right_edge[sz_levenshtein_simt_micro_side_k];
        sz_u32_t shuffled_topright;
#pragma unroll
        for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
            shuffled_right_edge[element] = sz_shuffle_up_simt_(previous_right_edge[element], 1);
        shuffled_topright = sz_shuffle_up_simt_(previous_topright, 1);
        if (micro_row >= sz_levenshtein_simt_micro_rows_k) continue;

        sz_u32_t const micro_first_row = tile_first_row + micro_row * sz_levenshtein_simt_micro_side_k;
        sz_u32_t left_column[sz_levenshtein_simt_micro_side_k], diagonal_corner;
        if (lane_index == 0) {
#pragma unroll
            for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
                left_column[element] = shared_left[micro_row * sz_levenshtein_simt_micro_side_k + element];
            diagonal_corner = micro_row == 0 ? tile_corner
                                             : shared_left[micro_row * sz_levenshtein_simt_micro_side_k - 1];
        }
        else {
#pragma unroll
            for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
                left_column[element] = shuffled_right_edge[element];
            diagonal_corner = shuffled_topright;
        }

        // The top-right corner the lane to the right needs is the last cell of this micro-tile's incoming top
        // row, which the march below overwrites, so it is captured first.
        sz_u32_t const topright_for_next_lane = carry_top[sz_levenshtein_simt_micro_side_k - 1];
        sz_u32_t above_row[sz_levenshtein_simt_micro_side_k + 1];
        sz_u32_t right_edge[sz_levenshtein_simt_micro_side_k];
        above_row[0] = diagonal_corner;
#pragma unroll
        for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
            above_row[element + 1] = carry_top[element];

#pragma unroll
        for (micro_row_cell = 1; micro_row_cell <= sz_levenshtein_simt_micro_side_k; ++micro_row_cell) {
            sz_u32_t current_row[sz_levenshtein_simt_micro_side_k + 1];
            char const query_char = shared_query[micro_row * sz_levenshtein_simt_micro_side_k + micro_row_cell - 1];
            current_row[0] = left_column[micro_row_cell - 1];
#pragma unroll
            for (micro_column_cell = 1; micro_column_cell <= sz_levenshtein_simt_micro_side_k; ++micro_column_cell) {
                sz_u32_t const substitution = query_char == target_chars[micro_column_cell - 1] ? 0u : 1u;
                sz_u32_t const cell = sz_levenshtein_simt_min3_(above_row[micro_column_cell - 1] + substitution,
                                                                above_row[micro_column_cell] + 1u,
                                                                current_row[micro_column_cell - 1] + 1u);
                current_row[micro_column_cell] = cell;
                if (march == sz_levenshtein_simt_march_checked_k) {
                    sz_u32_t const matrix_row = micro_first_row + micro_row_cell;
                    sz_u32_t const matrix_column = tile_first_column + lane_index * sz_levenshtein_simt_micro_side_k +
                                                   micro_column_cell;
                    if (matrix_row == shorter_length && matrix_column == longer_length) *distance = (sz_size_t)cell;
                }
            }
            right_edge[micro_row_cell - 1] = current_row[sz_levenshtein_simt_micro_side_k];
#pragma unroll
            for (element = 0; element <= sz_levenshtein_simt_micro_side_k; ++element)
                above_row[element] = current_row[element];
        }

#pragma unroll
        for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
            carry_top[element] = above_row[element + 1], previous_right_edge[element] = right_edge[element];
        previous_topright = topright_for_next_lane;
        // The rightmost micro-column is the tile's right column, which is the next tile-column's left one.
        if (lane_index == sz_levenshtein_simt_lanes_k - 1)
#pragma unroll
            for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element)
                row_frontier[micro_first_row + element + 1] = right_edge[element];
    }
}

/** Stores a distance known before any cell is filled, which is every pair with an empty text. */
static __global__ void sz_levenshtein_simt_store_kernel_(sz_size_t *distance, sz_size_t value) { *distance = value; }

/**
 *  @brief One pair's Levenshtein distance as an anti-diagonal wavefront spanning the whole device.
 *
 *  A warp owns a 128-wide tile-column and marches it top to bottom, taking the tile-column a
 *  whole grid further on when it runs out, so the launched grid can be capped at what stays
 *  co-resident. Every tile's top edge is free - it stays in the marching warp's registers - and
 *  only the left edge crosses warps, through @p row_frontier under the @p progress counters.
 *  Tile-column @e c reads the band tile-column @e c-1 released and overwrites it with its own
 *  right column; the counters serialize that into one read and one write per band per column,
 *  which is also what lets a warp reuse the band for a later tile-column. There is no grid
 *  barrier and no cooperative launch.
 *
 *  @param[in] shorter_text Text along the row axis, device-reachable, @p shorter_length bytes.
 *  @param[in] shorter_length Its length, at most @c STRINGZILLA_LEVENSHTEIN_SIMT_TILED_LENGTH_MAX.
 *  @param[in] longer_text Text along the column axis, device-reachable, @p longer_length bytes.
 *  @param[in] longer_length Its length, at most @c STRINGZILLA_LEVENSHTEIN_SIMT_TILED_LENGTH_MAX.
 *  @param[out] row_frontier Scratch of `round_up(shorter_length, 128) + 1` unseeded cells.
 *  @param[out] progress One counter per tile-column, zeroed before the launch.
 *  @param[out] distance The pair's distance, written once by the thread owning the matrix corner.
 */
static __global__ __launch_bounds__(sz_levenshtein_simt_tiled_threads_per_block_k) void //
    sz_levenshtein_simt_tiled_kernel_(                                                  //
        sz_cptr_t shorter_text, sz_u32_t shorter_length,                                //
        sz_cptr_t longer_text, sz_u32_t longer_length,                                  //
        sz_u32_t *row_frontier, sz_u32_t *progress, sz_size_t *distance) {

    // Each warp stages its tile's query window and its incoming left boundary once per tile-row, so the
    // wavefront's scattered lane-0 boundary reads come off-chip once instead of once per micro-tile.
    __shared__ char shared_query[sz_levenshtein_simt_tiled_warps_per_block_k][sz_levenshtein_simt_tile_side_k];
    __shared__ sz_u32_t shared_left[sz_levenshtein_simt_tiled_warps_per_block_k][sz_levenshtein_simt_tile_side_k];

    unsigned const warp_in_block = threadIdx.x / sz_levenshtein_simt_tiled_warp_stride_k;
    unsigned const lane_index = threadIdx.x % sz_levenshtein_simt_tiled_warp_stride_k;
    sz_u32_t const tile_grid_rows = sz_u32_divide_round_up(shorter_length, sz_levenshtein_simt_tile_side_k);
    sz_u32_t const tile_grid_columns = sz_u32_divide_round_up(longer_length, sz_levenshtein_simt_tile_side_k);
    sz_u32_t const warps_in_grid = gridDim.x * sz_levenshtein_simt_tiled_warps_per_block_k;
    sz_u32_t const corner_tile_row = (shorter_length - 1u) / sz_levenshtein_simt_tile_side_k;
    sz_u32_t const corner_tile_column = (longer_length - 1u) / sz_levenshtein_simt_tile_side_k;
    sz_u32_t const first_tile_column = blockIdx.x * sz_levenshtein_simt_tiled_warps_per_block_k + warp_in_block;
    sz_u32_t tile_column, tile_row;
    unsigned element, stage_row;
    if (lane_index >= sz_levenshtein_simt_lanes_k) return;

    for (tile_column = first_tile_column; tile_column < tile_grid_columns; tile_column += warps_in_grid) {
        sz_u32_t const tile_first_column = tile_column * sz_levenshtein_simt_tile_side_k;
        char target_chars[sz_levenshtein_simt_micro_side_k];
        sz_u32_t carry_top[sz_levenshtein_simt_micro_side_k];
        // The row above the whole matrix costs one deletion per column, and this lane's target characters
        // never change down the column.
#pragma unroll
        for (element = 0; element != sz_levenshtein_simt_micro_side_k; ++element) {
            sz_u32_t const target_index = tile_first_column + lane_index * sz_levenshtein_simt_micro_side_k + element;
            target_chars[element] = target_index < longer_length ? longer_text[target_index]
                                                                 : (char)sz_levenshtein_simt_past_target_k;
            carry_top[element] = target_index + 1u;
        }
        // The leftmost column of the whole matrix costs one insertion per row, so the first tile-column
        // synthesizes its left boundary instead of reading a seeded one, and nothing has to seed it.
        sz_u32_t tile_corner = tile_first_column;

        for (tile_row = 0; tile_row != tile_grid_rows; ++tile_row) {
            sz_u32_t const tile_first_row = tile_row * sz_levenshtein_simt_tile_side_k;
            sz_u32_t tile_bottom_left;
            sz_levenshtein_simt_march_t march;

            if (tile_column != 0) sz_levenshtein_simt_await_(progress + (tile_column - 1u), tile_row);
            // Holds this tile-row's staging behind the previous one's reads of the same shared window.
            __syncwarp();
            if (tile_column == 0)
                for (stage_row = lane_index; stage_row < sz_levenshtein_simt_tile_side_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = tile_first_row + 1u + stage_row;
            else
                for (stage_row = lane_index; stage_row < sz_levenshtein_simt_tile_side_k; stage_row += 32u)
                    shared_left[warp_in_block][stage_row] = row_frontier[tile_first_row + 1u + stage_row];
            for (stage_row = lane_index; stage_row < sz_levenshtein_simt_tile_side_k; stage_row += 32u)
                shared_query[warp_in_block][stage_row] = tile_first_row + stage_row < shorter_length
                                                             ? shorter_text[tile_first_row + stage_row]
                                                             : (char)sz_levenshtein_simt_past_query_k;
            __syncwarp();

            // The tile's bottom-left cell is the diagonal corner of the tile below, and the march is about to
            // overwrite the frontier band it sits in.
            tile_bottom_left = shared_left[warp_in_block][sz_levenshtein_simt_tile_side_k - 1];
            // Exactly one tile of the whole matrix holds the corner cell, and only that tile pays the
            // per-cell test; keeping the guarded store out of every other tile's hot loop is what lets the
            // march stay in registers.
            march = tile_row == corner_tile_row && tile_column == corner_tile_column
                        ? sz_levenshtein_simt_march_checked_k
                        : sz_levenshtein_simt_march_fast_k;
            if (march == sz_levenshtein_simt_march_fast_k)
                sz_levenshtein_simt_march_tile_(sz_levenshtein_simt_march_fast_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            else
                sz_levenshtein_simt_march_tile_(sz_levenshtein_simt_march_checked_k, lane_index, tile_first_row,
                                                tile_first_column, shorter_length, longer_length,
                                                shared_query[warp_in_block], shared_left[warp_in_block], tile_corner,
                                                target_chars, carry_top, row_frontier, distance);
            tile_corner = tile_bottom_left;
            if (lane_index == sz_levenshtein_simt_lanes_k - 1)
                sz_levenshtein_simt_publish_(progress + tile_column, tile_row);
        }
    }
}

/**
 *  @brief One pair's Levenshtein distance through the tiled wavefront, on texts the
 *      device already reaches, on the caller's current device.
 *  @param[in] a First text, device-reachable.
 *  @param[in] a_length Its length in bytes.
 *  @param[in] b Second text, device-reachable.
 *  @param[in] b_length Its length in bytes.
 *  @param[in] scratch The frontier, device-reachable, at least
 *      @ref sz_levenshtein_distance_tiled_scratch_bytes bytes, and untouched by anything else until
 *      the caller joins @p stream.
 *  @param[out] distance Device-reachable slot the distance lands in once @p stream is joined.
 *  @param[in] stream The @c cudaStream_t or @c hipStream_t to schedule on, or
 *      @c STRINGZILLA_NULL for the default one.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k when either text is longer than the
 *      kernel indexes, or @c sz_device_memory_mismatch_k when a text, the scratch or the distance is
 *      not memory the device reaches.
 *  @note Enqueues and returns, allocating nothing and joining nothing.
 *  @sa sz_levenshtein_distances_simt_scoped_, which scores a whole batch through the
 *      bit-parallel rungs instead.
 */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_distance_tiled_simt_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                   sz_size_t b_length, void *scratch,
                                                                   sz_size_t *distance, void *stream) {
    if (a_length > STRINGZILLA_LEVENSHTEIN_SIMT_TILED_LENGTH_MAX ||
        b_length > STRINGZILLA_LEVENSHTEIN_SIMT_TILED_LENGTH_MAX)
        return sz_unexpected_dimensions_k;
    if (!sz_memory_reaches_device_(distance)) return sz_device_memory_mismatch_k;

    // The recurrence is symmetric, and putting the shorter text on the row axis keeps the frontier small and
    // makes the parallel axis the long one.
    sz_cptr_t const shorter_text = a_length <= b_length ? a : b;
    sz_cptr_t const longer_text = a_length <= b_length ? b : a;
    sz_u32_t const shorter_length = (sz_u32_t)(a_length <= b_length ? a_length : b_length);
    sz_u32_t const longer_length = (sz_u32_t)(a_length <= b_length ? b_length : a_length);
    if (shorter_length == 0) {
        // A copy out of a host local may join the stream, and a launch captures its arguments instead.
        sz_size_t *launch_distance = distance;
        sz_size_t launch_value = longer_length;
        void *arguments[2];
        dim3 one;
        one.x = 1, one.y = 1, one.z = 1;
        arguments[0] = &launch_distance, arguments[1] = &launch_value;
        return sz_device_launch_((void const *)sz_levenshtein_simt_store_kernel_, one, one, arguments, 0, stream);
    }
    if (!sz_memory_reaches_device_(shorter_text) || !sz_memory_reaches_device_(longer_text) ||
        !sz_memory_reaches_device_(scratch))
        return sz_device_memory_mismatch_k;

    sz_size_t const tile_grid_columns = sz_size_divide_round_up(longer_length, sz_levenshtein_simt_tile_side_k);
    sz_size_t const row_frontier_cells =
        sz_size_divide_round_up(shorter_length, sz_levenshtein_simt_tile_side_k) * sz_levenshtein_simt_tile_side_k + 1u;
    sz_u32_t *const row_frontier = (sz_u32_t *)scratch;
    sz_u32_t *const progress = row_frontier + row_frontier_cells;

    // Every launched block has to be resident, since a block spins on a tile-column another block owns.
    sz_size_t const resident_per_multiprocessor = sz_device_resident_blocks_(
        (void const *)sz_levenshtein_simt_tiled_kernel_, sz_levenshtein_simt_tiled_threads_per_block_k, 0);
    sz_size_t const resident_blocks = sz_device_multiprocessors_() *
                                      sz_max_of_two(resident_per_multiprocessor, (sz_size_t)1);
    if (!resident_blocks) return sz_missing_gpu_k;
    sz_status_t const cleared = sz_device_memset_(progress, 0, tile_grid_columns * sizeof(sz_u32_t), stream);
    if (cleared != sz_success_k) return cleared;

    // A warp waits on the tile-column to its left, so a block that never gets scheduled is a block its
    // neighbour spins on forever; the grid is capped at what the occupancy answer says stays resident.
    sz_size_t const wanted_blocks = sz_size_divide_round_up(tile_grid_columns,
                                                            sz_levenshtein_simt_tiled_warps_per_block_k);
    dim3 grid, block;
    grid.x = (unsigned)(wanted_blocks < resident_blocks ? wanted_blocks : resident_blocks), grid.y = 1, grid.z = 1;
    block.x = sz_levenshtein_simt_tiled_threads_per_block_k, block.y = 1, block.z = 1;

    sz_cptr_t launch_shorter_text = shorter_text, launch_longer_text = longer_text;
    sz_u32_t launch_shorter_length = shorter_length, launch_longer_length = longer_length;
    sz_u32_t *launch_row_frontier = row_frontier, *launch_progress = progress;
    sz_size_t *launch_distance = distance;
    void *arguments[7];
    arguments[0] = &launch_shorter_text, arguments[1] = &launch_shorter_length;
    arguments[2] = &launch_longer_text, arguments[3] = &launch_longer_length;
    arguments[4] = &launch_row_frontier, arguments[5] = &launch_progress, arguments[6] = &launch_distance;
    return sz_device_launch_((void const *)sz_levenshtein_simt_tiled_kernel_, grid, block, arguments, 0, stream);
}

#pragma endregion Tiled

#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cuda(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator, void *stream) {
    return sz_levenshtein_engine_init_simt_scoped_(engine, queries, symbol, ordinal, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, void *stream) {
    return sz_levenshtein_distances_simt_scoped_(engine, candidates, distances, distances_stride, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               void *stream) {
    return sz_levenshtein_distance_tiled_simt_(a, a_length, b, b_length, scratch, distance, stream);
}

#endif // STRINGZILLA_TARGET_CUDA

#if STRINGZILLA_TARGET_ROCM

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_rocm(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                            sz_memory_allocator_t *allocator, void *stream) {
    return sz_levenshtein_engine_init_simt_scoped_(engine, queries, symbol, ordinal, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_rocm(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, void *stream) {
    return sz_levenshtein_distances_simt_scoped_(engine, candidates, distances, distances_stride, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_rocm(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               void *stream) {
    return sz_levenshtein_distance_tiled_simt_(a, a_length, b, b_length, scratch, distance, stream);
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA || STRINGZILLA_TARGET_ROCM
#endif // STRINGZILLA_LEVENSHTEIN_SIMT_CUH_
