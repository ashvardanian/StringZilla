/**
 *  @file include/stringzilla/ngrams/simt.cuh
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief The device steps CUDA and ROCm share for N-grams: the N-gram residue and its roll, the
 *      overlap walk of a warp per candidate over one prepared query's table, and the sketch of a
 *      warp per text over ranks in shared memory.
 *
 *  The arithmetic is the serial tier's, reached from the device through
 *  @c --expt-relaxed-constexpr, so counts and sketches are exactly serial's rather than merely
 *  close. Integers, not doubles: a window's residue is its big-endian integer modulo the prime,
 *  eight bytes of which fit one @c u64, and the remainder by a constant lowers to a multiply-high
 *  and one Barrett correction, where an @c f64 reduction would run at the device's double-precision
 *  rate, a sixty-fourth of its single-precision one on RTX parts, and weak on B300 too.
 *
 *  @see https://ashvardanian.com/posts/stringwars-on-gpus/#52-bit-math-for-bioinformatics
 *
 *  A window's residue depends only on its own bytes, so the walks roll no chain and prime none: a
 *  lane takes whole aligned words and hashes the N-grams starting at their bytes straight from
 *  them. A kit walking one text in order rolls instead, through @ref sz_ngrams_roll_simt , over
 *  the same reduction.
 *
 *  Overlaps: a hit sets the bit of the query window's first byte in the warp's hit words, so a
 *  window either side repeats counts once, and the warp sums its lanes' first hits. The query axis
 *  rides @c blockIdx.y , one table per block, strided when a batch outruns a grid dimension. A walk
 *  scoring one query against scattered candidates prepares that query itself instead, a warp filing
 *  its table into a span the caller owns.
 *
 *  Sketches: the lanes split a text's N-grams and lower the ranks of their slots by @c atomicMin ,
 *  then fill every empty slot from the next filled one and write the bytes, so the sketch equals
 *  the serial one in every slot, byte for byte.
 *
 *  Only device code lives here; each vendor wraps the walks in its own kernels and sizes the grid
 *  and launches from its own host side, in <tt>cuda.cuh</tt> and <tt>rocm.cuh</tt>.
 *
 *  @sa c/stringzilla/ngrams.h
 *  @sa include/stringzilla/ngrams/cuda.cuh
 *  @sa include/stringzilla/ngrams/rocm.cuh
 */
#ifndef STRINGZILLA_NGRAMS_SIMT_CUH_
#define STRINGZILLA_NGRAMS_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/ngrams/serial.h"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Residues

/** @p value modulo @c sz_ngrams_modulus_k without a 64-bit division: the Barrett quotient from the
 *  high half of one product with the reciprocal is short by at most one, which one conditional
 *  subtraction settles, so the residue is exactly the serial tier's. */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_reduce_simt_(sz_u64_t value) {
    sz_u64_t const modulus = sz_ngrams_modulus_k, reciprocal = ~(sz_u64_t)0 / modulus;
    sz_u64_t const remainder = value - __umul64hi(value, reciprocal) * modulus;
    return (sz_u32_t)(remainder >= modulus ? remainder - modulus : remainder);
}

/**
 *  @brief The residue of the window one byte on from the window of @p residue : @p outgoing leaves,
 *      the rest shifts a byte up and @p incoming enters, under one reduction.
 *
 *  A window being primed rolls in its bytes with a zero @p outgoing , from a zero residue.
 *
 *  @param[in] power @ref sz_ngrams_power of the window's width.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_roll_simt(sz_u32_t residue, sz_u8_t incoming, sz_u8_t outgoing, sz_u32_t power) {
    // The prime shifted a byte up exceeds any leaving byte's share, so no step is negative.
    return sz_ngrams_reduce_simt_(((sz_u64_t)residue << 8) + incoming + ((sz_u64_t)sz_ngrams_modulus_k << 8) -
                                  (sz_u64_t)outgoing * power);
}

/** The big-endian integer of the four bytes at @p offset past the aligned @p words, read through
 *  plain loads of the one or two aligned words holding them. */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_big_endian_simt_(sz_u32_t const *words, sz_size_t offset) {
    unsigned const shift = (unsigned)(offset & 3) * 8;
    sz_u32_t const low = words[offset / 4], high = shift ? words[offset / 4 + 1] : 0;
    return __byte_perm(__funnelshift_r(low, high, shift), 0, 0x0123);
}

/** The big-endian integer of the eight bytes from byte @p byte of @p first, the first of three
 *  consecutive aligned words. */
STRINGZILLA_DEVICE sz_u64_t sz_ngrams_leading_simt_(sz_u32_t first, sz_u32_t second, sz_u32_t third, unsigned byte) {
    return (sz_u64_t)__byte_perm(__funnelshift_r(first, second, 8 * byte), 0, 0x0123) << 32 |
           __byte_perm(__funnelshift_r(second, third, 8 * byte), 0, 0x0123);
}

/**
 *  @brief The residue of the window of @p width bytes at @p offset past the aligned @p words, given
 *      @p leading , the big-endian integer of the eight bytes from @p offset .
 *
 *  A window of up to eight bytes is one integer, reduced once; a wider one folds four more bytes
 *  per reduction into its leading bytes' residue, so no window waits on another.
 *
 *  @param[in] words The text's aligned words, read through plain loads only past eight bytes.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_window_residue_simt_(sz_u64_t leading, sz_u32_t const *words, sz_size_t offset,
                                                           sz_size_t width) {
    // Past eight bytes, the leading ones are as many as leave whole words behind them.
    sz_size_t const lead = width <= 8 ? width : width - (width - 5) / 4 * 4;
    sz_u32_t residue = sz_ngrams_reduce_simt_(leading >> (64 - 8 * lead));
    for (sz_size_t tail = offset + lead; tail != offset + width; tail += 4)
        residue = sz_ngrams_reduce_simt_((sz_u64_t)residue << 32 | sz_ngrams_big_endian_simt_(words, tail));
    return residue;
}

#pragma endregion Residues

#pragma region Overlaps Walk

/** The entry of @p slots holding @p residue at width @p width_index , or zero, as
 *  @ref sz_ngrams_query_probe finds it but two slots a step, so the longest cluster a warp meets
 *  costs half the steps; through plain loads, so the same kernel may have written the slots. */
STRINGZILLA_DEVICE sz_u64_t sz_ngrams_slot_probe_simt_(sz_u64_t const *slots, sz_u32_t const *filter, sz_u32_t mask,
                                                       sz_u32_t residue, sz_u32_t width_index) {
    sz_size_t const bit = sz_ngrams_filter_bit_(residue, mask);
    if (!((filter[bit / 32] >> (bit % 32)) & 1)) return 0;
    // An occupied slot's low word is never zero, so it alone tells an empty slot.
    sz_u32_t const wanted = (width_index << sz_ngrams_slot_width_shift_k) | sz_ngrams_slot_occupied_k;
    sz_u32_t const key_mask = (sz_u32_t)sz_ngrams_slot_key_mask_k;
    for (sz_u32_t slot = (sz_u32_t)(bit >> 3);; slot = (slot + 2) & mask) {
        sz_u64_t const first = slots[slot], second = slots[(slot + 1) & mask];
        if (!(sz_u32_t)first || ((sz_u32_t)(first >> 32) == residue && ((sz_u32_t)first & key_mask) == wanted))
            return first;
        if (!(sz_u32_t)second || ((sz_u32_t)(second >> 32) == residue && ((sz_u32_t)second & key_mask) == wanted))
            return second;
    }
}

/**
 *  @brief Probes the query's table for the window of @p width bytes at @p offset past the aligned
 *      @p words , and sets the matching query window's bit past @p bits_base in @p hit_words .
 *
 *  @param[in] leading The big-endian integer of the eight bytes from @p offset .
 *  @param[in] skew The text's first byte among the words, and @p length its bytes.
 *  @return One where this lane set the bit first, else zero: a miss, a repeat, or a window
 *      past the text's end.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_window_hit_simt_(sz_u64_t const *slots, sz_u32_t const *filter, sz_u32_t mask,
                                                       sz_u64_t leading, sz_u32_t const *words, sz_size_t offset,
                                                       sz_size_t skew, sz_size_t length, sz_size_t width,
                                                       sz_u32_t width_index, sz_u32_t bits_base, sz_u32_t *hit_words) {
    if (offset < skew || offset - skew + width > length) return 0;
    sz_u32_t const residue = sz_ngrams_window_residue_simt_(leading, words, offset, width);
    sz_u64_t const entry = sz_ngrams_slot_probe_simt_(slots, filter, mask, residue, width_index);
    if (!entry) return 0;
    sz_u32_t const bit = bits_base + (sz_u32_t)sz_ngrams_slot_first(entry);
    sz_u32_t const bit_mask = 1u << (bit & 31);
    return (atomicOr(hit_words + (bit >> 5), bit_mask) & bit_mask) == 0;
}

/**
 *  @brief Sets the bits of the distinct N-grams of a prepared query that one candidate holds, every
 *      lane of an aligned group of @p lanes taking part, and returns this lane's share of them.
 *
 *  Each lane of the group takes every lanes-th aligned word of the candidate and probes the N-grams
 *  starting at its four bytes, so the group's loads are coalesced, and a hit sets the bit of the
 *  query window's first byte in @p hit_words , so a window either side repeats counts once. The
 *  caller sums the shares over the group, whose lanes have then all returned, reads the bits if it
 *  wants per-width counts, and clears them with @ref sz_ngrams_candidate_clear_simt .
 *
 *  @param[in] query Prepared by @ref sz_ngrams_query_prepare_simt or laid out by the host, its
 *      slots read through plain loads, so the same kernel may have written it.
 *  @param[in,out] hit_words The group's own @c (query->ngrams+31)/32 words in shared memory, zero
 *      on entry; groups scoring at once need words of their own.
 *  @param[in] lane The calling lane within its group, below @p lanes .
 *  @param[in] lanes The group's lanes, a power of two up to a warp; past a lane per aligned word of
 *      the candidate, more lanes only idle.
 *  @return This lane's share of the count over every width; the group's sum is the count.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_candidate_matches_simt(sz_ngrams_query_t const *query, sz_cptr_t text,
                                                             sz_size_t length, sz_u32_t *hit_words, unsigned lane,
                                                             unsigned lanes) {
    // The table's fields are read once, as a hit's atomic could otherwise alias them.
    sz_u64_t const *const slots = query->slots;
    sz_u32_t const *const filter = query->filter;
    sz_u32_t const mask = (sz_u32_t)(query->slots_count - 1);
    sz_size_t const query_length = query->length;
    sz_size_t const widths_count = query->widths_count;
    sz_size_t const skew = (sz_size_t)text & 3;
    sz_u32_t const *const words = (sz_u32_t const *)(text - skew);
    sz_size_t const words_count = sz_size_divide_round_up(skew + length, 4);
    sz_u32_t held = 0;
    for (sz_size_t word = lane; word < words_count; word += lanes) {
        sz_u32_t const first = words[word];
        sz_u32_t const second = word + 1 < words_count ? words[word + 1] : 0;
        sz_u32_t const third = word + 2 < words_count ? words[word + 2] : 0;
        sz_u32_t base = 0;
        for (sz_size_t index = 0; index != widths_count; ++index) {
            // A width past the query holds no bits, so it probes nothing.
            sz_size_t const width = query->widths[index];
            if (!width || width > query_length) continue;
            sz_u32_t const width_base = base;
            base += (sz_u32_t)(query_length - width + 1);
#pragma unroll
            for (unsigned byte = 0; byte != 4; ++byte)
                held += sz_ngrams_window_hit_simt_(
                    slots, filter, mask, sz_ngrams_leading_simt_(first, second, third, byte), words, 4 * word + byte,
                    skew, length, width, (sz_u32_t)index, width_base, hit_words);
        }
    }
    return held;
}

/**
 *  @brief Zeroes the hit words @ref sz_ngrams_candidate_matches_simt set, every lane of the group
 *      taking part, once the group's shares are summed.
 *  @return Whether this lane cleared a set bit; the caller joins the group on it, so no lane starts
 *      another candidate before every word is zero.
 */
STRINGZILLA_DEVICE int sz_ngrams_candidate_clear_simt(sz_ngrams_query_t const *query, sz_u32_t *hit_words,
                                                      unsigned lane, unsigned lanes) {
    sz_u32_t cleared = 0;
    for (sz_size_t word = lane; word < sz_size_divide_round_up(query->ngrams, 32); word += lanes)
        cleared |= atomicExch(hit_words + word, 0u);
    return cleared != 0;
}

/** One pair's walk for the overlaps kernels: the shares summed over @p group , per-width counts
 *  into @p counts by lane zero, and the words cleared before the group returns. */
STRINGZILLA_DEVICE sz_size_t sz_ngrams_candidate_walk_simt_(sz_ngrams_query_t const *query, sz_cptr_t text,
                                                            sz_size_t length, sz_u32_t *hit_words,
                                                            sz_simt_group_t const *group, sz_u32_t *counts) {
    unsigned const lane = group->lane, lanes = group->lanes;
    sz_size_t const query_length = query->length;
    sz_size_t const held = (sz_size_t)group->sum(
        sz_ngrams_candidate_matches_simt(query, text, length, hit_words, lane, lanes));
    // A width's count is its bits among the hit words, which start where the widths before it end.
    sz_u32_t base = 0;
    for (sz_size_t index = 0; index != query->widths_count; ++index) {
        sz_size_t const width = query->widths[index];
        sz_u32_t const bits = width && width <= query_length ? (sz_u32_t)(query_length - width + 1) : 0;
        sz_u32_t found = 0;
        for (sz_u32_t word = base / 32 + lane; word * 32 < base + bits; word += lanes) {
            sz_u32_t const low = sz_max_of_two(base, word * 32) - word * 32;
            sz_u32_t const high = sz_min_of_two(base + bits, word * 32 + 32) - word * 32;
            sz_u32_t const mask = (high - low == 32 ? 0xFFFFFFFFu : (1u << (high - low)) - 1u) << low;
            found += sz_u32_popcount_simt_(atomicOr(hit_words + word, 0u) & mask);
        }
        sz_u32_t const total = (sz_u32_t)group->sum(found);
        if (lane == 0) counts[index] = total;
        base += bits;
    }
    return group->any(sz_ngrams_candidate_clear_simt(query, hit_words, lane, lanes)) ? held : 0;
}

/** Scores one candidate against a prepared query into @p row , zeros past @p bound misses. */
STRINGZILLA_DEVICE void sz_ngrams_pair_overlaps_simt_(sz_ngrams_query_t const *query, sz_size_t bound, sz_cptr_t text,
                                                      sz_size_t length, sz_u32_t *hit_words,
                                                      sz_simt_group_t const *group, sz_u32_t *row) {
    sz_size_t reachable = 0, index;
    for (index = 0; index != query->widths_count; ++index) {
        sz_size_t const width = query->widths[index];
        reachable += width && width <= length && width <= query->length ? length - width + 1 : 0;
    }
    sz_size_t held = 0;
    sz_bool_t const reaches = query->distinct - sz_min_of_two(query->distinct, reachable) <= bound ? sz_true_k
                                                                                                   : sz_false_k;
    if (reaches) held = sz_ngrams_candidate_walk_simt_(query, text, length, hit_words, group, row);
    // Lane zero wrote the counts, so lane zero alone overwrites a pair past its budget.
    if (group->lane == 0 && (!reaches || query->distinct - held > bound))
        for (index = 0; index != query->widths_count; ++index) row[index] = 0;
}

/**
 *  @brief One block per tile of candidates and per prepared query, each warp scoring a candidate of
 *      the tile at a time, the query's table staged when the launch asked for it.
 *
 *  Dynamic shared memory holds the staged table, then each warp's @p hit_words_per_warp words,
 *  which stay zero between candidates. A pair is one call of @ref sz_ngrams_candidate_walk_simt_ ,
 *  dropped to zeros once its misses pass @p misses_bound of the query's distinct N-grams, or before
 *  a byte is read when its lengths alone pass it. The barriers around the staging are collective,
 *  so no thread leaves the query loop before the others.
 *
 *  @param[in] staged_slots_count Slots a block stages, the widest query's, or zero to read every
 *      table in place.
 *  @param[in] hit_words_per_warp One warp's hit words, at least the widest query's bits.
 *  @param[in] misses_bound The radius as @ref sz_ngrams_misses_bound_ rounds it.
 *  @param[in] group The calling warp's lane steps, from the vendor's kernel.
 */
STRINGZILLA_DEVICE void sz_ngrams_overlaps_simt_(sz_ngrams_engine_t engine, sz_strings_view_t candidates,
                                                 sz_u32_t *matches, sz_size_t matches_query_stride,
                                                 sz_size_t matches_candidate_stride, sz_size_t staged_slots_count,
                                                 sz_size_t hit_words_per_warp, sz_u32_t misses_bound,
                                                 sz_simt_group_t const *group) {
    extern __shared__ sz_u64_t sz_ngrams_shared_simt_[];
    __shared__ sz_tile_queue_t queue;
    sz_u64_t *const staged_slots = sz_ngrams_shared_simt_;
    unsigned const lane = threadIdx.x & 31u, warps = blockDim.x / 32;
    sz_u32_t *const hit_words = (sz_u32_t *)(staged_slots + staged_slots_count) +
                                (sz_size_t)(threadIdx.x / 32) * hit_words_per_warp;
    for (sz_size_t word = lane; word < hit_words_per_warp; word += 32) hit_words[word] = 0;
    sz_size_t staged_query = STRINGZILLA_SIZE_MAX;
    sz_u32_t tile_x = blockIdx.x, tile_y = blockIdx.y;
    sz_tile_queue_open_simt_(&queue, 1, gridDim.x, 0);
    do {
        sz_size_t const tile_first = (sz_size_t)tile_x * blockDim.x;
        sz_size_t const tile_count = sz_min_of_two((sz_size_t)blockDim.x, candidates.count - tile_first);
        for (sz_size_t query_index = tile_y; query_index < engine.count; query_index += gridDim.y) {
            sz_ngrams_query_t query = sz_ngrams_engine_query(&engine, query_index);
            sz_bool_t const restage = staged_slots_count && query_index != staged_query ? sz_true_k : sz_false_k;
            for (sz_size_t slot = threadIdx.x; restage && slot < query.slots_count; slot += blockDim.x)
                staged_slots[slot] = query.slots[slot];
            if (restage) staged_query = query_index;
            __syncthreads();
            if (staged_slots_count) query.slots = staged_slots;
            sz_size_t const bound = (sz_size_t)(((sz_u64_t)misses_bound * query.distinct) >> 24);
            for (sz_size_t candidate = threadIdx.x / 32; group->any(candidate < tile_count); candidate += warps) {
                sz_cptr_t const text = sz_strings_start_simt_(candidates, tile_first + candidate);
                sz_size_t const length = sz_strings_length_simt_(candidates, tile_first + candidate);
                sz_u32_t *const row = matches + query_index * matches_query_stride +
                                      (tile_first + candidate) * matches_candidate_stride;
                sz_ngrams_pair_overlaps_simt_(&query, bound, text, length, hit_words, group, row);
            }
            // The next query restages the table these walks still read.
            __syncthreads();
        }
    } while (sz_tile_queue_next_simt_(&queue, &tile_x, &tile_y));
}

#pragma endregion Overlaps Walk

#pragma region Prepared Queries

/**
 *  @brief Files the window of @p width bytes at @p offset past the aligned @p words into @p slots
 *      by compare-and-swap, a repeat keeping the smaller first byte and marking its slot repeated.
 *
 *  @param[in] leading The big-endian integer of the eight bytes from @p offset .
 *  @param[in] skew The text's first byte among the words, and @p length its bytes.
 *  @return One for a new window; zero for a repeat or a window past the text.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_window_file_simt_(sz_u64_t *slots, sz_u32_t *filter, sz_size_t mask,
                                                        sz_u64_t leading, sz_u32_t const *words, sz_size_t offset,
                                                        sz_size_t skew, sz_size_t length, sz_size_t width,
                                                        sz_size_t width_index) {
    if (offset < skew || offset - skew + width > length) return 0;
    sz_u32_t const residue = sz_ngrams_window_residue_simt_(leading, words, offset, width);
    sz_u64_t const wanted = ((sz_u64_t)residue << 32) | ((sz_u64_t)width_index << sz_ngrams_slot_width_shift_k) |
                            sz_ngrams_slot_occupied_k;
    sz_u64_t const entry = wanted | ((sz_u64_t)(offset - skew) << sz_ngrams_slot_first_shift_k);
    sz_size_t const bit = sz_ngrams_filter_bit_(residue, mask);
    atomicOr(filter + bit / 32, 1u << (bit % 32));
    for (sz_size_t slot = bit >> 3;; slot = (slot + 1) & mask) {
        sz_u64_t const seen = atomicCAS((unsigned long long *)(slots + slot), 0ull, (unsigned long long)entry);
        if (!seen) return 1;
        if (((seen ^ wanted) & sz_ngrams_slot_key_mask_k) != 0) continue;
        atomicMin((unsigned long long *)(slots + slot), (unsigned long long)(entry | sz_ngrams_slot_repeat_k));
        atomicOr((unsigned long long *)(slots + slot), (unsigned long long)sz_ngrams_slot_repeat_k);
        return 0;
    }
}

/**
 *  @brief Prepares one query into a warp's @p span , every lane of the warp taking part, so a
 *      device walk prepares each query where it scores it.
 *
 *  Each lane hashes the N-grams starting in its aligned words and files each into the table: a new
 *  window claims an empty slot by compare-and-swap, and a repeat keeps the smaller first byte by an
 *  atomic minimum and marks the slot repeated, so the table is the host's whatever order the lanes
 *  run in. The warp is synchronized on return.
 *
 *  On a 64-lane AMD wavefront only the lanes below 32 call it, as a second half filing the same
 *  words would find every slot taken and mark it repeated. The upper half takes @p query from
 *  lane 0 by shuffle, which is exact, as every field is the same in every calling lane.
 *
 *  The count of distinct N-grams is the sum of every lane's return, which the caller stores in
 *  @p query's @c distinct with its own lane sum.
 *
 *  @param[in] text The query, read through plain loads, so it may be written by the same kernel.
 *  @param[in] length The query's bytes, at most @ref sz_ngrams_query_bytes_max_k .
 *  @param[in] widths The @b [widths_count] window widths, kept where every walk reads them, at most
 *      @ref sz_ngrams_widths_max_k of them.
 *  @param[out] span Eight-byte aligned, @ref sz_ngrams_query_span_bytes long: the slots, their
 *      miss filter, then a power per width.
 *  @param[out] query Points into @p span and at @p widths , the same in every lane, its
 *      @c distinct zero.
 *  @param[in] lane The calling lane, below 32.
 *  @return This lane's share of the distinct N-grams.
 */
STRINGZILLA_DEVICE sz_u32_t sz_ngrams_query_prepare_simt(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                                         sz_size_t widths_count, sz_ptr_t span,
                                                         sz_ngrams_query_t *query, unsigned lane) {
    sz_size_t const ngrams = sz_ngrams_count(length, widths, widths_count);
    sz_size_t const slots_count = sz_ngrams_slots_count_(ngrams), mask = slots_count - 1;
    sz_u64_t *const slots = (sz_u64_t *)span;
    sz_u32_t *const filter = (sz_u32_t *)(slots + slots_count);
    sz_u32_t *const powers = filter + slots_count / 4;
    sz_size_t index;
    // The span may still serve the previous query's readers until every lane gets here.
    __syncwarp();
    for (index = lane; index < slots_count; index += 32) slots[index] = 0;
    for (index = lane; index < slots_count / 4; index += 32) filter[index] = 0;
    for (index = lane; index < widths_count; index += 32) powers[index] = sz_ngrams_power(widths[index]);
    __syncwarp();

    // Each lane files the N-grams starting in every 32nd aligned word of the query.
    sz_size_t const skew = (sz_size_t)text & 3;
    sz_u32_t const *const words = (sz_u32_t const *)(text - skew);
    sz_size_t const words_count = sz_size_divide_round_up(skew + length, 4);
    sz_u32_t distinct = 0;
    for (sz_size_t word = lane; word < words_count; word += 32) {
        sz_u32_t const first = words[word];
        sz_u32_t const second = word + 1 < words_count ? words[word + 1] : 0;
        sz_u32_t const third = word + 2 < words_count ? words[word + 2] : 0;
        for (index = 0; index != widths_count; ++index) {
            sz_size_t const width = widths[index];
            if (!width) continue;
#pragma unroll
            for (unsigned byte = 0; byte != 4; ++byte)
                distinct += sz_ngrams_window_file_simt_(slots, filter, mask,
                                                        sz_ngrams_leading_simt_(first, second, third, byte), words,
                                                        4 * word + byte, skew, length, width, index);
        }
    }
    __syncwarp();
    query->slots = slots;
    query->slots_count = slots_count;
    query->filter = filter;
    query->widths = widths;
    query->powers = powers;
    query->widths_count = widths_count;
    query->ngrams = ngrams;
    query->distinct = 0;
    query->length = length;
    __syncwarp();
    return distinct;
}

#pragma endregion Prepared Queries

#pragma region Sketches

/**
 *  @brief Lowers the rank of the slot of the window of @p width bytes at @p offset past the aligned
 *      @p words , as @c sz_ngrams_sketch_block_ ranks it on the host.
 *
 *  @param[in] leading The big-endian integer of the eight bytes from @p offset .
 *  @param[in] skew The text's first byte among the words, and @p length its bytes.
 */
STRINGZILLA_DEVICE void sz_ngrams_window_rank_simt_(sz_u32_t *ranks, sz_size_t slots, sz_u64_t leading,
                                                    sz_u32_t const *words, sz_size_t offset, sz_size_t skew,
                                                    sz_size_t length, sz_size_t width) {
    if (offset < skew || offset - skew + width > length) return;
    sz_u32_t const residue = sz_ngrams_window_residue_simt_(leading, words, offset, width);
    sz_u32_t const mixed = sz_ngrams_mix_(residue + (sz_u32_t)width);
    sz_size_t const slot = (sz_size_t)(((sz_u64_t)mixed * slots) >> 32);
    atomicMin(ranks + slot, sz_ngrams_mix_(mixed ^ 0x9E3779B9u));
}

/**
 *  @brief The @p slots bytes of the sketch of one text, as @ref sz_ngrams_sketch_serial writes it;
 *      every lane of one warp calls it for the same text.
 *
 *  The lanes split the text's aligned words and lower each window's slot by @c atomicMin on
 *  @p ranks , then each lane fills a run of slots: a slot no window reached borrows the rank of the
 *  nearest reached slot after it, wrapping past the last, mixed with its distance. The ranks are
 *  read and never overwritten past the minimums, so no lane waits on another's fill.
 *
 *  @param[in] text The text's first byte, read through plain loads.
 *  @param[in] widths Bytes a window spans at each of @p widths_count widths, each at least one.
 *  @param[in,out] ranks The warp's own @p slots words, shared memory where it fits; overwritten.
 *  @param[in] slots Bytes of the sketch, at least one.
 *  @param[out] sketch The @p slots bytes, which the lanes write between them.
 *  @param[in] lane The caller's lane in its warp, taken modulo 32, so both halves of a 64-lane
 *      wavefront calling for one text over one @p ranks write the same bytes.
 */
STRINGZILLA_DEVICE void sz_ngrams_sketch_simt(sz_cptr_t text, sz_size_t length, sz_size_t const *widths,
                                              sz_size_t widths_count, sz_u32_t *ranks, sz_size_t slots, sz_u8_t *sketch,
                                              unsigned lane) {
    lane &= 31u;
    sz_size_t slot;
    // The ranks may still serve the previous text's fills until every lane gets here.
    __syncwarp();
    for (slot = lane; slot < slots; slot += 32) ranks[slot] = 0xFFFFFFFFu;
    __syncwarp();

    sz_size_t const skew = (sz_size_t)text & 3;
    sz_u32_t const *const words = (sz_u32_t const *)(text - skew);
    sz_size_t const words_count = sz_size_divide_round_up(skew + length, 4);
    for (sz_size_t word = lane; word < words_count; word += 32) {
        sz_u32_t const first = words[word];
        sz_u32_t const second = word + 1 < words_count ? words[word + 1] : 0;
        sz_u32_t const third = word + 2 < words_count ? words[word + 2] : 0;
        for (sz_size_t index = 0; index != widths_count; ++index) {
            sz_size_t const width = widths[index];
            if (!width || width > length) continue;
#pragma unroll
            for (unsigned byte = 0; byte != 4; ++byte)
                sz_ngrams_window_rank_simt_(ranks, slots, sz_ngrams_leading_simt_(first, second, third, byte), words,
                                            4 * word + byte, skew, length, width);
        }
    }
    __syncwarp();

    // Each lane fills its run of slots downwards from the nearest reached slot past it, found once.
    sz_size_t const run = sz_size_divide_round_up(slots, (sz_size_t)32);
    sz_size_t const begin = sz_min_of_two(lane * run, slots), end = sz_min_of_two(begin + run, slots);
    sz_size_t next = slots;
    for (sz_size_t step = 0, at = end; begin != end && step != slots; ++step, ++at) {
        if (at == slots) at = 0;
        if (ranks[at] != 0xFFFFFFFFu) {
            next = at;
            break;
        }
    }
    for (slot = end; slot != begin; --slot) {
        sz_u32_t rank = ranks[slot - 1];
        if (rank != 0xFFFFFFFFu) next = slot - 1;
        else if (next != slots)
            rank = sz_ngrams_mix_(ranks[next] +
                                  (sz_u32_t)(next >= slot ? next - (slot - 1) : next + slots - (slot - 1)));
        sketch[slot - 1] = (sz_u8_t)sz_ngrams_mix_(rank);
    }
}

/**
 *  @brief One warp per text writes the text's sketch, each warp with its own @p slots ranks in
 *      dynamic shared memory.
 *
 *  A warp takes every text a grid's worth of warps apart, so a batch past one grid dimension is
 *  still covered by the blocks it has; a warp's whole loop is uniform, so its barriers meet.
 */
STRINGZILLA_DEVICE void sz_ngrams_sketches_simt_(sz_size_t const *widths, sz_size_t widths_count,
                                                 sz_strings_view_t texts, sz_size_t slots, sz_u8_t *sketches,
                                                 sz_size_t sketches_stride) {
    extern __shared__ sz_u32_t sz_ngrams_ranks_simt_[];
    sz_size_t const warps = blockDim.x / 32, stride = (sz_size_t)gridDim.x * warps;
    sz_u32_t *const ranks = sz_ngrams_ranks_simt_ + (sz_size_t)(threadIdx.x / 32) * slots;
    for (sz_size_t index = (sz_size_t)blockIdx.x * warps + threadIdx.x / 32; index < texts.count; index += stride)
        sz_ngrams_sketch_simt(sz_strings_start_simt_(texts, index), sz_strings_length_simt_(texts, index), widths,
                              widths_count, ranks, slots, sketches + index * sketches_stride, threadIdx.x);
}

#pragma endregion Sketches

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_NGRAMS_SIMT_CUH_
