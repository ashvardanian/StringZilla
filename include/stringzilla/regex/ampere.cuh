/**
 *  @file include/stringzilla/regex/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of regular expressions: the start-marking walk staging each tile's text with
 *      @c cp.async , and the @c _ampere exports, over the CUDA host side.
 *
 *  The CUDA tier's marking walk stages a tile's text through registers, a granule per loop step,
 *  after thread zero alone has found the tile's first haystack behind two barriers. Here every warp
 *  gallops to that haystack itself, before the barrier that frees the last tile's text, and then
 *  issues all of its copies at once with @c cp.async , which waits on no register; a tile takes two
 *  barriers rather than three. The text keeps one buffer per block: a second, to stage the next
 *  tile while this one is walked, would take the shared memory of resident blocks, which this
 *  latency-bound walk needs more than it needs the staging hidden.
 *
 *  @sa include/stringzilla/regex/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_REGEX_AMPERE_CUH_
#define STRINGZILLA_REGEX_AMPERE_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#include "stringzilla/regex/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_commit_async_ampere_`, `sz_copy_b32_async_ampere_` & kin

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Copies the bytes from @p begin to @p end of a tape spanning @p tape to @p tape_end into
 *      @p staged, as @ref sz_regex_stage_text_simt_ lays them out, every thread of the block
 *      issuing its 4-byte words with @c cp.async and committing them as one group.
 *  @return The address staged position zero stands for, @p begin rounded down to a granule.
 *
 *  A word is the widest copy each padded row of 64 bytes keeps aligned. The word holding the
 *  tape's last byte reads only up to it, zero-filling the rest; a word holding its first byte is
 *  copied byte by byte, as no copy starts past its source.
 */
STRINGZILLA_DEVICE sz_u8_t const *sz_regex_stage_text_async_ampere_(sz_u8_t *staged, sz_u8_t const *begin,
                                                                    sz_u8_t const *end, sz_u8_t const *tape,
                                                                    sz_u8_t const *tape_end) {
    sz_u8_t const *const origin = (sz_u8_t const *)((sz_size_t)begin & ~(sz_size_t)15);
    sz_size_t const bytes = (sz_size_t)(end - origin);
    for (sz_size_t word = (sz_size_t)threadIdx.x * 4; word < bytes; word += (sz_size_t)blockDim.x * 4) {
        sz_u8_t const *const source = origin + word;
        sz_u8_t *const target = staged + word + word / 64 * 4;
        if (source >= tape && source < tape_end)
            sz_copy_b32_async_ampere_(sz_shared_address_ampere_(target), source,
                                      (sz_u32_t)sz_min_of_two((sz_size_t)(tape_end - source), (sz_size_t)4));
        else if (source < tape && source + 4 > tape)
            for (sz_size_t byte = (sz_size_t)(tape - source); byte != 4 && source + byte < tape_end; ++byte)
                target[byte] = source[byte];
    }
    sz_commit_async_ampere_();
    return origin;
}

/** @ref sz_regex_mark_simt_ with each warp finding its tile's first haystack and the tile's
 *  text staged by @ref sz_regex_stage_text_async_ampere_ . */
STRINGZILLA_DEVICE void sz_regex_mark_ampere_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                              sz_regex_round_simt_t round) {
    extern __shared__ sz_u8_t sz_regex_staged_simt_[];
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_u8_t *const staged = sz_regex_staged_simt_ + sz_regex_staged_bytes_simt_k +
                            round.staged_cells * sizeof(sz_u32_t);
    sz_u8_t const *const tape = (sz_u8_t const *)sz_strings_start_simt_(haystacks, 0);
    sz_u8_t const *const tape_end = (sz_u8_t const *)sz_strings_start_simt_(haystacks, haystacks.count);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, tile_slots = width * blockDim.x;
    sz_u32_t const accepts = engine.reverse_accepts_begin;
    sz_size_t tile, tile_haystack = 0;
    // The scan reads one entry past the chunks, which ends up holding the candidates' total.
    if (blockIdx.x == 0 && threadIdx.x == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (tile = blockIdx.x; tile * blockDim.x < round.chunks_count; tile += gridDim.x) {
        sz_size_t const chunk = tile * blockDim.x + threadIdx.x, tile_low = round.segment_begin + tile * tile_slots;
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t index, word, first_byte, found = 0;
        sz_u64_t *const words = round.starts + (low - round.segment_begin) / 64;
        sz_u8_t const *origin;
        sz_u32_t marked = 0;
        // Tiles move forward only, so the last tile's first haystack is at or before this one's.
        if ((threadIdx.x & 31u) == 0) found = sz_regex_haystack_at_simt_(haystacks, tile_low, tile_haystack);
        tile_haystack = (sz_size_t)sz_shuffle_index_cuda_((sz_u32_t)found, 0) |
                        ((sz_size_t)sz_shuffle_index_cuda_((sz_u32_t)(found >> 32), 0) << 32);
        // The last tile's walks are done with its text before it is replaced.
        __syncthreads();
        // A tile's bytes run from the byte before its first slot, as each slot past the first of a
        // haystack stands for the byte before it, and span at most its slots.
        first_byte = tile_low > tile_haystack ? tile_low - tile_haystack - 1 : 0;
        origin = sz_regex_stage_text_async_ampere_(
            staged, tape + first_byte, tape + sz_min_of_two(first_byte + tile_slots + 1, (sz_size_t)(tape_end - tape)),
            tape, tape_end);
        sz_wait_async_ampere_(0);
        __syncthreads();
        if (chunk >= round.chunks_count) continue;
        index = sz_regex_haystack_at_simt_(haystacks, low, tile_haystack);
        round.chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) words[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_size_t begin_byte, end_byte, at;
            sz_u32_t state, bit;
            if (haystack.first >= high) break;
            begin_byte = low > haystack.first ? low - haystack.first - 1 : 0;
            end_byte = sz_min_of_two(haystack.length, high - haystack.first - 1);
            state = end_byte == haystack.length ? engine.reverse_starts[0] : round.entries[chunk];
            for (at = end_byte; at != begin_byte;) {
                sz_u32_t const count = (sz_u32_t)sz_min_of_two(at - begin_byte, (sz_size_t)sz_regex_run_bytes_simt_k);
                sz_u32_t const mask = sz_regex_run_back_staged_simt_(&table, &state, staged,
                                                                     (sz_size_t)(haystack.text + at - origin),
                                                                     haystack.first + at, count, accepts);
                at -= count;
                if (!mask) continue;
                // A byte's own slot marks the match starting right after it.
                bit = (sz_u32_t)(haystack.first + 1 + at - low);
                words[bit / 64] |= (sz_u64_t)mask << (bit % 64);
                if (bit % 64 + count > 64) words[bit / 64 + 1] |= (sz_u64_t)mask >> (64 - bit % 64);
                marked += (sz_u32_t)sz_u32_popcount_simt_(mask);
            }
            if (haystack.first < low || sz_regex_step_edge_simt_(&table, state, haystack.first) < accepts) continue;
            bit = (sz_u32_t)(haystack.first - low);
            words[bit / 64] |= (sz_u64_t)1 << (bit % 64), ++marked;
        }
        round.chunk_offsets[chunk] = marked;
    }
}

#if STRINGZILLA_TARGET_AMPERE

/** @ref sz_regex_scan_tile_simt_ over CUDA's warps in 32 bits, for the starts per chunk,
 *  of which a step of a tile holds at most 64 per value. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_scan_bounded_ampere_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                         sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    sz_regex_scan_tile_simt_(values, count, elements_per_tile, chain, states, stale, sz_block_scan_bounded_cuda_,
                             sz_block_sum_cuda_, sz_chain_tile_simt_, sz_chain_look_back_cuda_);
}

/** @ref sz_regex_scan_tile_simt_ over CUDA's warps. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_scan_ampere_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile, sz_size_t *chain,
                                 sz_u64_t *states, sz_u64_t *stale) {
    sz_regex_scan_tile_simt_(values, count, elements_per_tile, chain, states, stale, sz_block_scan_cuda_,
                             sz_block_sum_cuda_, sz_chain_tile_simt_, sz_chain_look_back_cuda_);
}

/** The warp walk of the links that outlived their first thread, with CUDA's cross-lane steps. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_continue_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                     sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_false_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** The warp walk of the folded links that outlived their first thread, with CUDA's cross-lane
 *  steps. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_continue_folded_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                            sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_true_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** @ref sz_regex_bound_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_dfa_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_bound_pieces_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_pieces_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_dfa_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_affine_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_affine_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_dfa_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                       sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_affine_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_double_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_affine_double_simt_(round);
}

/** @ref sz_regex_looks_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_looks_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_dfa_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_fill_singles_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_singles_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_fill_singles_simt_(engine, haystacks, round);
}

/** @ref sz_regex_prime_chunks_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_prime_chunks_simt_(engine, haystacks, round);
}

/** @ref sz_regex_prime_pieces_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_pieces_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_prime_pieces_simt_(engine, haystacks, round);
}

/** @ref sz_regex_compose_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_compose_simt_(engine, haystacks, round);
}

/** @ref sz_regex_compose_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_double_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_compose_double_simt_(round);
}

/** @ref sz_regex_mark_prefixes_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_prefixes_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                          sz_regex_round_simt_t round) {
    sz_regex_mark_prefixes_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_folded_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_folded_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                        sz_regex_round_simt_t round) {
    sz_regex_mark_folded_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_positions_simt_ over bytes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                           sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_mark_positions_simt_ over folded runes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_folded_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                  sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_list_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_list_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_list_simt_(round);
}

/** @ref sz_regex_ends_pike_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_pike_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_ends_pike_simt_(engine, haystacks, round);
}

/** @ref sz_regex_ends_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_ends_simt_(engine, haystacks, round);
}

/** @ref sz_regex_links_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_links_simt_(engine, haystacks, round);
}

/** @ref sz_regex_resume_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_resume_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_resume_simt_(engine, haystacks, round);
}

/** @ref sz_regex_links_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_double_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_links_double_simt_(round);
}

/** @ref sz_regex_group_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_group_ampere_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_group_simt_(engine, round);
}

/** @ref sz_regex_point_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_point_ampere_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_point_simt_(engine, round);
}

/** @ref sz_regex_groups_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_groups_double_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_groups_double_simt_(round);
}

/** @ref sz_regex_take_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_take_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_take_simt_(round);
}

/** @ref sz_regex_retry_folded_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_retry_folded_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_retry_folded_simt_(engine, haystacks, round);
}

/** @ref sz_regex_chain_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_chain_ampere_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_chain_simt_(haystacks, round);
}

/** @ref sz_regex_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_double_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_double_simt_(round);
}

/** @ref sz_regex_sweep_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_sweep_ampere_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_sweep_simt_(haystacks, round);
}

/** @ref sz_regex_carry_simt_ . */
static __global__ void __launch_bounds__(sz_regex_carry_threads_per_block_simt_k)
    sz_regex_carry_ampere_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_carry_simt_(haystacks, round);
}

/** @ref sz_regex_counts_edges_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_edges_ampere_kernel_(sz_size_t const *counts, sz_size_t counts_stride, sz_size_t count,
                                         sz_size_t *edges) {
    sz_regex_counts_edges_simt_(counts, counts_stride, count, edges);
}

/** @ref sz_regex_counts_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_ampere_kernel_(sz_size_t *counts, sz_size_t counts_stride, sz_size_t count,
                                   sz_size_t const *edges) {
    sz_regex_counts_simt_(counts, counts_stride, count, edges);
}

/** @ref sz_regex_expand_kept_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_expand_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_expand_kept_simt_(engine, haystacks, round);
}

/** @ref sz_regex_drift_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_drift_ampere_kernel_(sz_regex_round_simt_t round) {
    sz_regex_drift_simt_(round);
}

/** @ref sz_regex_rewrite_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rewrite_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_rewrite_simt_(engine, haystacks, round);
}

/** @ref sz_regex_firsts_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_firsts_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_firsts_simt_(engine, haystacks, round);
}

/** @ref sz_regex_rows_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rows_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_rows_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_ampere_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_ampere_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_mark_ampere_(engine, haystacks, round);
}

/** The Ampere tier's entry points, staging text with @c cp.async . */
static sz_regex_entry_points_cuda_t const sz_regex_entry_points_ampere_ = {
    (void const *)sz_regex_mark_ampere_kernel_,
    (void const *)sz_regex_scan_ampere_kernel_,
    (void const *)sz_regex_scan_bounded_ampere_kernel_,
    (void const *)sz_regex_bound_ampere_kernel_,
    (void const *)sz_regex_bound_dfa_ampere_kernel_,
    (void const *)sz_regex_bound_pieces_ampere_kernel_,
    (void const *)sz_regex_bound_pieces_dfa_ampere_kernel_,
    (void const *)sz_regex_affine_ampere_kernel_,
    (void const *)sz_regex_affine_dfa_ampere_kernel_,
    (void const *)sz_regex_affine_double_ampere_kernel_,
    (void const *)sz_regex_looks_ampere_kernel_,
    (void const *)sz_regex_looks_dfa_ampere_kernel_,
    (void const *)sz_regex_singles_ampere_kernel_,
    (void const *)sz_regex_prime_ampere_kernel_,
    (void const *)sz_regex_prime_pieces_ampere_kernel_,
    (void const *)sz_regex_compose_ampere_kernel_,
    (void const *)sz_regex_compose_double_ampere_kernel_,
    (void const *)sz_regex_mark_prefixes_ampere_kernel_,
    (void const *)sz_regex_mark_folded_ampere_kernel_,
    (void const *)sz_regex_mark_positions_ampere_kernel_,
    (void const *)sz_regex_mark_positions_folded_ampere_kernel_,
    (void const *)sz_regex_list_ampere_kernel_,
    (void const *)sz_regex_ends_pike_ampere_kernel_,
    (void const *)sz_regex_ends_ampere_kernel_,
    (void const *)sz_regex_links_ampere_kernel_,
    (void const *)sz_regex_continue_ampere_kernel_,
    (void const *)sz_regex_continue_folded_ampere_kernel_,
    (void const *)sz_regex_resume_ampere_kernel_,
    (void const *)sz_regex_links_double_ampere_kernel_,
    (void const *)sz_regex_group_ampere_kernel_,
    (void const *)sz_regex_point_ampere_kernel_,
    (void const *)sz_regex_groups_double_ampere_kernel_,
    (void const *)sz_regex_take_ampere_kernel_,
    (void const *)sz_regex_retry_folded_ampere_kernel_,
    (void const *)sz_regex_chain_ampere_kernel_,
    (void const *)sz_regex_double_ampere_kernel_,
    (void const *)sz_regex_sweep_ampere_kernel_,
    (void const *)sz_regex_carry_ampere_kernel_,
    (void const *)sz_regex_counts_edges_ampere_kernel_,
    (void const *)sz_regex_counts_ampere_kernel_,
    (void const *)sz_regex_expand_ampere_kernel_,
    (void const *)sz_regex_drift_ampere_kernel_,
    (void const *)sz_regex_rewrite_ampere_kernel_,
    (void const *)sz_regex_firsts_ampere_kernel_,
    (void const *)sz_regex_rows_ampere_kernel_,
    1,
    sz_cap_ampere_k,
};

STRINGZILLA_API sz_status_t sz_regex_engine_init_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                        sz_case_sensitivity_t case_sensitivity,
                                                        sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_engine_init_scoped_cuda_(engine, patterns, case_sensitivity, malformed_index,
                                               &sz_regex_entry_points_ampere_, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_contains_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_u1x8_t *contained, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_contains_scoped_cuda_(engine, haystacks, contained, &sz_regex_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_counts_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_counts_scoped_cuda_(engine, haystacks, counts, counts_stride, &sz_regex_entry_points_ampere_,
                                          stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_find_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                        &sz_regex_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_replace_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_strings_view_t const *replacements, sz_ptr_t target,
                                                    sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                           &sz_regex_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_extract_ampere(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t pattern_index, sz_match_t *groups,
                                                    sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                    sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_extract_scoped_cuda_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                           &sz_regex_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_REGEX_AMPERE_CUH_
