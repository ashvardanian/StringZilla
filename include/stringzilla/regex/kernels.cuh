/**
 *  @file include/stringzilla/regex/kernels.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief StringZilla's own launch bodies over the walks of <tt>regex/simt.cuh</tt>, which products never
 *      include, as only <tt>regex/cuda.cuh</tt> and <tt>regex/rocm.cuh</tt> launch them.
 */
#ifndef STRINGZILLA_REGEX_KERNELS_CUH_
#define STRINGZILLA_REGEX_KERNELS_CUH_

#include "stringzilla/regex/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

/*  StringZilla's own kernels and the block bodies they share, which read @c blockIdx, declare
 *  @c __shared__ and drive the multi-launch rounds each vendor's host side sizes; the walks of
 *  <tt>regex/simt.cuh</tt> do the walking. */
#pragma region Kernels

/** Scans one ticket's own tile of @p values in place, as @c sz_substrings_scan_tile_simt_ does,
 *  seeded by the totals the tiles before it publish in @p states, so one launch makes a whole scan;
 *  it clears @p stale, the states the next scan takes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_scan_simt_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile, sz_size_t *chain,
                               sz_u64_t *states, sz_u64_t *stale) {
    __shared__ sz_size_t shared[sz_regex_threads_per_block_simt_k];
    sz_size_t const tiles = sz_size_divide_round_up(count, elements_per_tile);
    sz_size_t tile, begin, end, running = 0, first;
    if (blockIdx.x >= tiles) return;
    tile = sz_chain_ticket_simt_(chain);
    begin = tile * elements_per_tile;
    end = sz_min_of_two(begin + elements_per_tile, count);
    for (first = tile * blockDim.x + threadIdx.x; first < sz_chain_tiles_max_k; first += tiles * blockDim.x)
        stale[first] = 0;
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
}

/**
 *  @brief Bounds the set each chunk's walk of the round's flow hands downstream, one thread per
 *      chunk in each thread's share of the walks.
 *
 *  The bound is exact where the walk starts at its haystack's edge or at the segment's, from the
 *  set the segment edge keeps, else the set handed on from the least input, kept in @c handoffs,
 *  and the undecided elements the set handed on from every input adds.
 *
 *  Downstream is left for positions and lookahead passes, right for lookbehind passes. While more
 *  elements stay undecided than a composition holds, the walk starts further upstream, never past
 *  the segment's edge; past @ref sz_regex_widening_bytes_simt_k it marks the chunk wide, one past a
 *  composition, for the pieces kernel.
 */
STRINGZILLA_DEVICE void sz_regex_bound_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_regex_undecided_simt_t *const undecided = round->undecided + chunk;
        sz_u64_t *const handoff = round->handoffs + chunk * stride;
        sz_u64_t const *kept;
        sz_regex_haystack_simt_t haystack;
        sz_size_t begin, end, from, to, edge, span, word, count;
        sz_u64_t *result, bits;
        undecided->count = 0;
        if (!sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack)) continue;
        sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
        from = backward ? end : begin, to = backward ? begin : end;
        kept = sz_regex_flow_edge_simt_(engine, round, &haystack, &edge);
        for (span = width;; span *= 2) {
            sz_bool_t const exact = (sz_bool_t)(from == edge);
            if (exact && kept)
                for (word = 0; word != words; ++word) sets[word] = kept[word];
            else if (exact) sz_regex_flow_start_simt_(engine, round, sets);
            else sz_regex_flow_least_simt_(engine, round, sets);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            for (word = 0; word != words; ++word) handoff[word] = result[word];
            if (exact) break;
            sz_regex_flow_every_simt_(engine, round, sets);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            for (word = 0, count = 0; word != words; ++word)
                count += (sz_size_t)__popcll(result[word] & ~handoff[word]);
            // A body's DFA holds one state, so one it reaches from every state is the answer.
            if (round->body != sz_regex_none_simt_k && count == 1) {
                for (word = 0; word != words; ++word) handoff[word] = result[word];
                break;
            }
            if (count <= sz_regex_undecided_simt_k) {
                for (word = 0, count = 0; word != words; ++word)
                    for (bits = result[word] & ~handoff[word]; bits; bits &= bits - 1)
                        undecided->indices[count++] = (sz_u32_t)(word * 64 + sz_regex_lowest_simt_(bits));
                undecided->count = (sz_u32_t)count;
                break;
            }
            if (span >= sz_regex_widening_bytes_simt_k) {
                undecided->count = sz_regex_undecided_simt_k + 1;
                break;
            }
            // More elements stay undecided than a composition holds, so the walk starts further up.
            if (backward)
                from = edge - from > span ? sz_regex_flow_align_simt_(engine, round, &haystack, from + span) : edge;
            else from = from - edge > span ? sz_regex_flow_align_simt_(engine, round, &haystack, from - span) : edge;
        }
    }
}

/** @ref sz_regex_bound_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Settles every wide chunk of the round's flow as @ref sz_regex_prime_pieces_simt_kernel_
 *      settles priming, one thread per run of unsettled chunks holding one in each thread's share
 *      of the walks.
 *
 *  The run's most upstream wide chunk walks its haystack downstream from the nearest exact set
 *  upstream of it: a settled chunk's handoff, or the set the segment edge keeps, so no walk leaves
 *  its segment. It writes the exact handoff of every chunk it passes down to the run's most
 *  downstream wide one, which then holds nothing undecided, and the chunks past them compose onto
 *  those handoffs as onto any settled chunk's.
 */
STRINGZILLA_DEVICE void sz_regex_bound_pieces_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                    sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_u64_t const *anchor = STRINGZILLA_NULL;
        sz_u64_t *state = sets, *other = sets + stride, *result;
        sz_regex_haystack_simt_t haystack;
        sz_size_t next, last, previous, walked, begin, end, word;
        sz_bool_t owner = sz_true_k;
        if (round->undecided[chunk].count <= sz_regex_undecided_simt_k) continue;
        sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack);

        // The upstream anchor, past undecided chunks only, as a wide chunk further up owns the run.
        for (next = backward ? chunk + 1 : chunk - 1;; next = backward ? next + 1 : next - 1) {
            sz_u32_t count;
            if (next == (backward ? round->chunks_count : STRINGZILLA_SIZE_MAX)) {
                anchor = round->handoff_edges + (backward ? round->segment + 1 : round->segment) * stride;
                break;
            }
            count = round->undecided[next].count;
            if (count > sz_regex_undecided_simt_k) {
                owner = sz_false_k;
                break;
            }
            if (count) continue;
            anchor = round->handoffs + next * stride;
            break;
        }
        if (!owner) continue;

        // The run's most downstream wide chunk, where a settled chunk or a haystack edge ends it.
        for (last = chunk, previous = chunk; backward ? previous != 0 : previous + 1 != round->chunks_count;) {
            sz_size_t previous_low, previous_high;
            sz_u32_t count;
            previous = backward ? previous - 1 : previous + 1;
            previous_low = round->segment_begin + previous * width;
            previous_high = sz_min_of_two(previous_low + width, round->segment_end);
            count = round->undecided[previous].count;
            if (!count ||
                (backward ? haystack.first >= previous_low : haystack.first + haystack.length < previous_high))
                break;
            if (count > sz_regex_undecided_simt_k) last = previous;
        }
        for (word = 0; word != words; ++word) state[word] = anchor[word];
        for (walked = next; walked != last;) {
            sz_size_t walked_low, walked_high;
            walked = backward ? walked - 1 : walked + 1;
            walked_low = round->segment_begin + walked * width;
            walked_high = sz_min_of_two(walked_low + width, round->segment_end);
            sz_regex_flow_piece_simt_(engine, round, &haystack, walked_low, walked_high, &begin, &end);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, backward ? end : begin,
                                              backward ? begin : end, state, other, sz_false_k, dfa);
            other = result == state ? other : state, state = result;
            for (word = 0; word != words; ++word) round->handoffs[walked * stride + word] = state[word];
            round->undecided[walked].count = 0;
        }
    }
}

/** @ref sz_regex_bound_pieces_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                       sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_pieces_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                           sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Maps the undecided elements each chunk hands on from those its upstream neighbour hands
 *      it, one thread per chunk in each thread's share of the walks.
 *
 *  The map is affine while the neighbour's stay undecided, else the answer, the neighbour past the
 *  segment's edge handing on the set the segment there kept.
 */
STRINGZILLA_DEVICE void sz_regex_affine_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                              sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_regex_undecided_simt_t const *const undecided = round->undecided + chunk;
        sz_regex_affine_simt_t *const affine = round->affines + chunk;
        sz_bool_t const edge = (sz_bool_t)(backward ? chunk + 1 == round->chunks_count : chunk == 0);
        sz_size_t const upstream = backward ? chunk + 1 : chunk - 1;
        sz_u64_t const *known = STRINGZILLA_NULL;
        sz_regex_haystack_simt_t haystack;
        sz_size_t begin, end, from, to, word;
        sz_u32_t row, index;
        sz_u64_t *result;
        if (!undecided->count) continue;
        sz_regex_flow_handing_simt_(haystacks, round, low, high, &haystack);
        sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
        from = backward ? end : begin, to = backward ? begin : end;
        if (edge) known = round->handoff_edges + (backward ? round->segment + 1 : round->segment) * stride;
        else if (!round->undecided[upstream].count) known = round->handoffs + upstream * stride;
        for (word = 0; word != words; ++word)
            sets[word] = known ? known[word] : round->handoffs[upstream * stride + word];
        result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                          sz_false_k, dfa);
        affine->fixed = sz_regex_decided_simt_(undecided, result);
        affine->target = known ? sz_regex_none_simt_k : (sz_u32_t)upstream;
        affine->rows_count = known ? 0 : round->undecided[upstream].count;
        for (row = 0; row != affine->rows_count; ++row) {
            index = round->undecided[upstream].indices[row];
            sz_regex_flow_least_simt_(engine, round, sets);
            sets[index / 64] |= (sz_u64_t)1 << (index % 64);
            result = sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, from, to, sets, sets + stride,
                                              sz_false_k, dfa);
            affine->rows[row] = sz_regex_decided_simt_(undecided, result);
        }
    }
}

/** @ref sz_regex_affine_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_affine_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                     sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief One round of pointer doubling over the chunks' affine maps: each composes its own with
 *      the maps a round's hops upstream until one is answered.
 *
 *  The last round adds the answer to the chunk's handoff and keeps the handoff of the segment's
 *  last chunk downstream for the segment beyond.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0), last = (sz_bool_t)(round.round + 1 == round.rounds);
    sz_bool_t const backward = (sz_bool_t)(round.flow != sz_regex_flow_behind_simt_k);
    sz_regex_affine_simt_t const *source = even ? round.affines : round.affines_next;
    sz_regex_affine_simt_t *target = even ? round.affines_next : round.affines;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_regex_undecided_simt_t const *const undecided = round.undecided + chunk;
        sz_u64_t *const handoff = round.handoffs + chunk * round.set_words;
        sz_regex_affine_simt_t affine;
        sz_u64_t rows[sz_regex_undecided_simt_k];
        sz_u32_t hop, row, index;
        sz_size_t word;
        if (undecided->count) {
            affine = source[chunk];
            for (hop = 1; hop != sz_regex_round_hops_simt_k && affine.target != sz_regex_none_simt_k; ++hop) {
                sz_regex_affine_simt_t const *const next = source + affine.target;
                affine.fixed |= sz_regex_apply_simt_(affine.rows, next->fixed);
                for (row = 0; next->target != sz_regex_none_simt_k && row != next->rows_count; ++row)
                    rows[row] = sz_regex_apply_simt_(affine.rows, next->rows[row]);
                for (row = 0; next->target != sz_regex_none_simt_k && row != next->rows_count; ++row)
                    affine.rows[row] = rows[row];
                affine.target = next->target, affine.rows_count = next->rows_count;
            }
            target[chunk] = affine;
            if (last)
                for (index = 0; index != undecided->count; ++index)
                    if (affine.fixed >> index & 1)
                        handoff[undecided->indices[index] / 64] |= (sz_u64_t)1 << (undecided->indices[index] % 64);
        }
        if (last && chunk == (backward ? 0 : round.chunks_count - 1))
            for (word = 0; word != round.set_words; ++word)
                round.handoff_edges[(backward ? round.segment : round.segment + 1) * round.set_words + word] =
                    handoff[word];
    }
}

/** Walks each chunk's pieces from the sets handed to them, one thread per chunk in each thread's
 *  share of the walks, recording where the lookarounds of the round's pass hold. */
STRINGZILLA_DEVICE void sz_regex_looks_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t const *round, sz_bool_t dfa) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, stride = round->set_words;
    sz_bool_t const backward = (sz_bool_t)(round->flow != sz_regex_flow_behind_simt_k);
    sz_size_t const words = sz_regex_flow_words_simt_(engine, round);
    sz_regex_pike_simt_t pike;
    sz_regex_table_simt_t table;
    sz_u64_t *sets;
    sz_size_t chunk;
    // Every thread stages the body's rows before any leaves.
    if (dfa)
        table = sz_regex_stage_simt_(engine, round, engine->bodies[round->body].rows,
                                     round->body_states * (engine->classes_count + 1));
    if (thread >= round->walks_count) return;
    pike = sz_regex_pike_carve_simt_(round, thread, engine), pike.slots_count = 0;
    sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes + round->walk_sets);
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), begin, end, word;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_u64_t const *handed = STRINGZILLA_NULL;
            if (haystack.first >= high) break;
            sz_regex_flow_piece_simt_(engine, round, &haystack, low, high, &begin, &end);
            if (backward && end <= haystack.length)
                handed = chunk + 1 == round->chunks_count ? round->handoff_edges + (round->segment + 1) * stride
                                                          : round->handoffs + (chunk + 1) * stride;
            else if (!backward && begin)
                handed = chunk == 0 ? round->handoff_edges + round->segment * stride
                                    : round->handoffs + (chunk - 1) * stride;
            if (handed)
                for (word = 0; word != words; ++word) sets[word] = handed[word];
            else sz_regex_flow_start_simt_(engine, round, sets);
            sz_regex_flow_walk_simt_(engine, round, &pike, &table, &haystack, backward ? end : begin,
                                     backward ? begin : end, sets, sets + stride, sz_true_k, dfa);
        }
    }
}

/** @ref sz_regex_looks_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_looks_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_dfa_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_true_k);
}

/**
 *  @brief Writes the one-byte lookarounds holding at every slot of the round's segment into its
 *      half of the table, negated ones flipped, as @c sz_regex_fill_looks_serial_ does, which the
 *      passes then add their own bits to.
 *
 *  Each block takes a run of slots a tile at a time, its first thread galloping to the haystack of
 *  the tile's first slot from the last tile's, and the threads past that haystack on from it.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_singles_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    __shared__ sz_u32_t tables[512];
    __shared__ sz_size_t tile_index;
    __shared__ sz_regex_haystack_simt_t tile_haystack;
    sz_size_t const share = sz_size_divide_round_up(
                                sz_size_divide_round_up(round.segment_end - round.segment_begin, gridDim.x),
                                blockDim.x) *
                            blockDim.x;
    sz_size_t const low = round.segment_begin + blockIdx.x * share;
    sz_size_t const high = sz_min_of_two(low + share, round.segment_end);
    sz_u32_t flipped = 0, index;
    sz_size_t tile;
    for (index = threadIdx.x; index < 512; index += blockDim.x) tables[index] = engine.looks_bytes[1 + index];
    for (index = 0; index != engine.lookarounds_count; ++index)
        if (engine.lookarounds[index].kind & 1) flipped |= (sz_u32_t)sz_regex_look_lookaround_k << index;
    flipped &= engine.looks_bytes[0];
    if (threadIdx.x == 0) tile_index = 0;
    for (tile = low; tile < high; tile += blockDim.x) {
        sz_size_t const slot = tile + threadIdx.x;
        sz_regex_haystack_simt_t haystack;
        sz_size_t position;
        __syncthreads();
        if (threadIdx.x == 0)
            tile_index = sz_regex_haystack_at_simt_(haystacks, tile, tile_index),
            tile_haystack = sz_regex_haystack_simt_(haystacks, tile_index);
        __syncthreads();
        if (slot >= high) continue;
        haystack = tile_haystack;
        if (slot > haystack.first + haystack.length)
            haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, slot, tile_index + 1));
        position = slot - haystack.first;
        round.looks.table[slot & round.looks.mask] =
            ((position ? tables[haystack.text[position - 1]] : 0) |
             (position != haystack.length ? tables[256 + haystack.text[position]] : 0)) ^
            flipped;
    }
}

/**
 *  @brief Primes every chunk whose last haystack runs past its right edge, one thread per chunk, as
 *      the Metal kernel does.
 *
 *  It leaves the state in @c entries, or, when priming does not collapse within its reach, the
 *  uncollapsed set and where each of its states would leave the chunk, or, when no column narrows
 *  within its reach, a frontier one wider than a priming set holds for the pieces kernel. The first
 *  chunk's entry also lands in the segment's edge, which priming the segment before reads.
 *
 *  A folded walk's chunk edges move to the next unit boundary, where it steps the
 *  folds of whole units.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_regex_frontier_simt_t *const frontier = round.frontiers + chunk;
        sz_regex_haystack_simt_t haystack;
        sz_u32_t set[sz_regex_priming_states_simt_k], count, state, index, byte, block;
        sz_size_t edge, stop, at;
        frontier->count = 0;
        if (high == round.slots_count) continue;
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, high - 1, 0));
        if (haystack.first + haystack.length < high) continue;
        edge = folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, high - haystack.first)
                      : high - haystack.first - 1;
        state = folded ? sz_regex_prime_folded_simt_(&engine, &table, round.reverse_images, &haystack, edge,
                                                     sz_regex_priming_bytes_simt_k, set, &count)
                       : sz_regex_prime_simt_(&engine, &table, round.reverse_images, &haystack, edge,
                                              sz_regex_priming_bytes_simt_k, set, &count);
        round.entries[chunk] = state;
        if (chunk == 0) round.edges[round.segment] = state;
        if (state) continue;
        frontier->count = count ? count : sz_regex_priming_states_simt_k + 1;
        if (!count) continue;

        // Each state walks the chunk alone, as the chunk left of it depends on where each leaves.
        for (index = 0; index != count; ++index) frontier->states[index] = set[index];
        if (haystack.first < low && folded)
            for (at = edge, stop = sz_regex_boundary_simt_(haystack.text, haystack.length, low - haystack.first);
                 at > stop;) {
                sz_utf8_folded_image_t image;
                block = sz_regex_block_at_simt_(&table, haystack.first + at);
                at = sz_regex_unit_before_simt_(sz_true_k, haystack.text, at, &image);
                for (byte = image.length; byte--; block = 0) {
                    sz_u32_t const column = block + table.classes[sz_regex_image_byte_simt_(&image, byte)];
                    for (index = 0; index != count; ++index)
                        if (set[index]) set[index] = sz_regex_move_simt_(&table, set[index], column);
                }
            }
        else if (haystack.first < low)
            for (at = edge; at != low - haystack.first - 1; --at) {
                sz_u32_t const column = sz_regex_block_at_simt_(&table, haystack.first + at) +
                                        table.classes[haystack.text[at - 1]];
#pragma unroll
                for (index = 0; index != sz_regex_priming_states_simt_k; ++index)
                    if (index < count && set[index]) set[index] = sz_regex_move_simt_(&table, set[index], column);
            }
        for (index = 0; index != count; ++index) frontier->exits[index] = set[index];
    }
}

/**
 *  @brief Settles every wide chunk, one thread per run of unsettled chunks holding one, as
 *      @c sz_regex_prime_pieces_metal_kernel_ does.
 *
 *  The run's rightmost wide chunk walks its haystack right to left from the nearest exact state
 *  right of it: a settled chunk's entry, the reverse start where the haystack ends, or the next
 *  segment's edge, so no walk leaves its segment. It writes the entry of every chunk it passes down
 *  to the run's leftmost wide one, and the frontiers between them compose onto those entries as
 *  onto any settled chunk's.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_pieces_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                       sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t high, last, next, at = 0, leftmost, previous, walked;
        sz_regex_haystack_simt_t haystack;
        sz_u32_t state = 0;
        sz_bool_t owner = sz_true_k;
        if (round.frontiers[chunk].count <= sz_regex_priming_states_simt_k) continue;
        high = sz_min_of_two(begin + (chunk + 1) * width, round.segment_end);
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, high - 1, 0));
        last = haystack.first + haystack.length;

        // The anchor right of the chunk, past frontiers only, as a wide chunk past it owns the run.
        for (next = chunk + 1;; ++next) {
            sz_size_t const next_high = sz_min_of_two(begin + (next + 1) * width, round.slots_count);
            sz_u32_t const count = next == round.chunks_count ? 0 : round.frontiers[next].count;
            if (last < next_high) {
                state = engine.reverse_starts[0], at = haystack.length;
                break;
            }
            if (count > sz_regex_priming_states_simt_k) {
                owner = sz_false_k;
                break;
            }
            if (count) continue;
            state = next == round.chunks_count ? round.edges[round.segment + 1] : round.entries[next];
            at = sz_regex_prime_edge_simt_(folded, &haystack, next_high);
            break;
        }
        if (!owner) continue;

        // The run's leftmost wide chunk, where a settled chunk or the haystack's start ends it.
        for (leftmost = chunk, previous = chunk; previous-- != 0;) {
            sz_u32_t const count = round.frontiers[previous].count;
            if (!count || begin + (previous + 1) * width <= haystack.first) break;
            if (count > sz_regex_priming_states_simt_k) leftmost = previous;
        }
        for (walked = next; walked-- != leftmost;) {
            sz_size_t const edge = sz_regex_prime_edge_simt_(folded, &haystack, begin + (walked + 1) * width);
            state = folded ? sz_regex_walk_folded_back_simt_(&table, state, &haystack, at, edge)
                           : sz_regex_walk_back_simt_(&table, state, &haystack, at, edge);
            round.entries[walked] = state, at = edge;
            if (walked == 0) round.edges[round.segment] = state;
        }
    }
}

/** Points every unsettled chunk at what settles it, as @c sz_regex_compose_metal_kernel_ does; the
 *  last chunk's right neighbour is the next segment's first, whose entry its edge holds. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t const count = round.frontiers[chunk].count;
        sz_u32_t states[sz_regex_priming_states_simt_k], index;
        sz_regex_composition_simt_t composition;
        sz_size_t next, low, high;
        sz_regex_haystack_simt_t haystack;
        sz_bool_t ends;
        if (!sz_regex_composing_simt_(count)) continue;
        for (index = 0; index != count; ++index) states[index] = round.frontiers[chunk].states[index];
        composition.target = sz_regex_none_simt_k, composition.known = sz_regex_none_simt_k;
        next = chunk + 1, low = round.segment_begin + next * width;
        high = sz_min_of_two(low + width, round.slots_count);
        haystack = sz_regex_haystack_simt_(haystacks, sz_regex_haystack_at_simt_(haystacks, low, 0));
        // The haystack either ends inside the next chunk or spans it, which then is settled or not.
        ends = (sz_bool_t)(haystack.first + haystack.length < high);
        if (!ends && next != round.chunks_count && sz_regex_composing_simt_(round.frontiers[next].count)) {
            composition.target = (sz_u32_t)next;
            for (index = 0; index != round.frontiers[next].count; ++index)
                composition.indices[index] = sz_regex_index_of_simt_(states, count, round.frontiers[next].exits[index]);
        }
        else {
            sz_u32_t const entry = ends                         ? engine.reverse_starts[0]
                                   : next == round.chunks_count ? round.edges[round.segment + 1]
                                                                : round.entries[next];
            sz_size_t const at = ends ? haystack.length : sz_regex_prime_edge_simt_(folded, &haystack, high);
            sz_size_t const edge = sz_regex_prime_edge_simt_(folded, &haystack, low);
            sz_u32_t const state = folded ? sz_regex_walk_folded_back_simt_(&table, entry, &haystack, at, edge)
                                          : sz_regex_walk_back_simt_(&table, entry, &haystack, at, edge);
            composition.known = sz_regex_index_of_simt_(states, count, state);
        }
        round.compositions[chunk] = composition;
    }
}

/** One round of pointer doubling over the unsettled chunks; the last round writes the entry. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_composition_simt_t const *source = even ? round.compositions : round.compositions_next;
    sz_regex_composition_simt_t *target = even ? round.compositions_next : round.compositions;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_regex_composition_simt_t composition;
        sz_u32_t hop, index;
        if (!sz_regex_composing_simt_(round.frontiers[chunk].count)) continue;
        composition = source[chunk];
        for (hop = 1; hop != sz_regex_round_hops_simt_k && composition.target != sz_regex_none_simt_k; ++hop) {
            sz_regex_composition_simt_t const next = source[composition.target];
            sz_u8_t indices[sz_regex_priming_states_simt_k];
            sz_u32_t const next_count = round.frontiers[next.target == sz_regex_none_simt_k ? 0 : next.target].count;
            if (next.known != sz_regex_none_simt_k) {
                composition.known = composition.indices[next.known];
                composition.target = sz_regex_none_simt_k;
                break;
            }
            for (index = 0; index != next_count; ++index)
                indices[index] = next.indices[index] == 0xFF ? 0xFF : composition.indices[next.indices[index]];
            for (index = 0; index != next_count; ++index) composition.indices[index] = indices[index];
            composition.target = next.target;
        }
        target[chunk] = composition;
        if (round.round + 1 != round.rounds) continue;
        round.entries[chunk] = round.frontiers[chunk].states[composition.known];
        if (chunk == 0) round.edges[round.segment] = round.entries[chunk];
    }
}

/** Marks every slot of one chunk where a match starts, one thread per chunk, as the Metal kernel
 *  does, each block first staging its tile's text, which its threads read a chunk apart. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    extern __shared__ sz_u8_t sz_regex_staged_simt_[];
    __shared__ sz_size_t tile_haystack;
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_u8_t *const staged = sz_regex_staged_simt_ + sz_regex_staged_bytes_simt_k +
                            round.staged_cells * sizeof(sz_u32_t);
    sz_u8_t const *const tape = (sz_u8_t const *)sz_strings_start_simt_(haystacks, 0);
    sz_u8_t const *const tape_end = (sz_u8_t const *)sz_strings_start_simt_(haystacks, haystacks.count);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, tile_slots = width * blockDim.x;
    sz_u32_t const accepts = engine.reverse_accepts_begin;
    sz_size_t tile;
    // The scan reads one entry past the chunks, which ends up holding the candidates' total.
    if (blockIdx.x == 0 && threadIdx.x == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (tile = blockIdx.x; tile * blockDim.x < round.chunks_count; tile += gridDim.x) {
        sz_size_t const chunk = tile * blockDim.x + threadIdx.x, tile_low = round.segment_begin + tile * tile_slots;
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t index, word, first_byte;
        sz_u64_t *const words = round.starts + (low - round.segment_begin) / 64;
        sz_u8_t const *origin;
        sz_u32_t marked = 0;
        // The last tile's walks are done with its text and its haystack before either is replaced.
        __syncthreads();
        if (threadIdx.x == 0) tile_haystack = sz_regex_haystack_at_simt_(haystacks, tile_low, 0);
        __syncthreads();
        // A tile's bytes run from the byte before its first slot, as each slot past the first of a
        // haystack stands for the byte before it, and span at most its slots.
        first_byte = tile_low > tile_haystack ? tile_low - tile_haystack - 1 : 0;
        origin = sz_regex_stage_text_simt_(
            staged, tape + first_byte, tape + sz_min_of_two(first_byte + tile_slots + 1, (sz_size_t)(tape_end - tape)),
            tape, tape_end);
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
                marked += (sz_u32_t)__popc(mask);
            }
            if (haystack.first < low || sz_regex_step_edge_simt_(&table, state, haystack.first) < accepts) continue;
            bit = (sz_u32_t)(haystack.first - low);
            words[bit / 64] |= (sz_u64_t)1 << (bit % 64), ++marked;
        }
        round.chunk_offsets[chunk] = marked;
    }
}

/**
 *  @brief Marks every slot whose bytes keep some bucket of the engine's prefixes at every offset
 *      inside its haystack, a slot per thread at a time, as @c sz_regex_find_prefixes_serial_ finds
 *      them, in place of the start-finding walk and its priming; the end walks settle each one.
 *
 *  Each block takes a run of whole chunks a tile at a time, where a thread per chunk gallops to the
 *  haystack of its first slot from the last tile's last, and the slots past that haystack on from
 *  it. A haystack's last slot, past its bytes, is never marked, as no match is empty.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_prefixes_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                        sz_regex_round_simt_t round) {
    __shared__ sz_u8_t buckets[sz_regex_prefixes_depth_k][256];
    __shared__ sz_u64_t words[sz_regex_prefix_chunks_simt_k];
    __shared__ sz_size_t indices[sz_regex_prefix_chunks_simt_k];
    __shared__ sz_regex_haystack_simt_t heads[sz_regex_prefix_chunks_simt_k];
    sz_size_t const tile_slots = (sz_size_t)sz_regex_prefix_chunks_simt_k * sz_regex_chunk_slots_simt_k;
    sz_size_t const chunks = sz_size_divide_round_up(round.chunks_count, gridDim.x);
    sz_size_t const share = sz_size_divide_round_up(chunks * sz_regex_chunk_slots_simt_k, tile_slots) * tile_slots;
    sz_size_t const low = round.segment_begin + blockIdx.x * share;
    sz_size_t const high = sz_min_of_two(low + share, round.segment_end);
    sz_u32_t const depth = engine.prefixes->depth;
    sz_size_t tile, index, last = 0;
    for (index = threadIdx.x; index < sz_regex_prefixes_depth_k * 256; index += blockDim.x)
        buckets[index / 256][index % 256] = engine.prefixes->buckets[index / 256][index % 256];
    if (blockIdx.x == 0 && threadIdx.x == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (tile = low; tile < high; tile += tile_slots) {
        __syncthreads();
        if (threadIdx.x < sz_regex_prefix_chunks_simt_k) {
            sz_size_t const first = tile + threadIdx.x * sz_regex_chunk_slots_simt_k;
            words[threadIdx.x] = 0;
            if (first < high) {
                indices[threadIdx.x] = sz_regex_haystack_at_simt_(haystacks, first, last);
                heads[threadIdx.x] = sz_regex_haystack_simt_(haystacks, indices[threadIdx.x]);
                round.chunk_haystacks[(first - round.segment_begin) / sz_regex_chunk_slots_simt_k] =
                    indices[threadIdx.x];
            }
        }
        __syncthreads();
        last = indices[sz_min_of_two((high - tile - 1) / sz_regex_chunk_slots_simt_k,
                                     (sz_size_t)sz_regex_prefix_chunks_simt_k - 1)];
        // Each thread tests a run of consecutive slots, so the haystack it holds serves them all.
        {
            sz_size_t const run = tile + threadIdx.x * sz_regex_prefix_slots_simt_k;
            sz_size_t const chunk = threadIdx.x * sz_regex_prefix_slots_simt_k / sz_regex_chunk_slots_simt_k;
            sz_regex_haystack_simt_t haystack = heads[chunk];
            sz_size_t haystack_index = indices[chunk], slot, position, offset;
            sz_u64_t marks = 0;
            for (slot = run; slot != run + sz_regex_prefix_slots_simt_k && slot < high; ++slot) {
                sz_u8_t held = 0;
                if (slot > haystack.first + haystack.length)
                    haystack_index = sz_regex_haystack_at_simt_(haystacks, slot, haystack_index + 1),
                    haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
                position = slot - haystack.first;
                if (position != haystack.length) held = buckets[0][haystack.text[position]];
                for (offset = 1; held && offset != depth && position + offset != haystack.length; ++offset)
                    held &= buckets[offset][haystack.text[position + offset]];
                marks |= (sz_u64_t)(held != 0) << (slot - run);
            }
            if (marks)
                atomicOr((unsigned long long *)words + chunk,
                         (unsigned long long)(marks << (run - tile) % sz_regex_chunk_slots_simt_k));
        }
        __syncthreads();
        for (index = threadIdx.x; index < sz_regex_prefix_chunks_simt_k && tile + index * 64 < high;
             index += blockDim.x) {
            sz_size_t const chunk = (tile - round.segment_begin) / sz_regex_chunk_slots_simt_k + index;
            round.starts[chunk] = words[index];
            round.chunk_offsets[chunk] = (sz_size_t)__popcll(words[index]);
        }
    }
}

/**
 *  @brief Marks every slot of one chunk where a match of a folded walk starts, one thread per
 *      chunk, as the Metal kernel does.
 *
 *  A start inside the folds of one codepoint snaps outward to the codepoint's own start, as
 *  @c sz_regex_mark_folded_ snaps it, so the walk begins past the unit holding the chunk's last
 *  slot, where priming left its entry.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_folded_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.reverse_rows, round.reverse_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_u32_t const accepts = engine.reverse_accepts_begin;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0) round.chunk_offsets[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = round.segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), word;
        sz_u64_t *const words = round.starts + (low - round.segment_begin) / 64;
        sz_u32_t marked = 0;
        round.chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) words[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_size_t first, last, at, stop;
            sz_u32_t state;
            if (haystack.first >= high) break;
            first = low > haystack.first ? low - haystack.first : 0;
            last = sz_min_of_two(haystack.length, high - haystack.first - 1);
            at = last == haystack.length ? last : sz_regex_boundary_simt_(haystack.text, haystack.length, last + 1);
            state = at == haystack.length ? engine.reverse_starts[0] : round.entries[chunk];
            stop = first ? first - 1 : 0;
            while (at > stop) {
                sz_utf8_folded_image_t image;
                sz_u32_t block = sz_regex_block_at_simt_(&table, haystack.first + at), byte;
                sz_size_t const begin = sz_regex_unit_before_simt_(sz_true_k, haystack.text, at, &image);
                for (byte = image.length; byte--; block = 0) {
                    sz_size_t const start = byte + 1u == image.length ? at : begin;
                    state = sz_regex_move_simt_(&table, state,
                                                block + table.classes[sz_regex_image_byte_simt_(&image, byte)]);
                    if (state >= accepts && start >= first && start <= last)
                        words[(haystack.first + start - low) / 64] |= (sz_u64_t)1
                                                                      << ((haystack.first + start - low) % 64);
                }
                at = begin;
            }
            if (first == 0 && sz_regex_step_edge_simt_(&table, state, haystack.first) >= accepts)
                words[(haystack.first - low) / 64] |= (sz_u64_t)1 << ((haystack.first - low) % 64);
        }
        for (word = 0; word != width / 64; ++word) marked += (sz_u32_t)__popcll(words[word]);
        round.chunk_offsets[chunk] = marked;
    }
}

/**
 *  @brief Marks every unit boundary of one chunk where a match starts, stepping position sets right
 *      to left from the set handed to each piece, as @c sz_regex_mark_positions_serial_ does, or
 *      over folded runes when @p folded, as @c sz_regex_mark_positions_folded_ does.
 *
 *  Each thread keeps two sets in its share of the walks, so a launch takes as many threads as
 *  shares fit and each walks every chunk that many apart.
 */
STRINGZILLA_DEVICE void sz_regex_mark_positions_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                      sz_regex_round_simt_t const *round, sz_bool_t folded) {
    sz_size_t const width = (sz_size_t)1 << round->chunk_shift, words = engine->positions_words;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = round->set_words;
    sz_u64_t *const sets = (sz_u64_t *)(round->walks + thread * round->walk_bytes);
    sz_size_t chunk;
    if (thread == 0) round->chunk_offsets[round->chunks_count] = 0;
    if (thread >= round->walks_count) return;
    for (chunk = thread; chunk < round->chunks_count; chunk += round->walks_count) {
        sz_size_t const low = round->segment_begin + chunk * width;
        sz_size_t const high = sz_min_of_two(low + width, round->segment_end);
        sz_size_t index = sz_regex_haystack_at_simt_(haystacks, low, 0), word;
        sz_u64_t *const bits = round->starts + (low - round->segment_begin) / 64;
        sz_u32_t marked = 0;
        round->chunk_haystacks[chunk] = index;
        for (word = 0; word != width / 64; ++word) bits[word] = 0;
        for (; index < haystacks.count; ++index) {
            sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks, index);
            sz_u8_t const *const text = haystack.text;
            sz_size_t const length = haystack.length;
            sz_u64_t const *handed = STRINGZILLA_NULL;
            sz_size_t begin, end;
            sz_u64_t *state, *other;
            sz_utf8_folded_image_t image;
            if (haystack.first >= high) break;
            begin = low > haystack.first ? low - haystack.first : 0;
            end = sz_min_of_two(high - haystack.first, length + 1);
            // A piece running past the chunk takes the set the chunk right of it hands on,
            // past its unit boundary.
            if (end <= length)
                handed = chunk + 1 == round->chunks_count ? round->handoff_edges + (round->segment + 1) * stride
                                                          : round->handoffs + (chunk + 1) * stride;
            for (word = 0; word != words; ++word) sets[word] = handed ? handed[word] : word == 0;
            state = sz_regex_positions_walk_simt_(engine, round, &haystack, folded,
                                                  end > length ? length : sz_regex_boundary_simt_(text, length, end),
                                                  begin ? begin - 1 : 0, sets, sets + words, bits, low, high);
            if (begin) continue;
            // The start of the haystack takes one more step, over the edge class.
            other = state == sets ? sets + words : sets;
            if (length) sz_regex_unit_after_simt_(folded, text, length, 0, &image);
            sz_regex_positions_step_simt_(engine, state, other, 1,
                                          length ? sz_regex_byte_kind_(sz_regex_image_byte_simt_(&image, 0)) : 0,
                                          sz_regex_looks_at_simt_(&round->looks, haystack.first));
            if (other[0] >> 1 & 1) bits[(haystack.first - low) / 64] |= (sz_u64_t)1 << ((haystack.first - low) % 64);
        }
        for (word = 0; word != width / 64; ++word) marked += (sz_u32_t)__popcll(bits[word]);
        round->chunk_offsets[chunk] = marked;
    }
}

/** @ref sz_regex_mark_positions_simt_ over bytes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_mark_positions_simt_ over folded runes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_folded_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_true_k);
}

/** Lists the candidates of one chunk in slot order, one thread per chunk, for the end walks taking
 *  one candidate per thread. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_list_simt_kernel_(sz_regex_round_simt_t round) {
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const offset = chunk * width;
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk];
        sz_size_t word;
        for (word = 0; word != width / 64; ++word) {
            sz_u64_t bits = round.starts[offset / 64 + word];
            for (; bits; bits &= bits - 1)
                round.slots[index++] = (sz_u32_t)(offset + word * 64 + sz_regex_lowest_simt_(bits));
        }
    }
}

/** Walks every candidate to its end with a PikeVM in each thread's share of the
 *  walks, linking nothing. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_pike_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const count = round.chunk_offsets[round.chunks_count];
    sz_regex_pike_simt_t pike;
    sz_size_t item;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    pike.slots_count = 0;
    for (item = thread; item < count; item += round.walks_count) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        sz_regex_result_simt_t result;
        sz_size_t end, offset = 0;
        sz_u32_t pattern;
        result.link = sz_regex_none_simt_k, result.pattern = sz_regex_none_simt_k, result.end = 0;
        if (sz_regex_match_at_simt_(&engine, STRINGZILLA_NULL, &pike, &haystack, mark - haystack.first, &end, &pattern,
                                    &offset))
            result.pattern = pattern, result.end = haystack.first + end;
        round.results[item] = result;
    }
}

/**
 *  @brief Walks every candidate of one chunk to its end, right to left, one thread per chunk, which
 *      lists the candidates in slot order as it goes, as @c sz_regex_ends_metal_kernel_ does.
 *
 *  A walk leaves its state at the checkpoint of every 16-slot mark it passes, the first unit
 *  boundary at or past the mark, and a later walk, starting further left, that holds the same state
 *  at the same checkpoint stops there and takes that walk's outcome. A walk reaching the chunk's
 *  edge leaves a link there instead: the state it would walk on in, which the links kernel follows.
 *  A folded walk starts at the first rune its unit folds to, and once the links settle
 *  @ref sz_regex_retry_folded_simt_kernel_ tries the others.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const width = (sz_size_t)1 << round.chunk_shift, stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const begin = round.segment_begin;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_u32_t const accepts = engine.forward_accepts_begin;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t low, high, checkpoint, haystack_index, word;
        sz_regex_haystack_simt_t haystack;
        if (index == round.chunk_offsets[chunk]) continue;
        low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        // Only a chunk with candidates has its checkpoints read, so only such a chunk clears them.
        for (checkpoint = (low - begin) / sz_regex_checkpoint_slots_simt_k + 1;
             checkpoint <= (high - begin) / sz_regex_checkpoint_slots_simt_k; ++checkpoint)
            round.checkpoints[checkpoint].state = 0;
        haystack_index = sz_regex_haystack_at_simt_(haystacks, high - 1, round.chunk_haystacks[chunk]);
        haystack = sz_regex_haystack_simt_(haystacks, haystack_index);

        for (word = (high - low + 63) / 64; word-- != 0;) {
            sz_u64_t bits = round.starts[(low - begin) / 64 + word];
            while (bits) {
                sz_u32_t const bit = 63 - (sz_u32_t)__clzll((long long)bits);
                sz_size_t const mark = low + word * 64 + bit;
                sz_size_t start, at, stepped = 1, kind = 0, last_end = 0, last_at = 0, inherited_end = 0;
                sz_size_t recorded_low = 0, recorded_high = 0;
                sz_u32_t state, last = 0, link = sz_regex_none_simt_k, inherited = sz_regex_none_simt_k;
                sz_regex_result_simt_t result;
                bits ^= (sz_u64_t)1 << bit;
                while (mark < haystack.first) haystack = sz_regex_haystack_simt_(haystacks, --haystack_index);
                round.slots[--index] = (sz_u32_t)(mark - begin);

                start = mark - haystack.first;
                if (start && folded) {
                    sz_utf8_folded_image_t previous;
                    stepped = start - sz_regex_unit_before_simt_(sz_true_k, haystack.text, start, &previous);
                    kind = sz_regex_byte_kind_(sz_regex_image_byte_simt_(&previous, previous.length - 1u));
                }
                else if (start) kind = sz_regex_byte_kind_simt_(haystack.text[start - 1]);
                state = engine.forward_starts[kind];
                for (at = start; state != 0;) {
                    sz_size_t due;
                    if (at == haystack.length) {
                        state = sz_regex_step_edge_simt_(&table, state, haystack.first + at);
                        if (state >= accepts) last = state, last_end = last_at = at;
                        break;
                    }
                    // The mark at the chunk's own start is the edge of the chunk before,
                    // which owns its checkpoint.
                    due = sz_regex_mark_at_simt_(haystack.first + 1 + at, stepped);
                    if (due > low) {
                        sz_size_t const visited = (due - begin) / sz_regex_checkpoint_slots_simt_k;
                        sz_regex_checkpoint_simt_t *const entry = round.checkpoints + visited;
                        if (entry->state == state) {
                            link = entry->link;
                            if (entry->pattern != sz_regex_none_simt_k)
                                inherited = entry->pattern, inherited_end = begin + entry->end, last_at = at;
                            break;
                        }
                        entry->state = state;
                        if (due == high) {
                            link = index;
                            entry->link = index, entry->pattern = sz_regex_none_simt_k;
                            round.links[index].link = state;
                            round.links_list[atomicAdd(round.counters + sz_regex_links_counter_simt_k, 1u)] = index;
                            break;
                        }
                        if (recorded_high == 0) recorded_low = visited;
                        recorded_high = visited + 1;
                    }
                    at = sz_regex_advance_simt_(&table, folded, &haystack, at, haystack.length, accepts, &state, &last,
                                                &last_end, &last_at, &stepped);
                }
                result.link = link;
                result.end = inherited != sz_regex_none_simt_k ? inherited_end : haystack.first + last_end;
                result.pattern = inherited != sz_regex_none_simt_k ? inherited
                                 : last                            ? sz_regex_pattern_simt_(&engine, last)
                                                                   : sz_regex_none_simt_k;
                // The checkpoints passed learn the outcome, keeping the match only when the step
                // reaching it was past them, which a match ending inside the folds of the unit
                // before a checkpoint was not.
                for (checkpoint = recorded_low; checkpoint < recorded_high; ++checkpoint) {
                    sz_bool_t const past = (sz_bool_t)(result.pattern != sz_regex_none_simt_k &&
                                                       haystack.first + 1 + last_at >=
                                                           begin + checkpoint * sz_regex_checkpoint_slots_simt_k);
                    round.checkpoints[checkpoint].link = result.link;
                    round.checkpoints[checkpoint].pattern = past ? result.pattern : sz_regex_none_simt_k;
                    round.checkpoints[checkpoint].end = (sz_u32_t)(result.end - begin);
                }
                round.results[index] = result;
            }
        }
    }
}

/** Walks every link on from its chunk's edge, at the first unit boundary past it for a folded walk,
 *  as @ref sz_regex_link_walk_simt_ walks it. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const begin = round.segment_begin, end = round.segment_end;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_u32_t const link = round.links_list[item];
        sz_size_t const chunk = round.slots[link] >> round.chunk_shift;
        sz_size_t const edge = sz_min_of_two(begin + ((chunk + 1) << round.chunk_shift), end);
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks, sz_regex_haystack_at_simt_(haystacks, begin + round.slots[link], round.chunk_haystacks[chunk]));
        sz_size_t const at = edge - haystack.first - 1;
        sz_regex_link_walk_simt_(&engine, &round, &table, folded, link, round.links[link].link, &haystack,
                                 folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, at) : at);
    }
}

/** Walks every group the last slide made on from where its walks paused, as the links kernel walks
 *  links, the window having moved two segments on from its end there. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_resume_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const paused = round.looks.begin;
    sz_bool_t const folded = (sz_bool_t)(engine.case_sensitivity != sz_cased_k);
    sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(haystacks,
                                                                      sz_regex_haystack_at_simt_(haystacks, paused, 0));
    sz_size_t const at = folded ? sz_regex_boundary_simt_(haystack.text, haystack.length, paused - haystack.first)
                                : paused - haystack.first;
    sz_u32_t const groups_count = round.counters[sz_regex_groups_counter_simt_k];
    sz_u32_t group = round.groups_begin + blockIdx.x * blockDim.x + threadIdx.x;
    for (; group < groups_count; group += gridDim.x * blockDim.x)
        sz_regex_link_walk_simt_(&engine, &round, &table, folded, sz_regex_group_simt_k | group,
                                 round.groups[group].link, &haystack, at);
}

/**
 *  @brief Walks every link or group handed to a warp on to its end, one warp each, each lane taking
 *      the next piece of the haystack in turn, as @c sz_regex_continue_metal_kernel_ does.
 *
 *  The pieces are of whole units when @p folded. A walk whose next step would read the
 *  lookaround table past its window pauses there, which its continuation keeps, counted; any
 *  other ends with a zero state.
 *
 *  Lane @c turn 's answers reach every lane by a shuffle down to lane zero and a broadcast, the two
 *  cross-lane steps each vendor binds.
 */
STRINGZILLA_DEVICE void sz_regex_continue_simt_(sz_regex_engine_t const *engine, sz_strings_view_t haystacks,
                                                sz_regex_round_simt_t const *round, sz_bool_t folded,
                                                int (*shuffle_down)(int, unsigned), sz_u32_t (*broadcast)(sz_u32_t),
                                                int (*any)(int)) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(engine, round, engine->forward_rows, round->forward_cells);
    sz_regex_continuation_simt_t *const continuations = (sz_regex_continuation_simt_t *)round->links_next;
    sz_size_t const columns = sz_regex_stride_(engine->classes_count, engine->lookarounds_count);
    sz_u32_t const count = round->counters[sz_regex_handed_counter_simt_k], accepts = engine->forward_accepts_begin;
    sz_u32_t const lane = threadIdx.x % 32, warps = gridDim.x * (blockDim.x / 32);
    sz_u32_t item = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    for (; item < count; item += warps) {
        sz_regex_continuation_simt_t const continuation = continuations[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks, sz_regex_haystack_at_simt_(haystacks, continuation.slot, 0));
        sz_u8_t const *const text = haystack.text;
        sz_size_t const length = haystack.length;
        // The pieces end where the window does, at the first unit boundary at or past it.
        sz_size_t const stop = !sz_regex_past_simt_(table.looks, haystack.first + length) ? length
                               : folded ? sz_regex_boundary_simt_(text, length, table.looks->end - haystack.first)
                                        : table.looks->end - haystack.first;
        sz_u32_t state = continuation.state, last = 0;
        sz_size_t at = continuation.slot - haystack.first, last_end = 0, last_at, stepped;
        sz_bool_t paused = sz_false_k;
        while (state != 0) {
            sz_u32_t domain[sz_regex_priming_states_simt_k], set[sz_regex_priming_states_simt_k];
            sz_u32_t lasts[sz_regex_priming_states_simt_k], lasts_offsets[sz_regex_priming_states_simt_k];
            sz_u32_t column = 0, domain_count = 0, index, turn, byte;
            sz_size_t piece, piece_end, position, unit_end, round_at;
            sz_utf8_folded_image_t image;
            int wide = 0;
            if (sz_regex_past_simt_(table.looks, haystack.first + at)) {
                paused = sz_true_k;
                break;
            }
            if (at == length) {
                state = sz_regex_step_edge_simt_(&table, state, haystack.first + at);
                if (state >= accepts) last = state, last_end = length;
                break;
            }
            piece = sz_regex_piece_simt_(folded, &haystack, at, lane, stop);
            piece_end = sz_regex_piece_simt_(folded, &haystack, at, lane + 1, stop), unit_end = piece;
            if (piece < piece_end) {
                unit_end = sz_regex_unit_after_simt_(folded, text, length, piece, &image);
                column = sz_regex_block_at_simt_(&table, haystack.first + piece) +
                         table.classes[sz_regex_image_byte_simt_(&image, 0)];
            }
            if (piece < piece_end && lane == 0) {
                domain[0] = sz_regex_move_simt_(&table, state, column);
                domain_count = domain[0] != 0;
            }
            else if (piece < piece_end) {
                domain_count = round->forward_images[column];
                wide = domain_count > sz_regex_priming_states_simt_k;
                for (index = 0; !wide && index != domain_count; ++index)
                    domain[index] = round->forward_images[columns + column * sz_regex_priming_states_simt_k + index];
            }
            if (any(wide)) {
                sz_size_t const reach = sz_regex_piece_simt_(folded, &haystack, at, 32, stop);
                while (lane == 0 && state != 0 && at != reach)
                    at = sz_regex_advance_simt_(&table, folded, &haystack, at, reach, accepts, &state, &last, &last_end,
                                                &last_at, &stepped);
                state = broadcast(state), last = broadcast(last);
                last_end = (sz_size_t)broadcast((sz_u32_t)last_end) |
                           ((sz_size_t)broadcast((sz_u32_t)(last_end >> 32)) << 32);
                at = state ? reach : at;
                continue;
            }

            // Every state of the domain walks the piece on its own, remembering its last accepting
            // step, the first folded byte already stepped into the domain.
#pragma unroll
            for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
                set[index] = index < domain_count ? domain[index] : 0, lasts_offsets[index] = 0;
                lasts[index] = set[index] >= accepts ? set[index] : 0;
            }
            for (position = piece, byte = 1; position < piece_end; ++byte) {
                sz_u32_t byte_column;
                sz_bool_t alive = sz_false_k;
                if (byte == image.length) {
                    position = unit_end, byte = 0;
                    if (position == piece_end) break;
                    unit_end = sz_regex_unit_after_simt_(folded, text, length, position, &image);
                }
                byte_column = (byte ? 0 : sz_regex_block_at_simt_(&table, haystack.first + position)) +
                              table.classes[sz_regex_image_byte_simt_(&image, byte)];
#pragma unroll
                for (index = 0; index != sz_regex_priming_states_simt_k; ++index) {
                    if (index >= domain_count || !set[index]) continue;
                    set[index] = sz_regex_move_simt_(&table, set[index], byte_column);
                    if (set[index] >= accepts)
                        lasts[index] = set[index],
                        lasts_offsets[index] = (sz_u32_t)((byte ? unit_end : position) - piece);
                    alive = (sz_bool_t)(alive || set[index]);
                }
                if (!alive) break;
            }

            // The lanes settle in turn, each from the state the one before it left.
            round_at = at;
            for (turn = 0; turn != 32 && state != 0; ++turn) {
                sz_size_t const turn_piece = sz_regex_piece_simt_(folded, &haystack, round_at, turn, stop);
                sz_size_t const turn_end = sz_regex_piece_simt_(folded, &haystack, round_at, turn + 1, stop);
                sz_u32_t exit = 0, turn_last = 0, turn_offset = 0;
                if (turn_piece >= turn_end) break;
                if (lane == turn) {
                    sz_u32_t const entered = sz_regex_move_simt_(&table, state, column);
#pragma unroll
                    for (index = 0; index != sz_regex_priming_states_simt_k; ++index)
                        if (index < domain_count && entered && domain[index] == entered)
                            exit = set[index], turn_last = lasts[index], turn_offset = lasts_offsets[index];
                }
                state = broadcast((sz_u32_t)shuffle_down((int)exit, turn));
                turn_last = broadcast((sz_u32_t)shuffle_down((int)turn_last, turn));
                turn_offset = broadcast((sz_u32_t)shuffle_down((int)turn_offset, turn));
                if (turn_last) last = turn_last, last_end = turn_piece + turn_offset;
                at = turn_end;
            }
        }
        if (lane != 0) continue;
        if (last) {
            sz_regex_result_simt_t *const walked = sz_regex_walked_simt_(round, continuation.link);
            walked->pattern = sz_regex_pattern_simt_(engine, last), walked->end = haystack.first + last_end;
        }
        continuations[item].state = paused ? state : 0;
        if (paused)
            continuations[item].slot = haystack.first + at,
            atomicAdd(round->counters + sz_regex_paused_counter_simt_k, 1u);
    }
}

/** One round of pointer doubling over the links, each taking the deepest match its hops reach. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_result_simt_t const *source = even ? round.links : round.links_next;
    sz_regex_result_simt_t *target = even ? round.links_next : round.links;
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_u32_t const link = round.links_list[item];
        target[link] = sz_regex_hops_simt_(source, source[link]);
    }
}

/**
 *  @brief Gives each distinct state among the walks paused at the end of the lookaround table's
 *      window one group, one thread per walk handed on.
 *
 *  The first walk to take its state's leader allocates the group and publishes it there, and one
 *  past the groups' capacity raises the refusal instead.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_group_simt_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_continuation_simt_t const *const continuations = (sz_regex_continuation_simt_t const *)round.links_next;
    sz_size_t const columns = sz_regex_stride_(engine.classes_count, engine.lookarounds_count);
    sz_u32_t const count = round.counters[sz_regex_handed_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += gridDim.x * blockDim.x) {
        sz_u32_t const state = continuations[item].state;
        sz_u32_t *const leader = round.leaders + state / columns;
        sz_u32_t group;
        if (!state || atomicCAS(leader, sz_regex_none_simt_k, item) != sz_regex_none_simt_k) continue;
        group = atomicAdd(round.counters + sz_regex_groups_counter_simt_k, 1u);
        if (group >= round.groups_capacity) {
            *round.looks.refused = 1;
            continue;
        }
        round.groups[group].link = state, round.groups[group].pattern = sz_regex_none_simt_k;
        round.groups[group].end = 0;
        *leader = sz_regex_group_simt_k | group;
    }
}

/** Points every walk paused at the end of the lookaround table's window at its state's group, which
 *  a link or a group of the window's last end then follows as the link whose match wins. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_point_simt_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_continuation_simt_t const *const continuations = (sz_regex_continuation_simt_t const *)round.links_next;
    sz_size_t const columns = sz_regex_stride_(engine.classes_count, engine.lookarounds_count);
    sz_u32_t const count = round.counters[sz_regex_handed_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += gridDim.x * blockDim.x) {
        sz_regex_continuation_simt_t const continuation = continuations[item];
        sz_u32_t group;
        if (!continuation.state) continue;
        group = round.leaders[continuation.state / columns];
        if (group & sz_regex_group_simt_k) sz_regex_walked_simt_(&round, continuation.link)->link = group;
    }
}

/** One round of pointer doubling over the groups, as over the links, each taking the deepest match
 *  a round's hops ahead among the groups of later window ends. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_groups_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_regex_result_simt_t const *source = even ? round.groups : round.groups_next;
    sz_regex_result_simt_t *target = even ? round.groups_next : round.groups;
    sz_u32_t const count = round.counters[sz_regex_groups_counter_simt_k];
    sz_u32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    for (; group < count; group += gridDim.x * blockDim.x) target[group] = sz_regex_hops_simt_(source, source[group]);
}

/** Gives every link that paused at the end of the lookaround table's window its settled group's
 *  match, when it has one, and unlinks it, so the links double as though it never paused. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_take_simt_kernel_(sz_regex_round_simt_t round) {
    sz_u32_t const links_count = round.counters[sz_regex_links_counter_simt_k];
    sz_u32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < links_count; item += gridDim.x * blockDim.x) {
        sz_regex_result_simt_t *const link = round.links + round.links_list[item];
        sz_regex_result_simt_t group;
        if (link->link == sz_regex_none_simt_k || !(link->link & sz_regex_group_simt_k)) continue;
        group = round.groups[link->link & ~sz_regex_group_simt_k];
        link->link = sz_regex_none_simt_k;
        if (group.pattern != sz_regex_none_simt_k) link->pattern = group.pattern, link->end = group.end;
    }
}

/**
 *  @brief Tries the later runes of every folded candidate whose unit folds to several and whose
 *      first rune found no match once the links settled, as @ref sz_regex_match_at_simt_ does.
 *
 *  It takes one thread per candidate, each walked to its end without links, raising the refusal
 *  rather than reading the lookaround table past its window.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_retry_folded_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                       sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_stage_simt_(&engine, &round, engine.forward_rows, round.forward_cells);
    sz_size_t const count = round.chunk_offsets[round.chunks_count], stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t item = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; item < count; item += stride) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t const haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        sz_size_t const start = mark - haystack.first;
        sz_utf8_folded_image_t image;
        sz_size_t end, offset;
        sz_u32_t pattern;
        if (start == haystack.length ||
            sz_regex_settle_simt_(&round, (sz_u32_t)item, mark, &haystack).pattern != sz_regex_none_simt_k)
            continue;
        sz_regex_unit_after_simt_(sz_true_k, haystack.text, haystack.length, start, &image);
        if (!(image.rune_ends & (image.rune_ends - 1))) continue;
        offset = sz_regex_lowest_simt_(image.rune_ends) + 1;
        if (sz_regex_match_at_simt_(&engine, &table, STRINGZILLA_NULL, &haystack, start, &end, &pattern, &offset))
            round.results[item].link = sz_regex_none_simt_k, round.results[item].pattern = pattern,
            round.results[item].end = haystack.first + end;
    }
}

/** Follows the search through one chunk from every candidate as an entry, right to left, one thread
 *  per chunk, and seeds each haystack's chain, as @c sz_regex_chain_metal_kernel_ does. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_chain_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_size_t const begin = round.segment_begin, cursor = round.carry->cursor;
    sz_bool_t const refusing = (sz_bool_t)(round.carry->refusing != 0);
    sz_u32_t const count = (sz_u32_t)round.chunk_offsets[round.chunks_count];
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_u32_t const first = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t haystack_index;
        sz_regex_haystack_simt_t haystack;
        sz_bool_t leaves = sz_false_k;
        sz_u32_t index;
        if (first == end) continue;
        haystack_index = sz_regex_haystack_at_simt_(haystacks, begin + round.slots[end - 1],
                                                    round.chunk_haystacks[chunk]);
        haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        for (index = end; index-- != first;) {
            sz_size_t const mark = begin + round.slots[index];
            sz_size_t last;
            sz_regex_result_simt_t result;
            sz_u32_t next = index + 1, jump;
            sz_bool_t seed;
            while (mark < haystack.first) haystack = sz_regex_haystack_simt_(haystacks, --haystack_index);
            last = haystack.first + haystack.length;
            result = sz_regex_settle_simt_(&round, index, mark, &haystack);
            if (result.pattern != sz_regex_none_simt_k) {
                next = result.end >= round.segment_end
                           ? count
                           : sz_regex_lower_bound_simt_(round.slots, index, count, result.end - begin);
                if (next < count && begin + round.slots[next] == result.end) {
                    sz_regex_result_simt_t const after = sz_regex_settle_simt_(&round, next, result.end, &haystack);
                    if (after.pattern != sz_regex_none_simt_k && after.end == result.end) ++next;
                }
            }
            jump = next < count && begin + round.slots[next] <= last ? next : sz_regex_none_simt_k;
            round.nexts[index] = jump;
            round.exits[index] = jump == sz_regex_none_simt_k || jump >= end ? jump : round.exits[jump];
            leaves = (sz_bool_t)(leaves || round.exits[index] != sz_regex_none_simt_k);

            // A haystack started in this segment is seeded at its first candidate; the one
            // straddling from the last segment at its first candidate past the cursor, unless that
            // is a refused empty match.
            if (haystack.first >= begin)
                seed = (sz_bool_t)(index == 0 || begin + round.slots[index - 1] < haystack.first);
            else {
                sz_bool_t const first_past = (sz_bool_t)(mark >= cursor &&
                                                         (index == 0 || begin + round.slots[index - 1] < cursor));
                sz_bool_t const refused = (sz_bool_t)(first_past && refusing && mark == cursor &&
                                                      result.pattern != sz_regex_none_simt_k && result.end == mark);
                seed = (sz_bool_t)(first_past && !refused);
                if (index != 0 && !first_past && mark > cursor) {
                    sz_size_t const previous = begin + round.slots[index - 1];
                    sz_bool_t const previous_first =
                        (sz_bool_t)(previous >= cursor && (index == 1 || begin + round.slots[index - 2] < cursor));
                    if (previous_first && refusing && previous == cursor) {
                        sz_regex_result_simt_t const before = sz_regex_settle_simt_(&round, index - 1, previous,
                                                                                    &haystack);
                        seed = (sz_bool_t)(before.pattern != sz_regex_none_simt_k && before.end == previous);
                    }
                }
            }
            round.keep[index] = seed;
        }
        if (leaves) round.flags[0] = 1;
    }
}

/** One round of pointer doubling over the chunk exits: every kept candidate keeps those its next
 *  hops reach, and jumps that far again; a round after one that left nothing to follow retires. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_double_simt_kernel_(sz_regex_round_simt_t round) {
    sz_bool_t const even = (sz_bool_t)(round.round % 2 == 0);
    sz_u32_t const *source = even ? round.exits : round.exits_next;
    sz_u32_t *target = even ? round.exits_next : round.exits;
    sz_u32_t const count = (sz_u32_t)round.chunk_offsets[round.chunks_count];
    sz_u32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!round.flags[round.round]) return;
    for (; index < count; index += gridDim.x * blockDim.x) {
        // A mark set in this round only reaches candidates on the chain anyway.
        sz_bool_t const kept = (sz_bool_t)(round.keep[index] != 0);
        sz_u32_t jump = source[index], hop;
        for (hop = 1; hop != sz_regex_round_hops_simt_k && jump != sz_regex_none_simt_k; ++hop) {
            if (kept) round.keep[jump] = 1;
            jump = source[jump];
        }
        if (kept && jump != sz_regex_none_simt_k) round.keep[jump] = 1;
        target[index] = jump == sz_regex_none_simt_k ? jump : source[jump];
        if (target[index] != sz_regex_none_simt_k) round.flags[round.round + 1] = 1;
    }
}

/**
 *  @brief Follows the chain through one chunk from the candidates the doubling kept, left to right,
 *      one thread per chunk, as the Metal kernel does.
 *
 *  It counts what it keeps, flagging each kept candidate and counting those of the extracted
 *  pattern, or, once the counts are scanned, stores what the call asks.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_sweep_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0 && !round.storing)
        round.chunk_kept[round.chunks_count] = 0, round.chunk_sums[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_u32_t const first = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t const base = round.storing ? round.carry->running + round.chunk_kept[chunk] : 0;
        sz_size_t haystack_index = round.chunk_haystacks[chunk], pending = haystack_index;
        sz_size_t last_start = 0, last_end = 0, kept_pattern = 0, marked = STRINGZILLA_SIZE_MAX;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t expected = sz_regex_none_simt_k, kept = 0, index;
        if (haystack.first < low) ++pending;
        for (index = first; index != end; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            if (round.storing)
                for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) <= mark; ++pending)
                    sz_regex_boundary_store_simt_(&round, pending, base + kept);
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.storing) round.kept[index] = 0;
            if (!round.keep[index] && index != expected) continue;
            expected = round.nexts[index];
            result = sz_regex_settle_simt_(&round, index, mark, &haystack);
            if (result.pattern == sz_regex_none_simt_k) continue;
            if (round.storing && round.contained && haystack_index != marked)
                sz_regex_contained_set_simt_(&round, haystacks.count, marked = haystack_index);
            if (round.storing && base + kept < round.matches_capacity) {
                sz_match_t match;
                match.haystack_index = haystack_index, match.needle_index = result.pattern;
                match.byte_offset = mark - haystack.first, match.byte_length = result.end - mark;
                round.matches[base + kept] = match;
            }
            if (!round.storing) round.kept[index] = 1, kept_pattern += result.pattern == round.pattern_index;
            last_start = mark, last_end = result.end, ++kept;
        }
        if (!round.storing) {
            round.chunk_kept[chunk] = kept, round.chunk_sums[chunk] = kept_pattern;
            continue;
        }
        for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) < high; ++pending)
            sz_regex_boundary_store_simt_(&round, pending, base + kept);
        // The chunk with the segment's last kept match tells the carry where its haystack resumes.
        if (kept && round.chunk_kept[chunk] + kept == round.chunk_kept[round.chunks_count])
            round.carry->last_start = last_start, round.carry->last_end = last_end;
    }
}

/** Hands the next segment its carry, launched on one thread, as the Metal kernel does, the last
 *  boundary of whatever the call writes, and the edges of a contains call's bitmap. */
static __global__ void __launch_bounds__(sz_regex_carry_threads_per_block_simt_k)
    sz_regex_carry_simt_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_carry_simt_t *const carry = round.carry;
    sz_size_t const kept = round.chunk_kept[round.chunks_count];
    if (round.segment_end < round.slots_count) {
        sz_size_t const first = sz_regex_first_slot_simt_(haystacks,
                                                          sz_regex_haystack_at_simt_(haystacks, round.segment_end, 0));
        if (kept && carry->last_start >= first) carry->cursor = carry->last_end, carry->refusing = 1;
        else if (first >= round.segment_begin) carry->cursor = first, carry->refusing = 0;
        if (round.rows_offsets && first >= round.segment_begin && first < round.segment_end)
            carry->straddle = carry->counted + sz_regex_counted_before_simt_(&round, first);
    }
    carry->running += kept;
    sz_regex_boundary_store_simt_(&round, haystacks.count, carry->running);
    if (round.offsets) {
        carry->summed += round.chunk_sums[round.chunks_count];
        round.offsets[haystacks.count] = round.slots_count - haystacks.count + carry->summed;
    }
    if (round.rows_offsets) {
        carry->counted += round.chunk_sums[round.chunks_count];
        carry->summed += round.chunk_covers[round.chunks_count];
        round.rows_offsets[haystacks.count] = carry->summed;
    }
    if (round.contained) sz_regex_contained_edges_simt_(&round, haystacks.count);
}

/** Saves the total each thread's range of counts follows, before the differences overwrite it. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_edges_simt_kernel_(sz_size_t const *counts, sz_size_t counts_stride, sz_size_t count,
                                       sz_size_t *edges) {
    sz_size_t const threads = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = sz_min_of_two(thread * sz_size_divide_round_up(count, threads), count);
    edges[thread] = begin ? counts[(begin - 1) * counts_stride] : 0;
}

/** Turns the running totals the sweep left in @p counts into each haystack's own count, every
 *  thread over its own range right to left, so it reads each total before overwriting it. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_simt_kernel_(sz_size_t *counts, sz_size_t counts_stride, sz_size_t count, sz_size_t const *edges) {
    sz_size_t const threads = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const per_thread = sz_size_divide_round_up(count, threads);
    sz_size_t const begin = sz_min_of_two(thread * per_thread, count), end = sz_min_of_two(begin + per_thread, count);
    sz_size_t index;
    for (index = end; index-- != begin;)
        counts[index * counts_stride] -= index == begin ? edges[thread] : counts[(index - 1) * counts_stride];
}

/** Measures every kept match's replacement, one thread per candidate in each thread's share of the
 *  walks, leaving how far it moves what follows. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_expand_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const count = round.chunk_offsets[round.chunks_count];
    sz_regex_pike_simt_t pike;
    sz_size_t item;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (item = thread; item < count; item += round.walks_count) {
        sz_size_t const mark = round.segment_begin + round.slots[item];
        sz_regex_haystack_simt_t haystack;
        sz_regex_result_simt_t result;
        sz_size_t expanded;
        round.drifts[item] = 0;
        if (!round.kept[item]) continue;
        haystack = sz_regex_haystack_simt_(
            haystacks,
            sz_regex_haystack_at_simt_(haystacks, mark, round.chunk_haystacks[round.slots[item] >> round.chunk_shift]));
        result = sz_regex_linked_simt_(&round, (sz_u32_t)item);
        expanded = sz_regex_expand_simt_(&engine, &table, &pike, &haystack,
                                         (sz_u8_t const *)sz_strings_start_simt_(round.replacements, result.pattern),
                                         sz_strings_length_simt_(round.replacements, result.pattern),
                                         mark - haystack.first, result.end - haystack.first, result.pattern,
                                         round.pike_slots, STRINGZILLA_NULL, 0, 0);
        round.drifts[item] = expanded - (result.end - mark);
    }
}

/** Sums the drift of every chunk's kept matches, one thread per chunk, and tells the chunks the
 *  last of them reaches into that they start covered. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_drift_simt_kernel_(sz_regex_round_simt_t round) {
    sz_size_t const stride = (sz_size_t)gridDim.x * blockDim.x;
    sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk == 0) round.chunk_sums[round.chunks_count] = 0;
    for (; chunk < round.chunks_count; chunk += stride) {
        sz_size_t drift = 0, covered = 0, next;
        sz_u32_t index;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index)
            if (round.kept[index]) drift += round.drifts[index], covered = sz_regex_linked_simt_(&round, index).end;
        round.chunk_sums[chunk] = drift;
        for (next = chunk + 1;
             next < round.chunks_count && round.segment_begin + (next << round.chunk_shift) <= covered; ++next)
            round.chunk_covers[next] = covered;
    }
}

/**
 *  @brief Writes one chunk's share of the rewritten haystacks, one thread per chunk in each
 *      thread's share of the walks: every byte no kept match covers, every kept match's
 *      replacement, and where each haystack starting in the chunk begins.
 *
 *  Byte @c b of haystack @c h, in slot @c s, lands at <tt>s - h - 1</tt> plus the drift of the kept
 *  matches before it, and a match starting in slot @c s at <tt>s - h</tt> plus the same.
 */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rewrite_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_u32_t index = (sz_u32_t)round.chunk_offsets[chunk], end = (sz_u32_t)round.chunk_offsets[chunk + 1];
        sz_size_t drift = round.carry->summed + round.chunk_sums[chunk];
        sz_size_t covered = sz_max_of_two(round.carry->last_end, round.chunk_covers[chunk]);
        sz_size_t haystack_index = round.chunk_haystacks[chunk], slot;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        for (slot = low; slot != high; ++slot) {
            if (haystack_index + 1 < haystacks.count &&
                sz_regex_first_slot_simt_(haystacks, haystack_index + 1) == slot)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (slot == haystack.first) round.offsets[haystack_index] = slot - haystack_index + drift;
            else if (slot > covered && slot - haystack_index - 1 + drift < round.target_capacity)
                round.target[slot - haystack_index - 1 + drift] = (char)haystack.text[slot - haystack.first - 1];
            if (index == end || begin + round.slots[index] != slot) continue;
            if (round.kept[index]) {
                sz_regex_result_simt_t const result = sz_regex_linked_simt_(&round, index);
                sz_regex_expand_simt_(&engine, &table, &pike, &haystack,
                                      (sz_u8_t const *)sz_strings_start_simt_(round.replacements, result.pattern),
                                      sz_strings_length_simt_(round.replacements, result.pattern),
                                      slot - haystack.first, result.end - haystack.first, result.pattern,
                                      round.pike_slots, round.target, round.target_capacity,
                                      slot - haystack_index + drift);
                drift += round.drifts[index], covered = result.end;
            }
            ++index;
        }
    }
}

/** Finds every haystack's first kept match of the extracted pattern and counts its rows, one thread
 *  per chunk in each thread's share of the walks: a match is the first when as many kept matches of
 *  the pattern precede it as precede its haystack. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_firsts_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread == 0) round.chunk_covers[round.chunks_count] = 0;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t ranked = round.carry->counted + round.chunk_sums[chunk], preceding = 0, rows = 0;
        sz_size_t haystack_index = round.chunk_haystacks[chunk], ranked_haystack = STRINGZILLA_SIZE_MAX;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t index;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            round.exits[index] = 0;
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.kept[index]) continue;
            result = sz_regex_linked_simt_(&round, index);
            if (result.pattern != round.pattern_index) continue;
            if (ranked_haystack != haystack_index)
                ranked_haystack = haystack_index,
                preceding = haystack.first >= begin
                                ? round.carry->counted + sz_regex_counted_before_simt_(&round, haystack.first)
                                : round.carry->straddle;
            if (ranked++ != preceding) continue;
            round.exits[index] = (sz_u32_t)sz_regex_rows_write_simt_(&engine, &table, &pike, &haystack, haystack_index,
                                                                     mark - haystack.first, result.end - haystack.first,
                                                                     result.pattern, STRINGZILLA_NULL, 0, 0);
            rows += round.exits[index];
        }
        round.chunk_covers[chunk] = rows;
    }
}

/** Writes the rows the firsts kernel counted and where each haystack starting in a chunk begins
 *  among them, one thread per chunk in each thread's share of the walks. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rows_simt_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_table_simt_t const table = sz_regex_direct_simt_(&engine, &round);
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_size_t const begin = round.segment_begin, width = (sz_size_t)1 << round.chunk_shift;
    sz_regex_pike_simt_t pike;
    sz_size_t chunk;
    if (thread >= round.walks_count) return;
    pike = sz_regex_pike_carve_simt_(&round, thread, &engine);
    for (chunk = thread; chunk < round.chunks_count; chunk += round.walks_count) {
        sz_size_t const low = begin + chunk * width, high = sz_min_of_two(low + width, round.segment_end);
        sz_size_t const base = round.carry->summed + round.chunk_covers[chunk];
        sz_size_t haystack_index = round.chunk_haystacks[chunk], pending = haystack_index, written = 0;
        sz_regex_haystack_simt_t haystack = sz_regex_haystack_simt_(haystacks, haystack_index);
        sz_u32_t index;
        if (haystack.first < low) ++pending;
        for (index = (sz_u32_t)round.chunk_offsets[chunk]; index != (sz_u32_t)round.chunk_offsets[chunk + 1]; ++index) {
            sz_size_t const mark = begin + round.slots[index];
            sz_regex_result_simt_t result;
            for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) <= mark; ++pending)
                round.rows_offsets[pending] = base + written;
            while (haystack_index + 1 < haystacks.count &&
                   sz_regex_first_slot_simt_(haystacks, haystack_index + 1) <= mark)
                haystack = sz_regex_haystack_simt_(haystacks, ++haystack_index);
            if (!round.exits[index]) continue;
            result = sz_regex_linked_simt_(&round, index);
            sz_regex_rows_write_simt_(&engine, &table, &pike, &haystack, haystack_index, mark - haystack.first,
                                      result.end - haystack.first, result.pattern, round.rows, round.rows_capacity,
                                      base + written);
            written += round.exits[index];
        }
        for (; pending < haystacks.count && sz_regex_first_slot_simt_(haystacks, pending) < high; ++pending)
            round.rows_offsets[pending] = base + written;
    }
}

#pragma endregion Kernels

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_REGEX_KERNELS_CUH_
