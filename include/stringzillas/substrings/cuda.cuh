/**
 *  @brief  CUDA backend for multi-pattern exact and case-folded substring search.
 *  @file   include/stringzillas/substrings/cuda.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/substrings/serial.hpp
 *
 *  `try_build` uploads a host-built `aho_corasick_view`, so this backend consumes the published contract
 *  rather than the builder's internals.
 *
 *  Each haystack travels as its own pointer-and-length descriptor, so no layout is assumed: one packed
 *  tape and scattered device allocations chunk identically. One thread owns one contiguous slice of one
 *  haystack, sized against the corpus's total byte count rather than the haystack count, so occupancy does
 *  not depend on how the corpus is split, and no needle can straddle the seam between two documents.
 *
 *  A thread starts its walk `max_source_match_bytes - 1` bytes before its chunk, clamped to its own haystack's
 *  start. Aho-Corasick is self-synchronizing, so that warm-up makes chunking exact rather than approximate:
 *  the automaton reaches the same state at the chunk boundary wherever the walk began.
 *
 *  A transition is one data-dependent load, so the walk is latency-bound rather than bandwidth-bound and
 *  resident warps are what hides it. The automaton is therefore staged into shared memory only when the
 *  whole of it fits without costing a resident block, and read through the cache hierarchy otherwise.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_CUDA_CUH_
#define STRINGZILLAS_SUBSTRINGS_CUDA_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stringzillas/types.cuh"             // `unified_alloc_t`, `cuda_status_t`
#include "stringzillas/substrings/serial.hpp" // `aho_corasick_view`, the host-built contract

namespace ashvardanian {
namespace stringzillas {

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.
using ashvardanian::stringzilla::byte_t;
using ashvardanian::stringzilla::size_t;
using ashvardanian::stringzilla::small_size_t;
using ashvardanian::stringzilla::span;
using ashvardanian::stringzilla::status_t;
using ashvardanian::stringzilla::to_bytes_view;

#pragma region Device Kernels

/** @brief Block size every `substrings_cuda` kernel launches with; occupancy is shared-memory-bound, not
 *         thread-count-bound, so a modest fixed block keeps the launch geometry simple. */
static constexpr unsigned substrings_threads_per_block_k = 256;

/** @brief Candidates one thread will scan quadratically before a segment falls back to emitted order. */
static constexpr size_t substrings_cover_segment_limit_k = 4096;

/** @brief Output bytes one block of a rewrite's copy owns, so no block's work scales with a run's width. */
static constexpr size_t substrings_rewrite_tile_bytes_k = 4096;

/** @brief Counter slots a scoring block keeps in shared memory, sized by the document rather than by the
 *         dictionary. 64 KiB is the largest power of two that still leaves two blocks resident per
 *         multiprocessor, and it seats the distinct needles of a 40 KiB document below half load. */
static constexpr size_t substrings_bm25_slots_k = 8192;

/** @brief Probe distance a scoring insert gives up at, spilling to the overflow row. */
static constexpr size_t substrings_bm25_probes_k = 16;

/** @brief Fractional bits a block's running score carries, leaving three orders of headroom over the largest
 *         score a unit-weighted full vocabulary reaches. */
static constexpr int substrings_bm25_scale_k = 32;

/** @brief A counter slot no needle has claimed. Needle indices are dense from zero, so the top value is
 *         never one of them. */
static constexpr u32_t substrings_bm25_empty_slot_k = 0xFFFFFFFFu;

/**
 *  @brief Selects what a chunk walk does at each match: size the output, write it, or count per needle.
 *
 *  `sizing_k` only sizes the output so the caller can scan it, `writing_k` writes each match at its
 *  chunk's precomputed offset, and `counting_k` increments a per-needle counter instead of reporting
 *  anything - the shape BM25 needs, which wants how often each needle occurred and never where.
 *
 *  Consumed with `if constexpr`, matching `tile_march_t` in `stringzillas/types.cuh`.
 */
enum class substrings_pass_t : u8_t { sizing_k = 0, writing_k = 1, counting_k = 2 };

/**
 *  @brief Adds to a counter without reading it back - a reduction, not an atomic exchange.
 *
 *  `red` is issued to the L2 slice and retires without a round-trip, so the warp never stalls on it, which
 *  is what counting wants: the old count is never the question. `atomicAdd` lowers to the same
 *  instruction only when the compiler proves the returned value is dead, which is a property of the
 *  optimizer rather than of the source; spelling the reduction out states the intent and cannot regress.
 */
SZ_DEVICE_INLINE void cuda_increment_global_(u32_t *counter, u32_t addend) noexcept {
    asm volatile("red.global.add.u32 [%0], %1;" ::"l"(__cvta_generic_to_global(counter)), "r"(addend) : "memory");
}

/** @brief The same reduction against a block's own shared memory, for counters that never leave it. */
SZ_DEVICE_INLINE void cuda_increment_shared_(u32_t *counter, u32_t addend) noexcept {
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"((u32_t)__cvta_generic_to_shared(counter)), "r"(addend)
                 : "memory");
}

SZ_DEVICE_INLINE void cuda_increment_shared_(u64_t *counter, u64_t addend) noexcept {
    asm volatile("red.shared.add.u64 [%0], %1;" ::"r"((u32_t)__cvta_generic_to_shared(counter)), "l"(addend)
                 : "memory");
}

/** @brief Where a counting walk puts its counts: the block's own table first, the overflow row when full.
 *         An empty @p overflow means the dictionary fits the table, which is what lifts the probe bound. */
struct substrings_bm25_counters_t {
    substrings_bm25_counter_t *slots = nullptr;
    span<u32_t> overflow {};
    int *overflowed = nullptr;
};

/**
 *  @brief Counts one occurrence of @p needle_index into the block's table, or into the overflow row.
 *
 *  Every lane of the block counts into one table, so every write here races and every write here is atomic.
 *  The probe re-reads its slot rather than hoisting it: a cached read would make a full table look like one
 *  slot reused forever, which scores wrong rather than hanging.
 */
SZ_DEVICE_INLINE void substrings_bm25_count_(substrings_bm25_counters_t const &counters, u32_t needle_index) noexcept {
    // A dictionary that fits the table gets a slot per needle, so the index @b is the slot - no hash, no probe.
    if (counters.overflow.empty()) return cuda_increment_shared_(&counters.slots[needle_index].frequency, 1u);

    substrings_bm25_counter_t volatile *const slots = counters.slots; // ? Volatile: re-read on every probe
    size_t slot = substrings_bm25_probe_of_(needle_index) & (substrings_bm25_slots_k - 1u);
    for (size_t probe = 0; probe != substrings_bm25_probes_k;
         ++probe, slot = (slot + 1u) & (substrings_bm25_slots_k - 1u)) {
        // Whoever the exchange hands the slot to owns it; a rival wanting the same needle joins them.
        u32_t seated = slots[slot].needle_index;
        if (seated == substrings_bm25_empty_slot_k)
            seated = atomicCAS(&counters.slots[slot].needle_index, substrings_bm25_empty_slot_k, needle_index);
        if (seated != substrings_bm25_empty_slot_k && seated != needle_index) continue;
        return cuda_increment_shared_(&counters.slots[slot].frequency, 1u);
    }
    cuda_increment_global_(counters.overflow.data() + needle_index, 1u);
    *counters.overflowed = 1;
}

/** @brief One contribution as a fixed-point integer. The scaling runs in double so the `f32` contribution
 *         survives it exactly; scaling in `f32` would quantize back to 24 significant bits and waste it. */
SZ_DEVICE_INLINE i64_t substrings_bm25_to_fixed_(f32_t contribution) noexcept {
    return (i64_t)__double2ll_rn((double)contribution * (double)(1ull << (unsigned)substrings_bm25_scale_k));
}

/** @brief The block's fixed-point total, back as the score the caller reads. */
SZ_DEVICE_INLINE f32_t substrings_bm25_from_fixed_(i64_t total) noexcept {
    return (f32_t)((double)total / (double)(1ull << (unsigned)substrings_bm25_scale_k));
}

/**
 *  @brief Cooperatively fills @p shared_hot_rows from the head of the hot tier and @p shared_accepts_words
 *         from the acceptance bitmap, once per block. The hot tier's out-degree ordering makes its head the
 *         best prefix to stage; the bitmap span is empty when `try_build` budgeted it out of shared memory.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE void substrings_stage_automaton_(aho_corasick_view<state_id_type_> const &view,
                                                  span<state_id_type_> shared_hot_rows, span<u32_t const> accepts_words,
                                                  span<u32_t> shared_accepts_words) noexcept {
    for (size_t cell = threadIdx.x; cell < shared_hot_rows.size(); cell += blockDim.x)
        shared_hot_rows[cell] = view.hot_rows[cell];
    for (size_t word = threadIdx.x; word < shared_accepts_words.size(); word += blockDim.x)
        shared_accepts_words[word] = accepts_words[word];
    __syncthreads();
}

/**
 *  @brief One byte's transition, staged-shared-memory-first: the staged prefix resolves branch-free from
 *         shared memory, and everything else - hot tier beyond the prefix, and the whole cold tier - defers
 *         to @ref aho_corasick_step, the single transition definition every backend shares.
 *
 *  A single cold lane still makes the whole warp pay that lane's failure-chase depth, which is the cost the
 *  shared-memory staging exists to shrink.
 */
template <typename state_id_type_>
SZ_DEVICE_INLINE state_id_type_ substrings_step_device_(aho_corasick_view<state_id_type_> const &view,
                                                        span<state_id_type_ const> shared_hot_rows,
                                                        small_size_t staged_rows_count, state_id_type_ state,
                                                        u8_t byte) noexcept {
    if (static_cast<small_size_t>(state) < staged_rows_count)
        return hot_row_of<small_size_t>(shared_hot_rows, state)[byte];
    return aho_corasick_step(view, state, byte);
}

/**
 *  @brief Finds, via binary search, which haystack owns global chunk @p chunk_index, given the exclusive
 *         prefix sum of chunk counts per haystack (`haystack_chunk_offsets[haystack_count]` is the grand
 *         total chunk count, mirroring how `outputs_offsets` bounds `outputs_counts`).
 */
SZ_DEVICE_INLINE size_t substrings_resolve_haystack_(span<size_t const> haystack_chunk_offsets,
                                                     size_t chunk_index) noexcept {
    size_t low = 0, high = haystack_chunk_offsets.size() - 1;
    while (low + 1 < high) {
        size_t const mid = low + (high - low) / 2;
        if (haystack_chunk_offsets[mid] <= chunk_index) low = mid;
        else high = mid;
    }
    return low;
}

/**
 *  @brief Walks one chunk's transitions, warming up `max_source_match_bytes - 1` bytes before @p chunk_begin -
 *         clamped to the haystack's own start, never earlier - so a match ending inside the chunk is found
 *         regardless of where its needle started, without reading another haystack. Counts or writes
 *         every match ending in `[chunk_begin, chunk_end)` whose start offset is still within this haystack,
 *         per @p pass_.
 *  @return The number of matches found in the chunk.
 */
template <typename state_id_type_, substrings_pass_t pass_>
SZ_DEVICE_INLINE size_t substrings_walk_chunk_( //
    aho_corasick_view<state_id_type_> const &view, span<state_id_type_ const> shared_hot_rows,
    span<u32_t const> accepts_words, span<byte_t const> haystack, size_t chunk_begin, size_t chunk_end,
    size_t haystack_index, size_t output_base_offset, span<substrings_match_t> matches_out,
    substrings_bm25_counters_t counters = {}) noexcept {

    // Offsets are relative to this haystack, so the warm-up clamps against its own start at zero. Sized in
    // source bytes, the unit a haystack window is measured in, not in the folded bytes a needle is.
    size_t const warm_up_bytes = view.max_source_match_bytes > 0 ? (size_t)view.max_source_match_bytes - 1 : 0;
    size_t const walk_begin = chunk_begin >= warm_up_bytes ? chunk_begin - warm_up_bytes : 0;

    // Every 64-bit quantity is resolved here, once, and the per-byte loops below ride 32-bit deltas from it.
    byte_t const *const walk_base = haystack.data() + walk_begin;
    substrings_match_t *const matches_at_chunk = matches_out.data() + output_base_offset;
    small_size_t const walk_span = static_cast<small_size_t>(chunk_end - walk_begin);
    small_size_t const emit_from = static_cast<small_size_t>(chunk_begin - walk_begin);
    sz_assert_(shared_hot_rows.size() / substrings_alphabet_size_k <= std::numeric_limits<small_size_t>::max() &&
               "The staged prefix is budgeted against one multiprocessor's shared memory in `try_build`");
    small_size_t const staged_rows_count = static_cast<small_size_t>(shared_hot_rows.size() /
                                                                     substrings_alphabet_size_k);

    state_id_type_ state = view.root; // ? Fresh at walk_begin - no state ever crosses a haystack boundary.
    small_size_t matches_found = 0;

    // Every match ending at `position` under the state just entered. The bit answers "does anything end
    // here" - the common no-match byte never touches the global counts array, which at scale costs nearly
    // as much as the tape read itself. Counts ride the state id; offsets index a pool that is O(states
    // squared) and so stays 64-bit, but only as a base hoisted out of the inner loop.
    auto const emit_matches_at = [&](small_size_t position) {
        if (((accepts_words[state >> 5] >> (state & 31u)) & 1u) == 0) return;
        state_id_type_ const output_count = view.outputs_counts[state];
        substrings_output<state_id_type_> const *const outputs_at_state = view.outputs + view.outputs_offsets[state];
        for (state_id_type_ output_index = 0; output_index < output_count; ++output_index) {
            substrings_output<state_id_type_> const &output = outputs_at_state[output_index];
            // `walk_begin` is clamped to the haystack's own start, so underflowing the walk and underflowing
            // the haystack are the same test - and this one needs no absolute offset.
            if (position + 1 < static_cast<small_size_t>(output.folded_match_bytes)) continue;
            if constexpr (pass_ == substrings_pass_t::writing_k) {
                size_t const match_end = walk_begin + position + 1;
                matches_at_chunk[matches_found] = substrings_match_t {haystack_index, (size_t)output.needle_index,
                                                                      match_end - output.folded_match_bytes,
                                                                      (size_t)output.folded_match_bytes};
            }
            else if constexpr (pass_ == substrings_pass_t::counting_k)
                substrings_bm25_count_(counters, (u32_t)output.needle_index);
            ++matches_found;
        }
    };

    // The warm-up primes the state and reports nothing, so once it ends the emit test vanishes from the
    // loop rather than being re-asked on every byte.
    small_size_t delta = 0;
    for (; delta < emit_from; ++delta)
        state = substrings_step_device_(view, shared_hot_rows, staged_rows_count, state, walk_base[delta]);

    // Peeled to the load's own alignment, so the body pays one 4-byte load per four transitions - the
    // transition chain stays strictly serial; only the tape reads widen.
    for (; delta < walk_span && ((size_t)(walk_base + delta) & 3u) != 0; ++delta) {
        state = substrings_step_device_(view, shared_hot_rows, staged_rows_count, state, walk_base[delta]);
        emit_matches_at(delta);
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        u32_vec_t const quad = sz_u32_load_aligned(walk_base + delta);
#pragma unroll
        for (small_size_t lane = 0; lane < 4; ++lane) {
            state = substrings_step_device_(view, shared_hot_rows, staged_rows_count, state,
                                            static_cast<u8_t>(quad.u32 >> (lane * 8)));
            emit_matches_at(delta + lane);
        }
    }
    for (; delta < walk_span; ++delta) {
        state = substrings_step_device_(view, shared_hot_rows, staged_rows_count, state, walk_base[delta]);
        emit_matches_at(delta);
    }
    return matches_found;
}

/**
 *  @brief Walks one chunk as folded bytes, the case-insensitive twin of `substrings_walk_chunk_`.
 *
 *  Folding makes the walk restart-safe only at a codepoint start, so the warm-up snaps back to one before it
 *  begins - three bytes at most, and always earlier, so the extra transitions only prime state further. Match
 *  ends are reported at the source codepoint's end, which is what keeps chunk ownership comparable against
 *  the unsnapped `[chunk_begin, chunk_end)` the planner handed out.
 */
template <typename state_id_type_, substrings_pass_t pass_>
SZ_DEVICE_INLINE size_t substrings_walk_chunk_uncased_( //
    aho_corasick_view<state_id_type_> const &view, span<state_id_type_ const> shared_hot_rows,
    span<u32_t const> accepts_words, span<byte_t const> haystack, size_t chunk_begin, size_t chunk_end,
    size_t haystack_index, size_t output_base_offset, span<substrings_match_t> matches_out,
    substrings_bm25_counters_t counters = {}) noexcept {

    size_t const warm_up_bytes = view.max_source_match_bytes > 0 ? (size_t)view.max_source_match_bytes - 1 : 0;
    size_t walk_begin = chunk_begin >= warm_up_bytes ? chunk_begin - warm_up_bytes : 0;
    walk_begin = sz_utf8_rune_start_at_((cptr_t)haystack.data(), haystack.size(), walk_begin);

    substrings_match_t *const matches_at_chunk = matches_out.data() + output_base_offset;
    small_size_t const staged_rows_count = static_cast<small_size_t>(shared_hot_rows.size() /
                                                                     substrings_alphabet_size_k);
    state_id_type_ state = view.root;
    small_size_t matches_found = 0;

    span<byte_t const> const walked {haystack.data() + walk_begin, haystack.size() - walk_begin};
    substrings_folded_cursor_t cursor;
    substrings_folded_cursor_init(cursor, walked.cast<char const>());

    size_t folded = 0, last_break_folded_end = 0;
    substrings_folded_byte_t step;
    while (substrings_folded_cursor_next(cursor, step)) {
        size_t const source_end = walk_begin + step.codepoint_end;
        if (source_end > chunk_end) break;
        ++folded;
        if (step.malformed) {
            state = view.root;
            continue;
        }

        state = substrings_step_device_(view, shared_hot_rows, staged_rows_count, state, step.byte);
        if (!step.rune_end) continue;
        if (step.breaks_boundary) last_break_folded_end = folded + step.trailing;
        // The warm-up primes state without reporting, exactly as the byte-exact walk's prefix does.
        if (source_end <= chunk_begin) continue;
        if (((accepts_words[state >> 5] >> (state & 31u)) & 1u) == 0) continue;

        state_id_type_ const output_count = view.outputs_counts[state];
        substrings_output<state_id_type_> const *const outputs_at_state = view.outputs + view.outputs_offsets[state];
        for (state_id_type_ output_index = 0; output_index < output_count; ++output_index) {
            substrings_output<state_id_type_> const &output = outputs_at_state[output_index];
            size_t const folded_length = output.folded_match_bytes;
            if (folded < folded_length) continue;

            substrings_resolved_match_t const resolved = substrings_folded_span(walked.cast<char const>(), step, folded,
                                                                                last_break_folded_end, folded_length);
            if (resolved.repeats) continue;
            size_t const match_offset = resolved.source_offset;
            if constexpr (pass_ == substrings_pass_t::writing_k)
                matches_at_chunk[matches_found] = substrings_match_t {haystack_index, (size_t)output.needle_index,
                                                                      walk_begin + match_offset,
                                                                      step.codepoint_end - match_offset};
            else if constexpr (pass_ == substrings_pass_t::counting_k)
                substrings_bm25_count_(counters, (u32_t)output.needle_index);
            ++matches_found;
        }
    }
    return matches_found;
}

/**
 *  @brief How many equal-sized chunks of `chunk_bytes` a haystack of @p haystack_length bytes needs - at
 *         least one, so even an empty or shorter-than-a-chunk haystack still gets a thread.
 */
constexpr size_t substrings_chunks_for_haystack_(size_t haystack_length, size_t chunk_bytes) noexcept {
    return haystack_length == 0 ? (size_t)1 : divide_round_up(haystack_length, chunk_bytes);
}

/**
 *  @brief Walks every chunk of every haystack, one thread per chunk, in whichever pass @p pass_ names.
 *
 *  Both passes share one @p chunk_match_slots buffer, because the host's in-place exclusive scan already
 *  makes them the same allocation: the counting pass writes each chunk's match count into its slot, and the
 *  scattering pass reads the exclusive offset the scan left there. Every chunk owns a private,
 *  non-overlapping output range, so placing a write needs no atomics.
 */
template <typename state_id_type_, substrings_pass_t pass_>
__global__ void substrings_walk_per_cuda_chunk_(aho_corasick_view<state_id_type_> view,
                                                state_id_type_ staged_rows_count, span<u32_t const> accepts_words,
                                                u32_t staged_accepts_words, span<span<byte_t const> const> haystacks,
                                                span<size_t const> haystack_chunk_offsets, size_t chunk_bytes,
                                                size_t chunk_count, span<size_t> chunk_match_slots,
                                                span<substrings_match_t> matches_out) {
    extern __shared__ unsigned char substrings_shared_bytes_[];
    span<state_id_type_> const shared_hot_rows {reinterpret_cast<state_id_type_ *>(substrings_shared_bytes_),
                                                (size_t)staged_rows_count * substrings_alphabet_size_k};
    // The bitmap words land right after the rows, whose byte count is a multiple of four at either id width.
    span<u32_t> const shared_accepts_words {
        reinterpret_cast<u32_t *>(substrings_shared_bytes_ + shared_hot_rows.size() * sizeof(state_id_type_)),
        staged_accepts_words};
    substrings_stage_automaton_(view, shared_hot_rows, accepts_words, shared_accepts_words);
    // Resolved once per block: the staged copy when `try_build` budgeted it in, the global array otherwise.
    span<u32_t const> const accepts = staged_accepts_words
                                          ? span<u32_t const> {shared_accepts_words.data(), accepts_words.size()}
                                          : accepts_words;

    for (size_t chunk_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk_index < chunk_count;
         chunk_index += (size_t)gridDim.x * blockDim.x) {
        size_t const haystack_index = substrings_resolve_haystack_(haystack_chunk_offsets, chunk_index);
        span<byte_t const> const haystack = haystacks[haystack_index];
        size_t const local_chunk_index = chunk_index - haystack_chunk_offsets[haystack_index];
        size_t const chunk_begin = local_chunk_index * chunk_bytes;
        size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, haystack.size());

        size_t const output_base_offset = pass_ == substrings_pass_t::writing_k ? chunk_match_slots[chunk_index]
                                                                                : (size_t)0;
        // One dictionary is byte-exact or folded for its whole lifetime, so every thread takes the same
        // side and the branch costs no divergence. Policy no longer reaches here: the walk emits every
        // match, and a cover - when one is asked for - is resolved afterwards over what it emitted.
        size_t const matches_in_chunk = view.case_sensitivity == substrings_uncased_k
                                            ? substrings_walk_chunk_uncased_<state_id_type_, pass_>( //
                                                  view, shared_hot_rows, accepts, haystack, chunk_begin, chunk_end,
                                                  haystack_index, output_base_offset, matches_out)
                                            : substrings_walk_chunk_<state_id_type_, pass_>( //
                                                  view, shared_hot_rows, accepts, haystack, chunk_begin, chunk_end,
                                                  haystack_index, output_base_offset, matches_out);
        if constexpr (pass_ == substrings_pass_t::sizing_k) chunk_match_slots[chunk_index] = matches_in_chunk;
        else sz_unused_(matches_in_chunk);
    }
}

/**
 *  @brief Decides which overlapping matches survive a leftmost cover, one segment per thread.
 *
 *  A cover is a property of the matches, not of the bytes, so it is resolved here rather than inside the
 *  walk - where it cost every thread a ring wide enough for the longest match, and a second walk to find a
 *  safe place to start. Both are gone: the walk emits every match and this pass decides between them.
 *
 *  Within a haystack the walk emits in non-decreasing end order, so the running maximum end is simply the
 *  previous match's end. A boundary sits there when nothing still to come reaches back across it - and only
 *  matches ending within `max_source_match_bytes` of it can, since no match is longer than that. So the test
 *  is bounded: look ahead while ends stay inside that window and check that no start falls behind. Ends
 *  alone would not do, because the list is ordered by end and a later match can begin earlier.
 *
 *  Nothing before such a boundary can reach past it, so each segment resolves against a cursor of zero,
 *  independently of every other. Segments are short in real text - a needle set drawn from a vocabulary
 *  leaves a median of one match between boundaries - so one thread takes a whole one.
 *
 *  That is a measurement, not a guarantee. A dictionary of a needle and its own suffixes over a repetitive
 *  haystack makes one segment of the whole document, and the greedy below is quadratic in a segment, so the
 *  scan is capped: past `substrings_cover_segment_limit_k` candidates a segment falls back to accepting in
 *  emitted order, which is the same cover whenever starts ascend with ends and a documented approximation
 *  when they do not. Without the cap one thread could hold the grid for the length of a document.
 */
static __global__ void substrings_cover_resolve_(span<substrings_match_t const> matches, size_t longest_match_bytes,
                                                 substrings_overlap_policy_t policy, span<size_t> keep) {

    // Whether the boundary before `index` is real: no match still to come starts before the maximum end
    // already reached. Only matches ending within one match's length of it can, which bounds the look-ahead.
    auto const boundary_before = [&](size_t index) noexcept {
        if (index == 0) return true;
        substrings_match_t const &previous = matches[index - 1];
        if (previous.haystack_index != matches[index].haystack_index) return true;
        size_t const reached = previous.byte_offset + previous.byte_length;
        for (size_t ahead = index; ahead < matches.size(); ++ahead) {
            substrings_match_t const &candidate = matches[ahead];
            if (candidate.haystack_index != previous.haystack_index) break;
            if (candidate.byte_offset + candidate.byte_length >= reached + longest_match_bytes) break;
            if (candidate.byte_offset < reached) return false;
        }
        return true;
    };

    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; index < matches.size();
         index += (size_t)gridDim.x * blockDim.x) {

        // Only a segment's first match works; the rest are decided by whoever owns their segment.
        if (!boundary_before(index)) continue;

        size_t segment_end = index + 1;
        for (; segment_end < matches.size(); ++segment_end)
            if (boundary_before(segment_end)) break;

        // A segment past the cap is resolved in one linear sweep instead, so no thread can stall the grid.
        if (segment_end - index > substrings_cover_segment_limit_k) {
            size_t reached = 0;
            for (size_t slot = index; slot < segment_end; ++slot) {
                substrings_match_t const &candidate = matches[slot];
                bool const accepted = candidate.byte_offset >= reached;
                keep[slot] = accepted;
                if (accepted) reached = candidate.byte_offset + candidate.byte_length;
            }
            continue;
        }

        // The greedy cover: take the earliest start at or past the cursor, breaking ties by policy, and
        // repeat. Quadratic in the segment, which is why the segment is one thread's worth and no more.
        for (size_t slot = index; slot < segment_end; ++slot) keep[slot] = 0;
        size_t cursor = 0;
        for (;;) {
            size_t chosen = segment_end;
            for (size_t slot = index; slot < segment_end; ++slot) {
                substrings_match_t const &candidate = matches[slot];
                if (candidate.byte_offset < cursor) continue;
                if (chosen == segment_end) {
                    chosen = slot;
                    continue;
                }
                substrings_match_t const &incumbent = matches[chosen];
                if (candidate.byte_offset != incumbent.byte_offset) {
                    if (candidate.byte_offset < incumbent.byte_offset) chosen = slot;
                    continue;
                }
                if (policy == substrings_leftmost_longest_k && candidate.byte_length != incumbent.byte_length) {
                    if (candidate.byte_length > incumbent.byte_length) chosen = slot;
                    continue;
                }
                if (candidate.needle_index < incumbent.needle_index) chosen = slot;
            }
            if (chosen == segment_end) break;
            keep[chosen] = 1;
            cursor = matches[chosen].byte_offset + matches[chosen].byte_length;
        }
    }
}

/**
 *  @brief Gathers the surviving matches into their scanned slots, order preserved.
 *  @param[in] keep_offsets The scanned keep flags, one longer than @p matches so the last one has a successor.
 *
 *  The scan overwrote the flags it summed, so survival is read back out of it: a match was kept exactly when
 *  the scan steps across it.
 */
static __global__ void substrings_cover_compact_(span<substrings_match_t const> matches,
                                                 span<size_t const> keep_offsets, span<substrings_match_t> survivors) {
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x; index < matches.size();
         index += (size_t)gridDim.x * blockDim.x)
        if (keep_offsets[index + 1] > keep_offsets[index]) survivors[keep_offsets[index]] = matches[index];
}

/**
 *  @brief Maps each haystack's match range onto the boundaries its reported matches occupy.
 *  @param[in] keep_offsets The cover's scanned keep flags, or empty when every emitted match is reported.
 */
static __global__ void substrings_haystack_match_offsets_(span<size_t const> haystack_chunk_offsets,
                                                          span<size_t const> chunk_match_offsets,
                                                          span<size_t const> keep_offsets,
                                                          span<size_t> haystack_match_offsets) {
    for (size_t haystack_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         haystack_index < haystack_match_offsets.size(); haystack_index += (size_t)gridDim.x * blockDim.x) {
        size_t const emitted_before = chunk_match_offsets[haystack_chunk_offsets[haystack_index]];
        haystack_match_offsets[haystack_index] = keep_offsets.size() ? keep_offsets[emitted_before] : emitted_before;
    }
}

/** @brief Writes how many matches each haystack owns, as the gap between its two boundaries. */
static __global__ void substrings_counts_from_boundaries_(span<size_t const> haystack_match_offsets,
                                                          span<size_t> counts_per_haystack) {
    for (size_t haystack_index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         haystack_index < counts_per_haystack.size(); haystack_index += (size_t)gridDim.x * blockDim.x)
        counts_per_haystack[haystack_index] = haystack_match_offsets[haystack_index + 1] -
                                              haystack_match_offsets[haystack_index];
}

/**
 *  @brief Writes where each match's preceding gap lands, and how long each haystack becomes.
 *
 *  One block per haystack, threads striding its match range. A rewrite is a tiling of gaps and
 *  replacements, and every boundary in that tiling follows from one running quantity: how far the output
 *  has drifted from the input by the time a match is reached. So that drift is all this stores - one
 *  scanned offset per match - and the copy kernel derives the rest from the match list it already has.
 *
 *  Offsets are relative to the haystack's own start, because the base is only known after the scan across
 *  haystacks that this kernel feeds. The copy kernel adds it, having looked the haystack up anyway.
 */
static __global__ void substrings_rewrite_offsets_per_haystack_( //
    span<span<byte_t const> const> haystacks, span<size_t const> match_offsets, span<substrings_match_t const> matches,
    span<size_t const> replacement_offsets, span<size_t> match_gap_offsets, span<size_t> output_sizes) {
    using scan_t = cub::BlockScan<size_t, substrings_threads_per_block_k>;
    __shared__ typename scan_t::TempStorage scan_storage;
    __shared__ size_t drift_carry;

    for (size_t haystack_index = blockIdx.x; haystack_index < haystacks.size(); haystack_index += gridDim.x) {
        size_t const first = match_offsets[haystack_index], last = match_offsets[haystack_index + 1];
        if (threadIdx.x == 0) drift_carry = 0;
        __syncthreads();

        for (size_t tile_first = first; tile_first < last; tile_first += blockDim.x) {
            size_t const match_index = tile_first + threadIdx.x;
            bool const owns_match = match_index < last;
            size_t drift_here = 0, previous_end = 0;
            if (owns_match) {
                substrings_match_t const &match = matches[match_index];
                // Shrinking matches make this wrap, which is exactly right: only the prefix sums are ever
                // read, every one of them names a real offset, and modular arithmetic reproduces each.
                drift_here = replacement_offsets[match.needle_index + 1] - replacement_offsets[match.needle_index] -
                             match.byte_length;
                previous_end = match_index == first
                                   ? 0
                                   : matches[match_index - 1].byte_offset + matches[match_index - 1].byte_length;
            }

            size_t drift_before = 0, drift_in_tile = 0;
            scan_t(scan_storage).ExclusiveSum(drift_here, drift_before, drift_in_tile);
            if (owns_match) match_gap_offsets[match_index] = previous_end + drift_carry + drift_before;
            __syncthreads();
            if (threadIdx.x == 0) drift_carry += drift_in_tile;
            __syncthreads();
        }

        // The scan's own aggregate is the haystack's total drift, so no second pass reduces what it knows.
        if (threadIdx.x == 0) output_sizes[haystack_index] = haystacks[haystack_index].size() + drift_carry;
        __syncthreads(); // ! The next haystack resets the carry this one is still reading.
    }
}

/**
 *  @brief Copies one stretch, clipped to `[tile_begin, tile_end)`, with @p lane striding the surviving bytes.
 *
 *  A stretch that misses the tile entirely costs the clip and nothing else, which is what lets the caller
 *  hand every warp a stretch without first working out which ones land inside.
 */
SZ_DEVICE_INLINE void substrings_copy_clipped_(char *output, size_t tile_begin, size_t tile_end, size_t output_offset,
                                               byte_t const *source, size_t bytes, unsigned lane) noexcept {
    size_t const copy_begin = sz_max_of_two(output_offset, tile_begin);
    size_t const copy_end = sz_min_of_two(output_offset + bytes, tile_end);
    for (size_t position = copy_begin + lane; position < copy_end; position += 32)
        output[position] = (char)source[position - output_offset];
}

/** @brief Index of the last entry at or below @p value, in an ascending array; zero when none is. */
SZ_DEVICE_INLINE size_t substrings_last_not_above_(span<size_t const> ascending, size_t value) noexcept {
    size_t low = 0, high = ascending.size();
    while (low + 1 < high) {
        size_t const middle = low + (high - low) / 2;
        if (ascending[middle] <= value) low = middle;
        else high = middle;
    }
    return low;
}

/**
 *  @brief Copies the rewritten tape, one fixed-width output tile per block, one warp per gap or replacement.
 *
 *  Tiling the output rather than the matches bounds how long any one block works: a corpus of one huge
 *  document with a single match and a corpus of a million tiny ones give every block the same slice. Within
 *  a block the warps take stretches in parallel, because a rewrite over prose has stretches of tens of bytes
 *  and striding a whole block across one of them would leave most lanes idle.
 */
static __global__ void substrings_rewrite_copy_( //
    span<span<byte_t const> const> haystacks, span<size_t const> match_offsets, span<substrings_match_t const> matches,
    span<size_t const> match_gap_offsets, byte_t const *replacement_bytes, span<size_t const> replacement_offsets,
    span<size_t const> output_offsets, size_t tile_bytes, char *output) {

    // Read on the device, so nothing about the tape's size has to reach the host before this launch.
    size_t const output_bytes_total = output_offsets[output_offsets.size() - 1];
    size_t const tile_count = divide_round_up(output_bytes_total, tile_bytes);
    unsigned const warp_index = threadIdx.x / 32u, warps_per_block = blockDim.x / 32u, lane = threadIdx.x % 32u;

    for (size_t tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
        size_t const tile_begin = tile_index * tile_bytes;
        size_t const tile_end = sz_min_of_two(tile_begin + tile_bytes, output_bytes_total);

        for (size_t haystack_index = substrings_last_not_above_(output_offsets, tile_begin);
             haystack_index < haystacks.size() && output_offsets[haystack_index] < tile_end; ++haystack_index) {

            span<byte_t const> const haystack = haystacks[haystack_index];
            size_t const base = output_offsets[haystack_index];
            size_t const first = match_offsets[haystack_index], last = match_offsets[haystack_index + 1];

            // Every match contributes a gap and a replacement; one more stretch closes the haystack.
            size_t const stretches = last - first + 1;
            size_t const wanted = tile_begin > base ? tile_begin - base : 0;
            size_t const skip = first == last ? 0
                                              : substrings_last_not_above_(
                                                    {match_gap_offsets.data() + first, last - first}, wanted);

            // Past the last match the drift is whatever the whole haystack accumulated, which its rewritten
            // length already names - so the closing stretch needs no offset of its own.
            size_t const total_drift = (output_offsets[haystack_index + 1] - base) - haystack.size();

            for (size_t stretch = skip + warp_index; stretch < stretches; stretch += warps_per_block) {
                size_t const match_index = first + stretch;
                bool const closes_haystack = match_index == last;
                size_t const previous_end = match_index == first ? 0
                                                                 : matches[match_index - 1].byte_offset +
                                                                       matches[match_index - 1].byte_length;
                size_t const gap_source_end = closes_haystack ? haystack.size() : matches[match_index].byte_offset;
                size_t const gap_begin = base + (closes_haystack ? previous_end + total_drift
                                                                 : match_gap_offsets[match_index]);

                substrings_copy_clipped_(output, tile_begin, tile_end, gap_begin, haystack.data() + previous_end,
                                         gap_source_end - previous_end, lane);
                if (closes_haystack) continue;

                size_t const needle_index = matches[match_index].needle_index;
                size_t const replacement_first = replacement_offsets[needle_index];
                substrings_copy_clipped_(output, tile_begin, tile_end, gap_begin + (gap_source_end - previous_end),
                                         replacement_bytes + replacement_first,
                                         replacement_offsets[needle_index + 1] - replacement_first, lane);
            }
        }
    }
}

/**
 *  @brief One BM25 score per haystack: a block tallies its haystack's needle frequencies, then reduces them.
 *
 *  A block owns a haystack and its threads stride that haystack's chunks, so one long document still spreads
 *  across 256 lanes while the counter table stays private to the block and needs no cross-block traffic.
 *  Term frequencies are raw overlapping counts, which is what classic BM25 asks for - a cover would suppress
 *  genuine occurrences of a needle nested inside another - so no policy reaches here.
 *
 *  The counters are the block's own shared table, sized by the document rather than by the dictionary, so
 *  scoring reads back what this haystack hit. Contributions accumulate as fixed-point integers into one
 *  shared total, whose addition is associative, so the total does not depend on the order lanes finish in.
 */
template <typename state_id_type_>
__global__ void substrings_score_bm25_per_haystack_(aho_corasick_view<state_id_type_> view,
                                                    state_id_type_ staged_rows_count, span<u32_t const> accepts_words,
                                                    u32_t staged_accepts_words,
                                                    span<span<byte_t const> const> haystacks,
                                                    span<f32_t const> document_lengths, substrings_bm25_t parameters,
                                                    span<f32_t const> needle_weights, span<u32_t> overflow_per_block,
                                                    span<f32_t> scores) {

    extern __shared__ unsigned char substrings_shared_bytes_[];
    span<state_id_type_> const shared_hot_rows {reinterpret_cast<state_id_type_ *>(substrings_shared_bytes_),
                                                (size_t)staged_rows_count * substrings_alphabet_size_k};
    span<u32_t> const shared_accepts_words {
        reinterpret_cast<u32_t *>(substrings_shared_bytes_ + shared_hot_rows.size() * sizeof(state_id_type_)),
        (size_t)staged_accepts_words};
    substrings_stage_automaton_(view, shared_hot_rows, accepts_words, shared_accepts_words);
    span<u32_t const> const accepts = staged_accepts_words
                                          ? span<u32_t const> {shared_accepts_words.data(), shared_accepts_words.size()}
                                          : accepts_words;

    size_t const needle_count = needle_weights.size();
    size_t const table_slots = sz_min_of_two(needle_count, substrings_bm25_slots_k);
    span<substrings_match_t> const no_matches;

    // The table sits after the staged automaton, in the same dynamic allocation the host sized for both.
    substrings_bm25_counter_t *const slots = reinterpret_cast<substrings_bm25_counter_t *>(
        substrings_shared_bytes_ + shared_hot_rows.size() * sizeof(state_id_type_) +
        (size_t)staged_accepts_words * sizeof(u32_t));
    __shared__ u64_t block_score_fixed;
    __shared__ int block_overflowed;

    // Allocated only for a dictionary wider than the table; empty otherwise, draining this clear to nothing.
    span<u32_t> const overflow = overflow_per_block.size()
                                     ? span<u32_t> {overflow_per_block.data() + (size_t)blockIdx.x * needle_count,
                                                    needle_count}
                                     : span<u32_t> {};
    for (size_t needle_index = threadIdx.x; needle_index < overflow.size(); needle_index += blockDim.x)
        overflow[needle_index] = 0;
    __syncthreads();

    for (size_t haystack_index = blockIdx.x; haystack_index < haystacks.size(); haystack_index += gridDim.x) {
        span<byte_t const> const haystack = haystacks[haystack_index];

        for (size_t slot = threadIdx.x; slot < table_slots; slot += blockDim.x)
            slots[slot] = substrings_bm25_counter_t {substrings_bm25_empty_slot_k, 0u};
        if (threadIdx.x == 0) {
            block_score_fixed = 0ull;
            block_overflowed = 0;
        }
        __syncthreads();

        substrings_bm25_counters_t const counters {slots, overflow, &block_overflowed};

        // One chunk per lane where the haystack allows it, floored at one match width so no chunk re-walks
        // more warm-up than it covers. A width derived from the corpus cannot answer this: sized from the
        // mean, a haystack shorter than the mean leaves most of the block with nothing to walk.
        size_t const chunk_bytes = sz_max_of_two(divide_round_up(haystack.size(), (size_t)blockDim.x),
                                                 sz_max_of_two((size_t)view.max_source_match_bytes, (size_t)1));
        size_t const chunks = substrings_chunks_for_haystack_(haystack.size(), chunk_bytes);
        for (size_t chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
            size_t const chunk_begin = chunk * chunk_bytes;
            size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, haystack.size());
            if (view.case_sensitivity == substrings_uncased_k)
                substrings_walk_chunk_uncased_<state_id_type_, substrings_pass_t::counting_k>(
                    view, shared_hot_rows, accepts, haystack, chunk_begin, chunk_end, haystack_index, 0, no_matches,
                    counters);
            else
                substrings_walk_chunk_<state_id_type_, substrings_pass_t::counting_k>(
                    view, shared_hot_rows, accepts, haystack, chunk_begin, chunk_end, haystack_index, 0, no_matches,
                    counters);
        }
        __syncthreads();

        // A needle this document never hit scores `+0`, so skipping free slots is exact, and it keeps
        // `substrings_bm25_term` away from a zero frequency under a zero normalized length, where it is `0/0`.
        f32_t const document_length = document_lengths.size() ? document_lengths[haystack_index]
                                                              : (f32_t)haystack.size();
        auto contribution_of = [&](u32_t needle_index, u32_t frequency) noexcept {
            return substrings_bm25_to_fixed_(needle_weights[needle_index] *
                                             substrings_bm25_term(parameters, (f32_t)frequency, document_length));
        };

        // A seated slot is always incremented before this sync, so a zero frequency means untouched in both
        // layouts - and a slot's needle is its own index when the dictionary got one slot each.
        i64_t partial_fixed = 0;
        for (size_t slot = threadIdx.x; slot < table_slots; slot += blockDim.x) {
            u32_t const frequency = slots[slot].frequency;
            if (frequency == 0u) continue;
            partial_fixed += contribution_of(overflow.empty() ? (u32_t)slot : slots[slot].needle_index, frequency);
        }

        // Only a document that outgrew the table ever dirties the overflow row, so only that document pays a
        // pass over the vocabulary - and a document that large already amortizes it over its own bytes.
        if (block_overflowed)
            for (size_t needle_index = threadIdx.x; needle_index < overflow.size(); needle_index += blockDim.x) {
                u32_t const frequency = overflow[needle_index];
                if (frequency == 0u) continue;
                partial_fixed += contribution_of((u32_t)needle_index, frequency);
                overflow[needle_index] = 0u;
            }

        cuda_increment_shared_(&block_score_fixed, (u64_t)partial_fixed);
        __syncthreads();
        if (threadIdx.x == 0) scores[haystack_index] = substrings_bm25_from_fixed_((i64_t)block_score_fixed);
        __syncthreads(); // ! The next haystack clears this table and reuses this total.
    }
}

#pragma endregion Device Kernels

#pragma region Engine

/**
 *  @brief Aho-Corasick-based @b GPU multi-pattern exact/case-folded substring search.
 *  @tparam allocator_type_ The allocator backing this engine's automaton and device-resident scratch; unified
 *          memory by default, so the host can read match totals straight back after a stream synchronize.
 *  @tparam capability_ Any capability including `sz_cap_cuda_k` - the kernels need no generation-specific
 *          instructions, so every combination shares this specialization.
 *
 *  The automaton needs no upload: `allocator_t` already places the dictionary's arrays where the kernels read
 *  them, so the same `aho_corasick_dictionary` the host builds is the one the device walks. The state-id width
 *  follows from the needle set rather than from a template argument, so the engine holds whichever of the two
 *  automatons `try_build` settled on.
 *
 *  Move-only and owns its scratch: the automaton, and the per-call chunk-planning buffers. A moved-from engine
 *  holds no device memory and must not be used before another build.
 */
template <                                       //
    typename allocator_type_ = unified_alloc_t,  //
    sz_capability_t capability_ = sz_cap_cuda_k, //
    typename enable_ = void                      //
    >
struct substrings_cuda;

template <typename allocator_type_, sz_capability_t capability_>
struct substrings_cuda<allocator_type_, capability_, std::enable_if_t<(capability_ & sz_cap_cuda_k) != 0>> {

    using allocator_t = allocator_type_;
    using narrow_dictionary_t = aho_corasick_dictionary<u16_t, allocator_t>;
    using wide_dictionary_t = aho_corasick_dictionary<u32_t, allocator_t>;
    using match_t = substrings_match_t;
    static constexpr sz_capability_t capability_k = capability_;

    /**
     *  @brief How far beyond `state_count` the cold tier's `base`/`check`/`fail`/`outputs_counts`/
     *         `outputs_offsets` arrays must extend.
     *
     *  A cold transition's target is `base[state] + byte` for `byte` in `[0, 256)`, and the builder guarantees
     *  `base[state] < state_count`, so the highest slot ever addressed is `state_count + 254`.
     */
    static constexpr size_t substrings_cold_slot_headroom_k = substrings_alphabet_size_k - 1;

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    /** @brief Rebinds to `size_t`, for the output CSR and the chunk offsets, neither of which has a ceiling. */
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;
    using descriptor_allocator_t = typename allocator_traits_t::template rebind_alloc<span<byte_t const>>;
    using word_allocator_t = typename allocator_traits_t::template rebind_alloc<u32_t>;
    using byte_allocator_t = typename allocator_traits_t::template rebind_alloc<byte_t>;
    /*  Scratch no host ever touches - written by one kernel, read by the next, or drained by a copy. Unified
     *  memory would fault it in on first touch and migrate it again on the drain, so it is device-resident by
     *  the algorithm's nature rather than by the caller's allocator choice. @sa `similarities/cuda.cuh`. */
    using device_match_allocator_t = device_alloc<substrings_match_t>;
    using device_offset_allocator_t = device_alloc<size_t>;
    using device_byte_allocator_t = device_alloc<byte_t>;
    using device_word_allocator_t = device_alloc<u32_t>;

    /**
     *  @brief Dense acceptance bitmap: bit `state` is set when some needle ends at that state, hot or cold.
     *
     *  Derived from `outputs_counts` once the automaton is built, so the per-byte walk gate never touches that
     *  32x larger array. Words are 32-bit because that is one shared-memory bank: the gate reads exactly one
     *  bank-wide word, and lanes clustered near the root share it as a broadcast rather than a conflict.
     */
    safe_vector<u32_t, word_allocator_t> accepts_words_ {};

    /** @brief One descriptor per haystack. Unified, as the host writes them and every chunk thread reads them. */
    safe_vector<span<byte_t const>, descriptor_allocator_t> haystack_descriptors_ {};
    /** @brief Per-call scratch: chunk count per haystack, then - in place - the exclusive chunk-index offset
     *         per haystack, with the grand total chunk count trailing at `[haystack_count]`. */
    safe_vector<size_t, offset_allocator_t> haystack_chunk_offsets_ {};
    /** @brief Per-call scratch: holds per-chunk counts, then - in place - per-chunk exclusive offsets, with the
     *         grand total in the trailing slot. Grown as needed, reused across calls. */
    safe_vector<size_t, offset_allocator_t> chunk_match_offsets_ {};
    /**
     *  @brief Every match the walk emitted, before any cover has been applied to them.
     *
     *  Under a cover this is an intermediate rather than a result - all three entry points walk into it and
     *  then decide between what it holds - so it is engine scratch the caller never sees.
     */
    safe_vector<substrings_match_t, device_match_allocator_t> emitted_matches_ {};
    /** @brief One flag per emitted match going in, its scanned slot coming out, with the survivor count
     *         trailing - the same in-place trick `chunk_match_offsets_` plays, and for the same reason. */
    safe_vector<size_t, device_offset_allocator_t> cover_keep_ {};
    /** @brief Tile totals for the multi-block route of `cuda_launch_exclusive_sum_`, sized once at its grid
     *         ceiling so no scan ever has to fall back to one block for want of scratch. */
    safe_vector<size_t, device_offset_allocator_t> scan_partials_ {};
    /** @brief The survivors themselves, gathered out of the emitted list. */
    safe_vector<substrings_match_t, device_match_allocator_t> cover_survivors_ {};
    /** @brief Per-haystack match boundaries, the reported twin of `haystack_chunk_offsets_`. */
    safe_vector<size_t, offset_allocator_t> haystack_match_offsets_ {};
    /** @brief Where each match's preceding gap begins, relative to its haystack's rewritten start. The only
     *         thing the copy kernel cannot recompute, since it is the running drift the scan produced. */
    safe_vector<size_t, device_offset_allocator_t> rewrite_gap_offsets_ {};
    /** @brief The replacements as one tape, uploaded per call: the caller's container is host-addressed. */
    safe_vector<byte_t, byte_allocator_t> replacement_bytes_ {};
    /** @brief Where each needle's replacement starts in `replacement_bytes_`, with a trailing terminator. */
    safe_vector<size_t, offset_allocator_t> replacement_offsets_ {};

    /** @brief One `needle_count`-wide row per resident block, catching what will not seat in that block's
     *         table. Empty for any dictionary the table can hold, which is most of them. */
    safe_vector<u32_t, device_word_allocator_t> bm25_overflow_ {};

    /**
     *  @brief The automaton this engine compiles from its needles, at whichever state-id width it fits.
     *
     *  `allocator_t` places its arrays where the kernels read them, so `dictionary_.view()` is already the
     *  device view - there is no second copy to keep in step with it.
     */
    std::variant<narrow_dictionary_t, wide_dictionary_t> dictionary_;

    /** @brief Hot rows staged into shared memory at block start - all of them, or zero when they would cost
     *         a resident block and the kernels read them through the cache instead. */
    u32_t staged_rows_ {};
    /** @brief Acceptance bitmap words staged alongside `staged_rows_`, under the same all-or-nothing rule. */
    u32_t staged_accepts_words_ {};
    /** @brief Resident blocks per multiprocessor to budget shared memory against; zero derives it from the
     *         occupancy the kernel reaches with none, which is the default the accessor below overrides. */
    unsigned target_blocks_per_multiprocessor_ = 0;
    allocator_t alloc_ {};
    cuda_timer_t timer_ {};

  public:
    substrings_cuda() noexcept = default;
    substrings_cuda(substrings_cuda const &) = delete;
    substrings_cuda &operator=(substrings_cuda const &) = delete;
    substrings_cuda(substrings_cuda &&) noexcept = default;
    substrings_cuda &operator=(substrings_cuda &&) noexcept = default;

    /** @brief Releases the automaton and every device-resident buffer this engine owns; a fresh
     *         `try_insert_all` is required after. */
    void reset() noexcept {
        std::visit([](auto &dictionary) noexcept { dictionary.reset(); }, dictionary_);
        accepts_words_.reset();
        haystack_descriptors_.reset();
        haystack_chunk_offsets_.reset();
        chunk_match_offsets_.reset();
        emitted_matches_.reset();
        cover_keep_.reset();
        scan_partials_.reset();
        cover_survivors_.reset();
        haystack_match_offsets_.reset();
        rewrite_gap_offsets_.reset();
        replacement_bytes_.reset();
        replacement_offsets_.reset();
        bm25_overflow_.reset();
        staged_rows_ = u32_t {};
        staged_accepts_words_ = u32_t {};
    }

    /** @brief Overrides the resident-blocks-per-multiprocessor target the hot tier is budgeted against when the
     *         engine finalizes; call before the first operation, with zero restoring the automatic choice. */
    void target_blocks_per_multiprocessor(unsigned desired) noexcept { target_blocks_per_multiprocessor_ = desired; }
    unsigned target_blocks_per_multiprocessor() const noexcept { return target_blocks_per_multiprocessor_; }

    /** @brief The state-id width this engine's automaton settled on, once finalized. */
    substrings_state_width_t state_width() const noexcept {
        return std::holds_alternative<narrow_dictionary_t>(dictionary_) ? substrings_state_width_t::u16_k
                                                                        : substrings_state_width_t::u32_k;
    }
    size_t count_needles() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return dictionary.count_needles(); }, dictionary_);
    }
    size_t count_states() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return dictionary.count_states(); }, dictionary_);
    }
    size_t max_source_match_bytes() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return (size_t)dictionary.max_source_match_bytes(); },
                          dictionary_);
    }
    size_t min_source_match_bytes() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return (size_t)dictionary.min_source_match_bytes(); },
                          dictionary_);
    }
    size_t hot_count() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return dictionary.hot_count(); }, dictionary_);
    }

    /** @brief Runs @p callable against the automaton at whichever state-id width it settled on. */
    template <typename callable_type_>
    auto visit_dictionary(callable_type_ &&callable) const noexcept {
        return std::visit(std::forward<callable_type_>(callable), dictionary_);
    }

#pragma region Kernel Table

    struct kernels_t {
        /** @brief One shape per state-id width, for the three kernels that walk the automaton. The cover and
         *         rewrite kernels below take no view, so they are shared. @sa `levenshtein_distances::kernels_t`,
         *         which lists its cell widths the same way. */
        struct by_width_t {
            kernel_shape_t u16, u32;

            kernel_shape_t const &for_width(substrings_state_width_t width) const noexcept {
                return width == substrings_state_width_t::u16_k ? u16 : u32;
            }
        };
        by_width_t count_chunk;
        by_width_t scatter_chunk;
        exclusive_sum_shapes_t exclusive_sum;
        kernel_shape_t cover_resolve;
        kernel_shape_t cover_compact;
        kernel_shape_t haystack_match_offsets;
        kernel_shape_t counts_from_boundaries;
        kernel_shape_t rewrite_offsets;
        kernel_shape_t rewrite_copy;
        by_width_t score_bm25;
    };

    /** @brief Resolves every kernel handle for @p device_id into @p table, raising the dynamic shared-memory
     *         ceiling on the two chunk kernels to the device's opt-in maximum. The per-launch allocation
     *         depends on the dictionary's hot-tier size, so occupancy is queried per launch.
     *         Both state-id widths are resolved into one table, since the automaton picks its own. */
    static cuda_status_t resolve_kernels_(kernels_t &table, int device_id) noexcept {
        CUdevice const device = device_id;
        int shared_memory_ceiling = 0;
        cuDeviceGetAttribute(&shared_memory_ceiling, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device);

        cuda_status_t status {status_t::success_k, cudaSuccess};

        // One lambda per kernel family, invoked once per width, as `resolve_warp` does in `similarities/cuda.cuh`.
        auto const resolve_walk =
            [&]<typename state_id_type_, substrings_pass_t pass_>(kernel_shape_t &shape) noexcept -> cuda_status_t {
            return resolve_kernel_shape(
                shape, reinterpret_cast<void const *>(&substrings_walk_per_cuda_chunk_<state_id_type_, pass_>), 0,
                static_cast<unsigned>(shared_memory_ceiling), false);
        };
        status = resolve_walk.template operator()<u16_t, substrings_pass_t::sizing_k>(table.count_chunk.u16);
        if (status.status != status_t::success_k) return status;
        status = resolve_walk.template operator()<u32_t, substrings_pass_t::sizing_k>(table.count_chunk.u32);
        if (status.status != status_t::success_k) return status;
        status = resolve_walk.template operator()<u16_t, substrings_pass_t::writing_k>(table.scatter_chunk.u16);
        if (status.status != status_t::success_k) return status;
        status = resolve_walk.template operator()<u32_t, substrings_pass_t::writing_k>(table.scatter_chunk.u32);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.exclusive_sum.whole,
                                      reinterpret_cast<void const *>(&exclusive_sum_across_cuda_device_<size_t>), 0, 0,
                                      false);
        if (status.status != status_t::success_k) return status;

        // The tiled phases pick their grid from this occupancy, so unlike the whole-array kernel they precompute it.
        status = resolve_kernel_shape(
            table.exclusive_sum.reduce_tiles,
            reinterpret_cast<void const *>(&exclusive_sum_reduce_tiles_across_cuda_device_<size_t>),
            cuda_device_collective_threads_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.exclusive_sum.apply_tiles,
            reinterpret_cast<void const *>(&exclusive_sum_apply_tiles_across_cuda_device_<size_t>),
            cuda_device_collective_threads_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.cover_resolve, reinterpret_cast<void const *>(&substrings_cover_resolve_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.cover_compact, reinterpret_cast<void const *>(&substrings_cover_compact_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.haystack_match_offsets,
                                      reinterpret_cast<void const *>(&substrings_haystack_match_offsets_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.counts_from_boundaries,
                                      reinterpret_cast<void const *>(&substrings_counts_from_boundaries_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        // The rewrite kernels carry no automaton, so they stage nothing, need no raised ceiling, and - unlike
        // the walk - are not shared-memory-bound. Their own occupancy is precomputed here so neither has to
        // borrow a grid sized for a dictionary's hot tier.
        status = resolve_kernel_shape(table.rewrite_offsets,
                                      reinterpret_cast<void const *>(&substrings_rewrite_offsets_per_haystack_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(table.rewrite_copy, reinterpret_cast<void const *>(&substrings_rewrite_copy_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        // Scoring stages the same automaton the walk does, so it wants the same opt-in shared ceiling.
        auto const resolve_score = [&]<typename state_id_type_>(kernel_shape_t &shape) noexcept -> cuda_status_t {
            return resolve_kernel_shape(
                shape, reinterpret_cast<void const *>(&substrings_score_bm25_per_haystack_<state_id_type_>), 0,
                static_cast<unsigned>(shared_memory_ceiling), false);
        };
        status = resolve_score.template operator()<u16_t>(table.score_bm25.u16);
        if (status.status != status_t::success_k) return status;
        return resolve_score.template operator()<u32_t>(table.score_bm25.u32);
    }

    /** @brief This device's kernel table, resolved on first use. Check the status before reading it - a full
     *         cache hands back an unresolved table. @sa cuda_device_kernels */
    static expected<kernels_t const &, cuda_status_t> kernels(int device_id) noexcept {
        static cuda_device_kernels<kernels_t> per_device;
        auto *entry = per_device.acquire(device_id);
        if (!entry) return {per_device.unusable(), {status_t::missing_gpu_k, cudaSuccess, CUDA_ERROR_INVALID_DEVICE}};
        if (!entry->resolved) {
            cuda_status_t const status = resolve_kernels_(entry->table, device_id);
            if (status.status != status_t::success_k) {
                per_device.release();
                return {per_device.unusable(), status};
            }
            entry->resolved = true;
        }
        per_device.release();
        return {entry->table, {}};
    }

#pragma endregion Kernel Table

    /**
     *  @brief Indexes all of the @p needles strings into the FSM, at whichever state id it ends up fitting.
     *
     *  No upload follows: `allocator_t` is unified memory, reachable from every device, so the arrays this
     *  builds are already the ones the kernels walk. Construction runs wide, because a dictionary's state
     *  count is only known once it is built, and the narrowing attempt is itself the ceiling test.
     *  @param[in] executor Names the device whose context the non-unified scan scratch is allocated under.
     *  @param[in] specs Sizes the hot tier against that device's L2, the cache its walk reads through.
     *  @note Replaces any previously indexed needle set: the automaton is rebuilt from scratch and the old one
     *        released, so an engine can be re-indexed for a different vocabulary or a different device.
     *  @sa `aho_corasick_dictionary::try_insert` for the status codes this forwards.
     */
    template <typename needles_type_>
    cuda_status_t try_index(needles_type_ const &needles,
                            substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                            cuda_executor_t const &executor = {}, gpu_specs_t const &specs = {}) noexcept {
        // The scan scratch below is device-resident rather than unified, so it lands wherever the context
        // points; binding the named device first is what keeps it off whichever one happened to be current.
        if (cuda_status_t const current = executor.ensure_current(); current.status != status_t::success_k)
            return current;
        wide_dictionary_t wide(alloc_);
        wide.case_sensitivity(case_sensitivity);
        for (auto const &needle : needles) {
            status_t const status = wide.try_insert(to_bytes_view(needle));
            if (status != status_t::success_k) return {status, cudaSuccess};
        }
        // The tier split follows the cache the device walks through rather than the host's last level, which
        // a default `cpu_specs_t` would put at 8 MB whatever the GPU.
        wide.hot_count(specs.l2_bytes / (substrings_alphabet_size_k * sizeof(u32_t)));
        if (status_t const built = wide.try_build(); built != status_t::success_k) return {built, cudaSuccess};

        narrow_dictionary_t narrow(alloc_);
        status_t const narrowed = narrow.try_build(wide);
        if (narrowed != status_t::success_k && narrowed != status_t::overflow_risk_k) return {narrowed, cudaSuccess};
        if (narrowed == status_t::success_k) dictionary_.template emplace<narrow_dictionary_t>(std::move(narrow));
        else dictionary_.template emplace<wide_dictionary_t>(std::move(wide));

        // Sized at the scan's grid ceiling rather than per call: it is 8 KB whatever the corpus, and a scan
        // that found it short would silently drop back to one block.
        if (scan_partials_.try_resize_uninitialized(cuda_device_collective_max_blocks_k + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        return try_derive_accepts_();
    }

  private:
    /** @brief One cell's width in the settled automaton, which is what a staged hot row costs per entry. */
    size_t bytes_per_state_id_() const noexcept {
        return state_width() == substrings_state_width_t::u16_k ? sizeof(u16_t) : sizeof(u32_t);
    }

    /** @brief Shared bytes a scoring block's counter table costs, read by the occupancy query and by the
     *         launch alike so the two cannot disagree about the footprint they are sizing. A dictionary
     *         narrower than the table gets a slot per needle and pays for no more than that. */
    size_t substrings_bm25_counters_bytes_() const noexcept {
        return sz_min_of_two(count_needles(), substrings_bm25_slots_k) * sizeof(substrings_bm25_counter_t);
    }

    /** @brief The settled automaton at @p state_id_type_, which `try_build` has already pinned. */
    template <typename state_id_type_>
    aho_corasick_dictionary<state_id_type_, allocator_t> const &settled_dictionary_() const noexcept {
        return std::get<aho_corasick_dictionary<state_id_type_, allocator_t>>(dictionary_);
    }

    /**
     *  @brief Builds the dense acceptance bitmap the per-byte walk gate reads.
     *
     *  A pure function of the built automaton - no device involved - so it belongs to the build rather than
     *  to whichever executor happens to arrive first. The gate reads it instead of `outputs_counts` because
     *  that array is 32x larger.
     */
    cuda_status_t try_derive_accepts_() noexcept {
        return visit_dictionary([&](auto const &dictionary) noexcept -> cuda_status_t {
            auto const view = dictionary.view();
            size_t const state_capacity = (size_t)view.state_count + substrings_cold_slot_headroom_k;
            // `try_resize` leaves trivial words uninitialized, so every word is written before any bit is set.
            size_t const words_count = divide_round_up<size_t>(state_capacity, 32);
            if (accepts_words_.try_resize(words_count) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            for (size_t word = 0; word < words_count; ++word) accepts_words_[word] = 0;
            for (size_t slot = 0; slot < state_capacity; ++slot)
                if (view.outputs_counts[slot] != 0) accepts_words_[slot >> 5] |= u32_t(1) << (slot & 31u);
            return {status_t::success_k, cudaSuccess};
        });
    }

    /**
     *  @brief Settles how much of the automaton a block stages in shared memory on @p executor 's device.
     *
     *  Budgeted per call rather than once per build, because occupancy and the shared-memory ceiling belong to
     *  the device the caller names - and nothing stops two calls on one engine from naming different ones.
     */
    cuda_status_t try_budget_staging_(cuda_executor_t const &executor) noexcept {
        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;
        CUfunction const walk_function =
            kernel_table.count_chunk.for_width(state_width()).function; // ? The scatter kernel shares its shape

        // Occupancy first, staging only out of what is left over. The walk chases a data-dependent transition
        // load, so resident warps are the only thing hiding its latency, while the rows it would stage are
        // cache-resident already - which makes a block traded away for shared memory a straight loss.
        unsigned target_blocks = target_blocks_per_multiprocessor_;
        if (target_blocks == 0) {
            int blocks_without_staging = 0;
            CUresult const occupancy_error = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_without_staging, walk_function, (int)substrings_threads_per_block_k, 0);
            if (occupancy_error != CUDA_SUCCESS) return make_cuda_status(occupancy_error);
            target_blocks = (unsigned)sz_max_of_two(blocks_without_staging, 1);
        }

        size_t shared_memory_budget = 0;
        cuda_status_t const budget_status = shared_memory_budget_for_resident_blocks(
            shared_memory_budget, walk_function, substrings_threads_per_block_k, target_blocks, executor.device_id());
        if (budget_status.status != status_t::success_k) return budget_status;

        // Staging is all-or-nothing: a partial prefix still pays the copy per block and the bounds test per
        // byte, while most transitions miss it and fall through to memory anyway.
        size_t const bytes_per_state_id = state_width() == substrings_state_width_t::u16_k ? sizeof(u16_t)
                                                                                           : sizeof(u32_t);
        size_t const hot_rows = hot_count();
        size_t const accepts_bytes = accepts_words_.size() * sizeof(u32_t);
        size_t const whole_automaton_bytes = hot_rows * substrings_alphabet_size_k * bytes_per_state_id + accepts_bytes;
        // Scoring carries its counter table in the same allocation, so staging must fit beside it or the two
        // would compete for one budget. They do not today - staging is refused for every real dictionary -
        // but that is luck rather than design, and this keeps it true by construction.
        size_t const staging_budget = shared_memory_budget > substrings_bm25_counters_bytes_()
                                          ? shared_memory_budget - substrings_bm25_counters_bytes_()
                                          : 0;
        bool const stages_whole_automaton = whole_automaton_bytes <= staging_budget;
        staged_rows_ = stages_whole_automaton ? static_cast<u32_t>(hot_rows) : u32_t {0};
        staged_accepts_words_ = stages_whole_automaton ? static_cast<u32_t>(accepts_words_.size()) : u32_t {0};

        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Everything the counting pass establishes that a following scatter pass still needs. */
    struct planned_pass_t {
        kernels_t kernel_table {};
        unsigned shared_memory_bytes = 0;
        unsigned blocks_per_grid = 0;
        size_t chunk_bytes = 0;
        size_t chunk_count = 0;
        /** @brief False when the corpus is empty, so the caller returns success without launching anything. */
        bool has_work = false;
    };

    /**
     *  @brief Copies the caller's replacements into one unified tape the kernels can address.
     *
     *  Unlike haystacks, which are validated in place, replacements arrive through a callback-addressed
     *  container in host memory, so they have to be materialized. The needle set is query-sized and this
     *  runs against a whole corpus walk, so it is re-uploaded per call rather than cached and invalidated.
     */
    template <typename replacements_type_>
    cuda_status_t upload_replacements_(replacements_type_ const &replacements) noexcept {
        size_t const needle_count = count_needles();
        if (replacements.size() != needle_count) return {status_t::unexpected_dimensions_k, cudaSuccess};

        size_t total_bytes = 0;
        for (size_t needle_index = 0; needle_index < needle_count; ++needle_index)
            total_bytes += to_bytes_view(replacements[needle_index]).size();

        if (replacement_offsets_.try_resize_uninitialized(needle_count + 1) != status_t::success_k ||
            replacement_bytes_.try_resize_uninitialized(sz_max_of_two(total_bytes, (size_t)1)) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        size_t written = 0;
        for (size_t needle_index = 0; needle_index < needle_count; ++needle_index) {
            span<byte_t const> const replacement = to_bytes_view(replacements[needle_index]);
            replacement_offsets_[needle_index] = written;
            if (replacement.size())
                sz_copy((sz_ptr_t)(replacement_bytes_.data() + written), (sz_cptr_t)replacement.data(),
                        replacement.size());
            written += replacement.size();
        }
        replacement_offsets_[needle_count] = written;
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Records one pointer-and-length descriptor per haystack, sums their bytes, and reports the
     *         longest of them - which is what bounds the widest chunk any single haystack can ask for.
     *
     *  No layout is assumed: haystacks may sit in one tape or in separate allocations, and the input bytes
     *  are validated rather than copied, as in every other CUDA engine here.
     */
    template <typename haystacks_type_>
    cuda_status_t describe_haystacks_(haystacks_type_ const &haystacks, size_t &total_bytes,
                                      size_t &longest_bytes) noexcept {
        if (haystack_descriptors_.try_resize_uninitialized(haystacks.size()) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        total_bytes = 0;
        longest_bytes = 0;

        bool probed = false;
        for (size_t haystack_index = 0; haystack_index < haystacks.size(); ++haystack_index) {
            span<byte_t const> const haystack = to_bytes_view(haystacks[haystack_index]);
            haystack_descriptors_[haystack_index] = haystack;
            total_bytes += haystack.size();
            longest_bytes = sz_max_of_two(longest_bytes, haystack.size());
            // The probe is a driver round-trip, so one non-empty element decides for the whole batch.
            if (!probed && haystack.size() != 0) {
                if (!is_device_accessible_memory((void const *)haystack.data()))
                    return {status_t::device_memory_mismatch_k, cudaSuccess};
                probed = true;
            }
        }
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Runs everything `try_count` and `try_find` share: the guards, the kernel table, the timer's
     *         start, the chunk plan, and the counting pass whose exclusive scan both then read.
     */
    cuda_status_t plan_and_count_(size_t total_bytes, cuda_executor_t const &executor, gpu_specs_t const &specs,
                                  planned_pass_t &pass) noexcept {
        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;
        if (haystack_descriptors_.size() == 0 || total_bytes == 0) return {status_t::success_k, cudaSuccess};

        // The staging budget belongs to the device this call names, so it is settled here rather than at build.
        if (cuda_status_t const budgeted = try_budget_staging_(executor); budgeted.status != status_t::success_k)
            return budgeted;

        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;
        pass.kernel_table = kernel_table;

        CUresult const timer_error = timer_.ensure_created(executor.device_id());
        if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
        CUresult const start_error = timer_.record_start(executor.stream());
        if (start_error != CUDA_SUCCESS) return make_cuda_status(start_error);

        cuda_status_t const plan_status = plan_haystack_chunks_(total_bytes, specs, pass.kernel_table,
                                                                pass.shared_memory_bytes, pass.blocks_per_grid,
                                                                pass.chunk_bytes, pass.chunk_count);
        if (plan_status.status != status_t::success_k) return plan_status;

        cuda_status_t const count_status = count_into_offsets_(executor, specs, pass.kernel_table,
                                                               pass.shared_memory_bytes, pass.blocks_per_grid,
                                                               pass.chunk_bytes, pass.chunk_count);
        if (count_status.status != status_t::success_k) return count_status;

        pass.has_work = true;
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Bytes a rewrite of @p input_bytes can reach, from the dictionary and @p replacements alone.
     *  @sa `szs_substrings_replace_bound`, whose arithmetic this mirrors.
     *
     *  A caller sizing its output span to this cannot be refused, which lets `try_replace` skip the host
     *  round-trip its capacity check would otherwise need.
     */
    template <typename replacements_type_>
    size_t replace_bound_host_(size_t input_bytes, replacements_type_ const &replacements) const noexcept {
        // The densest rewrite tiles the input with the shortest match there is and swaps each one for the
        // widest replacement there is. The bytes past the last whole match survive verbatim, so they are
        // added rather than dropped - integer division alone under-counts, and a caller sizing to an
        // under-count would be handed an overflowing write by the pre-sized path below.
        size_t widest_replacement = 0;
        for (size_t needle_index = 0; needle_index < replacements.size(); ++needle_index)
            widest_replacement = sz_max_of_two(widest_replacement, to_bytes_view(replacements[needle_index]).size());
        size_t const shortest_match = sz_max_of_two(min_source_match_bytes(), (size_t)1);
        size_t const whole_matches = input_bytes / shortest_match;

        // A dictionary that only ever shrinks still bounds at the input length, never below it.
        return sz_max_of_two(input_bytes, whole_matches * widest_replacement + input_bytes % shortest_match);
    }

    /**
     *  @brief Grid for a kernel launched with one block per work item, from that kernel's own occupancy.
     *
     *  Clamped to the item count, because a block that finds nothing to do still costs its scratch - the
     *  BM25 frequency rows are sized from this, so an unclamped grid would allocate rows nobody fills.
     */
    static unsigned grid_for_items_(kernel_shape_t const &shape, size_t items, gpu_specs_t const &specs) noexcept {
        size_t const resident = (size_t)shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
        size_t const wanted = sz_min_of_two(sz_max_of_two(resident, (size_t)1), sz_max_of_two(items, (size_t)1));
        return (unsigned)wanted;
    }

    /**
     *  @brief Refuses a chunk width whose warm-up prefix or worst-case match count outgrows @ref small_size_t.
     *
     *  The walkers carry a chunk's reach and its match count in that narrow type, so both bounds belong to the
     *  chunk width rather than to the input: haystack offsets themselves stay 64-bit. Shared by the chunked
     *  plan and by the scoring pass, which sizes its own chunks and never builds a chunk-offsets map.
     */
    cuda_status_t check_chunk_bytes_fit_(size_t chunk_bytes) const noexcept {
        size_t const longest = max_source_match_bytes();
        size_t const warm_up_bytes = longest > 0 ? longest - 1 : 0;
        if (chunk_bytes + warm_up_bytes > (size_t)std::numeric_limits<small_size_t>::max())
            return {status_t::overflow_risk_k, cudaSuccess};

        // Worst case is a repeated byte against a nested-suffix dictionary, where every position emits
        // `max_outputs_per_state` merged outputs.
        size_t const outputs_per_state = visit_dictionary([](auto const &dictionary) noexcept { //
            return (size_t)dictionary.view().max_outputs_per_state;
        });
        size_t const worst_case_matches_per_chunk = chunk_bytes * outputs_per_state;
        if (outputs_per_state != 0 && (worst_case_matches_per_chunk / outputs_per_state != chunk_bytes ||
                                       worst_case_matches_per_chunk > (size_t)std::numeric_limits<small_size_t>::max()))
            return {status_t::overflow_risk_k, cudaSuccess};
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Sizes the chunk grid from the counting kernel's occupancy under the dictionary's actual
     *         shared-memory footprint, then lays out every haystack's chunks against that target.
     *
     *  The layout is arithmetic over lengths the descriptor walk already read, so the host writes
     *  `haystack_chunk_offsets_` outright rather than counting on the device and scanning back. The array stays
     *  unified because every chunk thread reads it.
     */
    cuda_status_t plan_haystack_chunks_(size_t total_bytes, gpu_specs_t const &specs, kernels_t const &kernel_table,
                                        unsigned &shared_memory_bytes, unsigned &blocks_per_grid, size_t &chunk_bytes,
                                        size_t &chunk_count) noexcept {
        size_t const haystack_count = haystack_descriptors_.size();

        size_t const staged_bytes = (size_t)staged_rows_ * substrings_alphabet_size_k * bytes_per_state_id_() +
                                    (size_t)staged_accepts_words_ * sizeof(u32_t); // ? Settled per call
        sz_assert_(staged_bytes <= std::numeric_limits<unsigned>::max() &&
                   "The staged prefix is budgeted against one multiprocessor's shared memory in `try_build`");
        shared_memory_bytes = static_cast<unsigned>(staged_bytes);
        cuda_status_t const occupancy_status = occupancy_grid_for(
            blocks_per_grid, kernel_table.count_chunk.for_width(state_width()).function, substrings_threads_per_block_k,
            shared_memory_bytes, specs);
        if (occupancy_status.status != status_t::success_k) return occupancy_status;

        size_t const target_threads = sz_max_of_two((size_t)blocks_per_grid * substrings_threads_per_block_k,
                                                    (size_t)1);
        chunk_bytes = sz_max_of_two(divide_round_up(total_bytes, target_threads), (size_t)1);

        if (cuda_status_t const fits = check_chunk_bytes_fit_(chunk_bytes); fits.status != status_t::success_k)
            return fits;

        if (haystack_chunk_offsets_.try_resize_uninitialized(haystack_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        chunk_count = 0;
        for (size_t haystack_index = 0; haystack_index < haystack_count; ++haystack_index) {
            haystack_chunk_offsets_[haystack_index] = chunk_count;
            chunk_count += substrings_chunks_for_haystack_(haystack_descriptors_[haystack_index].size(), chunk_bytes);
        }
        haystack_chunk_offsets_[haystack_count] = chunk_count;
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Runs the counting pass, then the in-place exclusive scan, so `chunk_match_offsets_` holds every
     *         chunk's write offset with the grand total trailing at `[chunk_count]` - the shared core of
     *         `try_count` and `try_find`. */
    cuda_status_t count_into_offsets_(cuda_executor_t const &executor, gpu_specs_t const &specs,
                                      kernels_t const &kernel_table, unsigned shared_memory_bytes,
                                      unsigned blocks_per_grid, size_t chunk_bytes, size_t chunk_count) noexcept {
        if (chunk_match_offsets_.try_resize_uninitialized(chunk_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        // Sizing hands the walk no output span: the pass writes each chunk's count into its slot instead.
        CUresult const count_error = launch_walk_(substrings_pass_t::sizing_k, kernel_table, blocks_per_grid,
                                                  shared_memory_bytes, chunk_bytes, chunk_count, span<match_t> {},
                                                  executor);
        if (count_error != CUDA_SUCCESS) return make_cuda_status(count_error);

        // In place: each chunk's slot holds its raw count going in and its exclusive offset coming out - the
        // scan kernel reads `input[i]` into a register before any thread writes `output[i]`, so reusing one
        // buffer for both is safe and skips a second chunk_count-sized allocation.
        return cuda_launch_exclusive_sum_(kernel_table.exclusive_sum, chunk_match_offsets_.data(), chunk_count,
                                          chunk_match_offsets_.data(), {scan_partials_.data(), scan_partials_.size()},
                                          specs, executor.stream());
    }

    /**
     *  @brief Lays each haystack's match range onto the boundaries its reported matches occupy.
     *  @param[in] keep_offsets The cover's scanned keep flags, or empty when every emitted match is reported.
     */
    cuda_status_t publish_haystack_match_offsets_(planned_pass_t const &pass, span<size_t const> keep_offsets,
                                                  size_t haystack_count, cuda_executor_t const &executor,
                                                  gpu_specs_t const &specs) noexcept {
        if (haystack_match_offsets_.try_resize_uninitialized(haystack_count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        span<size_t const> chunk_match_offsets_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        span<size_t const> keep_offsets_argument = keep_offsets;
        span<size_t> boundaries_argument {haystack_match_offsets_.data(), haystack_count + 1};
        void *boundary_arguments[4] = {&haystack_chunk_offsets_argument, &chunk_match_offsets_argument,
                                       &keep_offsets_argument, &boundaries_argument};
        unsigned const boundary_grid = grid_for_items_(pass.kernel_table.haystack_match_offsets, haystack_count + 1,
                                                       specs);
        CUresult const boundary_error = cuda_launch_t {}
                                            .grid(boundary_grid)
                                            .block(substrings_threads_per_block_k)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .launch(pass.kernel_table.haystack_match_offsets.function,
                                                    boundary_arguments);
        if (boundary_error != CUDA_SUCCESS) return make_cuda_status(boundary_error);
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Fences, then reads the rewritten tape's length out of the caller's own offsets array.
     *
     *  The trailing boundary is the total, and the array may be plain device memory, so it comes back through
     *  a driver copy rather than a dereference.
     */
    cuda_status_t read_rewritten_bytes_(span<size_t> output_offsets, size_t haystack_count,
                                        cuda_executor_t const &executor, size_t &rewritten_bytes) noexcept {
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        CUresult const read_error = cuMemcpyDtoH(&rewritten_bytes,
                                                 (CUdeviceptr)(output_offsets.data() + haystack_count), sizeof(size_t));
        if (read_error != CUDA_SUCCESS) return make_cuda_status(read_error);
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Zeroes the caller's counts through the driver, for a corpus the walk never reaches. */
    cuda_status_t clear_counts_(span<size_t> counts_per_haystack, cuda_executor_t const &executor) noexcept {
        if (counts_per_haystack.size() == 0) return {status_t::success_k, cudaSuccess};
        CUresult const clear_error = cuMemsetD8Async((CUdeviceptr)counts_per_haystack.data(), 0,
                                                     counts_per_haystack.size() * sizeof(size_t), executor.stream());
        if (clear_error != CUDA_SUCCESS) return make_cuda_status(clear_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Zeroes the caller's scores through the driver, for a batch no walk reaches. */
    cuda_status_t clear_scores_(span<f32_t> scores, cuda_executor_t const &executor) noexcept {
        if (scores.size() == 0) return {status_t::success_k, cudaSuccess};
        CUresult const clear_error = cuMemsetD8Async((CUdeviceptr)scores.data(), 0, scores.size() * sizeof(f32_t),
                                                     executor.stream());
        if (clear_error != CUDA_SUCCESS) return make_cuda_status(clear_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess};
    }

    /** @brief Differences the published boundaries into the caller's per-haystack counts, then fences once. */
    cuda_status_t count_from_boundaries_(planned_pass_t const &pass, span<size_t> counts_per_haystack,
                                         cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        span<size_t const> boundaries_argument {haystack_match_offsets_.data(), haystack_match_offsets_.size()};
        span<size_t> counts_argument = counts_per_haystack;
        void *counts_arguments[2] = {&boundaries_argument, &counts_argument};
        unsigned const counts_grid = grid_for_items_(pass.kernel_table.counts_from_boundaries,
                                                     counts_per_haystack.size(), specs);
        CUresult const counts_error = cuda_launch_t {}
                                          .grid(counts_grid)
                                          .block(substrings_threads_per_block_k)
                                          .shared(0)
                                          .stream(executor.stream())
                                          .launch(pass.kernel_table.counts_from_boundaries.function, counts_arguments);
        if (counts_error != CUDA_SUCCESS) return make_cuda_status(counts_error);

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

    /**
     *  @brief Resolves a leftmost cover over an already-emitted match list, and compacts the survivors.
     *
     *  The walk emits every match, which is the only thing it is fast at; deciding between them is a pass
     *  over a few million records rather than a few dozen million bytes, and it needs no per-thread ring and
     *  no second walk to find a safe cursor.
     *
     *  @param[in,out] matches In: every match, ascending by haystack and end. Out: the survivors, in place.
     *  @param[out] surviving How many survived, which is what sizes everything downstream.
     */
    cuda_status_t resolve_cover_(planned_pass_t const &pass, span<match_t> matches,
                                 substrings_overlap_policy_t overlap_policy, size_t haystack_count,
                                 cuda_executor_t const &executor, gpu_specs_t const &specs,
                                 size_t &surviving) noexcept {

        size_t const emitted = matches.size();
        surviving = emitted;
        if (overlap_policy == substrings_overlapping_k) return {status_t::success_k, cudaSuccess};

        // A corpus nothing matched still owes its caller a boundary per haystack; an empty keep span makes the
        // boundary kernel the identity over the emitted offsets, which are all zero in that case.
        if (emitted == 0) {
            surviving = 0;
            return publish_haystack_match_offsets_(pass, {}, haystack_count, executor, specs);
        }

        if (cover_keep_.try_resize_uninitialized(emitted + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        span<match_t const> matches_argument {matches.data(), emitted};
        size_t longest_argument = max_source_match_bytes();
        substrings_overlap_policy_t policy_argument = overlap_policy;
        span<size_t> keep_argument {cover_keep_.data(), emitted};
        void *resolve_arguments[4] = {&matches_argument, &longest_argument, &policy_argument, &keep_argument};
        unsigned const resolve_grid = grid_for_items_(pass.kernel_table.cover_resolve, emitted, specs);
        CUresult const resolve_error = cuda_launch_t {}
                                           .grid(resolve_grid)
                                           .block(substrings_threads_per_block_k)
                                           .shared(0)
                                           .stream(executor.stream())
                                           .launch(pass.kernel_table.cover_resolve.function, resolve_arguments);
        if (resolve_error != CUDA_SUCCESS) return make_cuda_status(resolve_error);

        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(
            pass.kernel_table.exclusive_sum, cover_keep_.data(), emitted, cover_keep_.data(),
            {scan_partials_.data(), scan_partials_.size()}, specs, executor.stream());
        if (scan_status.status != status_t::success_k) return scan_status;

        if (cuda_status_t const boundary_status = publish_haystack_match_offsets_(
                pass, {cover_keep_.data(), emitted + 1}, haystack_count, executor, specs);
            boundary_status.status != status_t::success_k)
            return boundary_status;

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        // The scan lives on the device, where it belongs - only its last element is the host's business.
        CUresult const total_error = cuMemcpyDtoH(&surviving, (CUdeviceptr)(cover_keep_.data() + emitted),
                                                  sizeof(size_t));
        if (total_error != CUDA_SUCCESS) return make_cuda_status(total_error);
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Gathers the cover's survivors into @p destination, which the caller sizes and owns.
     *
     *  Separate from `resolve_cover_` because the three callers want them in three different places:
     *  counting wants them nowhere, finding wants them in the caller's own span, and rewriting wants them in
     *  scratch its kernels read. Fusing this into the resolve would cost finding a whole extra copy.
     */
    CUresult compact_cover_(planned_pass_t const &pass, span<match_t const> matches, span<match_t> destination,
                            cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        span<size_t const> keep_argument {cover_keep_.data(), matches.size() + 1};
        span<match_t const> matches_argument = matches;
        span<match_t> destination_argument = destination;
        void *arguments[3] = {&matches_argument, &keep_argument, &destination_argument};
        return cuda_launch_t {}
            .grid(grid_for_items_(pass.kernel_table.cover_compact, matches.size(), specs))
            .block(substrings_threads_per_block_k)
            .shared(0)
            .stream(executor.stream())
            .launch(pass.kernel_table.cover_compact.function, arguments);
    }

    /**
     *  @brief Runs the scattering pass into @p target, which the counting pass has already sized and scanned.
     *
     *  Shared by all three entry points, so the ten arguments are written out once and stay in one order.
     */
    CUresult launch_scatter_(planned_pass_t const &pass, span<match_t> target,
                             cuda_executor_t const &executor) noexcept {
        // Bounded to what the counting pass actually found, not to the caller's capacity, so a debug index
        // assert inside the kernel catches an over-write rather than merely staying inside the allocation.
        return launch_walk_(substrings_pass_t::writing_k, pass.kernel_table, pass.blocks_per_grid,
                            pass.shared_memory_bytes, pass.chunk_bytes, pass.chunk_count, target, executor);
    }

    /**
     *  @brief Launches one walk over every chunk, in whichever pass @p pass_kind names.
     *
     *  Both passes take the same ten arguments in the same order - the counting one simply leaves @p target
     *  empty and tallies into the chunk slots - so the argument block is written here once.
     */
    CUresult launch_walk_(substrings_pass_t pass_kind, kernels_t const &kernel_table, unsigned blocks_per_grid,
                          unsigned shared_memory_bytes, size_t chunk_bytes, size_t chunk_count, span<match_t> target,
                          cuda_executor_t const &executor) noexcept {
        return state_width() == substrings_state_width_t::u16_k
                   ? launch_walk_at_<u16_t>(pass_kind, kernel_table, blocks_per_grid, shared_memory_bytes, chunk_bytes,
                                            chunk_count, target, executor)
                   : launch_walk_at_<u32_t>(pass_kind, kernel_table, blocks_per_grid, shared_memory_bytes, chunk_bytes,
                                            chunk_count, target, executor);
    }

    template <typename state_id_type_>
    CUresult launch_walk_at_(substrings_pass_t pass_kind, kernels_t const &kernel_table, unsigned blocks_per_grid,
                             unsigned shared_memory_bytes, size_t chunk_bytes, size_t chunk_count, span<match_t> target,
                             cuda_executor_t const &executor) noexcept {
        aho_corasick_view<state_id_type_> view_argument = settled_dictionary_<state_id_type_>().view();
        state_id_type_ staged_rows_argument = static_cast<state_id_type_>(staged_rows_);
        span<u32_t const> accepts_words_argument {accepts_words_.data(), accepts_words_.size()};
        u32_t staged_accepts_words_argument = staged_accepts_words_;
        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        span<size_t const> haystack_chunk_offsets_argument {haystack_chunk_offsets_.data(),
                                                            haystack_chunk_offsets_.size()};
        size_t chunk_bytes_argument = chunk_bytes;
        size_t chunk_count_argument = chunk_count;
        span<size_t> chunk_match_slots_argument {chunk_match_offsets_.data(), chunk_match_offsets_.size()};
        span<match_t> matches_out_argument = target;
        void *walk_arguments[10] = {&view_argument,
                                    &staged_rows_argument,
                                    &accepts_words_argument,
                                    &staged_accepts_words_argument,
                                    &haystacks_argument,
                                    &haystack_chunk_offsets_argument,
                                    &chunk_bytes_argument,
                                    &chunk_count_argument,
                                    &chunk_match_slots_argument,
                                    &matches_out_argument};
        auto const &shapes = pass_kind == substrings_pass_t::sizing_k ? kernel_table.count_chunk
                                                                      : kernel_table.scatter_chunk;
        return cuda_launch_t {}
            .grid(blocks_per_grid)
            .block(substrings_threads_per_block_k)
            .shared(shared_memory_bytes)
            .stream(executor.stream())
            .launch(shapes.for_width(state_width()).function, walk_arguments);
    }

  public:
    /**
     *  @brief Occurrences of all needles in each of the @p haystacks, for filtering and ranking.
     *  @param[in] haystacks Device-accessible, contiguously laid out; no needle is ever reported straddling
     *             two of them.
     *  @param[out] matches_total Sum of @p counts_per_haystack, which is what sizes a later `try_find` buffer.
     *
     *  Under `substrings_overlapping_k` this is strictly cheaper than `try_find` - the plan and the counting
     *  pass, no scatter - because chunks never cross a haystack boundary and the breakdown is a subtraction
     *  over the counting pass's offsets. A cover costs the same as finding one: the walk cannot know which
     *  matches survive, so they have to be emitted before anything can be counted.
     */
    template <typename haystacks_type_>
    cuda_status_t try_count(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                            span<size_t> counts_per_haystack, size_t &matches_total,
                            cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        matches_total = 0;
        if (counts_per_haystack.size() != haystacks.size()) return {status_t::unexpected_dimensions_k, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(counts_per_haystack);
            reachable != status_t::success_k)
            return {reachable, cudaSuccess};

        size_t total_bytes = 0;
        [[maybe_unused]] size_t longest_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes, longest_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;

        return count_described_(haystacks.size(), total_bytes, overlap_policy, counts_per_haystack, matches_total,
                                executor, specs);
    }

    /**
     *  @brief Everything `try_count` does once the haystacks are described, with the container behind it.
     *
     *  The descriptors are the type-erasure boundary, so the work below compiles once rather than once per
     *  input shape. @sa `describe_haystacks_`.
     */
    cuda_status_t count_described_(size_t haystack_count, size_t total_bytes,
                                   substrings_overlap_policy_t overlap_policy, span<size_t> counts_per_haystack,
                                   size_t &matches_total, cuda_executor_t const &executor, gpu_specs_t specs) noexcept {
        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(total_bytes, executor, specs, pass);
        if (pass_status.status != status_t::success_k) return pass_status;
        // A corpus with nothing to walk still owes its caller a count per haystack, all of them zero, and the
        // caller's span may be plain device memory - so the driver clears it rather than a host loop.
        if (!pass.has_work) return clear_counts_(counts_per_haystack, executor);

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_total = chunk_match_offsets_[pass.chunk_count];
        if (overlap_policy == substrings_overlapping_k) {
            if (cuda_status_t const boundary_status = publish_haystack_match_offsets_(pass, {}, haystack_count,
                                                                                      executor, specs);
                boundary_status.status != status_t::success_k)
                return boundary_status;
        }
        else {
            // A cover is decided between matches, so counting one means emitting them first - the walk cannot
            // know which survive. That makes a counted cover cost what a found one does.
            if (emitted_matches_.try_resize_uninitialized(sz_max_of_two(matches_total, (size_t)1)) !=
                status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            CUresult const scatter_error = launch_scatter_(pass, {emitted_matches_.data(), matches_total}, executor);
            if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

            // No gather: a count wants the boundaries, and those the resolve already wrote.
            cuda_status_t const cover_status = resolve_cover_(pass, {emitted_matches_.data(), matches_total},
                                                              overlap_policy, haystack_count, executor, specs,
                                                              matches_total);
            if (cover_status.status != status_t::success_k) return cover_status;
        }

        return count_from_boundaries_(pass, counts_per_haystack, executor, specs);
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks: count, then scatter every match
     *         at its chunk's precomputed offset without atomics, then - under a cover - resolve and gather.
     *  @param[out] matches_found Matches written, in ascending haystack order.
     *  @retval `status_t::unexpected_dimensions_k` @p matches_out is too small; nothing is written in that
     *          case. See `try_count` for the @p haystacks contract.
     */
    template <typename haystacks_type_>
    cuda_status_t try_find(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                           span<match_t> matches_out, size_t &matches_found, cuda_executor_t const &executor = {},
                           gpu_specs_t specs = {}) noexcept {
        matches_found = 0;
        if (status_t const reachable = check_device_accessible_memory(matches_out); reachable != status_t::success_k)
            return {reachable, cudaSuccess};

        size_t total_bytes = 0;
        [[maybe_unused]] size_t longest_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes, longest_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;

        return find_described_(haystacks.size(), total_bytes, overlap_policy, matches_out, matches_found, executor,
                               specs);
    }

    /** @brief Everything `try_find` does once the haystacks are described. @sa `count_described_`. */
    cuda_status_t find_described_(size_t haystack_count, size_t total_bytes, substrings_overlap_policy_t overlap_policy,
                                  span<match_t> matches_out, size_t &matches_found, cuda_executor_t const &executor,
                                  gpu_specs_t specs) noexcept {
        size_t const matches_capacity = matches_out.size();

        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(total_bytes, executor, specs, pass);
        if (pass_status.status != status_t::success_k || !pass.has_work) return pass_status;

        // The emitted total is only host-visible after a fence, and it sizes the scratch the walk writes to.
        CUresult const mid_sync_error = timer_.synchronize(executor.stream());
        if (mid_sync_error != CUDA_SUCCESS) return make_cuda_status(mid_sync_error);
        size_t const emitted = chunk_match_offsets_[pass.chunk_count];
        bool const covering = overlap_policy != substrings_overlapping_k;

        // Under a cover the walk's output is an intermediate, so it lands in scratch and only the survivors
        // reach the caller. Without one it is the answer, and the caller's span takes it directly.
        size_t matches_in_batch = emitted;
        span<match_t> scatter_target;
        if (covering) {
            if (emitted_matches_.try_resize_uninitialized(sz_max_of_two(emitted, (size_t)1)) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            scatter_target = {emitted_matches_.data(), emitted};
        }
        else {
            if (emitted > matches_capacity)
                return matches_found = emitted, cuda_status_t {status_t::unexpected_dimensions_k, cudaSuccess};
            scatter_target = {matches_out.data(), emitted};
        }

        CUresult const scatter_error = launch_scatter_(pass, scatter_target, executor);
        if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

        if (covering) {
            cuda_status_t const cover_status = resolve_cover_(pass, scatter_target, overlap_policy, haystack_count,
                                                              executor, specs, matches_in_batch);
            if (cover_status.status != status_t::success_k) return cover_status;
            // The survivors' count survives the refusal, so a caller that brought no buffer learns its size.
            if (matches_in_batch > matches_capacity)
                return matches_found = matches_in_batch, cuda_status_t {status_t::unexpected_dimensions_k, cudaSuccess};

            if (matches_in_batch) {
                CUresult const gather_error = compact_cover_(pass, {scatter_target.data(), emitted},
                                                             {matches_out.data(), matches_in_batch}, executor, specs);
                if (gather_error != CUDA_SUCCESS) return make_cuda_status(gather_error);
            }
        }

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);

        matches_found = matches_in_batch;
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

    /**
     *  @brief Rewrites every haystack into one tape, substituting each match with its needle's replacement.
     *  @param[in] overlap_policy Must name a leftmost policy; an overlapping rewrite is not a function.
     *  @param[in] replacements One per needle, inserted verbatim.
     *  @param[out] output_offsets Rewritten boundaries, `haystacks.size() + 1` entries; always filled.
     *  @param[out] output_bytes_written Bytes written, or - when @p output_bytes is short - the size needed.
     *  @retval `status_t::unexpected_dimensions_k` @p output_bytes is too small, and nothing was written.
     *
     *  Where the host walks each haystack twice, once to size and once to write, the device walks once and
     *  keeps the matches: the automaton is latency-bound on dependent transition loads, while everything
     *  after it is bandwidth. The scratch that buys is about 96 bytes per match, so a corpus matching every
     *  tenth byte should be handed over in batches.
     *
     *  Sizing @p output_bytes to `szs_substrings_replace_bound` makes the capacity check unfailable, which
     *  drops the host round-trip it would otherwise need; a device-accessible @p output_bytes then makes the
     *  whole call one uninterrupted stream of launches.
     */
    template <typename haystacks_type_, typename replacements_type_>
    cuda_status_t try_replace(haystacks_type_ const &haystacks, substrings_overlap_policy_t overlap_policy,
                              replacements_type_ const &replacements, span<char> output_bytes,
                              span<size_t> output_offsets, size_t &output_bytes_written,
                              cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        output_bytes_written = 0;
        if (output_offsets.size() != haystacks.size() + 1) return {status_t::unexpected_dimensions_k, cudaSuccess};
        if (status_t const rewritable = substrings_check_rewritable(overlap_policy); rewritable != status_t::success_k)
            return {rewritable, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(output_offsets); reachable != status_t::success_k)
            return {reachable, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(output_bytes); reachable != status_t::success_k)
            return {reachable, cudaSuccess};

        size_t total_bytes = 0;
        [[maybe_unused]] size_t longest_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes, longest_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;
        cuda_status_t const upload_status = upload_replacements_(replacements);
        if (upload_status.status != status_t::success_k) return upload_status;

        // The bound is arithmetic over the needle set, so it is settled here and the rewrite below never
        // names either container again.
        return replace_described_(haystacks.size(), total_bytes, replace_bound_host_(total_bytes, replacements),
                                  overlap_policy, output_bytes, output_offsets, output_bytes_written, executor, specs);
    }

    /** @brief Everything `try_replace` does once the haystacks and replacements are staged. @sa `count_described_`. */
    cuda_status_t replace_described_(size_t haystack_count, size_t total_bytes, size_t bound,
                                     substrings_overlap_policy_t overlap_policy, span<char> output_bytes,
                                     span<size_t> output_offsets, size_t &output_bytes_written,
                                     cuda_executor_t const &executor, gpu_specs_t specs) noexcept {
        planned_pass_t pass;
        cuda_status_t const pass_status = plan_and_count_(total_bytes, executor, specs, pass);
        if (pass_status.status != status_t::success_k) return pass_status;
        // A corpus with nothing to rewrite still owes its caller a boundary per haystack, all of them zero.
        if (!pass.has_work) return clear_counts_({output_offsets.data(), output_offsets.size()}, executor);

        // The match count sizes three buffers, so it has to reach the host before they can be allocated.
        CUresult const count_sync_error = timer_.synchronize(executor.stream());
        if (count_sync_error != CUDA_SUCCESS) return make_cuda_status(count_sync_error);

        size_t const matches_in_batch = chunk_match_offsets_[pass.chunk_count];
        if (emitted_matches_.try_resize_uninitialized(sz_max_of_two(matches_in_batch, (size_t)1)) !=
                status_t::success_k ||
            rewrite_gap_offsets_.try_resize_uninitialized(sz_max_of_two(matches_in_batch, (size_t)1)) !=
                status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        CUresult const scatter_error = launch_scatter_(pass, {emitted_matches_.data(), matches_in_batch}, executor);
        if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

        size_t surviving = 0;
        cuda_status_t const cover_status = resolve_cover_(pass, {emitted_matches_.data(), matches_in_batch},
                                                          overlap_policy, haystack_count, executor, specs, surviving);
        if (cover_status.status != status_t::success_k) return cover_status;
        if (cover_survivors_.try_resize_uninitialized(sz_max_of_two(surviving, (size_t)1)) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (surviving) {
            CUresult const gather_error = compact_cover_(pass, {emitted_matches_.data(), matches_in_batch},
                                                         {cover_survivors_.data(), surviving}, executor, specs);
            if (gather_error != CUDA_SUCCESS) return make_cuda_status(gather_error);
        }

        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        span<size_t const> match_offsets_argument {haystack_match_offsets_.data(), haystack_count + 1};
        span<match_t const> matches_argument {cover_survivors_.data(), surviving};
        span<size_t const> replacement_offsets_argument {replacement_offsets_.data(), replacement_offsets_.size()};
        span<size_t> gap_offsets_argument {rewrite_gap_offsets_.data(), surviving};
        span<size_t> output_sizes_argument {output_offsets.data(), haystack_count};
        void *offsets_arguments[6] = {&haystacks_argument,           &match_offsets_argument, &matches_argument,
                                      &replacement_offsets_argument, &gap_offsets_argument,   &output_sizes_argument};
        unsigned const offsets_grid = grid_for_items_(pass.kernel_table.rewrite_offsets, haystack_count, specs);
        CUresult const offsets_error = cuda_launch_t {}
                                           .grid(offsets_grid)
                                           .block(substrings_threads_per_block_k)
                                           .shared(0)
                                           .stream(executor.stream())
                                           .launch(pass.kernel_table.rewrite_offsets.function, offsets_arguments);
        if (offsets_error != CUDA_SUCCESS) return make_cuda_status(offsets_error);

        // The scan writes the caller's own array, so the boundaries are complete before any capacity check -
        // which is what lets a refused call name the exact size it wanted.
        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(
            pass.kernel_table.exclusive_sum, output_offsets.data(), haystack_count, output_offsets.data(),
            {scan_partials_.data(), scan_partials_.size()}, specs, executor.stream());
        if (scan_status.status != status_t::success_k) return scan_status;

        // A caller sized against the dictionary's own bound cannot be refused, so neither the check nor the
        // fence it needs happens at all - the copy kernel reads the tape's length from the scan itself.
        bool const pre_sized = output_bytes.size() >= bound;
        size_t rewritten_bytes = 0;
        if (!pre_sized) {
            cuda_status_t const size_status = read_rewritten_bytes_(output_offsets, haystack_count, executor,
                                                                    rewritten_bytes);
            if (size_status.status != status_t::success_k) return size_status;
            if (rewritten_bytes > output_bytes.size())
                return output_bytes_written = rewritten_bytes,
                       cuda_status_t {status_t::unexpected_dimensions_k, cudaSuccess};
        }

        size_t const copy_bytes = pre_sized ? bound : rewritten_bytes;
        char *copy_target = output_bytes.data();

        span<size_t const> gap_offsets_const_argument {rewrite_gap_offsets_.data(), surviving};
        byte_t const *replacement_bytes_argument = replacement_bytes_.data();
        span<size_t const> output_offsets_argument {output_offsets.data(), haystack_count + 1};
        size_t tile_bytes_argument = substrings_rewrite_tile_bytes_k;
        void *copy_arguments[9] = {
            &haystacks_argument,         &match_offsets_argument,     &matches_argument,
            &gap_offsets_const_argument, &replacement_bytes_argument, &replacement_offsets_argument,
            &output_offsets_argument,    &tile_bytes_argument,        &copy_target};
        unsigned const copy_grid = grid_for_items_(
            pass.kernel_table.rewrite_copy,
            divide_round_up(sz_max_of_two(copy_bytes, (size_t)1), substrings_rewrite_tile_bytes_k), specs);

        CUresult const copy_error = cuda_launch_t {}
                                        .grid(copy_grid)
                                        .block(substrings_threads_per_block_k)
                                        .shared(0)
                                        .stream(executor.stream())
                                        .launch(pass.kernel_table.rewrite_copy.function, copy_arguments);
        if (copy_error != CUDA_SUCCESS) return make_cuda_status(copy_error);

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);

        // A pre-sized caller never fetched the tape's length, so it is read past the one fence this call
        // always pays - the same fence that makes the caller's offsets readable.
        cuda_status_t const size_status = read_rewritten_bytes_(output_offsets, haystack_count, executor,
                                                                rewritten_bytes);
        if (size_status.status != status_t::success_k) return size_status;

        output_bytes_written = rewritten_bytes;
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

    /**
     *  @brief Scores every haystack against the compiled needle set in one walk.
     *  @param[in] document_lengths One per haystack; an empty span uses byte lengths.
     *  @param[in] needle_weights One IDF or boost per needle.
     *  @param[out] scores One per haystack.
     *
     *  One launch and one synchronize: nothing here is sized by a device result, so unlike `try_find` and
     *  `try_replace` this never stalls mid-call. Scores are bit-stable run to run because the block sums
     *  fixed-point integers, and integer addition is associative - no grid size, lane order or scheduling
     *  order can perturb the total.
     */
    template <typename haystacks_type_>
    cuda_status_t try_score_bm25(haystacks_type_ const &haystacks, span<f32_t const> document_lengths,
                                 substrings_bm25_t parameters, span<f32_t const> needle_weights, span<f32_t> scores,
                                 cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        size_t total_bytes = 0, longest_bytes = 0;
        cuda_status_t const describe_status = describe_haystacks_(haystacks, total_bytes, longest_bytes);
        if (describe_status.status != status_t::success_k) return describe_status;
        return score_bm25_described_(haystacks.size(), total_bytes, longest_bytes, document_lengths, parameters,
                                     needle_weights, scores, executor, specs);
    }

    /** @brief Everything `try_score_bm25` does once the haystacks are described. @sa `count_described_`. */
    cuda_status_t score_bm25_described_(size_t haystack_count, size_t total_bytes, size_t longest_bytes,
                                        span<f32_t const> document_lengths, substrings_bm25_t parameters,
                                        span<f32_t const> needle_weights, span<f32_t> scores,
                                        cuda_executor_t const &executor, gpu_specs_t specs) noexcept {
        size_t const needle_count = needle_weights.size();
        if (needle_count != count_needles() || scores.size() != haystack_count ||
            (document_lengths.size() != 0 && document_lengths.size() != haystack_count))
            return {status_t::unexpected_dimensions_k, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(needle_weights); reachable != status_t::success_k)
            return {reachable, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(document_lengths);
            reachable != status_t::success_k)
            return {reachable, cudaSuccess};
        if (status_t const reachable = check_device_accessible_memory(scores); reachable != status_t::success_k)
            return {reachable, cudaSuccess};

        cuda_status_t const current_status = executor.ensure_current();
        if (current_status.status != status_t::success_k) return current_status;
        // A dictionary with no needles scores every haystack zero, and the caller's span may be plain device
        // memory, so the driver clears it rather than a host loop.
        if (haystack_count == 0 || needle_count == 0) return clear_scores_(scores, executor);
        // Scoring bypasses `plan_and_count_`, so it budgets its own staging against this call's device.
        if (cuda_status_t const budgeted = try_budget_staging_(executor); budgeted.status != status_t::success_k)
            return budgeted;
        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;

        // The counter table shares the dynamic allocation with the staged automaton, so it must be counted
        // before the occupancy query - that query settles `blocks_per_grid`, which sizes the overflow rows.
        size_t const staged_bytes = (size_t)staged_rows_ * substrings_alphabet_size_k * bytes_per_state_id_() +
                                    (size_t)staged_accepts_words_ * sizeof(u32_t) +
                                    substrings_bm25_counters_bytes_(); // ? Settled per call
        sz_assert_(staged_bytes <= std::numeric_limits<unsigned>::max() &&
                   "A block's shared footprint must fit the launch parameter");
        unsigned const shared_memory_bytes = static_cast<unsigned>(staged_bytes);
        unsigned blocks_per_grid = 0;
        cuda_status_t const occupancy_status = occupancy_grid_for(
            blocks_per_grid, kernel_table.score_bm25.for_width(state_width()).function, substrings_threads_per_block_k,
            shared_memory_bytes, specs);
        if (occupancy_status.status != status_t::success_k) return occupancy_status;
        // One block owns one haystack, and each block owns an overflow row, so a grid wider than the corpus
        // buys nothing and costs `needle_count` counters per surplus block.
        blocks_per_grid = (unsigned)sz_min_of_two((size_t)blocks_per_grid, sz_max_of_two(haystack_count, (size_t)1));

        // Each block splits the one haystack it owns, so the chunk width is derived per haystack in the kernel.
        // Only the widest one any haystack can ask for has to clear the fit test, and that is the longest
        // haystack's share of a block.
        size_t const widest_chunk_bytes = sz_max_of_two(
            divide_round_up(longest_bytes, (size_t)substrings_threads_per_block_k),
            sz_max_of_two(max_source_match_bytes(), (size_t)1));
        if (cuda_status_t const fits = check_chunk_bytes_fit_(widest_chunk_bytes); fits.status != status_t::success_k)
            return fits;

        // A dictionary the table can hold never overflows, so most calls allocate nothing at all here.
        size_t const overflow_total = needle_count > substrings_bm25_slots_k ? (size_t)blocks_per_grid * needle_count
                                                                             : 0;
        if (bm25_overflow_.try_resize_uninitialized(overflow_total) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        span<f32_t const> weights_argument = needle_weights;
        span<f32_t const> lengths_argument = document_lengths;
        span<f32_t> scores_argument = scores;

        CUresult const timer_error = timer_.ensure_created(executor.device_id());
        if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
        CUresult const start_error = timer_.record_start(executor.stream());
        if (start_error != CUDA_SUCCESS) return make_cuda_status(start_error);

        CUresult const score_error =
            state_width() == substrings_state_width_t::u16_k
                ? launch_score_bm25_at_<u16_t>(kernel_table, blocks_per_grid, shared_memory_bytes, parameters,
                                               lengths_argument, weights_argument, scores_argument, executor)
                : launch_score_bm25_at_<u32_t>(kernel_table, blocks_per_grid, shared_memory_bytes, parameters,
                                               lengths_argument, weights_argument, scores_argument, executor);
        if (score_error != CUDA_SUCCESS) return make_cuda_status(score_error);

        CUresult const stop_error = timer_.record_stop(executor.stream());
        if (stop_error != CUDA_SUCCESS) return make_cuda_status(stop_error);

        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
    }

  private:
    /** @brief The scoring launch, at the width the automaton settled on; the ten arguments stay in one order. */
    template <typename state_id_type_>
    CUresult launch_score_bm25_at_(kernels_t const &kernel_table, unsigned blocks_per_grid,
                                   unsigned shared_memory_bytes, substrings_bm25_t parameters,
                                   span<f32_t const> lengths_argument, span<f32_t const> weights_argument,
                                   span<f32_t> scores_argument, cuda_executor_t const &executor) noexcept {
        aho_corasick_view<state_id_type_> view_argument = settled_dictionary_<state_id_type_>().view();
        state_id_type_ staged_rows_argument = static_cast<state_id_type_>(staged_rows_);
        span<u32_t const> accepts_words_argument {accepts_words_.data(), accepts_words_.size()};
        u32_t staged_accepts_words_argument = staged_accepts_words_;
        span<span<byte_t const> const> haystacks_argument {haystack_descriptors_.data(), haystack_descriptors_.size()};
        substrings_bm25_t parameters_argument = parameters;
        span<u32_t> overflow_argument {bm25_overflow_.data(), bm25_overflow_.size()};
        void *score_arguments[10] = {
            &view_argument,      &staged_rows_argument, &accepts_words_argument, &staged_accepts_words_argument,
            &haystacks_argument, &lengths_argument,     &parameters_argument,    &weights_argument,
            &overflow_argument,  &scores_argument};
        return cuda_launch_t {}
            .grid(blocks_per_grid)
            .block(substrings_threads_per_block_k)
            .shared(shared_memory_bytes)
            .stream(executor.stream())
            .launch(kernel_table.score_bm25.for_width(state_width()).function, score_arguments);
    }
};

using substrings_cuda_t = substrings_cuda<unified_alloc_t, sz_cap_cuda_k>;

#pragma endregion Engine

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SUBSTRINGS_CUDA_CUH_
