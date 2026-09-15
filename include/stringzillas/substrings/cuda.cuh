/**
 *  @brief  CUDA backend for multi-pattern exact and case-folded substring search.
 *  @file   include/stringzillas/substrings/cuda.cuh
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/substrings/serial.hpp
 *
 *  The automaton is derived here rather than uploaded: `aho_corasick_cuda_builder` runs the whole of
 *  Aho-Corasick construction in kernels and publishes an `aho_corasick_view`, the same contract the host
 *  dictionary publishes and the only thing the walks below consume.
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
#include "stringzillas/substrings/serial.hpp" // `aho_corasick_view`, the contract both backends publish

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

/** Block size every `substrings_cuda` kernel launches with; occupancy is shared-memory-bound, not thread-count-bound,
 *  so a modest fixed block keeps the launch geometry simple. */
static constexpr unsigned substrings_threads_per_block_k = 256;

/**
 *  @brief How far beyond the published bound the cold tier's `base`, `check`, `fail`, `outputs_counts` and
 *         `outputs_offsets` arrays extend.
 *
 *  A cold transition's target is `base[state] + byte` for `byte` in `[0, 256)`, and the builder guarantees
 *  `base[state] < state_count`, so the highest slot ever addressed is `state_count + 254`.
 */
static constexpr size_t substrings_cold_slot_headroom_k = substrings_alphabet_size_k - 1;

/** Candidates one thread will scan quadratically before a segment falls back to emitted order. */
static constexpr size_t substrings_cover_segment_limit_k = 4096;

/** Output bytes one block of a rewrite's copy owns, so no block's work scales with a run's width. */
static constexpr size_t substrings_rewrite_tile_bytes_k = 4096;

/** Counter slots a scoring block keeps in shared memory, sized by the document rather than by the dictionary. 64 KiB
 *  is the largest power of two that still leaves two blocks resident per multiprocessor, and it seats the distinct
 *  needles of a 40 KiB document below half load. */
static constexpr size_t substrings_bm25_slots_k = 8192;

/** Probe distance a scoring insert gives up at, spilling to the overflow row. */
static constexpr size_t substrings_bm25_probes_k = 16;

/** Fractional bits a block's running score carries, leaving three orders of headroom over the largest score a unit-
 *  weighted full vocabulary reaches. */
static constexpr int substrings_bm25_scale_k = 32;

/** A counter slot no needle has claimed. Needle indices are dense from zero, so the top value is never one of them. */
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

/** The same reduction against a block's own shared memory, for counters that never leave it. */
SZ_DEVICE_INLINE void cuda_increment_shared_(u32_t *counter, u32_t addend) noexcept {
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"((u32_t)__cvta_generic_to_shared(counter)), "r"(addend)
                 : "memory");
}

SZ_DEVICE_INLINE void cuda_increment_shared_(u64_t *counter, u64_t addend) noexcept {
    asm volatile("red.shared.add.u64 [%0], %1;" ::"r"((u32_t)__cvta_generic_to_shared(counter)), "l"(addend)
                 : "memory");
}

/** Where a counting walk puts its counts: the block's own table first, the overflow row when full. An empty @p
 *  overflow means the dictionary fits the table, which is what lifts the probe bound. */
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

/** One contribution as a fixed-point integer. The scaling runs in double so the `f32` contribution survives it
 *  exactly; scaling in `f32` would quantize back to 24 significant bits and waste it. */
SZ_DEVICE_INLINE i64_t substrings_bm25_to_fixed_(f32_t contribution) noexcept {
    return (i64_t)__double2ll_rn((double)contribution * (double)(1ull << (unsigned)substrings_bm25_scale_k));
}

/** The block's fixed-point total, back as the score the caller reads. */
SZ_DEVICE_INLINE f32_t substrings_bm25_from_fixed_(i64_t total) noexcept {
    return (f32_t)((double)total / (double)(1ull << (unsigned)substrings_bm25_scale_k));
}

/** Cooperatively fills @p shared_hot_rows from the head of the hot tier and @p shared_accepts_words from the
 *  acceptance bitmap, once per block. The hot tier's out-degree ordering makes its head the best prefix to stage; the
 *  bitmap span is empty when `try_build` budgeted it out of shared memory. */
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

/** Finds, via binary search, which haystack owns global chunk @p chunk_index, given the exclusive prefix sum of chunk
 *  counts per haystack (`haystack_chunk_offsets[haystack_count]` is the grand total chunk count, mirroring how
 *  `outputs_offsets` bounds `outputs_counts`). */
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

/** How many equal-sized chunks of `chunk_bytes` a haystack of @p haystack_length bytes needs - at least one, so even
 *  an empty or shorter-than-a-chunk haystack still gets a thread. */
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

/** Writes how many matches each haystack owns, as the gap between its two boundaries. */
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

/** Index of the last entry at or below @p value, in an ascending array; zero when none is. */
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

#pragma region Trie Derivation

/** Where one needle sits, in the tape and in the trie; the pair every depth reads together. */
struct substrings_trie_cursor_t {
    /** Where this needle's folded bytes begin in the tape. */
    small_size_t tape_start {};
    /** The state it currently sits on, advanced one depth per pass. */
    small_size_t state {};
};

/** What one needle is, and where it ended. The index half is uploaded, the state half is written once, by the pass
 *  that consumes the needle's last byte. */
struct substrings_trie_ending_t {
    /** The caller's own index for this needle, since the walk visits them length-ordered. */
    small_size_t needle_index {};
    /** The state its last byte lands on. */
    small_size_t terminal_state {};
    /** Its folded length, which is what a reported match spans in the bytes the automaton walks. */
    small_size_t folded_bytes {};
};

/** The two links out of a state, which a failure chase reads at one index and in one load. */
struct substrings_trie_links_t {
    /** Lowest child id; the row ends where the next state's own begins. */
    small_size_t first_child {};
    /** Deepest state spelling a proper suffix of this one; the root fails to itself. */
    small_size_t fail {};
};

/** @brief The vocabulary as the derivation reads it: one folded tape, and one entry per needle.
 *
 *  Needles are ordered by folded length ascending, so the ones still alive at a depth are the contiguous
 *  suffix starting at that depth's first live needle - which is what bounds the work at the sum of the
 *  lengths rather than at the needle count times the longest one.
 */
struct substrings_trie_needles_t {
    /** Every needle's folded bytes, laid end to end in the caller's own order. */
    byte_t const *bytes {};
    /** Where each needle sits, in the tape and in the trie, ordered by folded length ascending. */
    substrings_trie_cursor_t *cursor_of {};
    /** What each needle is and where it ended, at the same length-ordered position. */
    substrings_trie_ending_t *ending_of {};
};

/**
 *  @brief The trie as the derivation writes it: one entry per state, addressed by that state's own id.
 *
 *  `byte_of` stays a dense byte array rather than joining `links_of`, because it is what a row search
 *  bisects: a full row is four cache lines this way and fifty-two inside a struct. `parent_of` stays out
 *  for the opposite reason - a chase reads it at a different index than the links it then follows.
 *
 *  Ids are `small_size_t` rather than the published `state_id_t`, because sixteen bits is a publication
 *  width: `try_build(wider)` only ever narrows, so both backends derive wide and narrow when they publish.
 */
struct substrings_trie_arrays_t {
    /** Each state's parent; entry zero is the root and is never written. */
    small_size_t *parent_of {};
    /** The byte each state's parent edge spells. */
    u8_t *byte_of {};
    /** Where each state's children begin, and where it fails to. */
    substrings_trie_links_t *links_of {};
};

SZ_DEVICE_INLINE small_size_t substrings_trie_child_of_(substrings_trie_arrays_t const &arrays, small_size_t state,
                                                        u8_t byte) noexcept {
    small_size_t low = arrays.links_of[state].first_child, high = arrays.links_of[state + 1].first_child;
    while (low < high) {
        small_size_t const middle = low + (high - low) / 2;
        u8_t const spelled = arrays.byte_of[middle];
        if (spelled == byte) return middle;
        if (spelled < byte) low = middle + 1;
        else high = middle;
    }
    return 0;
}

/**
 *  @brief Resolves the failure link of every state in one depth band.
 *
 *  A failure state is strictly shallower, so a band resolves entirely against bands already final - which is
 *  what makes the band the parallel unit, exactly as it is on the host. Here a band is a contiguous id range
 *  rather than a slice of a permutation, because the derivation mints ids one depth at a time.
 */
static __global__ void substrings_trie_link_failures_(substrings_trie_arrays_t arrays, small_size_t band_first,
                                                      small_size_t band_last) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = band_first + blockIdx.x * blockDim.x + threadIdx.x; state < band_last; state += stride) {
        small_size_t const parent = arrays.parent_of[state];
        // A depth-one state fails to the root; chasing instead would land back on the state itself.
        if (parent == 0) {
            arrays.links_of[state].fail = 0;
            continue;
        }
        u8_t const byte = arrays.byte_of[state];
        small_size_t landed = 0;
        for (small_size_t chased = arrays.links_of[parent].fail;;) {
            small_size_t const child = substrings_trie_child_of_(arrays, chased, byte);
            if (child != 0) landed = child;
            if (child != 0 || chased == 0) break;
            chased = arrays.links_of[chased].fail;
        }
        arrays.links_of[state].fail = landed;
    }
}

/**
 *  @brief Marks one literal edge per live needle at @p depth, counting each state's out-degree.
 *
 *  A trie is one state per distinct needle prefix, so deriving it is a dedup over prefixes rather than a walk
 *  per needle. A row here is 256 bits wide, so a `(state, byte)` pair addresses one bit and no two pairs can
 *  collide - which is what makes the dedup exact with no key, no probe and no verification pass behind it.
 *
 *  The plain load before the exchange is what keeps the root's single row from serializing the whole
 *  vocabulary, and the exchange's own return value is what lets exactly one thread count each edge.
 */
static __global__ void substrings_trie_mark_edges_(substrings_trie_needles_t needles, small_size_t first_needle,
                                                   small_size_t needles_count, small_size_t depth,
                                                   small_size_t states_first, sz_byteset_t *rows,
                                                   small_size_t *degree_of) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t needle = first_needle + blockIdx.x * blockDim.x + threadIdx.x; needle < needles_count;
         needle += stride) {
        substrings_trie_cursor_t const cursor = needles.cursor_of[needle]; // ? Both halves in one load
        small_size_t const row = cursor.state - states_first;
        u8_t const byte = (u8_t)needles.bytes[cursor.tape_start + depth];
        u64_t *const quarter = &rows[row]._u64s[byte >> 6];
        u64_t const bit = (u64_t)1 << (byte & 63u);
        if (*quarter & bit) continue; // ? Already set, and setting it again would change nothing
        // ? Lost the race, and whoever won it is the one that counts the edge
        if (atomicOr((unsigned long long *)quarter, (unsigned long long)bit) & bit) continue;
        atomicAdd(degree_of + row, 1u);
    }
}

/**
 *  @brief Mints one state per marked bit, in byte-ascending order within each row.
 *
 *  The scan left each row's first child counted from the depth's own start; this turns it absolute. Minting
 *  in bit order is what leaves a state's children contiguous and their bytes ascending, so the CSR needs no
 *  row array of its own and a later goto can binary-search whichever row it lands on.
 */
static __global__ void substrings_trie_emit_states_(sz_byteset_t const *rows, small_size_t const *offset_of,
                                                    small_size_t states_first, small_size_t states_last,
                                                    substrings_trie_arrays_t arrays) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = states_first + blockIdx.x * blockDim.x + threadIdx.x; state < states_last;
         state += stride) {
        sz_byteset_t const &row = rows[state - states_first];
        small_size_t child = states_last + offset_of[state - states_first];
        arrays.links_of[state].first_child = child;
        for (small_size_t quarter = 0; quarter != 4; ++quarter)
            for (u64_t remaining = row._u64s[quarter]; remaining; remaining &= remaining - 1, ++child) {
                arrays.parent_of[child] = state;
                arrays.byte_of[child] = (u8_t)(quarter * 64u + __ffsll((long long)remaining) - 1);
            }
        // Closes the row of the band's last state, which no later thread owns. The deepest band emits nothing,
        // so without this its rows would keep the scan's relative zero as their end.
        if (state + 1 == states_last) arrays.links_of[states_last].first_child = child;
    }
}

/**
 *  @brief Moves every live needle onto the child its byte spells, and records where a needle ends.
 *
 *  The mint above laid each row out contiguously and byte-ascending, so the child is found by the same
 *  bisect a failure chase uses rather than by ranking the marked bits a second time.
 */
static __global__ void substrings_trie_advance_needles_(substrings_trie_needles_t needles, small_size_t first_needle,
                                                        small_size_t needles_count, small_size_t dying_last,
                                                        small_size_t depth, substrings_trie_arrays_t arrays) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t needle = first_needle + blockIdx.x * blockDim.x + threadIdx.x; needle < needles_count;
         needle += stride) {
        substrings_trie_cursor_t &cursor = needles.cursor_of[needle];
        u8_t const byte = (u8_t)needles.bytes[cursor.tape_start + depth];
        // The mark pass set this byte's bit, so the row holds it and the search always lands.
        small_size_t const child = substrings_trie_child_of_(arrays, cursor.state, byte);
        cursor.state = child;
        // Needles are length-ordered, so the ones ending here are the range the host already knows.
        if (needle < dying_last) needles.ending_of[needle].terminal_state = child;
    }
}

/**
 *  @brief Builds @p mask from a state's literal edges and returns the lowest byte it spells.
 *
 *  Set directly rather than through `sz_byteset_add_u8`, which is host-only `inline` and not `constexpr`, so
 *  no relaxed-constexpr flag brings it within reach of a kernel.
 */
SZ_DEVICE_INLINE u8_t substrings_pack_row_mask_(substrings_trie_arrays_t const &arrays, small_size_t state,
                                                sz_byteset_t &mask) noexcept {
    mask._u64s[0] = mask._u64s[1] = mask._u64s[2] = mask._u64s[3] = 0;
    small_size_t const first = arrays.links_of[state].first_child;
    small_size_t const last = arrays.links_of[state + 1].first_child;
    for (small_size_t child = first; child != last; ++child) {
        u8_t const byte = arrays.byte_of[child];
        mask._u64s[byte >> 6] |= (u64_t)1 << (byte & 63u);
    }
    for (u32_t quarter = 0; quarter != 4; ++quarter)
        if (mask._u64s[quarter]) return (u8_t)(quarter * 64u + __ffsll((long long)mask._u64s[quarter]) - 1);
    return 0;
}

/**
 *  @brief Whether a slot range is free, and claiming it, over the double array's occupancy bitmap.
 *
 *  The exchange hands back the word as it was, so a row learns from `wanted & ~observed` exactly which bits
 *  it set itself - and those are the only ones it may withdraw, since no rival can own them. That is what
 *  makes a lost race undoable without holding a lock across the whole row.
 */
SZ_DEVICE_INLINE bool substrings_pack_claim_word_(u64_t *occupied, u64_t wanted, u64_t &taken) noexcept {
    if (wanted == 0) return taken = 0, true;
    u64_t const observed = (u64_t)atomicOr((unsigned long long *)occupied, (unsigned long long)wanted);
    taken = wanted & ~observed;
    return taken == wanted;
}

/** Releases only the bits this row set itself, which a lost race leaves it holding. */
SZ_DEVICE_INLINE void substrings_pack_release_word_(u64_t *occupied, u64_t taken) noexcept {
    if (taken) atomicAnd((unsigned long long *)occupied, (unsigned long long)~taken);
}

/** Claims a whole row at @p base, withdrawing whatever it managed to take if any quarter was contested. */
SZ_DEVICE_INLINE bool substrings_pack_try_row_(u64_t *occupied, sz_byteset_t const &mask, small_size_t base) noexcept {
    u64_t taken[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool claimed = true;
    for (u32_t quarter = 0; claimed && quarter != 4; ++quarter) {
        u64_t const bits = mask._u64s[quarter];
        if (!bits) continue;
        small_size_t const at = base + quarter * 64u;
        u32_t const shift = at & 63u;
        claimed = substrings_pack_claim_word_(occupied + (at >> 6), bits << shift, taken[quarter * 2]);
        if (claimed && shift)
            claimed = substrings_pack_claim_word_(occupied + (at >> 6) + 1, bits >> (64u - shift),
                                                  taken[quarter * 2 + 1]);
    }
    if (claimed) return true;
    for (u32_t quarter = 0; quarter != 4; ++quarter) {
        small_size_t const at = base + quarter * 64u;
        substrings_pack_release_word_(occupied + (at >> 6), taken[quarter * 2]);
        substrings_pack_release_word_(occupied + (at >> 6) + 1, taken[quarter * 2 + 1]);
    }
    return false;
}

/** Splits every cold state that has children into the two lists the tiers below place separately. */
static __global__ void substrings_pack_partition_rows_(substrings_trie_arrays_t arrays, small_size_t states_count,
                                                       small_size_t hot_count, small_size_t *wide_rows,
                                                       small_size_t *narrow_rows, small_size_t *counts) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = hot_count + blockIdx.x * blockDim.x + threadIdx.x; state < states_count;
         state += stride) {
        small_size_t const degree = arrays.links_of[state + 1].first_child - arrays.links_of[state].first_child;
        if (degree >= 2) wide_rows[atomicAdd(counts, 1u)] = state;
        else if (degree == 1) narrow_rows[atomicAdd(counts + 1, 1u)] = state;
    }
}

/**
 *  @brief Gives every child of a hot parent a slot of its own, bumping one shared cursor.
 *
 *  A hot row addresses its children directly, so they share no base and need no search - and because the hot
 *  states own `[0, hot_count)` outright, the cursor starts on virgin ground and every bump lands where
 *  nothing else has reached. The slots it hands out are contiguous, so the caller marks them in one stroke
 *  rather than one atomic per child.
 */
static __global__ void substrings_pack_hot_children_(substrings_trie_arrays_t arrays, small_size_t hot_count,
                                                     small_size_t *slot_of, small_size_t *cursor) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = blockIdx.x * blockDim.x + threadIdx.x; state < hot_count; state += stride) {
        small_size_t const first = arrays.links_of[state].first_child;
        small_size_t const last = arrays.links_of[state + 1].first_child;
        // A hot child already owns the slot its own index names, so only the cold ones take from the bump -
        // and ids being depth-ordered puts every hot child of a row ahead of every cold one.
        small_size_t const cold_first = sz_max_of_two(first, hot_count);
        if (cold_first >= last) continue;
        small_size_t const base = atomicAdd(cursor, last - cold_first);
        for (small_size_t child = cold_first; child != last; ++child) slot_of[child] = base + (child - cold_first);
    }
}

/**
 *  @brief Places every cold row of out-degree two or more, one warp to a row, and lists what would not fit.
 *
 *  Thirty-two candidate bases are tested per ballot rather than one per probe, so the host's budget of two
 *  hundred and fifty-six interior probes becomes eight ballots. A row that loses a race withdraws only the
 *  bits it set, and a row that spends its budget is handed to `substrings_pack_stranded_rows_` instead of
 *  searching on - which is what keeps this bounded rather than quadratic in the vacancies it re-walks.
 */
static __global__ void substrings_pack_wide_rows_(substrings_trie_arrays_t arrays, small_size_t const *rows,
                                                  small_size_t rows_count, small_size_t floor_slot, u64_t *occupied,
                                                  small_size_t *base_of, small_size_t *stranded,
                                                  small_size_t *stranded_count) {

    small_size_t const warps = (blockDim.x * gridDim.x) / 32u;
    small_size_t const warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    small_size_t const lane = threadIdx.x & 31u;
    for (small_size_t index = warp; index < rows_count; index += warps) {
        small_size_t const state = rows[index];
        sz_byteset_t mask;
        u8_t const anchor = substrings_pack_row_mask_(arrays, state, mask);

        // Anchored at the row's own lowest byte, not at the floor: a candidate below it would put the base
        // before the arena, so windows under the anchor are infeasible for every lane and would spend the
        // budget without testing a single slot - which is what sends an otherwise placeable row stranded.
        bool placed = false;
        small_size_t const first_window = sz_max_of_two(floor_slot, (small_size_t)anchor);
        for (small_size_t window = first_window, ballots = 0; !placed && ballots != 8u; window += 32u, ++ballots) {
            small_size_t const candidate = window + lane;
            bool feasible = candidate >= anchor;
            small_size_t const base = feasible ? candidate - anchor : 0;
            // Every quarter has to read clear before any of it is claimed, so the whole row is tested first.
            for (u32_t quarter = 0; feasible && quarter != 4; ++quarter) {
                u64_t const bits = mask._u64s[quarter];
                if (!bits) continue;
                small_size_t const at = base + quarter * 64u;
                u64_t const low = occupied[at >> 6] >> (at & 63u);
                u64_t const high = (at & 63u) ? occupied[(at >> 6) + 1] << (64u - (at & 63u)) : 0;
                if ((low | high) & bits) feasible = false;
            }
            u32_t const ballot = __ballot_sync(0xFFFFFFFFu, feasible);
            if (ballot == 0) continue;

            // The lowest feasible lane wins the window, and only it attempts the claim.
            u32_t const winner = (u32_t)__ffs((int)ballot) - 1u;
            u32_t won = 0;
            if (lane == winner && substrings_pack_try_row_(occupied, mask, base)) base_of[state] = base, won = 1u;
            placed = __shfl_sync(0xFFFFFFFFu, won, winner) != 0u;
        }
        if (!placed && lane == 0) stranded[atomicAdd(stranded_count, 1u)] = state;
    }
}

/**
 *  @brief Places the rows that spent their ballot budget, each on a stride of its own past the frontier.
 *
 *  Every stranded row owns an alphabet's worth of ground no other row can reach, so its claim cannot fail -
 *  a termination proof rather than an expectation, and the reason the search above may give up at all.
 */
static __global__ void substrings_pack_stranded_rows_(substrings_trie_arrays_t arrays, small_size_t const *stranded,
                                                      small_size_t stranded_count, small_size_t frontier,
                                                      u64_t *occupied, small_size_t *base_of) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < stranded_count; index += stride) {
        small_size_t const state = stranded[index];
        sz_byteset_t mask;
        [[maybe_unused]] u8_t const anchor = substrings_pack_row_mask_(arrays, state, mask);
        small_size_t const base = frontier + index * substrings_alphabet_size_k;
        [[maybe_unused]] bool const claimed = substrings_pack_try_row_(occupied, mask, base);
        sz_assert_(claimed && "A stride past the frontier is reachable by exactly one row");
        base_of[state] = base;
    }
}

/**
 *  @brief The bits of @p word that name a slot a one-byte row may take.
 *
 *  A row's base is `slot - byte`, so a vacancy below the alphabet would place it before the arena begins,
 *  and one in the top `alphabet_size - 1` slots would put `base + byte` past the arena's end. Both ends are
 *  cut here rather than at the row, so the vacancy count and the placement agree on what a rank names.
 */
SZ_DEVICE_INLINE u64_t substrings_pack_placeable_(small_size_t word, small_size_t last_slot) noexcept {
    small_size_t const first_slot = substrings_alphabet_size_k - 1;
    small_size_t const word_first = word * 64u, word_last = word_first + 64u;
    if (word_last <= first_slot || word_first >= last_slot) return 0;
    u64_t const above = word_first >= first_slot ? ~(u64_t)0 : ~(u64_t)0 << (first_slot - word_first);
    u64_t const below = word_last <= last_slot ? ~(u64_t)0 : ~(u64_t)0 >> (word_last - last_slot);
    return above & below;
}

/** Counts the placeable vacancies in each bitmap word, which the scan below turns into where a word's rows begin. */
static __global__ void substrings_pack_count_vacancies_(u64_t const *occupied, small_size_t words,
                                                        small_size_t last_slot, small_size_t *vacancies_of_word) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t word = blockIdx.x * blockDim.x + threadIdx.x; word < words; word += stride)
        vacancies_of_word[word] = (small_size_t)__popcll(
            (unsigned long long)(~occupied[word] & substrings_pack_placeable_(word, last_slot)));
}

/**
 *  @brief Places every cold row of out-degree one, by rank rather than by search.
 *
 *  A one-bit mask fits the first vacancy it meets, so the rows that spell a single byte - most of any trie -
 *  need no probing at all. Each word owns a contiguous run of ranks, so one thread hands its own zero bits
 *  to consecutive rows and no two words can reach the same row.
 */
static __global__ void substrings_pack_narrow_rows_(substrings_trie_arrays_t arrays, small_size_t const *rows,
                                                    small_size_t rows_count, small_size_t const *rank_of_word,
                                                    small_size_t words, small_size_t last_slot, u64_t *occupied,
                                                    small_size_t *base_of) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t word = blockIdx.x * blockDim.x + threadIdx.x; word < words; word += stride) {
        small_size_t rank = rank_of_word[word];
        u64_t vacancies = ~occupied[word] & substrings_pack_placeable_(word, last_slot);
        u64_t claimed = 0;
        for (; vacancies && rank < rows_count; vacancies &= vacancies - 1, ++rank) {
            u64_t const lowest = vacancies & (~vacancies + 1);
            small_size_t const slot = word * 64u + (small_size_t)(__ffsll((long long)vacancies) - 1);
            small_size_t const state = rows[rank];
            base_of[state] = slot - arrays.byte_of[arrays.links_of[state].first_child];
            claimed |= lowest;
        }
        // One thread owns this word outright, so the write needs no exchange.
        occupied[word] |= claimed;
    }
}

/**
 *  @brief Derives every cold state's published slot from the base its parent settled on.
 *
 *  One past the highest slot any state ends up on is maxed into @p published_bound as they are derived: the
 *  arena is provisioned for a worst case the packing rarely reaches, and every cold array is sized against
 *  what was actually used rather than against what was reserved.
 */
static __global__ void substrings_pack_publish_ids_(substrings_trie_arrays_t arrays, small_size_t states_count,
                                                    small_size_t hot_count, small_size_t const *base_of,
                                                    small_size_t *slot_of, small_size_t *published_bound) {

    small_size_t const stride = blockDim.x * gridDim.x;
    small_size_t highest = 0;
    for (small_size_t state = blockIdx.x * blockDim.x + threadIdx.x; state < states_count; state += stride) {
        small_size_t const parent = arrays.parent_of[state];
        if (state >= hot_count && parent >= hot_count) // ? A hot parent's child was given its slot outright
            slot_of[state] = base_of[parent] + arrays.byte_of[state];
        else if (state < hot_count) slot_of[state] = state;
        highest = sz_max_of_two(highest, slot_of[state] + 1u);
    }
    if (highest) atomicMax(published_bound, highest);
}

/** Tallies how many needles end on each state, which is that state's own output run before any merge. */
static __global__ void substrings_outputs_count_own_(substrings_trie_needles_t needles, small_size_t needles_count,
                                                     small_size_t *own_of) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t needle = blockIdx.x * blockDim.x + threadIdx.x; needle < needles_count; needle += stride)
        atomicAdd(own_of + needles.ending_of[needle].terminal_state, 1u);
}

/**
 *  @brief Adds one band's failure-merged output totals, which its failure states already carry.
 *
 *  A failure state is strictly shallower, so its own total is final before this band reads it - the same
 *  property that makes the failure links themselves a band-parallel pass. The longest run of them all is
 *  maxed into @p longest_run as they are written, so the bound the walk budgets against needs no pass of
 *  its own - and the root is excluded only because no needle is empty, so its own run is zero.
 */
static __global__ void substrings_outputs_merge_band_(substrings_trie_arrays_t arrays, small_size_t band_first,
                                                      small_size_t band_last, small_size_t const *own_of,
                                                      small_size_t *total_of, small_size_t *longest_run) {

    small_size_t const stride = blockDim.x * gridDim.x;
    small_size_t longest = 0;
    for (small_size_t state = band_first + blockIdx.x * blockDim.x + threadIdx.x; state < band_last; state += stride) {
        small_size_t const total = own_of[state] + total_of[arrays.links_of[state].fail];
        total_of[state] = total;
        longest = sz_max_of_two(longest, total);
    }
    if (longest) atomicMax(longest_run, longest);
}

/**
 *  @brief Fills one band's output runs: its own needles first, then whatever its failure state reports.
 *
 *  The failure state's run is already flattened, so this copies it wholesale rather than chasing the chain -
 *  which is what keeps a nested-suffix vocabulary linear here instead of quadratic in the chain depth.
 */
static __global__ void substrings_outputs_fill_band_(substrings_trie_arrays_t arrays, substrings_trie_needles_t needles,
                                                     small_size_t needles_count, small_size_t band_first,
                                                     small_size_t band_last, small_size_t const *offset_of,
                                                     small_size_t const *total_of, small_size_t *written_of,
                                                     small_size_t *outputs) {

    small_size_t const stride = blockDim.x * gridDim.x;
    sz_unused_(needles), sz_unused_(needles_count);
    for (small_size_t state = band_first + blockIdx.x * blockDim.x + threadIdx.x; state < band_last; state += stride) {
        small_size_t const failure = arrays.links_of[state].fail;
        small_size_t const inherited = total_of[failure];
        small_size_t const at = offset_of[state] + written_of[state];
        for (small_size_t index = 0; index != inherited; ++index)
            outputs[at + index] = outputs[offset_of[failure] + index];
    }
}

/** Places each needle's own index into its terminal state's run, ahead of whatever the merge inherits. */
static __global__ void substrings_outputs_place_own_(substrings_trie_needles_t needles, small_size_t needles_count,
                                                     small_size_t const *offset_of, small_size_t *written_of,
                                                     small_size_t *outputs) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t needle = blockIdx.x * blockDim.x + threadIdx.x; needle < needles_count; needle += stride) {
        substrings_trie_ending_t const ending = needles.ending_of[needle];
        small_size_t const at = atomicAdd(written_of + ending.terminal_state, 1u);
        outputs[offset_of[ending.terminal_state] + at] = needle;
    }
}

/**
 *  @brief Writes the double array's owner column, one thread per state that has children.
 *
 *  A hot parent's child answers through the completed row rather than through a base, so it checks against
 *  itself; every cold row's child checks against the parent that placed it.
 */
template <typename state_id_type_>
__global__ void substrings_publish_check_(substrings_trie_arrays_t arrays, small_size_t states_count,
                                          small_size_t hot_count, small_size_t const *slot_of, state_id_type_ *check) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = blockIdx.x * blockDim.x + threadIdx.x; state < states_count; state += stride) {
        small_size_t const first = arrays.links_of[state].first_child;
        small_size_t const last = arrays.links_of[state + 1].first_child;
        for (small_size_t child = first; child != last; ++child) {
            small_size_t const slot = slot_of[child];
            check[slot] = (state_id_type_)(state < hot_count ? slot : slot_of[state]);
        }
    }
}

/**
 *  @brief Writes each published slot's base, failure link and output run.
 *
 *  One thread per state rather than per slot, and the columns arrive already cleared: the two orders differ
 *  wherever a state's slot is not its own id, which is every cold state, so a thread clearing slots while
 *  another writes them would race for exactly the entries the cold tier depends on.
 */
template <typename state_id_type_>
__global__ void substrings_publish_slots_(substrings_trie_arrays_t arrays, small_size_t states_count,
                                          small_size_t hot_count, small_size_t const *slot_of,
                                          small_size_t const *base_of, small_size_t const *totals_of,
                                          small_size_t const *offsets_of, state_id_type_ *base, state_id_type_ *fail,
                                          state_id_type_ *counts, size_t *offsets) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t state = blockIdx.x * blockDim.x + threadIdx.x; state < states_count; state += stride) {
        small_size_t const slot = slot_of[state];
        base[slot] = (state_id_type_)base_of[state];
        counts[slot] = (state_id_type_)totals_of[state];
        offsets[slot] = offsets_of[state];
        if (slot >= hot_count) fail[slot] = (state_id_type_)slot_of[arrays.links_of[state].fail];
    }
}

/**
 *  @brief Completes one hot row: its failure state's row, overwritten by its own literal edges.
 *
 *  Rows are filled shallowest first, so a row's failure state is already final when it is copied - the same
 *  depth ordering the failure links themselves rely on, and the reason a hot row needs no chase at all.
 */
template <typename state_id_type_>
__global__ void substrings_publish_hot_row_(substrings_trie_arrays_t arrays, small_size_t hot_index,
                                            small_size_t const *slot_of, state_id_type_ root,
                                            state_id_type_ *hot_rows) {

    state_id_type_ *const row = hot_rows + (size_t)hot_index * substrings_alphabet_size_k;
    small_size_t const state = hot_index; // ? Ids are depth-ordered, so the hot tier is the lowest range
    if (hot_index == 0)
        for (small_size_t byte = threadIdx.x; byte < substrings_alphabet_size_k; byte += blockDim.x) row[byte] = root;
    else {
        state_id_type_ const *const inherited = hot_rows + (size_t)slot_of[arrays.links_of[state].fail] *
                                                               substrings_alphabet_size_k;
        for (small_size_t byte = threadIdx.x; byte < substrings_alphabet_size_k; byte += blockDim.x)
            row[byte] = inherited[byte];
    }
    __syncthreads();
    small_size_t const first = arrays.links_of[state].first_child;
    small_size_t const last = arrays.links_of[state + 1].first_child;
    for (small_size_t child = first + threadIdx.x; child < last; child += blockDim.x)
        row[arrays.byte_of[child]] = (state_id_type_)slot_of[child];
}

/**
 *  @brief Sets one bit per published slot that some needle ends on, which is what the walk gate reads.
 *
 *  One word per warp-sized run of slots, assembled by ballot and written once, so the gate's array is built
 *  without an atomic and the automaton never has to be host-readable for it.
 */
template <typename state_id_type_>
__global__ void substrings_publish_accepts_(state_id_type_ const *counts, small_size_t slots, u32_t *words) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < ((slots + 31u) & ~31u); slot += stride) {
        bool const accepts = slot < slots && counts[slot] != 0;
        u32_t const lane_mask = __ballot_sync(0xFFFFFFFFu, accepts);
        if ((slot & 31u) == 0) words[slot >> 5] = lane_mask;
    }
}

/** Copies the merged output pool out at the published width, in the run order the slots now name. */
template <typename state_id_type_>
__global__ void substrings_publish_outputs_(substrings_trie_needles_t needles, small_size_t const *pool,
                                            small_size_t pool_size, substrings_output<state_id_type_> *outputs) {

    small_size_t const stride = blockDim.x * gridDim.x;
    for (small_size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < pool_size; index += stride) {
        substrings_trie_ending_t const ending = needles.ending_of[pool[index]];
        outputs[index] = {(state_id_type_)ending.needle_index, (state_id_type_)ending.folded_bytes};
    }
}

#pragma endregion Trie Derivation

#pragma endregion Device Kernels

#pragma region Kernel Table

/**
 *  @brief Every kernel this file launches, resolved once per device.
 *
 *  At namespace scope rather than nested in the engine because the device builder launches two thirds of it
 *  and the engine the rest, and a table one of them could not name would have to be duplicated.
 */
struct substrings_cuda_kernels_t {
    /** @brief One shape per state-id width, for the kernels that walk or write the automaton. The cover and
     *         rewrite kernels below take no view, so they are shared. @sa `levenshtein_distances::kernels_t`,
     *         which lists its cell widths the same way. */
    struct by_width_t {
        kernel_shape_t u16, u32;

        kernel_shape_t const &for_width(substrings_state_width_t width) const noexcept {
            return width == substrings_state_width_t::u16_k ? u16 : u32;
        }
        kernel_shape_t &for_width(substrings_state_width_t width) noexcept {
            return width == substrings_state_width_t::u16_k ? u16 : u32;
        }
    };
    by_width_t count_chunk;
    by_width_t scatter_chunk;
    /** The prefix sum at both widths the engine scans at: match and haystack counts, which can outgrow 32 bits,
     *  and the derivation's out-degrees, which provably cannot. */
    exclusive_sum_shapes_t exclusive_sum;
    exclusive_sum_shapes_t exclusive_sum_u32;
    kernel_shape_t cover_resolve;
    kernel_shape_t cover_compact;
    kernel_shape_t haystack_match_offsets;
    kernel_shape_t counts_from_boundaries;
    kernel_shape_t rewrite_offsets;
    kernel_shape_t rewrite_copy;
    by_width_t score_bm25;
    /** One depth of the derivation; none of the four walks the automaton, so none is width-typed. */
    kernel_shape_t trie_mark_edges;
    kernel_shape_t trie_emit_states;
    kernel_shape_t trie_advance_needles;
    kernel_shape_t trie_link_failures;
    /** One tier of the double-array packing each, plus the two passes that bracket them. */
    kernel_shape_t pack_partition_rows;
    kernel_shape_t pack_hot_children;
    kernel_shape_t pack_wide_rows;
    kernel_shape_t pack_stranded_rows;
    kernel_shape_t pack_count_vacancies;
    kernel_shape_t pack_narrow_rows;
    kernel_shape_t pack_publish_ids;
    /** The five publishing passes, each at both widths, since only these write at the settled one. */
    by_width_t publish_check;
    by_width_t publish_slots;
    by_width_t publish_hot_row;
    by_width_t publish_outputs;
    by_width_t publish_accepts;
    /** The failure-merged output pool, which the publish then writes out at the settled width. */
    kernel_shape_t outputs_count_own;
    kernel_shape_t outputs_merge_band;
    kernel_shape_t outputs_place_own;
    kernel_shape_t outputs_fill_band;
};

/** The half of a width-paired table entry that @p state_id_type_ names. */
template <typename state_id_type_>
inline kernel_shape_t &published_shape_of_(substrings_cuda_kernels_t::by_width_t &pair) noexcept {
    if constexpr (sizeof(state_id_type_) == sizeof(u16_t)) return pair.u16;
    else return pair.u32;
}

/** The same, for a table nobody may write. */
template <typename state_id_type_>
inline kernel_shape_t const &published_shape_of_(substrings_cuda_kernels_t::by_width_t const &pair) noexcept {
    if constexpr (sizeof(state_id_type_) == sizeof(u16_t)) return pair.u16;
    else return pair.u32;
}

/**
 *  @brief Grid for a kernel launched with one block per work item, from that kernel's own occupancy.
 *
 *  Clamped to the item count, because a block that finds nothing to do still costs its scratch - the
 *  BM25 frequency rows are sized from this, so an unclamped grid would allocate rows nobody fills.
 */
inline unsigned grid_for_items_(kernel_shape_t const &shape, size_t items, gpu_specs_t const &specs) noexcept {
    size_t const resident = (size_t)shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
    return (unsigned)sz_min_of_two(sz_max_of_two(resident, (size_t)1), sz_max_of_two(items, (size_t)1));
}

/** Launches @p shape over @p items with this file's block width, which every build phase shares. */
inline cuda_status_t launch_over_(kernel_shape_t const &shape, size_t items, gpu_specs_t const &specs,
                                  cuda_executor_t const &executor, void **arguments) noexcept {
    CUresult const launched = cuda_launch_t {}
                                  .grid(grid_for_items_(shape, items, specs))
                                  .block(substrings_threads_per_block_k)
                                  .shared(0)
                                  .stream(executor.stream())
                                  .launch(shape.function, arguments);
    return launched == CUDA_SUCCESS ? cuda_status_t {status_t::success_k, cudaSuccess} : make_cuda_status(launched);
}

#pragma endregion Kernel Table

#pragma region Device Dictionary

template <typename allocator_type_>
struct aho_corasick_cuda_builder;

/**
 *  @brief The automaton a device build publishes, at one state-id width.
 *
 *  The device sibling of `aho_corasick_dictionary`: a different construction path behind the same published
 *  contract, `aho_corasick_view<state_id_t>`, which is the only thing the walks consume. Its arrays are
 *  device-resident rather than unified, because every one of them is written by a kernel and read by a
 *  kernel - a host touch in between would migrate the pages twice for nothing.
 *
 *  Construction state lives in `aho_corasick_cuda_builder` and dies with it, so this holds one allocation
 *  plus the view over it however long the engine lives.
 */
template <typename state_id_type_>
struct aho_corasick_cuda_dictionary {

    using state_id_t = state_id_type_;
    using view_t = aho_corasick_view<state_id_t>;
    using output_t = substrings_output<state_id_t>;

  private:
    template <typename>
    friend struct aho_corasick_cuda_builder;

    using device_byte_allocator_t = device_alloc<byte_t>;
    using device_word_allocator_t = device_alloc<u32_t>;

    /** One block holding every published array, carved at this dictionary's own width. */
    safe_vector<byte_t, device_byte_allocator_t> automaton_ {};
    /** Dense acceptance bitmap: bit `slot` is set when some needle ends there, so the per-byte gate never
     *  touches the 32x larger `outputs_counts`. Words are 32-bit because that is one shared-memory bank. */
    safe_vector<u32_t, device_word_allocator_t> accepts_words_ {};
    /** The published automaton over that block, rebuilt by no one - the walks read exactly this. */
    view_t view_ {};
    /** Needles the vocabulary held, which is what "has this engine been indexed" asks. */
    size_t count_needles_ = 0;

  public:
    aho_corasick_cuda_dictionary() noexcept = default;
    aho_corasick_cuda_dictionary(aho_corasick_cuda_dictionary const &) = delete;
    aho_corasick_cuda_dictionary &operator=(aho_corasick_cuda_dictionary const &) = delete;
    aho_corasick_cuda_dictionary(aho_corasick_cuda_dictionary &&) noexcept = default;
    aho_corasick_cuda_dictionary &operator=(aho_corasick_cuda_dictionary &&) noexcept = default;

    /** Releases the automaton and the gate, leaving a dictionary a fresh build can fill again. */
    void reset() noexcept {
        automaton_.reset();
        accepts_words_.reset();
        view_ = view_t {};
        count_needles_ = 0;
    }

    /** The published automaton, which is the whole cross-backend contract. */
    view_t view() const noexcept { return view_; }
    /** The acceptance bitmap the per-byte walk gate reads, one bit per published slot. */
    span<u32_t const> accepts() const noexcept { return {accepts_words_.data(), accepts_words_.size()}; }
    /** Whether a walk folds the haystack as it consumes it, which the tape was folded under. */
    substrings_case_sensitivity_t case_sensitivity() const noexcept { return view_.case_sensitivity; }
    /** Needles the vocabulary held, which is what "has this engine been indexed" asks. */
    size_t count_needles() const noexcept { return count_needles_; }
    /** The exclusive published bound: the cold arrays reach 255 slots past it. */
    size_t count_states() const noexcept { return view_.state_count; }
    /** States living in `hot_rows` rather than the double array, so the tier test is `state < hot_count`. */
    size_t hot_count() const noexcept { return view_.hot_count; }
    /** Most haystack bytes one match can span, which is what every slice, halo and warm-up needs. */
    state_id_t max_source_match_bytes() const noexcept { return view_.max_source_match_bytes; }
    /** Fewest haystack bytes one match can span; the mirror bound. */
    state_id_t min_source_match_bytes() const noexcept { return view_.min_source_match_bytes; }
    /** Device bytes the transition tables occupy, which is what a build's footprint is reported as. */
    size_t transitions_bytes() const noexcept { return automaton_.size(); }

    /**
     *  @brief Writes @p builder 's packed trie out as an automaton at this width.
     *
     *  The mirror of the host's narrowing `try_build(wider)`: the derivation runs once, at the widest id,
     *  and this is the only pass that knows how wide a published cell is.
     *  @retval `status_t::overflow_risk_k` Some published value exceeds this width; @p builder stays usable.
     *  @retval `status_t::bad_alloc_k` A device allocation failed.
     */
    template <typename allocator_type_>
    cuda_status_t try_build(aho_corasick_cuda_builder<allocator_type_> const &builder,
                            substrings_cuda_kernels_t const &kernel_table, cuda_executor_t const &executor,
                            gpu_specs_t const &specs) noexcept;
};

/**
 *  @brief Compiles a vocabulary into an automaton without leaving the device.
 *
 *  Six phases: the tape is folded and length-ordered on the host, the trie is derived one depth at a time,
 *  failure links resolve band by band, the double array packs in three tiers, the outputs merge along those
 *  same bands, and `aho_corasick_cuda_dictionary::try_build` writes it out at the settled width.
 *
 *  Every array below is construction state, which is why this is a type of its own rather than more members
 *  on the dictionary or on the engine: it is a local of whatever indexes, and its device memory is returned
 *  the moment that scope ends.
 */
template <typename allocator_type_>
struct aho_corasick_cuda_builder {

    using allocator_t = allocator_type_;

  private:
    template <typename>
    friend struct aho_corasick_cuda_dictionary;

    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using word_allocator_t = typename allocator_traits_t::template rebind_alloc<u32_t>;
    using byte_allocator_t = typename allocator_traits_t::template rebind_alloc<byte_t>;
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;
    using cursor_allocator_t = typename allocator_traits_t::template rebind_alloc<substrings_trie_cursor_t>;
    using ending_allocator_t = typename allocator_traits_t::template rebind_alloc<substrings_trie_ending_t>;
    using device_byte_allocator_t = device_alloc<byte_t>;
    using device_word_allocator_t = device_alloc<u32_t>;
    using device_row_allocator_t = device_alloc<sz_byteset_t>;
    using device_bitmap_allocator_t = device_alloc<u64_t>;
    using device_cursor_allocator_t = device_alloc<substrings_trie_cursor_t>;
    using device_ending_allocator_t = device_alloc<substrings_trie_ending_t>;
    using device_links_allocator_t = device_alloc<substrings_trie_links_t>;

    /** The needles as one tape, folded when the mode asks, in the caller's own order. */
    safe_vector<byte_t, device_byte_allocator_t> needle_bytes_;
    /** Where each needle sits, in the tape and in the trie, ordered by folded length ascending. */
    safe_vector<substrings_trie_cursor_t, device_cursor_allocator_t> needle_cursors_;
    /** What each needle is and where it ended, at that same length-ordered position. */
    safe_vector<substrings_trie_ending_t, device_ending_allocator_t> needle_endings_;
    /** Host-side: needles already spent by each depth, so the live ones are the suffix from there. */
    safe_vector<small_size_t, word_allocator_t> needle_first_live_;
    /** Each state's parent; entry zero is the root and is never written. */
    safe_vector<small_size_t, device_word_allocator_t> trie_parent_of_;
    /** The byte each state's parent edge spells, dense so that a row search can bisect it. */
    safe_vector<byte_t, device_byte_allocator_t> trie_byte_of_;
    /** Where each state's children begin, and where it fails to. */
    safe_vector<substrings_trie_links_t, device_links_allocator_t> trie_links_;
    /** Host-side: where each depth's states begin, with the state total trailing. */
    safe_vector<small_size_t, word_allocator_t> trie_band_firsts_;
    /** One 256-bit row per state at the depth being derived, its own array so the alignment survives. */
    safe_vector<sz_byteset_t, device_row_allocator_t> trie_rows_;
    /** That depth's out-degrees, which the scan turns in place into where each of its rows begins. */
    safe_vector<small_size_t, device_word_allocator_t> trie_degrees_;
    /** One bit per double-array slot, with a word of headroom so a shifted row read never runs off the end. */
    safe_vector<u64_t, device_bitmap_allocator_t> pack_occupied_;
    /** Each state's published slot, which is its id in the automaton the device finally writes. */
    safe_vector<small_size_t, device_word_allocator_t> pack_slot_of_;
    /** Each cold parent's row base, so `base + byte` addresses the child that byte spells. */
    safe_vector<small_size_t, device_word_allocator_t> pack_base_of_;
    /** The cold rows split by out-degree, and the ones that spent their ballot budget, laid end to end. */
    safe_vector<small_size_t, device_word_allocator_t> pack_rows_;
    /** Vacancies per bitmap word, which the scan turns into where each word's narrow rows begin. */
    safe_vector<small_size_t, device_word_allocator_t> pack_ranks_;
    /** The five counters the packing reports through: two list lengths, the hot cursor, the stranded tally,
     *  and one past the highest slot any state ended up on. */
    safe_vector<small_size_t, device_word_allocator_t> pack_counters_;
    /** Needles ending on each state, before any failure merge folds a suffix's run into it. */
    safe_vector<small_size_t, device_word_allocator_t> outputs_own_;
    /** Each state's merged run length, with the longest of them all maxed into the trailing entry. */
    safe_vector<small_size_t, device_word_allocator_t> outputs_totals_;
    /** Where each state's run begins, with the pool size trailing, as the scan leaves it. */
    safe_vector<small_size_t, device_word_allocator_t> outputs_offsets_;
    /** How much of each state's run is already filled, so the merge appends behind its own needles. */
    safe_vector<small_size_t, device_word_allocator_t> outputs_written_;
    /** The flattened pool itself, one length-ordered needle position per entry. */
    safe_vector<small_size_t, device_word_allocator_t> outputs_pool_;
    /** Tile totals for `cuda_launch_exclusive_sum_`'s multi-block route, sized at its grid ceiling. */
    safe_vector<small_size_t, device_word_allocator_t> scan_partials_;

    /** Whether the walk folds the haystack as it consumes it, which the tape was folded under. */
    substrings_case_sensitivity_t case_sensitivity_ = substrings_cased_k;
    /** States the trie spells, the root included. */
    size_t count_states_ = 0;
    /** States the hot tier holds, which is the tier test `state < hot_count`. */
    size_t hot_count_ = 0;
    /** Published slots the automaton spans, which is what every cold array is sized against. */
    size_t slots_ = 0;
    /** Entries the merged output pool holds. */
    size_t outputs_total_ = 0;
    /** Most merged outputs any one state carries, which bounds one pass's match count. */
    size_t max_outputs_per_state_ = 0;
    /** Most haystack bytes one match can span, which is what every slice, halo and warm-up needs. */
    size_t max_source_match_bytes_ = 0;
    /** Fewest haystack bytes one match can span; the mirror bound. */
    size_t min_source_match_bytes_ = 0;
    allocator_t alloc_ {};

  public:
    explicit aho_corasick_cuda_builder(substrings_case_sensitivity_t case_sensitivity, allocator_t alloc = {}) noexcept
        : needle_bytes_(), needle_cursors_(), needle_endings_(), needle_first_live_(alloc), trie_parent_of_(),
          trie_byte_of_(), trie_links_(), trie_band_firsts_(alloc), trie_rows_(), trie_degrees_(), pack_occupied_(),
          pack_slot_of_(), pack_base_of_(), pack_rows_(), pack_ranks_(), pack_counters_(), outputs_own_(),
          outputs_totals_(), outputs_offsets_(), outputs_written_(), outputs_pool_(), scan_partials_(),
          case_sensitivity_(case_sensitivity), alloc_(alloc) {}

    aho_corasick_cuda_builder(aho_corasick_cuda_builder const &) = delete;
    aho_corasick_cuda_builder &operator=(aho_corasick_cuda_builder const &) = delete;

    /** Needles the vocabulary held, which the dictionary carries forward. */
    size_t count_needles() const noexcept { return needle_endings_.size(); }
    substrings_case_sensitivity_t case_sensitivity() const noexcept { return case_sensitivity_; }

    /**
     *  @brief The narrowest state-id width every published value fits, which is what the engine settles on.
     *
     *  Tests the same four ceilings the host's narrowing overload does, so a vocabulary refused there is
     *  refused here for the same reason rather than discovered mid-publish.
     */
    substrings_state_width_t published_width() const noexcept {
        constexpr size_t narrow_ceiling = (size_t)std::numeric_limits<u16_t>::max();
        bool const fits = slots_ <= narrow_ceiling && count_needles() <= narrow_ceiling &&
                          max_source_match_bytes_ <= narrow_ceiling && max_outputs_per_state_ <= narrow_ceiling;
        return fits ? substrings_state_width_t::u16_k : substrings_state_width_t::u32_k;
    }

    /**
     *  @brief Phases one through five: upload, derive, link, pack, merge.
     *
     *  Leaves the shape the publish needs settled - the state count, the hot tier, the slot span and the
     *  pool size - and every array it wrote still resident, since the publish is the only reader of them.
     */
    template <typename needles_type_>
    cuda_status_t try_derive(substrings_cuda_kernels_t const &kernel_table, needles_type_ const &needles,
                             cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {

        // Sized at the scans' grid ceiling rather than per phase: it is 4 KB whatever the corpus, and a scan
        // that found it short would silently drop back to one block.
        if (scan_partials_.try_resize_uninitialized(cuda_device_collective_max_blocks_k + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        if (cuda_status_t const uploaded = try_upload_needles_(needles, executor);
            uploaded.status != status_t::success_k)
            return uploaded;
        if (cuda_status_t const derived = try_derive_trie_(kernel_table, specs, executor);
            derived.status != status_t::success_k)
            return derived;

        // The tier split follows the cache the device walks through rather than the host's last level, which
        // a default `cpu_specs_t` would put at 8 MB whatever the GPU. The root is hot however small that
        // cache is: it is the one state with no parent edge to place it, and a cold root would also make
        // its own id indistinguishable from the unowned column below.
        hot_count_ = sz_max_of_two(
            sz_min_of_two(specs.l2_bytes / (substrings_alphabet_size_k * sizeof(u32_t)), count_states_), (size_t)1);
        if (count_states_ <= 1) { // ? An empty vocabulary spells only the root, which needs no packing
            slots_ = count_states_, outputs_total_ = 0, max_outputs_per_state_ = 0;
            return {status_t::success_k, cudaSuccess};
        }

        if (cuda_status_t const linked = try_link_failures_(kernel_table, specs, executor);
            linked.status != status_t::success_k)
            return linked;
        if (cuda_status_t const packed = try_pack_(kernel_table, specs, executor); packed.status != status_t::success_k)
            return packed;
        return try_merge_outputs_(kernel_table, specs, executor);
    }

  private:
    /** The trie arrays as the kernels take them, so no launch site assembles the bundle by hand. */
    substrings_trie_arrays_t trie_arrays_() const noexcept {
        substrings_trie_arrays_t arrays;
        arrays.parent_of = const_cast<small_size_t *>(trie_parent_of_.data());
        arrays.byte_of = (u8_t *)const_cast<byte_t *>(trie_byte_of_.data());
        arrays.links_of = const_cast<substrings_trie_links_t *>(trie_links_.data());
        return arrays;
    }

    /** The needle arrays as the kernels take them, the mirror of `trie_arrays_`. */
    substrings_trie_needles_t trie_needles_() const noexcept {
        substrings_trie_needles_t needles;
        needles.bytes = const_cast<byte_t *>(needle_bytes_.data());
        needles.cursor_of = const_cast<substrings_trie_cursor_t *>(needle_cursors_.data());
        needles.ending_of = const_cast<substrings_trie_ending_t *>(needle_endings_.data());
        return needles;
    }

    /**
     *  @brief Uploads the vocabulary as one tape, folding it when the mode asks and ordering it by length.
     *
     *  Folding stays on the host: it is one pass over the needle bytes, it is shared with every CPU backend
     *  through `substrings_fold_needle`, and the fold ladder is not reachable from device code.
     *
     *  The ordering is a counting sort on folded length, which turns "still alive at this depth" into a
     *  contiguous suffix - so the derivation walks the sum of the lengths rather than the needle count times
     *  the longest one, and a single long needle stops being a cost every other needle pays for.
     */
    template <typename needles_type_>
    cuda_status_t try_upload_needles_(needles_type_ const &needles, cuda_executor_t const &executor) noexcept {

        size_t const count = needles.size();
        safe_vector<byte_t, byte_allocator_t> folded(alloc_), tape(alloc_);
        safe_vector<size_t, offset_allocator_t> starts(alloc_), lengths(alloc_);
        if (starts.try_reserve(count) != status_t::success_k || lengths.try_reserve(count) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        // One folded byte can stand for up to `sz_utf8_fold_max_contraction_k` source bytes, and one source
        // byte for up to `sz_utf8_fold_max_expansion_k` folded ones, so a folded length brackets rather than
        // fixes the source span. Cased needles fold to themselves, so their bounds stay exact.
        size_t const contraction = case_sensitivity_ == substrings_uncased_k ? (size_t)sz_utf8_fold_max_contraction_k
                                                                             : (size_t)1;
        size_t const expansion = case_sensitivity_ == substrings_uncased_k ? (size_t)sz_utf8_fold_max_expansion_k
                                                                           : (size_t)1;

        // Staged host-side first, because the tape's total length is only known once every needle is folded.
        size_t longest = 0;
        max_source_match_bytes_ = 0, min_source_match_bytes_ = 0;
        for (auto const &needle : needles) {
            span<byte_t const> const source = to_bytes_view(needle);
            if (source.size() == 0) return {status_t::unexpected_dimensions_k, cudaSuccess};
            span<byte_t const> spelled = source;
            if (case_sensitivity_ == substrings_uncased_k) {
                folded.clear();
                if (status_t const status = substrings_fold_needle(source, folded); status != status_t::success_k)
                    return {status, cudaSuccess};
                spelled = {folded.data(), folded.size()};
            }
            if (starts.try_push_back(tape.size()) != status_t::success_k ||
                lengths.try_push_back(spelled.size()) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            if (tape.try_append(spelled) != status_t::success_k) return {status_t::bad_alloc_k, cudaSuccess};
            longest = sz_max_of_two(longest, spelled.size());

            // A folded walk snaps both ends of a match outward to whole codepoints, so a reported source span
            // reaches one rune past this ceiling - the same reach the walkers budget for.
            size_t const source_ceiling = spelled.size() * contraction;
            size_t const source_floor = (spelled.size() + expansion - 1) / expansion;
            if (source_ceiling + (size_t)sz_rune_4bytes_k > (size_t)std::numeric_limits<small_size_t>::max())
                return {status_t::overflow_risk_k, cudaSuccess};
            max_source_match_bytes_ = sz_max_of_two(max_source_match_bytes_, source_ceiling);
            min_source_match_bytes_ = min_source_match_bytes_ ? sz_min_of_two(min_source_match_bytes_, source_floor)
                                                              : source_floor;
        }
        // One state per tape byte plus the root is the most a vocabulary can spell, and the widest array
        // below runs one past that, so the tape is refused wherever `tape + 2` would not itself fit. Every
        // size and id derived from the tape is a `small_size_t`, and none of them may wrap.
        if (tape.size() + 2 > (size_t)std::numeric_limits<small_size_t>::max())
            return {status_t::unexpected_dimensions_k, cudaSuccess};

        // Counting sort by folded length. Its cursor array is left holding, for each depth, how many needles
        // are spent by then - which is exactly where that depth's live suffix begins.
        if (needle_first_live_.try_resize(longest + 2) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        // `try_resize` leaves a trivial type uninitialized, and this one is a histogram before it is a cursor.
        for (size_t depth = 0; depth != needle_first_live_.size(); ++depth) needle_first_live_[depth] = 0;
        for (size_t needle = 0; needle != count; ++needle) ++needle_first_live_[lengths[needle]];
        for (size_t depth = 0, running = 0; depth != needle_first_live_.size(); ++depth) {
            size_t const here = needle_first_live_[depth];
            needle_first_live_[depth] = (small_size_t)running;
            running += here;
        }

        safe_vector<substrings_trie_cursor_t, cursor_allocator_t> sorted_cursors(alloc_);
        safe_vector<substrings_trie_ending_t, ending_allocator_t> sorted_endings(alloc_);
        if (sorted_cursors.try_resize(count) != status_t::success_k ||
            sorted_endings.try_resize(count) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        for (size_t needle = 0; needle != count; ++needle) {
            small_size_t const slot = needle_first_live_[lengths[needle]]++;
            sorted_cursors[slot] = {(small_size_t)starts[needle], 0}; // ? Every needle starts on the root
            sorted_endings[slot] = {(small_size_t)needle, 0, (small_size_t)lengths[needle]};
        }

        if (needle_bytes_.try_resize_uninitialized(tape.size()) != status_t::success_k ||
            needle_cursors_.try_resize_uninitialized(count) != status_t::success_k ||
            needle_endings_.try_resize_uninitialized(count) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (CUresult const copied = cuMemcpyHtoDAsync((CUdeviceptr)needle_bytes_.data(), tape.data(), tape.size(),
                                                      executor.stream());
            copied != CUDA_SUCCESS)
            return make_cuda_status(copied);
        if (CUresult const copied = cuMemcpyHtoDAsync((CUdeviceptr)needle_cursors_.data(), sorted_cursors.data(),
                                                      count * sizeof(substrings_trie_cursor_t), executor.stream());
            copied != CUDA_SUCCESS)
            return make_cuda_status(copied);
        if (CUresult const copied = cuMemcpyHtoDAsync((CUdeviceptr)needle_endings_.data(), sorted_endings.data(),
                                                      count * sizeof(substrings_trie_ending_t), executor.stream());
            copied != CUDA_SUCCESS)
            return make_cuda_status(copied);
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Derives the trie on the device, one depth at a time: mark, scan, mint, advance.
     *
     *  A state's 256 possible edges are one `sz_byteset_t` row, so marking is exact and needs no key to say
     *  what it marked. Scanning one depth's out-degrees is what mints the next depth's ids, which is why they
     *  come out dense, depth-ascending, and grouped under each parent in byte order.
     *
     *  The per-state arrays are sized once, against the tape: the host learns a depth's size only from that
     *  depth's scan, and `try_resize_uninitialized` discards what it holds when it grows, so growing later
     *  would discard the depths already derived. One state per tape byte is the most any vocabulary spells.
     */
    cuda_status_t try_derive_trie_(substrings_cuda_kernels_t const &kernel_table, gpu_specs_t const &specs,
                                   cuda_executor_t const &executor) noexcept {

        size_t const count = needle_cursors_.size();
        size_t const tape = needle_bytes_.size();
        count_states_ = 1;
        if (count == 0) return {status_t::success_k, cudaSuccess};
        size_t const longest = needle_first_live_.size() - 2;

        // Sized in `size_t` and bounded by the tape ceiling `try_upload_needles_` enforced, so none of these
        // can wrap the width the ids themselves are carried at.
        if (trie_parent_of_.try_resize_uninitialized(tape + 1) != status_t::success_k ||
            trie_byte_of_.try_resize_uninitialized(tape + 1) != status_t::success_k ||
            trie_links_.try_resize_uninitialized(tape + 2) != status_t::success_k ||
            trie_rows_.try_resize_uninitialized(count) != status_t::success_k ||
            // One past the widest depth, because the scan leaves that depth's total trailing its offsets.
            trie_degrees_.try_resize_uninitialized(count + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        // The root is state zero at depth zero, it is its own parent, and every needle starts on it.
        if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)trie_parent_of_.data(), 0, sizeof(small_size_t),
                                                     executor.stream());
            cleared != CUDA_SUCCESS)
            return make_cuda_status(cleared);

        substrings_trie_needles_t needles_argument = trie_needles_();
        substrings_trie_arrays_t arrays_argument = trie_arrays_();
        sz_byteset_t *rows_argument = trie_rows_.data();
        small_size_t *degrees_argument = trie_degrees_.data();
        small_size_t count_argument = (small_size_t)count;
        // The launch protocol takes the address of every argument, so the loop's own state is what it passes -
        // a shadow copy per depth would only be one more thing to keep in step.
        small_size_t states_first = 0, states_last = 1, depth = 0, first_live = 0, dying = 0;
        trie_band_firsts_.clear();
        if (trie_band_firsts_.try_push_back(0u) != status_t::success_k) return {status_t::bad_alloc_k, cudaSuccess};
        for (; depth <= longest; ++depth) {
            small_size_t const level = states_last - states_first;
            first_live = needle_first_live_[depth];

            if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)rows_argument, 0, level * sizeof(sz_byteset_t),
                                                         executor.stream());
                cleared != CUDA_SUCCESS)
                return make_cuda_status(cleared);
            if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)degrees_argument, 0, level * sizeof(small_size_t),
                                                         executor.stream());
                cleared != CUDA_SUCCESS)
                return make_cuda_status(cleared);

            void *mark_arguments[7] = {&needles_argument, &first_live,    &count_argument,  &depth,
                                       &states_first,     &rows_argument, &degrees_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.trie_mark_edges, count - first_live, specs,
                                                            executor, mark_arguments);
                launched.status != status_t::success_k)
                return launched;

            cuda_status_t const scanned = cuda_launch_exclusive_sum_(
                kernel_table.exclusive_sum_u32, trie_degrees_.data(), level, trie_degrees_.data(),
                {scan_partials_.data(), scan_partials_.size()}, specs, executor.stream());
            if (scanned.status != status_t::success_k) return scanned;

            // The depth's own size is the one thing the host cannot predict, and the next resize waits on it.
            small_size_t minted = 0;
            if (CUresult const synchronized = cuStreamSynchronize(executor.stream()); synchronized != CUDA_SUCCESS)
                return make_cuda_status(synchronized);
            if (CUresult const read = cuMemcpyDtoH(&minted, (CUdeviceptr)(trie_degrees_.data() + level),
                                                   sizeof(small_size_t));
                read != CUDA_SUCCESS)
                return make_cuda_status(read);

            // Runs even when the depth minted nothing, because it is also what turns a childless row's own
            // entry from the scan's relative zero into the absolute end of the trie.
            void *emit_arguments[5] = {&rows_argument, &degrees_argument, &states_first, &states_last,
                                       &arrays_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.trie_emit_states, level, specs, executor,
                                                            emit_arguments);
                launched.status != status_t::success_k)
                return launched;
            if (minted == 0) break; // ? No needle reaches this deep, so none can move

            dying = needle_first_live_[depth + 1];
            void *advance_arguments[6] = {&needles_argument, &first_live, &count_argument, &dying, &depth,
                                          &arrays_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.trie_advance_needles, count - first_live,
                                                            specs, executor, advance_arguments);
                launched.status != status_t::success_k)
                return launched;

            states_first = states_last, states_last += minted;
            if (trie_band_firsts_.try_push_back(states_first) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
        }

        if (trie_band_firsts_.try_push_back(states_last) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        count_states_ = states_last;
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Resolves every failure link, one depth band at a time.
     *
     *  Bands run shallowest first because a failure state is strictly shallower, so each launch resolves
     *  against bands already final. The root's band is skipped: it fails to itself, which the zeroing below
     *  already says.
     */
    cuda_status_t try_link_failures_(substrings_cuda_kernels_t const &kernel_table, gpu_specs_t const &specs,
                                     cuda_executor_t const &executor) noexcept {

        substrings_trie_arrays_t arrays_argument = trie_arrays_();
        for (size_t band = 1; band + 1 < trie_band_firsts_.size(); ++band) {
            u32_t band_first = trie_band_firsts_[band], band_last = trie_band_firsts_[band + 1];
            if (band_first == band_last) break;
            void *arguments[3] = {&arrays_argument, &band_first, &band_last};
            if (cuda_status_t const launched = launch_over_(kernel_table.trie_link_failures, band_last - band_first,
                                                            specs, executor, arguments);
                launched.status != status_t::success_k)
                return launched;
        }
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Packs every state into the double array, in the three tiers the arena's own shape asks for.
     *
     *  Choosing a row's base depends only on the occupancy bitmap, and assigning published ids is a gather
     *  that runs after every base is chosen - so the tiers are free to run in whatever order packs best,
     *  rather than in the depth order the host is bound to.
     */
    cuda_status_t try_pack_(substrings_cuda_kernels_t const &kernel_table, gpu_specs_t const &specs,
                            cuda_executor_t const &executor) noexcept {

        small_size_t const states = (small_size_t)count_states_;
        small_size_t const hot = (small_size_t)hot_count_;

        // A stranded row takes an alphabet of virgin ground, so the arena is grown once the tally is known
        // rather than provisioned for a worst case no vocabulary reaches. The alphabet above the headroom is
        // what makes the one-byte rows placeable: they may take neither the lowest slots, which would put a
        // base before the arena, nor the highest, which would put `base + byte` past it, and every state
        // still has to fit in what is left.
        small_size_t slots = states + (small_size_t)substrings_cold_slot_headroom_k +
                             (small_size_t)substrings_alphabet_size_k;
        if (pack_slot_of_.try_resize_uninitialized(states) != status_t::success_k ||
            pack_base_of_.try_resize_uninitialized(states) != status_t::success_k ||
            pack_rows_.try_resize_uninitialized((size_t)states * 2) != status_t::success_k ||
            pack_counters_.try_resize_uninitialized(5) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        if (cuda_status_t const sized = try_grow_bitmap_(slots, executor); sized.status != status_t::success_k)
            return sized;
        if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)pack_counters_.data(), 0, 5 * sizeof(small_size_t),
                                                     executor.stream());
            cleared != CUDA_SUCCESS)
            return make_cuda_status(cleared);

        substrings_trie_arrays_t arrays_argument = trie_arrays_();
        small_size_t *counts_argument = pack_counters_.data();
        small_size_t *wide_argument = pack_rows_.data();
        small_size_t *narrow_argument = pack_rows_.data() + states;
        small_size_t *base_argument = pack_base_of_.data();
        small_size_t *slot_argument = pack_slot_of_.data();
        u64_t *occupied_argument = pack_occupied_.data();
        small_size_t states_argument = states, hot_argument = hot;

        void *partition_arguments[6] = {&arrays_argument, &states_argument, &hot_argument,
                                        &wide_argument,   &narrow_argument, &counts_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_partition_rows, states, specs, executor,
                                                        partition_arguments);
            launched.status != status_t::success_k)
            return launched;

        // The hot tier owns `[0, hot)` outright, and its children take the run directly above it.
        small_size_t *cursor_argument = pack_counters_.data() + 2;
        if (CUresult const seeded = cuMemcpyHtoDAsync((CUdeviceptr)cursor_argument, &hot, sizeof(small_size_t),
                                                      executor.stream());
            seeded != CUDA_SUCCESS)
            return make_cuda_status(seeded);
        void *hot_arguments[4] = {&arrays_argument, &hot_argument, &slot_argument, &cursor_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_hot_children, sz_max_of_two(hot, 1u), specs,
                                                        executor, hot_arguments);
            launched.status != status_t::success_k)
            return launched;

        small_size_t counters[5] = {0, 0, 0, 0, 0};
        if (cuda_status_t const read = read_counters_(counters, executor); read.status != status_t::success_k)
            return read;
        small_size_t wide_count = counters[0], narrow_count = counters[1];
        small_size_t const floor_slot = counters[2];

        // Every slot below the cursor is claimed and they are contiguous, so the bitmap is marked in whole
        // words plus one partial rather than one exchange per child.
        if (cuda_status_t const marked = mark_prefix_claimed_(floor_slot, executor);
            marked.status != status_t::success_k)
            return marked;

        small_size_t floor_argument = floor_slot;
        small_size_t *stranded_argument = pack_rows_.data() + states + narrow_count;
        small_size_t *stranded_count_argument = pack_counters_.data() + 3;
        void *wide_arguments[8] = {&arrays_argument,   &wide_argument, &wide_count,        &floor_argument,
                                   &occupied_argument, &base_argument, &stranded_argument, &stranded_count_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_wide_rows, (size_t)wide_count * 32, specs,
                                                        executor, wide_arguments);
            launched.status != status_t::success_k)
            return launched;

        if (cuda_status_t const read = read_counters_(counters, executor); read.status != status_t::success_k)
            return read;
        small_size_t const stranded_count = counters[3];
        if (stranded_count) {
            small_size_t const frontier = slots;
            slots += stranded_count * (small_size_t)substrings_alphabet_size_k;
            if (cuda_status_t const grown = try_grow_bitmap_(slots, executor); grown.status != status_t::success_k)
                return grown;
            occupied_argument = pack_occupied_.data();
            small_size_t frontier_argument = frontier, stranded_total = stranded_count;
            void *stranded_arguments[6] = {&arrays_argument,   &stranded_argument, &stranded_total,
                                           &frontier_argument, &occupied_argument, &base_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.pack_stranded_rows, stranded_count, specs,
                                                            executor, stranded_arguments);
                launched.status != status_t::success_k)
                return launched;
        }

        // The narrow rows take vacancies by rank, so each bitmap word needs to know where its own run begins.
        small_size_t const words = (slots + 63u) / 64u;
        if (pack_ranks_.try_resize_uninitialized((size_t)words + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        small_size_t *ranks_argument = pack_ranks_.data();
        small_size_t words_argument = words;
        small_size_t published_argument = slots - (small_size_t)substrings_cold_slot_headroom_k;
        void *vacancy_arguments[4] = {&occupied_argument, &words_argument, &published_argument, &ranks_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_count_vacancies, words, specs, executor,
                                                        vacancy_arguments);
            launched.status != status_t::success_k)
            return launched;
        if (cuda_status_t const scanned = cuda_launch_exclusive_sum_(
                kernel_table.exclusive_sum_u32, pack_ranks_.data(), words, pack_ranks_.data(),
                {scan_partials_.data(), scan_partials_.size()}, specs, executor.stream());
            scanned.status != status_t::success_k)
            return scanned;

        void *narrow_arguments[8] = {&arrays_argument, &narrow_argument,    &narrow_count,      &ranks_argument,
                                     &words_argument,  &published_argument, &occupied_argument, &base_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_narrow_rows, words, specs, executor,
                                                        narrow_arguments);
            launched.status != status_t::success_k)
            return launched;

        small_size_t *bound_argument = pack_counters_.data() + 4;
        void *publish_arguments[6] = {&arrays_argument, &states_argument, &hot_argument,
                                      &base_argument,   &slot_argument,   &bound_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.pack_publish_ids, states, specs, executor,
                                                        publish_arguments);
            launched.status != status_t::success_k)
            return launched;

        // One past the highest slot a state took, as the host derives it too - the arrays then run
        // `substrings_cold_slot_headroom_k` past that, so a `base[state] + byte` lookup from the highest
        // owned slot still lands inside them.
        small_size_t counters_after[5] = {0, 0, 0, 0, 0};
        if (cuda_status_t const read = read_counters_(counters_after, executor); read.status != status_t::success_k)
            return read;
        slots_ = counters_after[4];
        return {status_t::success_k, cudaSuccess};
    }

    /**
     *  @brief Builds the failure-merged output pool, one depth band at a time.
     *
     *  Totals come first for every band, because a state's run length is its own needles plus its failure
     *  state's whole run - and only once every length is known can the scan say where each run begins.
     */
    cuda_status_t try_merge_outputs_(substrings_cuda_kernels_t const &kernel_table, gpu_specs_t const &specs,
                                     cuda_executor_t const &executor) noexcept {

        small_size_t const states = (small_size_t)count_states_;
        small_size_t const needles = (small_size_t)needle_endings_.size();
        if (outputs_own_.try_resize_uninitialized(states) != status_t::success_k ||
            outputs_totals_.try_resize_uninitialized((size_t)states + 1) != status_t::success_k ||
            outputs_offsets_.try_resize_uninitialized((size_t)states + 1) != status_t::success_k ||
            outputs_written_.try_resize_uninitialized(states) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};
        for (auto *zeroed : {&outputs_own_, &outputs_written_})
            if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)zeroed->data(), 0, states * sizeof(small_size_t),
                                                         executor.stream());
                cleared != CUDA_SUCCESS)
                return make_cuda_status(cleared);
        if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)outputs_totals_.data(), 0,
                                                     ((size_t)states + 1) * sizeof(small_size_t), executor.stream());
            cleared != CUDA_SUCCESS)
            return make_cuda_status(cleared);

        substrings_trie_arrays_t arrays_argument = trie_arrays_();
        substrings_trie_needles_t needles_argument = trie_needles_();
        small_size_t needles_argument_count = needles;
        small_size_t *own_argument = outputs_own_.data();
        small_size_t *totals_argument = outputs_totals_.data();
        small_size_t *longest_argument = outputs_totals_.data() + states;
        void *count_arguments[3] = {&needles_argument, &needles_argument_count, &own_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.outputs_count_own, needles, specs, executor,
                                                        count_arguments);
            launched.status != status_t::success_k)
            return launched;
        // The root inherits nothing, so its own tally is already its total - and every depth-one state reads
        // it, which means it has to be there before the first band runs rather than after the last.
        if (CUresult const seeded = cuMemcpyDtoDAsync((CUdeviceptr)outputs_totals_.data(),
                                                      (CUdeviceptr)outputs_own_.data(), sizeof(small_size_t),
                                                      executor.stream());
            seeded != CUDA_SUCCESS)
            return make_cuda_status(seeded);

        // The root carries whatever ends on it and inherits nothing, so the bands walk from depth one.
        for (size_t band = 1; band + 1 < trie_band_firsts_.size(); ++band) {
            small_size_t band_first = trie_band_firsts_[band], band_last = trie_band_firsts_[band + 1];
            if (band_first == band_last) break;
            void *merge_arguments[6] = {&arrays_argument, &band_first,      &band_last,
                                        &own_argument,    &totals_argument, &longest_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.outputs_merge_band, band_last - band_first,
                                                            specs, executor, merge_arguments);
                launched.status != status_t::success_k)
                return launched;
        }

        // Read before the scan overwrites nothing it needs: the longest run rides the trailing entry, which
        // the scan below reads as one of its inputs and leaves alone in its output.
        small_size_t longest_run = 0;
        if (CUresult const synced = cuStreamSynchronize(executor.stream()); synced != CUDA_SUCCESS)
            return make_cuda_status(synced);
        if (CUresult const read = cuMemcpyDtoH(&longest_run, (CUdeviceptr)longest_argument, sizeof(small_size_t));
            read != CUDA_SUCCESS)
            return make_cuda_status(read);
        max_outputs_per_state_ = longest_run;

        if (cuda_status_t const scanned = cuda_launch_exclusive_sum_(
                kernel_table.exclusive_sum_u32, outputs_totals_.data(), states, outputs_offsets_.data(),
                {scan_partials_.data(), scan_partials_.size()}, specs, executor.stream());
            scanned.status != status_t::success_k)
            return scanned;

        small_size_t pool_size = 0;
        if (CUresult const synced = cuStreamSynchronize(executor.stream()); synced != CUDA_SUCCESS)
            return make_cuda_status(synced);
        if (CUresult const read = cuMemcpyDtoH(&pool_size, (CUdeviceptr)(outputs_offsets_.data() + states),
                                               sizeof(small_size_t));
            read != CUDA_SUCCESS)
            return make_cuda_status(read);
        if (outputs_pool_.try_resize_uninitialized(sz_max_of_two(pool_size, 1u)) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        small_size_t *offsets_argument = outputs_offsets_.data();
        small_size_t *written_argument = outputs_written_.data();
        small_size_t *pool_argument = outputs_pool_.data();
        void *place_arguments[5] = {&needles_argument, &needles_argument_count, &offsets_argument, &written_argument,
                                    &pool_argument};
        if (cuda_status_t const launched = launch_over_(kernel_table.outputs_place_own, needles, specs, executor,
                                                        place_arguments);
            launched.status != status_t::success_k)
            return launched;

        for (size_t band = 1; band + 1 < trie_band_firsts_.size(); ++band) {
            small_size_t band_first = trie_band_firsts_[band], band_last = trie_band_firsts_[band + 1];
            if (band_first == band_last) break;
            void *fill_arguments[9] = {&arrays_argument, &needles_argument, &needles_argument_count,
                                       &band_first,      &band_last,        &offsets_argument,
                                       &totals_argument, &written_argument, &pool_argument};
            if (cuda_status_t const launched = launch_over_(kernel_table.outputs_fill_band, band_last - band_first,
                                                            specs, executor, fill_arguments);
                launched.status != status_t::success_k)
                return launched;
        }
        outputs_total_ = pool_size;
        return {status_t::success_k, cudaSuccess};
    }

    /** Grows the occupancy bitmap to @p slots bits, zeroing whatever the growth exposed. */
    cuda_status_t try_grow_bitmap_(small_size_t slots, cuda_executor_t const &executor) noexcept {
        size_t const words = (size_t)(slots + 63u) / 64u + 1; // ? One spare, for a row read that straddles
        size_t const had = pack_occupied_.size();
        if (words > had) {
            safe_vector<u64_t, device_bitmap_allocator_t> grown;
            if (grown.try_resize_uninitialized(words) != status_t::success_k)
                return {status_t::bad_alloc_k, cudaSuccess};
            if (had)
                if (CUresult const copied = cuMemcpyDtoDAsync((CUdeviceptr)grown.data(),
                                                              (CUdeviceptr)pack_occupied_.data(), had * sizeof(u64_t),
                                                              executor.stream());
                    copied != CUDA_SUCCESS)
                    return make_cuda_status(copied);
            if (CUresult const synced = cuStreamSynchronize(executor.stream()); synced != CUDA_SUCCESS)
                return make_cuda_status(synced);
            pack_occupied_ = std::move(grown);
        }
        if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)(pack_occupied_.data() + had), 0,
                                                     (words - had) * sizeof(u64_t), executor.stream());
            words > had && cleared != CUDA_SUCCESS)
            return make_cuda_status(cleared);
        return {status_t::success_k, cudaSuccess};
    }

    /** Marks slots `[0, claimed)` as taken, which the hot tier leaves contiguous. */
    cuda_status_t mark_prefix_claimed_(small_size_t claimed, cuda_executor_t const &executor) noexcept {
        size_t const whole = claimed / 64u;
        if (whole)
            if (CUresult const set = cuMemsetD8Async((CUdeviceptr)pack_occupied_.data(), 0xFF, whole * sizeof(u64_t),
                                                     executor.stream());
                set != CUDA_SUCCESS)
                return make_cuda_status(set);
        if (u32_t const remainder = claimed & 63u) {
            u64_t const partial = ((u64_t)1 << remainder) - 1;
            if (CUresult const set = cuMemcpyHtoDAsync((CUdeviceptr)(pack_occupied_.data() + whole), &partial,
                                                       sizeof(u64_t), executor.stream());
                set != CUDA_SUCCESS)
                return make_cuda_status(set);
        }
        return {status_t::success_k, cudaSuccess};
    }

    /** Drains the stream and reads the packing's five counters back. */
    cuda_status_t read_counters_(small_size_t (&counters)[5], cuda_executor_t const &executor) noexcept {
        if (CUresult const synced = cuStreamSynchronize(executor.stream()); synced != CUDA_SUCCESS)
            return make_cuda_status(synced);
        if (CUresult const read = cuMemcpyDtoH(counters, (CUdeviceptr)pack_counters_.data(), sizeof(counters));
            read != CUDA_SUCCESS)
            return make_cuda_status(read);
        return {status_t::success_k, cudaSuccess};
    }
};

template <typename state_id_type_>
template <typename allocator_type_>
cuda_status_t aho_corasick_cuda_dictionary<state_id_type_>::try_build( //
    aho_corasick_cuda_builder<allocator_type_> const &builder, substrings_cuda_kernels_t const &kernel_table,
    cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {

    constexpr size_t ceiling = (size_t)std::numeric_limits<state_id_t>::max();
    if (builder.slots_ + substrings_cold_slot_headroom_k > ceiling || builder.count_needles() > ceiling ||
        builder.max_source_match_bytes_ > ceiling || builder.max_outputs_per_state_ > ceiling)
        return {status_t::overflow_risk_k, cudaSuccess};

    small_size_t const states = (small_size_t)builder.count_states_;
    small_size_t const hot = (small_size_t)builder.hot_count_;
    small_size_t const slots = (small_size_t)(builder.slots_ + substrings_cold_slot_headroom_k);
    small_size_t const pool = (small_size_t)builder.outputs_total_;

    // A device allocation is already aligned well past this, so the carve only has to keep each array
    // on its own element boundary.
    scratch_amount_t amount {alignof(std::max_align_t)};
    size_t const at_hot_rows = amount;
    amount += (size_t)hot * substrings_alphabet_size_k * sizeof(state_id_t);
    size_t const at_base = amount;
    amount += (size_t)slots * sizeof(state_id_t);
    size_t const at_check = amount;
    amount += (size_t)slots * sizeof(state_id_t);
    size_t const at_fail = amount;
    amount += (size_t)slots * sizeof(state_id_t);
    size_t const at_counts = amount;
    amount += (size_t)slots * sizeof(state_id_t);
    size_t const at_offsets = amount;
    amount += (size_t)slots * sizeof(size_t);
    size_t const at_outputs = amount;
    amount += (size_t)pool * sizeof(output_t);
    size_t const accepts_words = divide_round_up<size_t>(slots, 32);
    if (automaton_.try_resize_uninitialized(amount) != status_t::success_k ||
        accepts_words_.try_resize_uninitialized(accepts_words) != status_t::success_k)
        return {status_t::bad_alloc_k, cudaSuccess};

    auto *const hot_rows = (state_id_t *)(automaton_.data() + at_hot_rows);
    auto *const base = (state_id_t *)(automaton_.data() + at_base);
    auto *const check = (state_id_t *)(automaton_.data() + at_check);
    auto *const fail = (state_id_t *)(automaton_.data() + at_fail);
    auto *const counts = (state_id_t *)(automaton_.data() + at_counts);
    auto *const offsets = (size_t *)(automaton_.data() + at_offsets);
    auto *const outputs = (output_t *)(automaton_.data() + at_outputs);

    // `base` below it, and `fail`, `counts` and `outputs_offsets` above, all read zero on a slot no state
    // owns; only the publishing passes write the rest. Cleared here rather than in a first loop of the
    // slot pass, whose threads walk states while the writes land on slots - two orders that diverge for
    // every cold state, and would race on exactly the entries the cold tier depends on.
    if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)base, 0, at_check - at_base, executor.stream());
        cleared != CUDA_SUCCESS)
        return make_cuda_status(cleared);
    if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)fail, 0, at_outputs - at_fail, executor.stream());
        cleared != CUDA_SUCCESS)
        return make_cuda_status(cleared);

    // A slot nobody owns must name no state at all, so the whole column starts at the invalid id - the same
    // sentinel the host fills `check` with, and the reason a probe that lands on a vacancy reads as a miss
    // rather than as an edge belonging to whichever state that slot's bit pattern happens to spell.
    if (CUresult const cleared = cuMemsetD8Async((CUdeviceptr)check, 0xFF, at_fail - at_check, executor.stream());
        cleared != CUDA_SUCCESS)
        return make_cuda_status(cleared);

    substrings_trie_arrays_t arrays_argument = builder.trie_arrays_();
    substrings_trie_needles_t needles_argument = builder.trie_needles_();
    small_size_t states_argument = states, hot_argument = hot, slots_argument = slots, pool_argument = pool;
    small_size_t *slot_argument = const_cast<small_size_t *>(builder.pack_slot_of_.data());
    small_size_t *base_of_argument = const_cast<small_size_t *>(builder.pack_base_of_.data());
    small_size_t *totals_argument = const_cast<small_size_t *>(builder.outputs_totals_.data());
    small_size_t *offsets_of_argument = const_cast<small_size_t *>(builder.outputs_offsets_.data());
    small_size_t *pool_data_argument = const_cast<small_size_t *>(builder.outputs_pool_.data());
    state_id_t *check_argument = check, *base_argument = base, *fail_argument = fail;
    state_id_t *counts_argument = counts, *rows_argument = hot_rows;
    size_t *offsets_argument = offsets;
    output_t *outputs_argument = outputs;
    u32_t *accepts_argument = accepts_words_.data();

    void *check_arguments[5] = {&arrays_argument, &states_argument, &hot_argument, &slot_argument, &check_argument};
    if (cuda_status_t const launched = launch_over_(published_shape_of_<state_id_t>(kernel_table.publish_check), states,
                                                    specs, executor, check_arguments);
        launched.status != status_t::success_k)
        return launched;

    void *slot_arguments[11] = {&arrays_argument,  &states_argument, &hot_argument,        &slot_argument,
                                &base_of_argument, &totals_argument, &offsets_of_argument, &base_argument,
                                &fail_argument,    &counts_argument, &offsets_argument};
    if (cuda_status_t const launched = launch_over_(published_shape_of_<state_id_t>(kernel_table.publish_slots), states,
                                                    specs, executor, slot_arguments);
        launched.status != status_t::success_k)
        return launched;

    // Shallowest first, because a hot row is its failure state's row with its own edges written over it.
    state_id_t root_argument = 0;
    for (small_size_t hot_index = 0; hot_index != hot; ++hot_index) {
        small_size_t index_argument = hot_index;
        void *row_arguments[5] = {&arrays_argument, &index_argument, &slot_argument, &root_argument, &rows_argument};
        CUresult const launched = cuda_launch_t {}
                                      .grid(1u)
                                      .block(substrings_threads_per_block_k)
                                      .shared(0)
                                      .stream(executor.stream())
                                      .launch(published_shape_of_<state_id_t>(kernel_table.publish_hot_row).function,
                                              row_arguments);
        if (launched != CUDA_SUCCESS) return make_cuda_status(launched);
    }

    void *outputs_arguments[4] = {&needles_argument, &pool_data_argument, &pool_argument, &outputs_argument};
    if (cuda_status_t const launched = launch_over_(published_shape_of_<state_id_t>(kernel_table.publish_outputs),
                                                    sz_max_of_two((size_t)pool, (size_t)1), specs, executor,
                                                    outputs_arguments);
        launched.status != status_t::success_k)
        return launched;

    void *accepts_arguments[3] = {&counts_argument, &slots_argument, &accepts_argument};
    if (cuda_status_t const launched = launch_over_(published_shape_of_<state_id_t>(kernel_table.publish_accepts),
                                                    slots, specs, executor, accepts_arguments);
        launched.status != status_t::success_k)
        return launched;

    view_.hot_rows = hot_rows, view_.base = base, view_.check = check, view_.fail = fail;
    view_.outputs = outputs, view_.outputs_counts = counts, view_.outputs_offsets = offsets;
    view_.outputs_total = pool;
    view_.hot_count = (state_id_t)hot, view_.state_count = (state_id_t)builder.slots_, view_.root = 0;
    view_.max_source_match_bytes = (state_id_t)builder.max_source_match_bytes_;
    view_.min_source_match_bytes = (state_id_t)builder.min_source_match_bytes_;
    view_.max_outputs_per_state = (state_id_t)builder.max_outputs_per_state_;
    view_.case_sensitivity = builder.case_sensitivity_;
    count_needles_ = builder.count_needles();
    return make_cuda_status(cuStreamSynchronize(executor.stream()));
}

#pragma endregion Device Dictionary

#pragma region Engine

/**
 *  @brief Aho-Corasick-based @b GPU multi-pattern exact/case-folded substring search.
 *  @tparam allocator_type_ The allocator backing this engine's host-reachable scratch; unified memory by
 *          default, so the host can read match totals straight back after a stream synchronize.
 *  @tparam capability_ Any capability including `sz_cap_cuda_k` - the kernels need no generation-specific
 *          instructions, so every combination shares this specialization.
 *
 *  The automaton needs no upload because it never left: `aho_corasick_cuda_builder` derives it in kernels and
 *  publishes it into a device-resident `aho_corasick_cuda_dictionary`. The state-id width follows from the
 *  needle set rather than from a template argument, so the engine holds whichever of the two the build
 *  settled on, and construction state dies with the builder `try_index` scoped it to.
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
    using narrow_dictionary_t = aho_corasick_cuda_dictionary<u16_t>;
    using wide_dictionary_t = aho_corasick_cuda_dictionary<u32_t>;
    using builder_t = aho_corasick_cuda_builder<allocator_t>;
    using match_t = substrings_match_t;
    static constexpr sz_capability_t capability_k = capability_;

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    /** Rebinds to `size_t`, for the output CSR and the chunk offsets, neither of which has a ceiling. */
    using offset_allocator_t = typename allocator_traits_t::template rebind_alloc<size_t>;
    using descriptor_allocator_t = typename allocator_traits_t::template rebind_alloc<span<byte_t const>>;
    using byte_allocator_t = typename allocator_traits_t::template rebind_alloc<byte_t>;
    /*  Scratch no host ever touches - written by one kernel, read by the next, or drained by a copy. Unified
     *  memory would fault it in on first touch and migrate it again on the drain, so it is device-resident by
     *  the algorithm's nature rather than by the caller's allocator choice. @sa `similarities/cuda.cuh`. */
    using device_match_allocator_t = device_alloc<substrings_match_t>;
    using device_offset_allocator_t = device_alloc<size_t>;
    using device_word_allocator_t = device_alloc<u32_t>;

    /** One descriptor per haystack. Unified, as the host writes them and every chunk thread reads them. */
    safe_vector<span<byte_t const>, descriptor_allocator_t> haystack_descriptors_ {};
    /** Per-call scratch: chunk count per haystack, then - in place - the exclusive chunk-index offset per haystack,
     *  with the grand total chunk count trailing at `[haystack_count]`. */
    safe_vector<size_t, offset_allocator_t> haystack_chunk_offsets_ {};
    /** Per-call scratch: holds per-chunk counts, then - in place - per-chunk exclusive offsets, with the grand total
     *  in the trailing slot. Grown as needed, reused across calls. */
    safe_vector<size_t, offset_allocator_t> chunk_match_offsets_ {};
    /**
     *  @brief Every match the walk emitted, before any cover has been applied to them.
     *
     *  Under a cover this is an intermediate rather than a result - all three entry points walk into it and
     *  then decide between what it holds - so it is engine scratch the caller never sees.
     */
    safe_vector<substrings_match_t, device_match_allocator_t> emitted_matches_ {};
    /** One flag per emitted match going in, its scanned slot coming out, with the survivor count trailing - the same
     *  in-place trick `chunk_match_offsets_` plays, and for the same reason. */
    safe_vector<size_t, device_offset_allocator_t> cover_keep_ {};
    /** Tile totals for `cuda_launch_exclusive_sum_`'s multi-block route at `size_t`, sized at its grid ceiling. */
    safe_vector<size_t, device_offset_allocator_t> scan_partials_ {};
    /** The survivors themselves, gathered out of the emitted list. */
    safe_vector<substrings_match_t, device_match_allocator_t> cover_survivors_ {};
    /** Per-haystack match boundaries, the reported twin of `haystack_chunk_offsets_`. */
    safe_vector<size_t, offset_allocator_t> haystack_match_offsets_ {};
    /** Where each match's preceding gap begins, relative to its haystack's rewritten start. The only thing the copy
     *  kernel cannot recompute, since it is the running drift the scan produced. */
    safe_vector<size_t, device_offset_allocator_t> rewrite_gap_offsets_ {};
    /** The replacements as one tape, uploaded per call: the caller's container is host-addressed. */
    safe_vector<byte_t, byte_allocator_t> replacement_bytes_ {};
    /** Where each needle's replacement starts in `replacement_bytes_`, with a trailing terminator. */
    safe_vector<size_t, offset_allocator_t> replacement_offsets_ {};

    /** One `needle_count`-wide row per resident block, catching what will not seat in that block's table. Empty for
     *  any dictionary the table can hold, which is most of them. */
    safe_vector<u32_t, device_word_allocator_t> bm25_overflow_ {};

    /**
     *  @brief The automaton this engine compiles from its needles, at whichever state-id width it fits.
     *
     *  Built on the device and left there, so `dictionary_.view()` is already what the kernels read - there
     *  is no second copy to keep in step with it. Construction state belongs to the builder that filled it,
     *  which is a local of `try_index` and hands its device memory back the moment indexing ends.
     */
    std::variant<narrow_dictionary_t, wide_dictionary_t> dictionary_;

    /** Hot rows staged into shared memory at block start - all of them, or zero when they would cost a resident block
     *  and the kernels read them through the cache instead. */
    u32_t staged_rows_ {};
    /** Acceptance bitmap words staged alongside `staged_rows_`, under the same all-or-nothing rule. */
    u32_t staged_accepts_words_ {};
    allocator_t alloc_ {};
    cuda_timer_t timer_ {};

  public:
    substrings_cuda() noexcept = default;
    substrings_cuda(substrings_cuda const &) = delete;
    substrings_cuda &operator=(substrings_cuda const &) = delete;
    substrings_cuda(substrings_cuda &&) noexcept = default;
    substrings_cuda &operator=(substrings_cuda &&) noexcept = default;

    /** Releases the automaton and every device-resident buffer this engine owns; a fresh `try_insert_all` is required
     *  after. */
    void reset() noexcept {
        std::visit([](auto &dictionary) noexcept { dictionary.reset(); }, dictionary_);
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

    /** The state-id width this engine's automaton settled on, once finalized. */
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
    substrings_case_sensitivity_t case_sensitivity() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return dictionary.case_sensitivity(); }, dictionary_);
    }

    /** Runs @p callable against the automaton at whichever state-id width it settled on. */
    template <typename callable_type_>
    auto visit_dictionary(callable_type_ &&callable) const noexcept {
        return std::visit(std::forward<callable_type_>(callable), dictionary_);
    }

#pragma region Kernel Resolution

    using kernels_t = substrings_cuda_kernels_t;

    /** Resolves every kernel handle for @p device_id into @p table, raising the dynamic shared-memory ceiling on the
     *  two chunk kernels to the device's opt-in maximum. The per-launch allocation depends on the dictionary's hot-
     *  tier size, so occupancy is queried per launch. Both state-id widths are resolved into one table, since the
     *  automaton picks its own. */
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

        status = resolve_kernel_shape(table.exclusive_sum_u32.whole,
                                      reinterpret_cast<void const *>(&exclusive_sum_across_cuda_device_<u32_t>), 0, 0,
                                      false);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.exclusive_sum_u32.reduce_tiles,
            reinterpret_cast<void const *>(&exclusive_sum_reduce_tiles_across_cuda_device_<u32_t>),
            cuda_device_collective_threads_k, 0, true);
        if (status.status != status_t::success_k) return status;

        status = resolve_kernel_shape(
            table.exclusive_sum_u32.apply_tiles,
            reinterpret_cast<void const *>(&exclusive_sum_apply_tiles_across_cuda_device_<u32_t>),
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

        // The derivation stages nothing, so none of the three takes a shared ceiling.
        status = resolve_kernel_shape(table.trie_mark_edges,
                                      reinterpret_cast<void const *>(&substrings_trie_mark_edges_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;
        status = resolve_kernel_shape(table.trie_emit_states,
                                      reinterpret_cast<void const *>(&substrings_trie_emit_states_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;
        status = resolve_kernel_shape(table.trie_advance_needles,
                                      reinterpret_cast<void const *>(&substrings_trie_advance_needles_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;
        status = resolve_kernel_shape(table.trie_link_failures,
                                      reinterpret_cast<void const *>(&substrings_trie_link_failures_),
                                      substrings_threads_per_block_k, 0, true);
        if (status.status != status_t::success_k) return status;

        // The packing stages nothing either, so every tier resolves the same way the derivation does.
        auto const resolve_pack = [&](kernel_shape_t &shape, void const *function) noexcept -> cuda_status_t {
            return resolve_kernel_shape(shape, function, substrings_threads_per_block_k, 0, true);
        };
        status = resolve_pack(table.pack_partition_rows, (void const *)&substrings_pack_partition_rows_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_hot_children, (void const *)&substrings_pack_hot_children_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_wide_rows, (void const *)&substrings_pack_wide_rows_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_stranded_rows, (void const *)&substrings_pack_stranded_rows_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_count_vacancies, (void const *)&substrings_pack_count_vacancies_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_narrow_rows, (void const *)&substrings_pack_narrow_rows_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.pack_publish_ids, (void const *)&substrings_pack_publish_ids_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.outputs_count_own, (void const *)&substrings_outputs_count_own_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.outputs_merge_band, (void const *)&substrings_outputs_merge_band_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.outputs_place_own, (void const *)&substrings_outputs_place_own_);
        if (status.status != status_t::success_k) return status;
        status = resolve_pack(table.outputs_fill_band, (void const *)&substrings_outputs_fill_band_);
        if (status.status != status_t::success_k) return status;

        auto const resolve_publish = [&]<typename state_id_type_>() noexcept -> cuda_status_t {
            kernel_shape_t &check = published_shape_of_<state_id_type_>(table.publish_check);
            kernel_shape_t &slots = published_shape_of_<state_id_type_>(table.publish_slots);
            kernel_shape_t &row = published_shape_of_<state_id_type_>(table.publish_hot_row);
            kernel_shape_t &pool = published_shape_of_<state_id_type_>(table.publish_outputs);
            kernel_shape_t &gate = published_shape_of_<state_id_type_>(table.publish_accepts);
            cuda_status_t at = resolve_pack(check, (void const *)&substrings_publish_check_<state_id_type_>);
            if (at.status != status_t::success_k) return at;
            at = resolve_pack(slots, (void const *)&substrings_publish_slots_<state_id_type_>);
            if (at.status != status_t::success_k) return at;
            at = resolve_pack(row, (void const *)&substrings_publish_hot_row_<state_id_type_>);
            if (at.status != status_t::success_k) return at;
            at = resolve_pack(pool, (void const *)&substrings_publish_outputs_<state_id_type_>);
            if (at.status != status_t::success_k) return at;
            return resolve_pack(gate, (void const *)&substrings_publish_accepts_<state_id_type_>);
        };
        status = resolve_publish.template operator()<u16_t>();
        if (status.status != status_t::success_k) return status;
        status = resolve_publish.template operator()<u32_t>();
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

#pragma endregion Kernel Resolution

    /**
     *  @brief Indexes all of the @p needles strings into the FSM, at whichever state id it ends up fitting.
     *
     *  Only the vocabulary crosses the bus - the derivation, the packing and the publish all run in kernels,
     *  so the automaton is already where the walks read it. The builder is scoped to this call, which is
     *  what returns its device memory the moment the width is settled and the automaton written.
     *  @param[in] executor Names the device whose context every device allocation is made under.
     *  @param[in] specs Sizes the hot tier against that device's L2, the cache its walk reads through.
     *  @note Replaces any previously indexed needle set: the automaton is rebuilt from scratch and the old one
     *        released, so an engine can be re-indexed for a different vocabulary or a different device.
     *  @sa `aho_corasick_cuda_builder::try_derive` for the status codes this forwards.
     */
    template <typename needles_type_>
    cuda_status_t try_index(needles_type_ const &needles,
                            substrings_case_sensitivity_t case_sensitivity = substrings_cased_k,
                            cuda_executor_t const &executor = {}, gpu_specs_t const &specs = {}) noexcept {
        // Every array the build touches is device-resident rather than unified, so it lands wherever the
        // context points; binding the named device first is what keeps it off whichever one was current.
        if (cuda_status_t const current = executor.ensure_current(); current.status != status_t::success_k)
            return current;
        auto [kernel_table, kernels_status] = kernels(executor.device_id());
        if (kernels_status.status != status_t::success_k) return kernels_status;

        // Sized at the scan's grid ceiling rather than per call: it is 8 KB whatever the corpus, and a scan
        // that found it short would silently drop back to one block.
        if (scan_partials_.try_resize_uninitialized(cuda_device_collective_max_blocks_k + 1) != status_t::success_k)
            return {status_t::bad_alloc_k, cudaSuccess};

        builder_t builder(case_sensitivity, alloc_);
        if (cuda_status_t const derived = builder.try_derive(kernel_table, needles, executor, specs);
            derived.status != status_t::success_k)
            return derived;
        return builder.published_width() == substrings_state_width_t::u16_k
                   ? try_settle_dictionary_<u16_t>(builder, kernel_table, executor, specs)
                   : try_settle_dictionary_<u32_t>(builder, kernel_table, executor, specs);
    }

  private:
    /** Publishes @p builder 's automaton at @p state_id_type_ and makes it the one this engine answers from. */
    template <typename state_id_type_>
    cuda_status_t try_settle_dictionary_(builder_t const &builder, kernels_t const &kernel_table,
                                         cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        aho_corasick_cuda_dictionary<state_id_type_> settled;
        if (cuda_status_t const published = settled.try_build(builder, kernel_table, executor, specs);
            published.status != status_t::success_k)
            return published;
        dictionary_.template emplace<aho_corasick_cuda_dictionary<state_id_type_>>(std::move(settled));
        return {status_t::success_k, cudaSuccess};
    }

    /** One cell's width in the settled automaton, which is what a staged hot row costs per entry. */
    size_t bytes_per_state_id_() const noexcept {
        return state_width() == substrings_state_width_t::u16_k ? sizeof(u16_t) : sizeof(u32_t);
    }

    /** Shared bytes a scoring block's counter table costs, read by the occupancy query and by the launch alike so the
     *  two cannot disagree about the footprint they are sizing. A dictionary narrower than the table gets a slot per
     *  needle and pays for no more than that. */
    size_t substrings_bm25_counters_bytes_() const noexcept {
        return sz_min_of_two(count_needles(), substrings_bm25_slots_k) * sizeof(substrings_bm25_counter_t);
    }

    /** The settled automaton at @p state_id_type_, which `try_index` has already pinned. */
    template <typename state_id_type_>
    aho_corasick_cuda_dictionary<state_id_type_> const &settled_dictionary_() const noexcept {
        return std::get<aho_corasick_cuda_dictionary<state_id_type_>>(dictionary_);
    }

    /** The acceptance bitmap the settled automaton published, whatever width it settled on. */
    span<u32_t const> accepts_words_() const noexcept {
        return std::visit([](auto const &dictionary) noexcept { return dictionary.accepts(); }, dictionary_);
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
        int blocks_without_staging = 0;
        CUresult const occupancy_error = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_without_staging, walk_function, (int)substrings_threads_per_block_k, 0);
        if (occupancy_error != CUDA_SUCCESS) return make_cuda_status(occupancy_error);
        unsigned const target_blocks = (unsigned)sz_max_of_two(blocks_without_staging, 1);

        size_t shared_memory_budget = 0;
        cuda_status_t const budget_status = shared_memory_budget_for_resident_blocks(
            shared_memory_budget, walk_function, substrings_threads_per_block_k, target_blocks, executor.device_id());
        if (budget_status.status != status_t::success_k) return budget_status;

        // Staging is all-or-nothing: a partial prefix still pays the copy per block and the bounds test per
        // byte, while most transitions miss it and fall through to memory anyway.
        size_t const bytes_per_state_id = state_width() == substrings_state_width_t::u16_k ? sizeof(u16_t)
                                                                                           : sizeof(u32_t);
        size_t const hot_rows = hot_count();
        size_t const accepts_words = accepts_words_().size();
        size_t const accepts_bytes = accepts_words * sizeof(u32_t);
        size_t const whole_automaton_bytes = hot_rows * substrings_alphabet_size_k * bytes_per_state_id + accepts_bytes;
        // Scoring carries its counter table in the same allocation, so staging must fit beside it or the two
        // would compete for one budget. They do not today - staging is refused for every real dictionary -
        // but that is luck rather than design, and this keeps it true by construction.
        size_t const staging_budget = shared_memory_budget > substrings_bm25_counters_bytes_()
                                          ? shared_memory_budget - substrings_bm25_counters_bytes_()
                                          : 0;
        bool const stages_whole_automaton = whole_automaton_bytes <= staging_budget;
        staged_rows_ = stages_whole_automaton ? static_cast<u32_t>(hot_rows) : u32_t {0};
        staged_accepts_words_ = stages_whole_automaton ? static_cast<u32_t>(accepts_words) : u32_t {0};

        return {status_t::success_k, cudaSuccess};
    }

    /** Everything the counting pass establishes that a following scatter pass still needs. */
    struct planned_pass_t {
        /** This device's resolved table, which outlives every pass that reads it. Null until the plan has work to
         *  name, which is what `has_work` reads rather than a flag of its own. */
        kernels_t const *kernel_table = nullptr;
        /** Dynamic shared memory each walk block takes, and the grid the two passes share. */
        unsigned shared_memory_bytes = 0;
        unsigned blocks_per_grid = 0;
        /** The chunk the corpus was cut into, and how many of them cover it. */
        size_t chunk_bytes = 0;
        size_t chunk_count = 0;

        /** Whether the plan named any work; an empty corpus leaves it unplanned and launches nothing. */
        bool has_work() const noexcept { return kernel_table != nullptr; }
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

    /** Runs everything `try_count` and `try_find` share: the guards, the kernel table, the timer's start, the chunk
     *  plan, and the counting pass whose exclusive scan both then read. */
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
        CUresult const timer_error = timer_.ensure_created(executor.device_id());
        if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
        CUresult const start_error = timer_.record_start(executor.stream());
        if (start_error != CUDA_SUCCESS) return make_cuda_status(start_error);

        cuda_status_t const plan_status = plan_haystack_chunks_(total_bytes, specs, kernel_table,
                                                                pass.shared_memory_bytes, pass.blocks_per_grid,
                                                                pass.chunk_bytes, pass.chunk_count);
        if (plan_status.status != status_t::success_k) return plan_status;

        cuda_status_t const count_status = count_into_offsets_(executor, specs, kernel_table, pass.shared_memory_bytes,
                                                               pass.blocks_per_grid, pass.chunk_bytes,
                                                               pass.chunk_count);
        if (count_status.status != status_t::success_k) return count_status;

        // Published last, because it is also what tells the scatter the plan carries work.
        pass.kernel_table = &kernel_table;
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

    /** Runs the counting pass, then the in-place exclusive scan, so `chunk_match_offsets_` holds every chunk's write
     *  offset with the grand total trailing at `[chunk_count]` - the shared core of `try_count` and `try_find`. */
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
        unsigned const boundary_grid = grid_for_items_(pass.kernel_table->haystack_match_offsets, haystack_count + 1,
                                                       specs);
        CUresult const boundary_error = cuda_launch_t {}
                                            .grid(boundary_grid)
                                            .block(substrings_threads_per_block_k)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .launch(pass.kernel_table->haystack_match_offsets.function,
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

    /** Zeroes the caller's counts through the driver, for a corpus the walk never reaches. */
    cuda_status_t clear_counts_(span<size_t> counts_per_haystack, cuda_executor_t const &executor) noexcept {
        if (counts_per_haystack.size() == 0) return {status_t::success_k, cudaSuccess};
        CUresult const clear_error = cuMemsetD8Async((CUdeviceptr)counts_per_haystack.data(), 0,
                                                     counts_per_haystack.size() * sizeof(size_t), executor.stream());
        if (clear_error != CUDA_SUCCESS) return make_cuda_status(clear_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess};
    }

    /** Zeroes the caller's scores through the driver, for a batch no walk reaches. */
    cuda_status_t clear_scores_(span<f32_t> scores, cuda_executor_t const &executor) noexcept {
        if (scores.size() == 0) return {status_t::success_k, cudaSuccess};
        CUresult const clear_error = cuMemsetD8Async((CUdeviceptr)scores.data(), 0, scores.size() * sizeof(f32_t),
                                                     executor.stream());
        if (clear_error != CUDA_SUCCESS) return make_cuda_status(clear_error);
        CUresult const sync_error = timer_.synchronize(executor.stream());
        if (sync_error != CUDA_SUCCESS) return make_cuda_status(sync_error);
        return {status_t::success_k, cudaSuccess};
    }

    /** Differences the published boundaries into the caller's per-haystack counts, then fences once. */
    cuda_status_t count_from_boundaries_(planned_pass_t const &pass, span<size_t> counts_per_haystack,
                                         cuda_executor_t const &executor, gpu_specs_t const &specs) noexcept {
        span<size_t const> boundaries_argument {haystack_match_offsets_.data(), haystack_match_offsets_.size()};
        span<size_t> counts_argument = counts_per_haystack;
        void *counts_arguments[2] = {&boundaries_argument, &counts_argument};
        unsigned const counts_grid = grid_for_items_(pass.kernel_table->counts_from_boundaries,
                                                     counts_per_haystack.size(), specs);
        CUresult const counts_error = cuda_launch_t {}
                                          .grid(counts_grid)
                                          .block(substrings_threads_per_block_k)
                                          .shared(0)
                                          .stream(executor.stream())
                                          .launch(pass.kernel_table->counts_from_boundaries.function, counts_arguments);
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
        unsigned const resolve_grid = grid_for_items_(pass.kernel_table->cover_resolve, emitted, specs);
        CUresult const resolve_error = cuda_launch_t {}
                                           .grid(resolve_grid)
                                           .block(substrings_threads_per_block_k)
                                           .shared(0)
                                           .stream(executor.stream())
                                           .launch(pass.kernel_table->cover_resolve.function, resolve_arguments);
        if (resolve_error != CUDA_SUCCESS) return make_cuda_status(resolve_error);

        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(
            pass.kernel_table->exclusive_sum, cover_keep_.data(), emitted, cover_keep_.data(),
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
            .grid(grid_for_items_(pass.kernel_table->cover_compact, matches.size(), specs))
            .block(substrings_threads_per_block_k)
            .shared(0)
            .stream(executor.stream())
            .launch(pass.kernel_table->cover_compact.function, arguments);
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
        return launch_walk_(substrings_pass_t::writing_k, *pass.kernel_table, pass.blocks_per_grid,
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
        span<u32_t const> accepts_words_argument = accepts_words_();
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
        if (!pass.has_work()) return clear_counts_(counts_per_haystack, executor);

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
        if (pass_status.status != status_t::success_k || !pass.has_work()) return pass_status;

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
        if (!pass.has_work()) return clear_counts_({output_offsets.data(), output_offsets.size()}, executor);

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
        unsigned const offsets_grid = grid_for_items_(pass.kernel_table->rewrite_offsets, haystack_count, specs);
        CUresult const offsets_error = cuda_launch_t {}
                                           .grid(offsets_grid)
                                           .block(substrings_threads_per_block_k)
                                           .shared(0)
                                           .stream(executor.stream())
                                           .launch(pass.kernel_table->rewrite_offsets.function, offsets_arguments);
        if (offsets_error != CUDA_SUCCESS) return make_cuda_status(offsets_error);

        // The scan writes the caller's own array, so the boundaries are complete before any capacity check -
        // which is what lets a refused call name the exact size it wanted.
        cuda_status_t const scan_status = cuda_launch_exclusive_sum_(
            pass.kernel_table->exclusive_sum, output_offsets.data(), haystack_count, output_offsets.data(),
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
            pass.kernel_table->rewrite_copy,
            divide_round_up(sz_max_of_two(copy_bytes, (size_t)1), substrings_rewrite_tile_bytes_k), specs);

        CUresult const copy_error = cuda_launch_t {}
                                        .grid(copy_grid)
                                        .block(substrings_threads_per_block_k)
                                        .shared(0)
                                        .stream(executor.stream())
                                        .launch(pass.kernel_table->rewrite_copy.function, copy_arguments);
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
    /** The scoring launch, at the width the automaton settled on; the ten arguments stay in one order. */
    template <typename state_id_type_>
    CUresult launch_score_bm25_at_(kernels_t const &kernel_table, unsigned blocks_per_grid,
                                   unsigned shared_memory_bytes, substrings_bm25_t parameters,
                                   span<f32_t const> lengths_argument, span<f32_t const> weights_argument,
                                   span<f32_t> scores_argument, cuda_executor_t const &executor) noexcept {
        aho_corasick_view<state_id_type_> view_argument = settled_dictionary_<state_id_type_>().view();
        state_id_type_ staged_rows_argument = static_cast<state_id_type_>(staged_rows_);
        span<u32_t const> accepts_words_argument = accepts_words_();
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
