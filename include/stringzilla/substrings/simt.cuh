/**
 *  @file include/stringzilla/substrings/simt.cuh
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief The walks CUDA and ROCm share for multi-pattern search: one thread per haystack chunk,
 *      the automaton's hot head staged in shared memory, and the leftmost cover resolved after the
 *      walk rather than inside it.
 *
 *  The transition is the serial tier's, reached from the device through
 *  @c --expt-relaxed-constexpr, so both sides answer the same matches rather than similar ones.
 *
 *  Parallelism comes from @b chunks rather than from haystacks: a corpus of one long document and a
 *  corpus of a million short ones then fill the device the same way. A chunk reports every match
 *  @b ending inside it, priming itself from the <tt>max_source_match_bytes - 1</tt> bytes before its own
 *  start - clamped to its haystack, never earlier - so every match is found exactly once and no
 *  chunk reads a neighbour's text. The walk that counts a chunk's matches also keeps them, the
 *  first few in the chunk's own slots and the rest in a pool the round shares, so once the counts
 *  are scanned a gather copies them to their offsets rather than walking the text again.
 *
 *  A cover is a property of the matches, not of the bytes, so it is resolved after the walk: each
 *  match links to its successor, the first match a leftmost search takes past its end, and pointer
 *  doubling marks the chain from each haystack's first, exactly, in a logarithmic number of rounds.
 *  Inside the walk it would cost every thread a ring wide enough for the longest match, and a
 *  second walk to find a safe place to start; resolving it afterwards needs neither.
 *
 *  When a round outruns its match budget the gather, the cover, the compaction and the boundaries
 *  kernel each retire at their first instruction, so an output is left untouched rather than
 *  truncated. StringZilla's own kernels over the steps and walks here live in
 *  <tt>substrings/kernels.cuh</tt>, which <tt>substrings/cuda.cuh</tt> and <tt>substrings/rocm.cuh</tt> size and launch;
 *  a product including only this header compiles none of them.
 *
 *  @sa c/stringzilla/substrings.h
 */
#ifndef STRINGZILLA_SUBSTRINGS_SIMT_CUH_
#define STRINGZILLA_SUBSTRINGS_SIMT_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/substrings/serial.h"
#include "stringzilla/utf8_uncased_fold/simt.cuh" // `sz_unicode_fold_image_simt_`

#ifdef __cplusplus
extern "C" {
#endif

/*  Plans size a launch from the engine alone, as the host does before it, or the device inside one:
 *  the shapes a round's buffers take, and the bytes a block stages and a walk cuts a value into. */
#pragma region Plans

/** Threads every kernel here launches with; occupancy is shared-memory-bound rather than
 *  thread-bound, so a modest fixed block keeps the launch geometry simple and the block scan one
 *  power of two. */
enum { sz_substrings_threads_per_block_simt_k = 256 };

enum {

    /** Bits of a tally slot index: 4096 key-and-count slots fill 32 KB of static shared memory. */
    sz_substrings_tally_slot_bits_simt_k = 12,

    /** Slots one block's tally holds. */
    sz_substrings_tally_slots_simt_k = 1 << sz_substrings_tally_slot_bits_simt_k,

    /** Slots a hashed tally probes before spilling a needle to the block's overflow row. */
    sz_substrings_tally_probes_simt_k = 16,
};

/** How a tally maps a needle to a slot. */
typedef enum sz_substrings_tally_layout_simt_t {

    /** The vocabulary fits the slots, so a needle's index is its slot. */
    sz_substrings_tally_direct_simt_k = 0,

    /** A larger vocabulary, hashed into the slots with linear probing and an overflow
     *  row behind them. */
    sz_substrings_tally_hashed_simt_k = 1,
} sz_substrings_tally_layout_simt_t;

/** One block's per-needle counts for the haystack it is scoring. */
typedef struct sz_substrings_tally_simt_t {

    /** How @c counts is indexed. */
    sz_substrings_tally_layout_simt_t layout;

    /** In shared memory: each hashed slot's needle index plus one, zero while free;
     *  @c STRINGZILLA_NULL when direct. */
    sz_u32_t *keys;

    /** In shared memory: each slot's occurrences, one slot per needle when direct. */
    sz_u32_t *counts;

    /** The block's own @b [needles] global row for needles no probe could seat,
     *  @c STRINGZILLA_NULL when direct. */
    sz_u32_t *overflow;

    /** In shared memory: nonzero once anything reached @c overflow, so scoring scans
     *  it only then. */
    sz_u32_t *overflowed;
} sz_substrings_tally_simt_t;

/** A match past what its chunk keeps, with where in the chunk's range it lands. */
typedef struct sz_substrings_spill_simt_t {

    /** The chunk the match ends in, and where in its haystack the match starts. */
    sz_size_t chunk_index, byte_offset;

    /** Its place among the chunk's matches, its needle and its span, padded to a match's size. */
    sz_u32_t slot, needle_index, byte_length, padding;
} sz_substrings_spill_simt_t;

/** A scoring round's occurrences of each needle in each haystack, which every chunk of the round
 *  counts into, hashed in global memory. */
typedef struct sz_substrings_pairs_simt_t {

    /** @b [capacity] keys, each a haystack's index times the vocabulary plus the needle's index
     *  plus one, zero while free. */
    sz_u64_t *keys;

    /** @b [capacity] occurrences of each seated key. */
    sz_u32_t *counts;

    /** Slots in the table, and needles in the vocabulary, which a key multiplies a haystack by. */
    sz_size_t capacity, needles_count;

    /** @b [haystacks] flags, set where a haystack's needle found no slot. */
    sz_size_t *overflowed;
} sz_substrings_pairs_simt_t;

/** Where a chunk walk keeps the matches it counts, or tallies them. */
typedef struct sz_substrings_sink_simt_t {

    /** Sizing: words keeping the chunk's first matches, three per match, each @c stride apart, so
     *  neighbouring chunks' stores coalesce, or @c STRINGZILLA_NULL to only count them. */
    sz_u32_t *kept;

    /** Words from one word of a kept match to the next. */
    sz_size_t stride;

    /** Sizing: the round's pool for matches past what their chunk keeps, its fill counter, its
     *  capacity, and the chunk being walked. */
    sz_substrings_spill_simt_t *spills;
    sz_size_t *spilled;
    sz_size_t spills_capacity, chunk_index;

    /** Tallying: the block's tally, or @c STRINGZILLA_NULL to count into the round's @c pairs under
     *  the walked haystack. */
    sz_substrings_tally_simt_t const *tally;
    sz_substrings_pairs_simt_t const *pairs;
    sz_size_t haystack_index;
} sz_substrings_sink_simt_t;

/**
 *  @brief What a block keeps in shared memory beside the engine, which a walk reads in place of the
 *      engine's own arrays.
 *
 *  The engine stays the kernel's read-only parameter: rebinding its pointers to staged copies would
 *  make it a local variable, spilled to the stack and reloaded on every byte.
 */
typedef struct sz_substrings_staged_simt_t {

    /** The engine's class map, in shared memory. */
    sz_u8_t const *classes;

    /** The head of the hot tier, in shared memory. */
    sz_u32_t const *rows;

    /** States the staged head covers. */
    sz_u32_t rows_count;

    /** The acceptance bitmap, staged by the scoring kernel when it fits, else the engine's own. */
    sz_u32_t const *accepts_words;
} sz_substrings_staged_simt_t;

/** Slots a round's pairs table probes before flagging the haystack, whose scoring then falls back
 *  to a tally of its own. */
enum { sz_substrings_pairs_probes_simt_k = 32 };

/** Matches the sizing walk keeps in each chunk's own slots; a busier chunk's others spill into a
 *  pool the round shares. */
enum { sz_substrings_kept_matches_simt_k = 4 };

/** Chunks the engine's budget cuts a round into per resident thread. A walk's cost follows its
 *  chunk's matches and failure chases, so more, smaller chunks spread a heavy region over more
 *  threads and leave the slowest thread less to finish alone. */
enum { sz_substrings_chunks_per_thread_simt_k = 8 };

/** Slots a block's tally takes: one per needle when the vocabulary fits, a
 *  hashed table otherwise. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_tally_slots_for_simt_(sz_size_t needles_count) {
    return needles_count <= sz_substrings_tally_slots_simt_k ? needles_count : sz_substrings_tally_slots_simt_k;
}

/** Words of the acceptance bitmap, one bit per double-array slot. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_accepts_words_simt_(sz_substrings_engine_t const *engine) {
    return sz_size_divide_round_up(engine->slots_count, 32);
}

/** Slots a scoring round's pairs table holds per budgeted match, so a round within its match budget
 *  fills at most half of it. */
enum { sz_substrings_pairs_per_match_simt_k = 2 };

/** Slots of a pairs table each thread of the scoring kernel loads at once. */
enum { sz_substrings_pairs_unroll_simt_k = 4 };

/** Jumps one doubling round chains, so each round reaches this many times as far as the last; many,
 *  as every round costs a launch while a chain's hops stop where it ends. */
enum { sz_substrings_round_hops_simt_k = 8 };

/** Bytes of shared memory a block stages before a walk: the class map, then @p staged_count rows of
 *  the hot tier, as @ref sz_substrings_stage_simt_ lays them out from the buffer's start. */
STRINGZILLA_INLINE sz_size_t sz_substrings_staged_bytes_simt_(sz_substrings_engine_t const *engine,
                                                              sz_u32_t staged_count) {
    return (STRINGZILLA_U8_MAX + 1) + (sz_size_t)staged_count * engine->classes_count * sizeof(sz_u32_t);
}

/** The narrowest chunk a walk cuts a value into: four times the longest match, so the warm-up
 *  before a chunk never outgrows a quarter of it, and at least one byte. */
STRINGZILLA_CONSTEXPR sz_size_t sz_substrings_chunk_bytes_simt_(sz_substrings_engine_t const *engine) {
    return sz_max_of_two(4 * (sz_size_t)engine->max_source_match_bytes, (sz_size_t)1);
}

/**
 *  @brief A copy of @p engine reading its tables from @p device_block, for a product that keeps
 *      them in device memory it owns.
 *  @param[in] device_block The @c engine->memory_bytes bytes of @c engine->memory, copied.
 *
 *  A kernel takes the copy by value, as it takes a compiled engine whose tables it reads in place;
 *  @c scratch and @c report stay where they are.
 */
STRINGZILLA_INLINE sz_substrings_engine_t sz_substrings_engine_rebase_simt_(sz_substrings_engine_t const *engine,
                                                                            void *device_block) {
    sz_substrings_engine_t moved = *engine;
    moved.hot_rows = (sz_u32_t const *)sz_pointer_rebased_(engine->hot_rows, engine->memory, engine->memory_bytes,
                                                           device_block);
    moved.byte_to_class = (sz_u8_t const *)sz_pointer_rebased_(engine->byte_to_class, engine->memory,
                                                               engine->memory_bytes, device_block);
    moved.base = (sz_u32_t const *)sz_pointer_rebased_(engine->base, engine->memory, engine->memory_bytes,
                                                       device_block);
    moved.check = (sz_u32_t const *)sz_pointer_rebased_(engine->check, engine->memory, engine->memory_bytes,
                                                        device_block);
    moved.fail = (sz_u32_t const *)sz_pointer_rebased_(engine->fail, engine->memory, engine->memory_bytes,
                                                       device_block);
    moved.accepts_words = (sz_u32_t const *)sz_pointer_rebased_(engine->accepts_words, engine->memory,
                                                                engine->memory_bytes, device_block);
    moved.outputs = (sz_substrings_output_t const *)sz_pointer_rebased_(engine->outputs, engine->memory,
                                                                        engine->memory_bytes, device_block);
    moved.outputs_counts = (sz_u32_t const *)sz_pointer_rebased_(engine->outputs_counts, engine->memory,
                                                                 engine->memory_bytes, device_block);
    moved.outputs_offsets = (sz_size_t const *)sz_pointer_rebased_(engine->outputs_offsets, engine->memory,
                                                                   engine->memory_bytes, device_block);
    moved.memory = device_block;
    return moved;
}

#pragma endregion Plans

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

/*  Steps advance one byte, word or transition, or carry one block-wide collective, for the walks
 *  and the kernels below. */
#pragma region Steps

/** Four haystack bytes as one read-only load; the address is peeled to its own alignment by the
 *  caller. */
STRINGZILLA_DEVICE sz_u32_t sz_substrings_load_quad_simt_(sz_u8_t const *pointer) {
    return __ldg((unsigned int const *)pointer);
}

/**
 *  @brief One byte's transition, staged-shared-memory-first.
 *
 *  The probe is @ref sz_substrings_step's, spelled again so every automaton load takes the
 *  read-only global path: through the engine's plain pointers the device can only issue generic
 *  loads, which it cannot move past the walk's own match stores. A single cold lane still makes the
 *  whole warp pay that lane's failure-chase depth, which is the cost the staging exists to shrink.
 */
STRINGZILLA_DEVICE sz_u32_t sz_substrings_step_simt_(sz_substrings_engine_t const *engine,
                                                     sz_substrings_staged_simt_t const *staged, sz_u32_t state,
                                                     sz_u8_t byte) {
    sz_u32_t const byte_class = staged->classes[byte];
    // Asked on every chase step, so a failure link into the staged head stays in shared memory.
    for (;;) {
        if (state < staged->rows_count) return staged->rows[(sz_size_t)state * engine->classes_count + byte_class];
        if (state < engine->hot_count)
            return __ldg(engine->hot_rows + (sz_size_t)state * engine->classes_count + byte_class);
        sz_size_t const candidate = (sz_size_t)__ldg(engine->base + state) + byte;
        if (__ldg(engine->check + candidate) == state) return (sz_u32_t)candidate;
        if (state == engine->root) return engine->root;
        state = __ldg(engine->fail + state);
    }
}

/** Whether any needle ends on @p state, from the staged acceptance bitmap. */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_accepts_simt_(sz_substrings_staged_simt_t const *staged, sz_u32_t state) {
    return (sz_bool_t)((staged->accepts_words[state >> 5] >> (state & 31u)) & 1u);
}

/**
 *  @brief Cooperatively stages the class map and the head of the hot tier, once per block.
 *
 *  The hot tier's out-degree ordering makes its head the best prefix to stage, and every step
 *  reads the map.
 */
STRINGZILLA_DEVICE void sz_substrings_stage_simt_(sz_substrings_engine_t const *engine, sz_u8_t *staged_classes,
                                                  sz_u32_t *staged_rows, sz_u32_t staged_count) {
    sz_size_t const cells = (sz_size_t)staged_count * engine->classes_count;
    sz_size_t cell;
    for (cell = threadIdx.x; cell < STRINGZILLA_U8_MAX + 1; cell += blockDim.x)
        staged_classes[cell] = engine->byte_to_class[cell];
    for (cell = threadIdx.x; cell < cells; cell += blockDim.x) staged_rows[cell] = engine->hot_rows[cell];
    __syncthreads();
}

/** The state a walk from the root reaches at @p chunk_begin of @p text, primed without reporting
 *  over the bytes before it a match ending in the chunk may start at, never before the text. */
STRINGZILLA_DEVICE sz_u32_t sz_substrings_prime_simt_(sz_substrings_engine_t const *engine,
                                                      sz_substrings_staged_simt_t const *staged, sz_cptr_t text,
                                                      sz_size_t chunk_begin) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t position = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u32_t state = engine->root;
    for (; position < chunk_begin; ++position) state = sz_substrings_step_simt_(engine, staged, state, bytes[position]);
    return state;
}

/** How many chunks of @p chunk_bytes a haystack of @p length bytes needs - at least one, so even an
 *  empty haystack still gets a thread and still lands its own boundary. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_chunks_for_simt_(sz_size_t length, sz_size_t chunk_bytes) {
    return length == 0 ? 1 : sz_size_divide_round_up(length, chunk_bytes);
}

/** Which haystack owns global chunk @p chunk_index, from the exclusive prefix sum
 *  of per-haystack counts. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_haystack_of_simt_(sz_size_t const *chunk_offsets, sz_size_t haystacks_count,
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
STRINGZILLA_DEVICE sz_size_t sz_substrings_last_not_above_simt_(sz_size_t const *ascending, sz_size_t count,
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
STRINGZILLA_DEVICE void sz_substrings_tally_simt_(sz_substrings_tally_simt_t const *tally, sz_u32_t needle) {
    sz_u32_t const key = needle + 1;
    // Volatile, so a probe re-reads a slot a rival may have seated since.
    sz_u32_t const volatile *const seated_keys = tally->keys;
    sz_u32_t slot, probe;
    if (tally->layout == sz_substrings_tally_direct_simt_k) {
        atomicAdd(tally->counts + needle, 1u);
        return;
    }
    slot = (sz_u32_t)(((sz_u64_t)key * 0x9E3779B97F4A7C15ull) >> (64 - sz_substrings_tally_slot_bits_simt_k));
    for (probe = 0; probe != sz_substrings_tally_probes_simt_k; ++probe) {
        sz_u32_t seated = seated_keys[slot];
        if (seated == 0) seated = atomicCAS(tally->keys + slot, 0u, key);
        if (seated == 0 || seated == key) {
            atomicAdd(tally->counts + slot, 1u);
            return;
        }
        slot = (slot + 1) & (sz_substrings_tally_slots_simt_k - 1);
    }
    atomicAdd(tally->overflow + needle, 1u);
    *tally->overflowed = 1;
}

/** Counts one occurrence of @p needle in the sink's haystack, seating the pair on first sight. */
STRINGZILLA_DEVICE void sz_substrings_pair_simt_(sz_substrings_sink_simt_t const *sink, sz_u32_t needle) {
    sz_substrings_pairs_simt_t const *const pairs = sink->pairs;
    sz_u64_t const key = (sz_u64_t)sink->haystack_index * pairs->needles_count + needle + 1;
    // A seated key never changes within a launch, so a cached read can only miss a rival's seating,
    // which the compare-and-swap then reports.
    sz_u64_t const *const seated_keys = pairs->keys;
    // The high half of a product scales the hash onto any capacity, a power of two or not.
    sz_size_t slot = (sz_size_t)__umul64hi(key * 0x9E3779B97F4A7C15ull, (sz_u64_t)pairs->capacity);
    sz_u32_t probe;
    for (probe = 0; probe != sz_substrings_pairs_probes_simt_k; ++probe) {
        sz_u64_t seated = seated_keys[slot];
        if (seated == 0) seated = atomicCAS((unsigned long long *)pairs->keys + slot, 0ull, key);
        if (seated == 0 || seated == key) {
            atomicAdd(pairs->counts + slot, 1u);
            return;
        }
        slot = slot + 1 == pairs->capacity ? 0 : slot + 1;
    }
    pairs->overflowed[sink->haystack_index] = 1;
}

/**
 *  @brief This round's chunk width: the tape's bytes over the engine's chunk budget, and at least
 *      four times the longest match, so a warm-up never outgrows a quarter of a chunk.
 *
 *  There is no ceiling on the width, and that is what makes the budget a bound rather than a hope:
 *  a chunk holds at least @c total/budget bytes, so the corpus contributes at most @c budget
 *  chunks, and each haystack's own remainder contributes at most one more. A ceiling would let a
 *  large corpus outrun any fixed budget, which is the readback this inversion exists to remove.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_round_chunk_bytes_simt_(sz_substrings_engine_t const *engine,
                                                                   sz_strings_view_t haystacks) {
    sz_size_t const bytes = (sz_size_t)(sz_strings_start_simt_(haystacks, haystacks.count) -
                                        sz_strings_start_simt_(haystacks, 0));
    sz_size_t const budget = engine->chunk_budget ? engine->chunk_budget : 1;
    sz_size_t const share = sz_size_divide_round_up(bytes, budget);
    return sz_max_of_two(share, sz_substrings_chunk_bytes_simt_(engine));
}

/** Keeps the chunk's match @p slot in its own slots, its offset cut to 32 bits, which the chunk's
 *  start restores, or past them in the spill pool, returning the spill index when spilled. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_keep_simt_(sz_substrings_sink_simt_t const *sink, sz_size_t slot,
                                                      sz_u32_t needle_index, sz_size_t byte_offset,
                                                      sz_size_t byte_length) {
    sz_size_t spill;
    if (slot < sz_substrings_kept_matches_simt_k) {
        sz_u32_t *const entry = sink->kept + slot * 3 * sink->stride;
        entry[0] = needle_index, entry[sink->stride] = (sz_u32_t)byte_offset;
        entry[2 * sink->stride] = (sz_u32_t)byte_length;
        return STRINGZILLA_SIZE_MAX;
    }
    spill = (sz_size_t)atomicAdd((unsigned long long *)sink->spilled, 1ull);
    if (spill >= sink->spills_capacity) return spill; // ? Past the budget, which the round's shortfall reports.
    sink->spills[spill].chunk_index = sink->chunk_index, sink->spills[spill].byte_offset = byte_offset;
    sink->spills[spill].slot = (sz_u32_t)slot, sink->spills[spill].needle_index = needle_index;
    sink->spills[spill].byte_length = (sz_u32_t)byte_length;
    return spill;
}

/** The chunk walk's most recently kept match, so consecutive prefix matches sharing a start offset
 *  collapse in-place under leftmost overlap policies without consuming kept or spill slots. */
typedef struct sz_substrings_last_simt_t {
    sz_size_t byte_offset;
    sz_u32_t byte_length;
    sz_u32_t needle_index;
    sz_size_t spill;
} sz_substrings_last_simt_t;

STRINGZILLA_DEVICE void sz_substrings_record_simt_(sz_substrings_engine_t const *engine,
                                                   sz_substrings_sink_simt_t const *sink, sz_size_t *found,
                                                   sz_substrings_last_simt_t *last, sz_u32_t needle_index,
                                                   sz_size_t byte_offset, sz_size_t byte_length) {
    if (engine->overlap_policy != sz_substrings_overlapping_k && byte_offset == last->byte_offset) {
        sz_bool_t const better = engine->overlap_policy == sz_substrings_leftmost_longest_k
                                     ? (sz_bool_t)(byte_length > last->byte_length ||
                                                   (byte_length == last->byte_length &&
                                                    needle_index < last->needle_index))
                                     : (sz_bool_t)(needle_index < last->needle_index);
        if (!better) return;
        last->byte_length = (sz_u32_t)byte_length, last->needle_index = needle_index;
        if (!sink->kept) return;
        sz_size_t const slot = *found - 1;
        if (slot < sz_substrings_kept_matches_simt_k) {
            sz_u32_t *const entry = sink->kept + slot * 3 * sink->stride;
            entry[0] = needle_index, entry[sink->stride] = (sz_u32_t)byte_offset;
            entry[2 * sink->stride] = (sz_u32_t)byte_length;
        }
        else if (last->spill < sink->spills_capacity) {
            sink->spills[last->spill].needle_index = needle_index;
            sink->spills[last->spill].byte_length = (sz_u32_t)byte_length;
        }
        return;
    }
    last->byte_offset = byte_offset, last->byte_length = (sz_u32_t)byte_length, last->needle_index = needle_index;
    if (sink->kept) last->spill = sz_substrings_keep_simt_(sink, *found, needle_index, byte_offset, byte_length);
    ++*found;
}

/**
 *  @brief Reports every match ending at @p delta, the chunk's matches from @p found on,
 *      keeping or tallying them as @p pass asks.
 *
 *  The acceptance bit answers "does anything end here" without touching the counts array, which at
 *  scale costs nearly as much as the haystack read itself.
 */
STRINGZILLA_DEVICE void sz_substrings_emit_simt_(sz_substrings_engine_t const *engine,
                                                 sz_substrings_staged_simt_t const *staged, sz_u32_t state,
                                                 sz_size_t walk_begin, sz_u32_t delta, sz_substrings_gpu_pass_t pass,
                                                 sz_substrings_sink_simt_t const *sink, sz_size_t *found,
                                                 sz_substrings_last_simt_t *last) {
    sz_size_t output_offset, index;
    sz_u32_t output_count;
    if (!sz_substrings_accepts_simt_(staged, state)) return;
    output_count = __ldg(engine->outputs_counts + state);
    output_offset = (sz_size_t)__ldg((unsigned long long const *)engine->outputs_offsets + state);
    for (index = 0; index != output_count; ++index) {
        sz_substrings_output_t const *const stored = engine->outputs + output_offset + index;
        sz_substrings_output_t output;
        output.needle_index = __ldg(&stored->needle_index);
        output.folded_match_bytes = __ldg(&stored->folded_match_bytes);
        // `walk_begin` is clamped to the haystack's own start, so underflowing the walk and underflowing
        // the haystack are the same test.
        if (delta + 1 < output.folded_match_bytes) continue;
        if (pass == sz_substrings_gpu_tallying_k) {
            if (sink->tally) sz_substrings_tally_simt_(sink->tally, output.needle_index);
            else sz_substrings_pair_simt_(sink, output.needle_index);
            ++*found;
        }
        else
            sz_substrings_record_simt_(engine, sink, found, last, output.needle_index,
                                       walk_begin + delta + 1 - output.folded_match_bytes, output.folded_match_bytes);
    }
}

/**
 *  @brief The device twin of @c sz_substrings_folded_span, stepping back one folded codepoint at a
 *      time through the fold tables, as @c sz_substrings_resolve_match_metal_ does.
 *
 *  A byte that ends no well-formed codepoint passes through as itself and steps back by one, which
 *  keeps the backward byte stream the exact reverse of the forward walk's.
 */
STRINGZILLA_DEVICE sz_substrings_resolved_match_t
sz_substrings_folded_span_simt_(sz_cptr_t haystack, sz_substrings_folded_byte_t const *step, sz_size_t folded,
                                sz_size_t last_break_folded_end, sz_size_t folded_match_bytes) {
    sz_u8_t const *const haystack_bytes = (sz_u8_t const *)haystack;
    sz_size_t const trailing = step->trailing, shift = step->shift;
    sz_size_t const wanted = folded_match_bytes + shift;
    sz_size_t position = step->codepoint_end, stepped;
    sz_size_t start_here = STRINGZILLA_SIZE_MAX, start_earlier = STRINGZILLA_SIZE_MAX;
    sz_utf8_folded_image_t image;
    sz_u64_t ring = 0;
    sz_u8_t pending = 0;
    sz_bool_t periodic = (sz_bool_t)(shift != 0);
    sz_substrings_resolved_match_t resolved;
    resolved.source_offset = step->codepoint_end - folded_match_bytes;
    resolved.repeats = sz_false_k;
    if (folded - folded_match_bytes >= last_break_folded_end) return resolved;

    for (stepped = 0; stepped < trailing + wanted; ++stepped) {
        sz_u8_t byte;
        sz_size_t taken;
        if (!pending) {
            sz_size_t candidate, source_bytes;
            if (!position) break;
            candidate = position - 1;
            for (int back = 0; back != 3 && candidate && (haystack_bytes[candidate] & 0xC0) == 0x80; ++back)
                --candidate;
            source_bytes = sz_utf8_fold_next_simt_(haystack_bytes + candidate, haystack_bytes + position, &image);
            if (!image.rune_ends || candidate + source_bytes != position)
                image.bytes = haystack_bytes[--position], image.length = 1;
            else position = candidate;
            pending = image.length;
        }
        byte = (sz_u8_t)(image.bytes >> (8 * --pending));
        if (stepped < trailing) continue; // ? Folded bytes of the ending codepoint past the match's own end

        taken = stepped - trailing + 1;
        // A shift stays within one codepoint's image of at most six bytes, so eight ring bytes do.
        if (shift) {
            if (taken > shift && (sz_u8_t)(ring >> (8 * ((taken - shift) & 7))) != byte) periodic = sz_false_k;
            ring = (ring & ~(0xFFull << (8 * (taken & 7)))) | ((sz_u64_t)byte << (8 * (taken & 7)));
        }
        if (taken == folded_match_bytes) start_here = position;
        if (taken == wanted) start_earlier = position;
        if (!periodic && taken >= folded_match_bytes) break;
    }

    resolved.source_offset = start_here != STRINGZILLA_SIZE_MAX ? start_here : step->codepoint_end;
    resolved.repeats = (sz_bool_t)(periodic && start_here != STRINGZILLA_SIZE_MAX && start_here == start_earlier);
    return resolved;
}

/** One needle's term, in the fixed point every device sum accumulates in. */
STRINGZILLA_DEVICE sz_i64_t sz_substrings_bm25_fixed_term_simt_(sz_substrings_bm25_t const *parameters, sz_f64_t norm,
                                                                sz_f32_t weight, sz_u32_t frequency) {
    return __double2ll_rn(sz_substrings_bm25_term(parameters, norm, weight, frequency) *
                          (sz_f64_t)((sz_u64_t)1 << sz_substrings_gpu_bm25_fraction_bits_k));
}

/** Sums in fixed point the terms of the @p tally slots and overflow entries from @p first on, every
 *  @p step th, zeroing each as it reads it, so the next haystack starts clean; its callers add up
 *  what each thread returns. */
STRINGZILLA_DEVICE sz_i64_t sz_substrings_bm25_terms_simt_(sz_substrings_tally_simt_t const *tally,
                                                           sz_size_t table_slots, sz_size_t needles_count,
                                                           sz_substrings_bm25_t const *parameters, sz_f64_t norm,
                                                           sz_f32_t const *needle_weights, sz_size_t first,
                                                           sz_size_t step) {
    sz_i64_t mine = 0;
    sz_size_t slot, needle;
    for (slot = first; slot < table_slots; slot += step) {
        sz_u32_t const frequency = tally->counts[slot];
        if (!frequency) continue;
        needle = tally->keys ? (sz_size_t)tally->keys[slot] - 1 : slot;
        mine += sz_substrings_bm25_fixed_term_simt_(parameters, norm, needle_weights[needle], frequency);
        tally->counts[slot] = 0;
        if (tally->keys) tally->keys[slot] = 0;
    }
    if (*tally->overflowed)
        for (needle = first; needle < needles_count; needle += step) {
            sz_u32_t const frequency = tally->overflow[needle];
            if (!frequency) continue;
            mine += sz_substrings_bm25_fixed_term_simt_(parameters, norm, needle_weights[needle], frequency);
            tally->overflow[needle] = 0;
        }
    return mine;
}

/** The fixed-point @p sum of a haystack's terms as its score. */
STRINGZILLA_DEVICE sz_f32_t sz_substrings_bm25_fixed_score_simt_(unsigned long long sum) {
    return (sz_f32_t)((sz_f64_t)(sz_i64_t)sum / (sz_f64_t)((sz_u64_t)1 << sz_substrings_gpu_bm25_fraction_bits_k));
}

/** Sums the terms of the block's own @p tally into @p score, zeroing every slot, overflow entry and
 *  the sum itself as it reads them, so the next haystack starts clean. */
STRINGZILLA_DEVICE void sz_substrings_bm25_score_simt_(sz_substrings_tally_simt_t const *tally, sz_size_t table_slots,
                                                       sz_size_t needles_count, sz_substrings_bm25_t const *parameters,
                                                       sz_f64_t norm, sz_f32_t const *needle_weights,
                                                       unsigned long long *block_sum, sz_f32_t *score) {
    sz_i64_t const mine = sz_substrings_bm25_terms_simt_(tally, table_slots, needles_count, parameters, norm,
                                                         needle_weights, threadIdx.x, blockDim.x);
    // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
    if (mine) atomicAdd(block_sum, (unsigned long long)mine);
    __syncthreads();
    if (threadIdx.x == 0) {
        *score = sz_substrings_bm25_fixed_score_simt_(*block_sum);
        *block_sum = 0, *tally->overflowed = 0;
    }
    __syncthreads();
}

/** Adds the term of the pair seated under @p key at @p slot into its haystack's word of @p sums,
 *  and frees the slot. */
STRINGZILLA_DEVICE void sz_substrings_bm25_pair_term_simt_(sz_substrings_pairs_simt_t const *pairs, sz_size_t slot,
                                                           sz_u64_t key, sz_strings_view_t haystacks,
                                                           sz_f32_t const *document_lengths,
                                                           sz_substrings_bm25_t const *parameters,
                                                           sz_f32_t const *needle_weights, unsigned long long *sums) {
    sz_size_t const haystack_index = (sz_size_t)((key - 1) / pairs->needles_count);
    sz_size_t const needle = (sz_size_t)(key - 1) - haystack_index * pairs->needles_count;
    sz_f64_t const norm = sz_substrings_bm25_norm(
        parameters, document_lengths ? (sz_f64_t)document_lengths[haystack_index]
                                     : (sz_f64_t)sz_strings_length_simt_(haystacks, haystack_index));
    sz_i64_t const term = sz_substrings_bm25_fixed_term_simt_(parameters, norm, needle_weights[needle],
                                                              pairs->counts[slot]);
    // Two's complement makes the unsigned sum of signed terms the signed sum, bit for bit.
    if (term) atomicAdd(sums + haystack_index, (unsigned long long)term);
    pairs->keys[slot] = 0, pairs->counts[slot] = 0;
}

/** Whether @p candidate is the one a leftmost @p policy takes over @p incumbent,
 *  both at or past the cursor. */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_precedes_simt_(sz_match_t const *candidate, sz_match_t const *incumbent,
                                                          sz_substrings_overlap_policy_t policy) {
    if (candidate->byte_offset != incumbent->byte_offset)
        return (sz_bool_t)(candidate->byte_offset < incumbent->byte_offset);
    if (policy == sz_substrings_leftmost_longest_k && candidate->byte_length != incumbent->byte_length)
        return (sz_bool_t)(candidate->byte_length > incumbent->byte_length);
    return (sz_bool_t)(candidate->needle_index < incumbent->needle_index);
}

/** The match a leftmost cover takes first at or past @p reached, scanning from @p from
 *  in end order, or @p count. */
STRINGZILLA_DEVICE sz_size_t sz_substrings_successor_simt_(sz_match_t const *matches, sz_size_t count,
                                                           sz_size_t longest, sz_substrings_overlap_policy_t policy,
                                                           sz_size_t from, sz_size_t reached) {
    sz_size_t best = count, slot;
    for (slot = from; slot < count && matches[slot].haystack_index == matches[from].haystack_index; ++slot) {
        // Every later match ends no earlier, so it starts after the best one
        // once it ends `longest` past it.
        if (best != count &&
            matches[slot].byte_offset + matches[slot].byte_length > matches[best].byte_offset + longest)
            break;
        if (matches[slot].byte_offset < reached) continue;
        if (best == count || sz_substrings_precedes_simt_(&matches[slot], &matches[best], policy)) best = slot;
    }
    return best;
}

/** Copies one stretch, clipped to the tile, with @p lane striding the surviving bytes. */
STRINGZILLA_DEVICE void sz_substrings_copy_clipped_simt_(sz_ptr_t output, sz_size_t tile_begin, sz_size_t tile_end,
                                                         sz_size_t output_offset, sz_cptr_t source, sz_size_t bytes,
                                                         unsigned lane) {
    sz_size_t const copy_begin = sz_max_of_two(output_offset, tile_begin);
    sz_size_t const copy_end = sz_min_of_two(output_offset + bytes, tile_end);
    sz_size_t position;
    for (position = copy_begin + lane; position < copy_end; position += 32)
        output[position] = source[position - output_offset];
}

#pragma endregion Steps

/*  A walk answers for one value at one granularity: a lane walking it alone, a @c sz_simt_group_t
 *  walking it together, or a lane walking one chunk of it with the automaton's state carried in and
 *  out. A walk reads no @c blockIdx and declares no @c __shared__, so a product calls it from a
 *  kernel of its own under any launch, having staged the engine with @c sz_substrings_stage_simt_
 *  into @c sz_substrings_staged_bytes_simt_ bytes. Its engine is one
 *  @c sz_substrings_engine_init_serial compiled into memory the host writes and the device reads,
 *  taken by value with its tables read where they were compiled, or rebased onto a device block
 *  with @c sz_substrings_engine_rebase_simt_; the @c first walks take a byte-exact one. */
#pragma region Walks

/**
 *  @brief Walks one chunk of a byte-exact haystack, counting every match ending inside it, and
 *      keeping or tallying them as @p pass asks.
 *  @param[in] chunk_begin First byte of the chunk, relative to the haystack's own start.
 *  @return The number of matches the chunk holds.
 *
 *  The warm-up, @ref sz_substrings_prime_simt_, primes the state from before the chunk and reports
 *  nothing, so once it ends the emit test leaves the loop rather than being re-asked per byte.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_walk_chunk_cased_simt_(sz_substrings_engine_t const *engine,
                                                                  sz_substrings_staged_simt_t const *staged,
                                                                  sz_cptr_t haystack, sz_size_t length,
                                                                  sz_size_t chunk_begin, sz_size_t chunk_end,
                                                                  sz_substrings_gpu_pass_t pass,
                                                                  sz_substrings_sink_simt_t const *sink) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
    sz_size_t const walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const walk_base = (sz_u8_t const *)haystack + walk_begin;
    // Every 64-bit quantity is resolved here, once; the per-byte loops ride 32-bit deltas from it.
    sz_u32_t const walk_span = (sz_u32_t)(chunk_end - walk_begin);
    sz_u32_t const emit_from = (sz_u32_t)(chunk_begin - walk_begin);
    sz_size_t found = 0;
    sz_substrings_last_simt_t last = {STRINGZILLA_SIZE_MAX, 0, 0, STRINGZILLA_SIZE_MAX};
    sz_u32_t delta = emit_from, lane;
    sz_u32_t state = sz_substrings_prime_simt_(engine, staged, haystack, chunk_begin);
    sz_unused_(length);

    // Peeled to the load's own alignment, so the body pays one four-byte load per four transitions;
    // the transition chain stays strictly serial, only the haystack reads widen.
    for (; delta < walk_span && (((sz_size_t)(walk_base + delta)) & 3u) != 0; ++delta) {
        state = sz_substrings_step_simt_(engine, staged, state, walk_base[delta]);
        sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta, pass, sink, &found, &last);
        if (pass == sz_substrings_gpu_marking_k && found) return found;
    }
    for (; delta + 4 <= walk_span; delta += 4) {
        sz_u32_t const quad = sz_substrings_load_quad_simt_(walk_base + delta);
#pragma unroll
        for (lane = 0; lane != 4; ++lane) {
            state = sz_substrings_step_simt_(engine, staged, state, (sz_u8_t)(quad >> (lane * 8)));
            sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta + lane, pass, sink, &found, &last);
            if (pass == sz_substrings_gpu_marking_k && found) return found;
        }
    }
    for (; delta < walk_span; ++delta) {
        state = sz_substrings_step_simt_(engine, staged, state, walk_base[delta]);
        sz_substrings_emit_simt_(engine, staged, state, walk_begin, delta, pass, sink, &found, &last);
        if (pass == sz_substrings_gpu_marking_k && found) return found;
    }
    return found;
}

/**
 *  @brief Walks one chunk as folded bytes, the case-insensitive twin of the walk above.
 *
 *  Folding makes a walk restart-safe only at a codepoint start, so the warm-up snaps back to one
 *  before it begins - three bytes at most, and always earlier, so the extra transitions only prime
 *  state further. Match ends are reported at the source codepoint's end, which keeps chunk
 *  ownership comparable against the unsnapped chunk bounds the planner handed out.
 */
STRINGZILLA_DEVICE sz_size_t sz_substrings_walk_chunk_uncased_simt_(sz_substrings_engine_t const *engine,
                                                                    sz_substrings_staged_simt_t const *staged,
                                                                    sz_cptr_t haystack, sz_size_t length,
                                                                    sz_size_t chunk_begin, sz_size_t chunk_end,
                                                                    sz_substrings_gpu_pass_t pass,
                                                                    sz_substrings_sink_simt_t const *sink) {
    sz_size_t const warm_up = engine->max_source_match_bytes > 0 ? (sz_size_t)engine->max_source_match_bytes - 1 : 0;
    sz_size_t walk_begin = chunk_begin >= warm_up ? chunk_begin - warm_up : 0;
    sz_u8_t const *const haystack_bytes = (sz_u8_t const *)haystack;
    sz_substrings_folded_byte_t step;
    sz_utf8_folded_image_t image;
    sz_u32_t state = engine->root;
    sz_size_t folded = 0, last_break_folded_end = 0, found = 0, position;
    sz_substrings_last_simt_t last = {STRINGZILLA_SIZE_MAX, 0, 0, STRINGZILLA_SIZE_MAX};
    sz_u8_t image_index = 0, previous_rune_end = 0;
    sz_bool_t breaks_boundary = sz_false_k;
    // The aligned word under the cursor, so an ASCII run costs one load per four bytes.
    sz_u8_t const *loaded_word = STRINGZILLA_NULL;
    sz_u32_t loaded_quad = 0;

    // A fold restarts only on a lead byte, so the walk snaps outward to one - never inward, which
    // would drop a match whose needle began mid-codepoint.
    while (walk_begin != 0 && haystack_bytes[walk_begin] >= 0x80 && (haystack_bytes[walk_begin] & 0xC0) == 0x80)
        --walk_begin;

    // Each codepoint folds into an image held in registers, which the automaton steps through a
    // byte at a time; only a byte ending a folded rune can end a match.
    image.length = 0;
    for (position = walk_begin;;) {
        sz_size_t output_offset, output_count, index;
        if (image_index == image.length) {
            sz_size_t source_bytes;
            if (position >= length) break;
            sz_u8_t const *const word = (sz_u8_t const *)((sz_size_t)(haystack_bytes + position) & ~(sz_size_t)3);
            if (word != loaded_word)
                loaded_word = word,
                loaded_quad = sz_u32_load_bounded_simt_((sz_u32_t const *)word, haystack_bytes + length);
            sz_u8_t const lead = (sz_u8_t)(loaded_quad >> (((sz_size_t)(haystack_bytes + position) & 3u) * 8));
            if (lead < 0x80)
                image.bytes = sz_ascii_fold_(lead), image.length = 1, image.rune_ends = 1, source_bytes = 1;
            else source_bytes = sz_utf8_fold_next_simt_(haystack_bytes + position, haystack_bytes + length, &image);
            breaks_boundary = (sz_bool_t)(sz_u32_popcount_simt_(image.rune_ends) != 1 || image.length != source_bytes);
            position += source_bytes;
            step.codepoint_end = position - walk_begin;
            image_index = 0, previous_rune_end = 0;
        }
        ++folded;
        // A byte beginning no codepoint is its own folded byte, and no match runs through it.
        if (!image.rune_ends) {
            ++image_index, state = engine->root;
            continue;
        }
        state = sz_substrings_step_simt_(engine, staged, state, (sz_u8_t)(image.bytes >> (8 * image_index)));
        if (!((image.rune_ends >> image_index++) & 1u)) continue;
        // A codepoint's first rune end has nothing before it to repeat, which a zero shift names.
        step.trailing = (sz_u8_t)(image.length - image_index);
        step.shift = previous_rune_end ? (sz_u8_t)(image_index - previous_rune_end) : 0;
        previous_rune_end = image_index;
        if (breaks_boundary) last_break_folded_end = folded + step.trailing;

        if (position > chunk_end) return found; // ? Past this chunk's share; the next chunk owns these ends.
        if (!sz_substrings_accepts_simt_(staged, state)) continue;
        if (position <= chunk_begin) continue; // ? Still warming up, where this chunk reports nothing.

        output_count = engine->outputs_counts[state];
        output_offset = engine->outputs_offsets[state];
        for (index = 0; index != output_count; ++index) {
            sz_substrings_output_t const output = engine->outputs[output_offset + index];
            sz_size_t const folded_length = output.folded_match_bytes;
            sz_substrings_resolved_match_t resolved;
            if (folded < folded_length) continue;
            resolved = sz_substrings_folded_span_simt_(haystack + walk_begin, &step, folded, last_break_folded_end,
                                                       folded_length);
            if (resolved.repeats) continue;
            if (pass == sz_substrings_gpu_tallying_k) {
                if (sink->tally) sz_substrings_tally_simt_(sink->tally, output.needle_index);
                else sz_substrings_pair_simt_(sink, output.needle_index);
                ++found;
            }
            else
                sz_substrings_record_simt_(engine, sink, &found, &last, output.needle_index,
                                           walk_begin + resolved.source_offset,
                                           step.codepoint_end - resolved.source_offset);
            if (pass == sz_substrings_gpu_marking_k) return found;
        }
    }
    return found;
}

/** Walks one haystack's share of chunks, the @p thread th of @p threads, into @p tally. */
STRINGZILLA_DEVICE void sz_substrings_bm25_walk_simt_(sz_substrings_engine_t const *engine,
                                                      sz_substrings_staged_simt_t const *staged, sz_cptr_t haystack,
                                                      sz_size_t length, sz_size_t thread, sz_size_t threads,
                                                      sz_substrings_tally_simt_t const *tally) {
    // Never narrower than the longest match, past which a chunk re-walks more warm-up than it owns.
    sz_size_t const warm_up = sz_max_of_two((sz_size_t)engine->max_source_match_bytes, (sz_size_t)1);
    sz_size_t const chunk_bytes = sz_max_of_two(sz_size_divide_round_up(length, threads), warm_up);
    sz_size_t const chunk_begin = thread * chunk_bytes;
    if (chunk_begin >= length) return;
    sz_size_t const chunk_end = sz_min_of_two(chunk_begin + chunk_bytes, length);
    sz_substrings_sink_simt_t sink;
    sink.kept = STRINGZILLA_NULL, sink.stride = 0, sink.spills = STRINGZILLA_NULL, sink.spilled = STRINGZILLA_NULL;
    sink.spills_capacity = 0, sink.chunk_index = 0, sink.tally = tally;
    sink.pairs = STRINGZILLA_NULL, sink.haystack_index = 0;
    if (engine->case_sensitivity == sz_uncased_k)
        sz_substrings_walk_chunk_uncased_simt_(engine, staged, haystack, length, chunk_begin, chunk_end,
                                               sz_substrings_gpu_tallying_k, &sink);
    else
        sz_substrings_walk_chunk_cased_simt_(engine, staged, haystack, length, chunk_begin, chunk_end,
                                             sz_substrings_gpu_tallying_k, &sink);
}

/**
 *  @brief Walks bytes @p chunk_begin to @p chunk_end of a byte-exact @p text from the state
 *      @p carried, answering whether a match ends inside them.
 *  @param[in] length Bytes of @p text, past which the chunk is cut.
 *  @param[in,out] carried The state at @p chunk_begin, which the root,
 *      @ref sz_substrings_prime_simt_ or the previous chunk's walk leaves; the state at
 *      @p chunk_end on return, or at the byte the first match ends on.
 *
 *  A match may start before @p chunk_begin, as far back as @p carried remembers, so a value cut
 *  into chunks walked in order, each carrying the last one's state, answers exactly as one walk
 *  over all of it.
 */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_first_chunk_simt_(sz_substrings_engine_t const *engine,
                                                             sz_substrings_staged_simt_t const *staged, sz_cptr_t text,
                                                             sz_size_t length, sz_size_t chunk_begin,
                                                             sz_size_t chunk_end, sz_u32_t *carried) {
    sz_u8_t const *const bytes = (sz_u8_t const *)text;
    sz_size_t const end = sz_min_of_two(chunk_end, length);
    sz_size_t position = chunk_begin;
    sz_u32_t state = *carried, lane;
    // Peeled to the load's alignment, as the chunk walk is, so the body reads four bytes per load.
    for (; position < end && (((sz_size_t)(bytes + position)) & 3u) != 0; ++position) {
        state = sz_substrings_step_simt_(engine, staged, state, bytes[position]);
        if (sz_substrings_accepts_simt_(staged, state)) {
            *carried = state;
            return sz_true_k;
        }
    }
    for (; position + 4 <= end; position += 4) {
        sz_u32_t const quad = sz_substrings_load_quad_simt_(bytes + position);
#pragma unroll
        for (lane = 0; lane != 4; ++lane) {
            state = sz_substrings_step_simt_(engine, staged, state, (sz_u8_t)(quad >> (lane * 8)));
            if (sz_substrings_accepts_simt_(staged, state)) {
                *carried = state;
                return sz_true_k;
            }
        }
    }
    for (; position < end; ++position) {
        state = sz_substrings_step_simt_(engine, staged, state, bytes[position]);
        if (sz_substrings_accepts_simt_(staged, state)) {
            *carried = state;
            return sz_true_k;
        }
    }
    *carried = state;
    return sz_false_k;
}

/** Whether any needle of a byte-exact @p engine occurs in @p text, one lane walking all of it from
 *  the root. */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_first_thread_simt_(sz_substrings_engine_t const *engine,
                                                              sz_substrings_staged_simt_t const *staged, sz_cptr_t text,
                                                              sz_size_t length) {
    sz_u32_t carried = engine->root;
    return sz_substrings_first_chunk_simt_(engine, staged, text, length, 0, length, &carried);
}

/**
 *  @brief Whether any needle of a byte-exact @p engine occurs in @p text, the lanes of @p group
 *      walking it together, and every lane answering the same.
 *
 *  Lane @c l walks chunk @c l of @p text cut into @c group->lanes, primed as the chunk walk primes,
 *  so a match is found by the lane owning its last byte, whichever lane its first byte falls to.
 */
STRINGZILLA_DEVICE sz_bool_t sz_substrings_first_group_simt_(sz_substrings_engine_t const *engine,
                                                             sz_substrings_staged_simt_t const *staged, sz_cptr_t text,
                                                             sz_size_t length, sz_simt_group_t const *group) {
    sz_size_t const chunk_bytes = sz_size_divide_round_up(length, (sz_size_t)group->lanes);
    sz_size_t const chunk_begin = sz_min_of_two((sz_size_t)group->lane * chunk_bytes, length);
    sz_u32_t carried = sz_substrings_prime_simt_(engine, staged, text, chunk_begin);
    sz_bool_t const found = sz_substrings_first_chunk_simt_(engine, staged, text, length, chunk_begin,
                                                            chunk_begin + chunk_bytes, &carried);
    return group->any(found) ? sz_true_k : sz_false_k;
}

#pragma endregion Walks

#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_SUBSTRINGS_SIMT_CUH_
