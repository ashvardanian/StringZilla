/**
 *  @file include/stringzilla/substrings/metal.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Metal backend for multi-pattern search on Apple7 and newer: the vocabulary
 *      compiled on the host into one shared block, and one call per round, the CUDA tier's
 *      launches as dispatches.
 *
 *  @sa include/stringzilla/substrings/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/substrings/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @c id<MTLCommandQueue> as the stream. Every block
 *  the kernels read must come from @ref sz_allocator_init_unified_metal on that stream's device,
 *  and haystacks and replacements must be tapes from @ref sz_sequence_realloc_metal, since the
 *  kernels read the tapes themselves. A round's dispatches run in order inside one command buffer,
 *  so each sees what the one before it wrote, and nothing is read back until the caller
 *  synchronizes.
 *
 *  A case-insensitive vocabulary is refused at init: its walk folds the haystack through the
 *  Unicode tables the host tiers read, which have no Metal port.
 */
#ifndef STRINGZILLA_SUBSTRINGS_METAL_H_
#define STRINGZILLA_SUBSTRINGS_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/memory/serial.h" // `sz_fill_serial_`
#include "stringzilla/substrings/serial.h"

#if STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the substrings kernels, compiled once per device behind the shared prelude. */
static char const sz_substrings_source_metal_[] = {
#embed "stringzilla/substrings/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Threads every threadgroup runs, as in the kernels. */
enum { sz_substrings_threads_per_group_metal_k = 256 };

/** Threadgroups a grid-strided dispatch takes at most, past which its threads stride instead. */
enum { sz_substrings_groups_max_metal_k = 1024 };

/** Bits of a tally slot index: 2048 slots of a key and a count each leave room in 32 KB of
 *  threadgroup memory. */
enum { sz_substrings_tally_slot_bits_metal_k = 11 };

/** Slots one threadgroup's tally holds. */
enum { sz_substrings_tally_slots_metal_k = 1 << sz_substrings_tally_slot_bits_metal_k };

/** Slots a hashed tally probes before spilling a needle to the threadgroup's overflow row. */
enum { sz_substrings_tally_probes_metal_k = 16 };

/** Threadgroups a BM25 dispatch runs, each with its own overflow row when the tally hashes. */
enum { sz_substrings_bm25_groups_metal_k = 64 };

/** Chunks one round cuts its haystacks into beyond one per haystack, enough to fill
 *  an Apple GPU. */
enum { sz_substrings_chunk_budget_metal_k = 1 << 16 };

/** The buffers a round binds, in the order the kernels number them, the launch record after the
 *  last. A dispatch that clears or scans binds its target in the cleared or scanned slot. */
enum {
    sz_substrings_memory_buffer_metal_k,
    sz_substrings_scratch_buffer_metal_k,
    sz_substrings_haystacks_buffer_metal_k,
    sz_substrings_boundaries_buffer_metal_k,
    sz_substrings_scanned_buffer_metal_k,
    sz_substrings_cleared_buffer_metal_k,
    sz_substrings_counts_buffer_metal_k,
    sz_substrings_matches_buffer_metal_k,
    sz_substrings_replacements_buffer_metal_k,
    sz_substrings_target_buffer_metal_k,
    sz_substrings_offsets_buffer_metal_k,
    sz_substrings_lengths_buffer_metal_k,
    sz_substrings_weights_buffer_metal_k,
    sz_substrings_scores_buffer_metal_k,
    sz_substrings_buffers_metal_k,
};

/** Every kernel of `metal.metal`, built at init so no round compiles one. */
static char const *const sz_substrings_kernels_metal_[] = {
    "sz_substrings_clear_metal_kernel_",        "sz_substrings_scan_reduce_metal_kernel_",
    "sz_substrings_scan_carry_metal_kernel_",   "sz_substrings_scan_apply_metal_kernel_",
    "sz_substrings_chunk_bytes_metal_kernel_",  "sz_substrings_chunk_counts_metal_kernel_",
    "sz_substrings_walk_metal_kernel_",         "sz_substrings_sized_metal_kernel_",
    "sz_substrings_cover_metal_kernel_",        "sz_substrings_compact_metal_kernel_",
    "sz_substrings_covered_metal_kernel_",      "sz_substrings_haystack_offsets_metal_kernel_",
    "sz_substrings_counts_metal_kernel_",       "sz_substrings_store_matches_metal_kernel_",
    "sz_substrings_stored_metal_kernel_",       "sz_substrings_rewrite_offsets_metal_kernel_",
    "sz_substrings_rewrite_copy_metal_kernel_", "sz_substrings_target_metal_kernel_",
    "sz_substrings_bm25_metal_kernel_",
};

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host, scratch_host;
    sz_u64_t hot_rows, byte_to_class, base, check, fail, accepts_words, outputs, outputs_counts, outputs_offsets;
    sz_u64_t hot_count, classes_count, root, needles_count, max_source_match_bytes, overlap_policy;
    sz_u64_t report, chunk_bytes, chunk_offsets, chunk_slots, tile_sums;
    sz_u64_t emitted, reported, keep_offsets, gap_offsets, overflow_rows;
    sz_u64_t slots_count, chunk_budget, chunk_floor, matches_budget, emitting;
    sz_u64_t haystacks_count, counts_stride, matches_capacity, target_ceiling;
    sz_u64_t document_lengths, scores_stride, tally_hashed;
    sz_f32_t saturation, normalization, average_length, unused;
    sz_u64_t pass, scan_count, scan_elements_per_tile, scan_tiles, clear_words;
} sz_substrings_arguments_metal_t;

/** Byte offsets of the round block every round runs out of, as the CUDA tier lays it out. */
typedef struct sz_substrings_arena_metal_t {
    sz_size_t report, chunk_bytes, chunk_offsets, haystack_offsets, chunk_slots, tile_sums;
    sz_size_t emitted, reported, keep_offsets, gap_offsets, overflow_rows, slots_count, total;
} sz_substrings_arena_metal_t;

/** Lays the round block out for one round over @p haystacks_count texts. */
STRINGZILLA_INLINE sz_substrings_arena_metal_t sz_substrings_arena_metal_(sz_substrings_engine_t const *engine,
                                                                          sz_size_t haystacks_count) {
    sz_size_t const boundaries = haystacks_count + 1, matches = engine->matches_budget;
    sz_size_t const match_bytes = matches * sizeof(sz_substrings_match_t);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_size_t const overflow_bytes = engine->needles_count > sz_substrings_tally_slots_metal_k
                                         ? sz_substrings_bm25_groups_metal_k * engine->needles_count * sizeof(sz_u32_t)
                                         : 0;
    sz_substrings_arena_metal_t arena;
    // A chunk is at least the corpus over the budget wide, so the corpus contributes at most
    // `chunk_budget` chunks and each haystack's own remainder at most one more.
    arena.slots_count = engine->chunk_budget + haystacks_count + 1;
    arena.report = 0;
    arena.chunk_bytes = arena.report + sizeof(sz_substrings_report_t);
    arena.chunk_offsets = arena.chunk_bytes + sizeof(sz_size_t);
    arena.haystack_offsets = arena.chunk_offsets + boundaries * sizeof(sz_size_t);
    arena.chunk_slots = arena.haystack_offsets + boundaries * sizeof(sz_size_t);
    arena.tile_sums = arena.chunk_slots + arena.slots_count * sizeof(sz_size_t);
    arena.emitted = arena.tile_sums + sz_substrings_gpu_scan_tiles_max_k * sizeof(sz_size_t);
    arena.reported = arena.emitted + match_bytes;
    arena.keep_offsets = arena.reported + (covering ? match_bytes : 0);
    arena.gap_offsets = arena.keep_offsets + (covering ? (matches + 1) * sizeof(sz_size_t) : 0);
    arena.overflow_rows = arena.gap_offsets + (covering ? matches * sizeof(sz_size_t) : 0);
    arena.total = arena.overflow_rows + overflow_bytes;
    return arena;
}

/** Threadgroups of the tier's width a grid-strided dispatch over @p items takes. */
STRINGZILLA_INLINE sz_metal_size_t sz_substrings_groups_metal_(sz_size_t items) {
    sz_size_t const groups = sz_size_divide_round_up(items, sz_substrings_threads_per_group_metal_k);
    sz_metal_size_t const grid = {groups ? sz_min_of_two(groups, (sz_size_t)sz_substrings_groups_max_metal_k) : 1, 1,
                                  1};
    return grid;
}

/** Encodes kernel @p name over @p groups of the tier's width, with the round's @p buffers. */
STRINGZILLA_INLINE sz_status_t sz_substrings_encode_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                           char const *name,
                                                           sz_substrings_arguments_metal_t const *arguments,
                                                           sz_metal_size_t groups) {
    sz_metal_size_t const threads = {sz_substrings_threads_per_group_metal_k, 1, 1};
    return sz_metal_encode_(call, sz_substrings_source_metal_, name, buffers, sz_substrings_buffers_metal_k, arguments,
                            sizeof(*arguments), groups, threads);
}

/** Encodes a single-thread kernel, the shape every report publication takes. */
STRINGZILLA_INLINE sz_status_t sz_substrings_encode_one_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                               char const *name,
                                                               sz_substrings_arguments_metal_t const *arguments) {
    sz_metal_size_t const one = {1, 1, 1};
    return sz_metal_encode_(call, sz_substrings_source_metal_, name, buffers, sz_substrings_buffers_metal_k, arguments,
                            sizeof(*arguments), one, one);
}

/** Zeroes @p bytes, a whole number of words, from @p begin. */
STRINGZILLA_INLINE sz_status_t sz_substrings_clear_metal_(sz_metal_call_t *call, sz_metal_bound_t *buffers,
                                                          sz_substrings_arguments_metal_t *arguments, void const *begin,
                                                          sz_size_t bytes) {
    if (!sz_metal_resolve_call_(call, begin, bytes, buffers + sz_substrings_cleared_buffer_metal_k))
        return sz_device_memory_mismatch_k;
    arguments->clear_words = bytes / 8;
    return sz_substrings_encode_metal_(call, buffers, "sz_substrings_clear_metal_kernel_", arguments,
                                       sz_substrings_groups_metal_(bytes / 8));
}

/** Turns @p count entries at @p values into their own exclusive prefix sum, in three dispatches:
 *  each tile reduced, the tile totals carried on one threadgroup, each tile's base added back. */
STRINGZILLA_INLINE sz_status_t sz_substrings_scan_metal_(sz_metal_call_t *call, sz_metal_bound_t *buffers,
                                                         sz_substrings_arguments_metal_t *arguments, void const *values,
                                                         sz_size_t count) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_substrings_threads_per_group_metal_k),
                                          (sz_size_t)sz_substrings_gpu_scan_tiles_max_k);
    sz_metal_size_t const tile_groups = {tiles, 1, 1}, one = {1, 1, 1};
    sz_status_t status;
    if (!count) return sz_success_k;
    if (!sz_metal_resolve_call_(call, values, count * sizeof(sz_size_t),
                                buffers + sz_substrings_scanned_buffer_metal_k))
        return sz_device_memory_mismatch_k;
    arguments->scan_count = count, arguments->scan_tiles = tiles;
    arguments->scan_elements_per_tile = sz_size_divide_round_up(count, tiles);
    status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_scan_reduce_metal_kernel_", arguments,
                                         tile_groups);
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_scan_carry_metal_kernel_", arguments, one);
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_scan_apply_metal_kernel_", arguments,
                                             tile_groups);
    return status;
}

/** Binds what every verb reads, the engine's two blocks and the haystacks' tape, the first thing
 *  every verb checks. */
STRINGZILLA_INLINE sz_status_t sz_substrings_resident_metal_(sz_substrings_engine_t const *engine,
                                                             sz_metal_call_t *call, sz_sequence_t const *haystacks,
                                                             sz_metal_bound_t *buffers) {
    if (!sz_metal_resolve_call_(call, engine->memory, engine->memory_bytes,
                                buffers + sz_substrings_memory_buffer_metal_k) ||
        !sz_metal_resolve_call_(call, engine->scratch, engine->scratch_bytes,
                                buffers + sz_substrings_scratch_buffer_metal_k))
        return sz_device_memory_mismatch_k;
    return sz_metal_tape_call_(call, haystacks, buffers + sz_substrings_haystacks_buffer_metal_k);
}

/** Zeroes @p arguments and fills what every dispatch of a round reads: the blocks' host addresses
 *  and the automaton. */
STRINGZILLA_INLINE void sz_substrings_arguments_metal_(sz_substrings_engine_t const *engine,
                                                       sz_sequence_t const *haystacks,
                                                       sz_substrings_arguments_metal_t *arguments) {
    sz_fill_serial_((char *)arguments, sizeof(*arguments), 0);
    arguments->engine_host = (sz_u64_t)engine->memory, arguments->scratch_host = (sz_u64_t)engine->scratch;
    arguments->hot_rows = (sz_u64_t)engine->hot_rows, arguments->byte_to_class = (sz_u64_t)engine->byte_to_class;
    arguments->base = (sz_u64_t)engine->base, arguments->check = (sz_u64_t)engine->check;
    arguments->fail = (sz_u64_t)engine->fail, arguments->accepts_words = (sz_u64_t)engine->accepts_words;
    arguments->outputs = (sz_u64_t)engine->outputs, arguments->outputs_counts = (sz_u64_t)engine->outputs_counts;
    arguments->outputs_offsets = (sz_u64_t)engine->outputs_offsets;
    arguments->hot_count = engine->hot_count, arguments->classes_count = engine->classes_count;
    arguments->root = engine->root, arguments->needles_count = engine->needles_count;
    arguments->max_source_match_bytes = engine->max_source_match_bytes;
    arguments->overlap_policy = (sz_u64_t)engine->overlap_policy;
    arguments->haystacks_count = haystacks->count;
}

/**
 *  @brief Walks every chunk and settles the cover, leaving @p arguments holding what the verbs'
 *      own dispatches read, all in the open @p call. @p arguments comes from
 *      @ref sz_substrings_arguments_metal_.
 *
 *  @param[in] emitting Whether the caller reads the matches, since an overlapping count answers
 *      from the boundaries its sizing walk already scanned. A cover always emits.
 *  @param[out] haystack_offsets Where the per-haystack boundaries land, or @c STRINGZILLA_NULL to
 *      leave them in the round block.
 */
STRINGZILLA_INLINE sz_status_t sz_substrings_walk_metal_(sz_substrings_engine_t const *engine, sz_metal_call_t *call,
                                                         sz_metal_bound_t *buffers, sz_sequence_t const *haystacks,
                                                         sz_bool_t emitting, sz_size_t *haystack_offsets,
                                                         sz_substrings_arguments_metal_t *arguments) {
    sz_substrings_arena_metal_t const arena = sz_substrings_arena_metal_(engine, haystacks->count);
    char *const block = (char *)engine->scratch;
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_size_t const boundaries = haystacks->count + 1;
    sz_metal_size_t const one_group = {1, 1, 1};
    sz_status_t status;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;
    void const *const boundaries_start = haystack_offsets ? (void const *)haystack_offsets
                                                          : (void const *)(block + arena.haystack_offsets);
    if (!sz_metal_resolve_call_(call, boundaries_start, boundaries * sizeof(sz_size_t),
                                buffers + sz_substrings_boundaries_buffer_metal_k))
        return sz_device_memory_mismatch_k;

    arguments->report = (sz_u64_t)(block + arena.report);
    arguments->chunk_bytes = (sz_u64_t)(block + arena.chunk_bytes);
    arguments->chunk_offsets = (sz_u64_t)(block + arena.chunk_offsets);
    arguments->chunk_slots = (sz_u64_t)(block + arena.chunk_slots),
    arguments->tile_sums = (sz_u64_t)(block + arena.tile_sums);
    arguments->emitted = (sz_u64_t)(block + arena.emitted);
    arguments->reported = (sz_u64_t)(block + (covering ? arena.reported : arena.emitted));
    arguments->keep_offsets = covering ? (sz_u64_t)(block + arena.keep_offsets) : 0;
    arguments->gap_offsets = covering ? (sz_u64_t)(block + arena.gap_offsets) : 0;
    arguments->overflow_rows = (sz_u64_t)(block + arena.overflow_rows);
    arguments->slots_count = arena.slots_count, arguments->chunk_budget = engine->chunk_budget;
    arguments->chunk_floor = sz_max_of_two(4 * (sz_size_t)engine->max_source_match_bytes, (sz_size_t)1);
    arguments->matches_budget = engine->matches_budget;
    arguments->emitting = covering || emitting;

    // The head of the round block, and under a cover every keep flag: the cover writes only those
    // below the emitted count, and its scan reads them all.
    status = sz_substrings_clear_metal_(call, buffers, arguments, block + arena.report, arena.tile_sums - arena.report);
    if (status == sz_success_k && covering)
        status = sz_substrings_clear_metal_(call, buffers, arguments, block + arena.keep_offsets,
                                            (engine->matches_budget + 1) * sizeof(sz_size_t));

    // The chunk width and each haystack's chunk range, both device-side, so the host never
    // reads the corpus.
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_chunk_bytes_metal_kernel_", arguments,
                                             one_group);
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_chunk_counts_metal_kernel_", arguments,
                                             sz_substrings_groups_metal_(haystacks->count));
    if (status == sz_success_k)
        status = sz_substrings_scan_metal_(call, buffers, arguments, block + arena.chunk_offsets, boundaries);

    // The sizing walk, over the chunk budget rather than a discovered chunk count, and
    // what it found.
    arguments->pass = sz_substrings_gpu_sizing_k;
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_walk_metal_kernel_", arguments,
                                             sz_substrings_groups_metal_(arena.slots_count));
    if (status == sz_success_k)
        status = sz_substrings_scan_metal_(call, buffers, arguments, block + arena.chunk_slots, arena.slots_count);
    if (status == sz_success_k)
        status = sz_substrings_encode_one_metal_(call, buffers, "sz_substrings_sized_metal_kernel_", arguments);

    // The writing walk, the cover over what it wrote, and the boundaries both feed, each retiring
    // at its first instruction when the sizing walk outran the budget.
    arguments->pass = sz_substrings_gpu_writing_k;
    if (status == sz_success_k && arguments->emitting)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_walk_metal_kernel_", arguments,
                                             sz_substrings_groups_metal_(arena.slots_count));
    if (status == sz_success_k && covering) {
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_cover_metal_kernel_", arguments,
                                             sz_substrings_groups_metal_(engine->matches_budget));
        if (status == sz_success_k)
            status = sz_substrings_scan_metal_(call, buffers, arguments, block + arena.keep_offsets,
                                               engine->matches_budget + 1);
        if (status == sz_success_k)
            status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_compact_metal_kernel_", arguments,
                                                 sz_substrings_groups_metal_(engine->matches_budget));
        if (status == sz_success_k)
            status = sz_substrings_encode_one_metal_(call, buffers, "sz_substrings_covered_metal_kernel_", arguments);
    }
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(call, buffers, "sz_substrings_haystack_offsets_metal_kernel_", arguments,
                                             sz_substrings_groups_metal_(boundaries));
    return status;
}

STRINGZILLA_API sz_status_t sz_substrings_engine_init_metal(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    sz_allocator_t unified;
    sz_metal_bound_t bound;
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (case_sensitivity == sz_substrings_uncased_k) return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    if (!matches_budget) matches_budget = (sz_size_t)sz_substrings_gpu_matches_budget_default_k;
    if (!haystacks_budget) haystacks_budget = (sz_size_t)sz_substrings_gpu_haystacks_budget_default_k;
    status = sz_substrings_engine_compile_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                           sz_cap_metal_k, &unified, stream, engine);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    engine->chunk_budget = sz_substrings_chunk_budget_metal_k;
    engine->haystacks_budget = haystacks_budget;

    sz_size_t const scratch_bytes = sz_substrings_arena_metal_(engine, haystacks_budget).total;
    engine->scratch = unified.allocate(scratch_bytes, unified.handle, stream);
    engine->scratch_bytes = engine->scratch ? scratch_bytes : 0;
    if (!engine->scratch) status = sz_bad_alloc_k;
    else if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &bound) ||
             !sz_metal_resolve_call_(&call, engine->scratch, engine->scratch_bytes, &bound))
        status = sz_device_memory_mismatch_k;
    // Building every pipeline here keeps the compiles out of the rounds.
    for (sz_size_t index = 0; status == sz_success_k &&
                              index != sizeof(sz_substrings_kernels_metal_) / sizeof(sz_substrings_kernels_metal_[0]);
         ++index)
        if (!sz_metal_pipeline_(call.context, sz_substrings_source_metal_, sz_substrings_kernels_metal_[index]))
            status = sz_device_code_mismatch_k;
    if (status != sz_success_k) {
        sz_substrings_engine_free_(engine, stream);
        return sz_metal_commit_(&call, status);
    }
    engine->report = (sz_substrings_report_t *)engine->scratch;
    return sz_metal_commit_(&call, sz_success_k);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_substrings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_substrings_arguments_metal_t arguments;
    sz_metal_call_t call;
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_substrings_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, counts, ((haystacks->count - 1) * counts_stride + 1) * sizeof(sz_size_t),
                                buffers + sz_substrings_counts_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);

    sz_substrings_arguments_metal_(engine, haystacks, &arguments);
    status = sz_substrings_walk_metal_(engine, &call, buffers, haystacks, sz_false_k, STRINGZILLA_NULL, &arguments);
    arguments.counts_stride = counts_stride;
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(&call, buffers, "sz_substrings_counts_metal_kernel_", &arguments,
                                             sz_substrings_groups_metal_(haystacks->count));
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_substrings_find_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_substrings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_substrings_arguments_metal_t arguments;
    sz_metal_bound_t bound;
    sz_metal_call_t call;
    sz_status_t status;
    if (!haystacks->count) return sz_success_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_substrings_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, matches_offsets, (haystacks->count + 1) * sizeof(sz_size_t), &bound))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (matches_capacity && !sz_metal_resolve_call_(&call, matches, matches_capacity * sizeof(sz_substrings_match_t),
                                                    buffers + sz_substrings_matches_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (haystacks->count > engine->haystacks_budget) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);

    // The boundaries kernel retires when the matches did not fit, so the caller's array
    // is zeroed first.
    sz_substrings_arguments_metal_(engine, haystacks, &arguments);
    status = sz_substrings_clear_metal_(&call, buffers, &arguments, matches_offsets,
                                        (haystacks->count + 1) * sizeof(sz_size_t));
    if (status == sz_success_k)
        status = sz_substrings_walk_metal_(engine, &call, buffers, haystacks, sz_true_k, matches_offsets, &arguments);
    arguments.matches_capacity = matches_capacity;
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(&call, buffers, "sz_substrings_store_matches_metal_kernel_", &arguments,
                                             sz_substrings_groups_metal_(matches_capacity));
    if (status == sz_success_k)
        status = sz_substrings_encode_one_metal_(&call, buffers, "sz_substrings_stored_metal_kernel_", &arguments);
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_sequence_t const *replacements, sz_ptr_t target,
                                                        sz_size_t target_capacity, sz_size_t *offsets,
                                                        sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_substrings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_size_t const boundaries = haystacks->count + 1;
    sz_substrings_arguments_metal_t arguments;
    sz_metal_size_t copy_groups = {1, 1, 1};
    sz_metal_call_t call;
    sz_status_t status;
    // A substitution over matches that share bytes is not a function, so there is no
    // cover to apply.
    if (engine->overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != engine->needles_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_substrings_resident_metal_(engine, &call, haystacks, buffers);
    if (status == sz_success_k)
        status = sz_metal_tape_call_(&call, replacements, buffers + sz_substrings_replacements_buffer_metal_k);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, offsets, boundaries * sizeof(sz_size_t),
                                buffers + sz_substrings_offsets_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (target_capacity &&
        !sz_metal_resolve_call_(&call, target, target_capacity, buffers + sz_substrings_target_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);

    sz_substrings_arguments_metal_(engine, haystacks, &arguments);
    status = sz_substrings_walk_metal_(engine, &call, buffers, haystacks, sz_true_k, STRINGZILLA_NULL, &arguments);
    arguments.target_ceiling = target_capacity;
    // The offsets kernels retire when the matches did not fit, so the caller's array
    // is zeroed first.
    if (status == sz_success_k)
        status = sz_substrings_clear_metal_(&call, buffers, &arguments, offsets, boundaries * sizeof(sz_size_t));
    // One threadgroup per haystack, so the drift scan a rewrite needs stays inside
    // one threadgroup's carry.
    if (status == sz_success_k) {
        sz_metal_size_t const haystack_groups = {
            sz_min_of_two(haystacks->count, (sz_size_t)sz_substrings_groups_max_metal_k), 1, 1};
        status = sz_substrings_encode_metal_(&call, buffers, "sz_substrings_rewrite_offsets_metal_kernel_", &arguments,
                                             haystack_groups);
    }
    if (status == sz_success_k) status = sz_substrings_scan_metal_(&call, buffers, &arguments, offsets, boundaries);
    if (status == sz_success_k)
        status = sz_substrings_encode_one_metal_(&call, buffers, "sz_substrings_target_metal_kernel_", &arguments);
    // The copy's grid comes from the caller's own capacity, the last host number a rewrite needs.
    copy_groups.width = sz_min_of_two(
        sz_max_of_two(sz_size_divide_round_up(target_capacity, sz_substrings_gpu_rewrite_tile_bytes_k), (sz_size_t)1),
        (sz_size_t)sz_substrings_groups_max_metal_k);
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(&call, buffers, "sz_substrings_rewrite_copy_metal_kernel_", &arguments,
                                             copy_groups);
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_metal(sz_substrings_engine_t *engine,
                                                            sz_sequence_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_substrings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_substrings_arena_metal_t const arena = sz_substrings_arena_metal_(engine, haystacks->count);
    sz_bool_t const hashed = (sz_bool_t)(engine->needles_count > sz_substrings_tally_slots_metal_k);
    sz_substrings_arguments_metal_t arguments;
    sz_metal_size_t groups = {1, 1, 1};
    sz_metal_call_t call;
    sz_status_t status = sz_substrings_bm25_check(parameters, needle_weights);
    if (status != sz_success_k) return status;
    if (!scores_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_device_enter_metal_(stream, &call);
    if (status == sz_success_k) status = sz_substrings_resident_metal_(engine, &call, haystacks, buffers);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, scores, ((haystacks->count - 1) * scores_stride + 1) * sizeof(sz_f32_t),
                                buffers + sz_substrings_scores_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (!sz_metal_resolve_call_(&call, needle_weights, engine->needles_count * sizeof(sz_f32_t),
                                buffers + sz_substrings_weights_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (document_lengths && !sz_metal_resolve_call_(&call, document_lengths, haystacks->count * sizeof(sz_f32_t),
                                                    buffers + sz_substrings_lengths_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (haystacks->count > engine->haystacks_budget) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);

    sz_substrings_arguments_metal_(engine, haystacks, &arguments);
    arguments.overflow_rows = (sz_u64_t)((char *)engine->scratch + arena.overflow_rows);
    arguments.document_lengths = (sz_u64_t)document_lengths;
    arguments.scores_stride = scores_stride, arguments.tally_hashed = hashed;
    arguments.saturation = parameters->term_frequency_saturation;
    arguments.normalization = parameters->length_normalization;
    arguments.average_length = parameters->average_document_length;

    // A hashed tally spills into one row per threadgroup, which the round block was sized for at
    // init.
    status = sz_substrings_clear_metal_(&call, buffers, &arguments, engine->report, sizeof(sz_substrings_report_t));
    if (status == sz_success_k && hashed)
        status = sz_substrings_clear_metal_(&call, buffers, &arguments, (char *)engine->scratch + arena.overflow_rows,
                                            arena.total - arena.overflow_rows);
    groups.width = sz_min_of_two(haystacks->count, (sz_size_t)sz_substrings_bm25_groups_metal_k);
    if (status == sz_success_k)
        status = sz_substrings_encode_metal_(&call, buffers, "sz_substrings_bm25_metal_kernel_", &arguments, groups);
    return sz_metal_commit_(&call, status);
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_SUBSTRINGS_METAL_H_
