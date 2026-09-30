/**
 *  @file include/stringzilla/substrings/simt.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Metal backend for multi-pattern search on Apple7 and newer: the vocabulary
 *      compiled on the host into the device's arena, and one call per round, the CUDA tier's
 *      launches as dispatches.
 *
 *  @sa include/stringzilla/substrings/simt.metal, the kernels this embeds
 *  @sa include/stringzilla/substrings/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @ref sz_metal_device_t where the stream goes.
 *  Every block the kernels read must come from that device's arena, and haystacks and replacements
 *  must be bound by @ref sz_sequence_from_string_views, since the kernels read the views
 *  themselves. A round's dispatches run in order inside one command buffer, so each sees what the
 *  one before it wrote, and nothing is read back until the caller synchronizes.
 *
 *  A case-insensitive vocabulary is refused at init: its walk folds the haystack through the
 *  Unicode tables the host tiers read, which have no Metal port.
 */
#ifndef STRINGZILLA_SUBSTRINGS_SIMT_H_
#define STRINGZILLA_SUBSTRINGS_SIMT_H_

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

/** The MSL source of the substrings kernels, compiled once per device. */
static char const sz_substrings_simt_source_[] = {
#embed "simt.metal"
    , 0};
#pragma clang diagnostic pop

/** Threads every threadgroup runs, as in the kernels. */
enum { sz_substrings_simt_threads_per_group_k = 256 };

/** Threadgroups a grid-strided dispatch takes at most, past which its threads stride instead. */
enum { sz_substrings_simt_groups_max_k = 1024 };

/** Tiles a scan cuts its input into, past which one threadgroup carries longer tiles. */
enum { sz_substrings_simt_scan_tiles_max_k = 1024 };

/** Output bytes one threadgroup of a rewrite's copy owns, as in the kernels. */
enum { sz_substrings_simt_rewrite_tile_bytes_k = 4096 };

/** Needles a threadgroup's BM25 tally holds by index; a wider vocabulary hashes, as
 *  in the kernels. */
enum { sz_substrings_simt_tally_slots_k = 2048 };

/** Threadgroups a BM25 dispatch runs, each with its own overflow row when the tally hashes. */
enum { sz_substrings_simt_bm25_groups_k = 64 };

/** Chunks one round cuts its haystacks into beyond one per haystack, enough to fill
 *  an Apple GPU. */
enum { sz_substrings_simt_chunk_budget_k = 1 << 16 };

/** Matches and haystacks a round carries when the caller names no budget, as on CUDA. */
enum {
    sz_substrings_simt_matches_budget_default_k = 1u << 20,
    sz_substrings_simt_haystacks_budget_default_k = 1u << 20
};

/** What a chunk walk does at each match: size the output, or write it. */
typedef enum { sz_substrings_simt_sizing_k = 0, sz_substrings_simt_writing_k = 1 } sz_substrings_simt_pass_t;

/** Every kernel of `simt.metal`, built at init so no round compiles one. */
static char const *const sz_substrings_simt_kernels_[] = {
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
    sz_u64_t host_base;
    sz_u64_t hot_rows, byte_to_class, base, check, fail, accepts_words, outputs, outputs_counts, outputs_offsets;
    sz_u64_t hot_count, classes_count, root, needles_count, max_source_match_bytes, overlap_policy;
    sz_u64_t report, chunk_bytes, chunk_offsets, haystack_offsets, chunk_slots, tile_sums;
    sz_u64_t emitted, reported, keep_offsets, gap_offsets, overflow_rows;
    sz_u64_t slots_count, chunk_budget, chunk_floor, matches_budget, emitting;
    sz_u64_t haystacks, haystacks_count, counts, counts_stride, matches, matches_capacity;
    sz_u64_t replacements, target, target_ceiling, output_offsets;
    sz_u64_t document_lengths, needle_weights, scores, scores_stride, tally_hashed;
    sz_f32_t saturation, normalization, average_length, unused;
    sz_u64_t pass, scan_values, scan_count, scan_elements_per_tile, scan_tiles, clear_begin, clear_words;
} sz_substrings_simt_arguments_t;

/** Byte offsets of the one arena every round runs out of, as the CUDA tier lays it out. */
typedef struct sz_substrings_simt_arena_t {
    sz_size_t report, chunk_bytes, chunk_offsets, haystack_offsets, chunk_slots, tile_sums;
    sz_size_t emitted, reported, keep_offsets, gap_offsets, overflow_rows, slots_count, total;
} sz_substrings_simt_arena_t;

/** Lays the arena out for one round over @p haystacks_count texts. */
STRINGZILLA_INLINE sz_substrings_simt_arena_t sz_substrings_simt_arena_(sz_substrings_engine_t const *engine,
                                                                        sz_size_t haystacks_count) {
    sz_size_t const boundaries = haystacks_count + 1, matches = engine->matches_budget;
    sz_size_t const match_bytes = matches * sizeof(sz_substrings_match_t);
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_size_t const overflow_bytes = engine->needles_count > sz_substrings_simt_tally_slots_k
                                         ? sz_substrings_simt_bm25_groups_k * engine->needles_count * sizeof(sz_u32_t)
                                         : 0;
    sz_substrings_simt_arena_t arena;
    // A chunk is at least the corpus over the budget wide, so the corpus contributes at most
    // `chunk_budget` chunks and each haystack's own remainder at most one more.
    arena.slots_count = engine->chunk_budget + haystacks_count + 1;
    arena.report = 0;
    arena.chunk_bytes = arena.report + sizeof(sz_substrings_report_t);
    arena.chunk_offsets = arena.chunk_bytes + sizeof(sz_size_t);
    arena.haystack_offsets = arena.chunk_offsets + boundaries * sizeof(sz_size_t);
    arena.chunk_slots = arena.haystack_offsets + boundaries * sizeof(sz_size_t);
    arena.tile_sums = arena.chunk_slots + arena.slots_count * sizeof(sz_size_t);
    arena.emitted = arena.tile_sums + sz_substrings_simt_scan_tiles_max_k * sizeof(sz_size_t);
    arena.reported = arena.emitted + match_bytes;
    arena.keep_offsets = arena.reported + (covering ? match_bytes : 0);
    arena.gap_offsets = arena.keep_offsets + (covering ? (matches + 1) * sizeof(sz_size_t) : 0);
    arena.overflow_rows = arena.gap_offsets + (covering ? matches * sizeof(sz_size_t) : 0);
    arena.total = arena.overflow_rows + overflow_bytes;
    return arena;
}

/** Threadgroups of the tier's width a grid-strided dispatch over @p items takes. */
STRINGZILLA_INLINE sz_metal_size_t sz_substrings_simt_groups_(sz_size_t items) {
    sz_size_t const groups = sz_size_divide_round_up(items, sz_substrings_simt_threads_per_group_k);
    sz_metal_size_t const grid = {groups ? sz_min_of_two(groups, (sz_size_t)sz_substrings_simt_groups_max_k) : 1, 1, 1};
    return grid;
}

/** Encodes kernel @p name over @p groups of the tier's width. */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_encode_(sz_metal_device_t *device, char const *name,
                                                          sz_substrings_simt_arguments_t const *arguments,
                                                          sz_metal_size_t groups) {
    sz_metal_size_t const threads = {sz_substrings_simt_threads_per_group_k, 1, 1};
    return sz_metal_encode_(device, sz_substrings_simt_source_, name, arguments, sizeof(*arguments), groups, threads);
}

/** Encodes a single-thread kernel, the shape every report publication takes. */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_encode_one_(sz_metal_device_t *device, char const *name,
                                                              sz_substrings_simt_arguments_t const *arguments) {
    sz_metal_size_t const one = {1, 1, 1};
    return sz_metal_encode_(device, sz_substrings_simt_source_, name, arguments, sizeof(*arguments), one, one);
}

/** Zeroes @p bytes, a whole number of words, from host address @p begin inside the arena. */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_clear_(sz_metal_device_t *device,
                                                         sz_substrings_simt_arguments_t *arguments, void const *begin,
                                                         sz_size_t bytes) {
    arguments->clear_begin = (sz_u64_t)begin, arguments->clear_words = bytes / 8;
    return sz_substrings_simt_encode_(device, "sz_substrings_clear_metal_kernel_", arguments,
                                      sz_substrings_simt_groups_(bytes / 8));
}

/** Turns @p count entries at host address @p values into their own exclusive prefix sum, in
 *  three dispatches: each tile reduced, the tile totals carried on one threadgroup, each tile's
 *  base added back. */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_scan_(sz_metal_device_t *device,
                                                        sz_substrings_simt_arguments_t *arguments, void const *values,
                                                        sz_size_t count) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_substrings_simt_threads_per_group_k),
                                          (sz_size_t)sz_substrings_simt_scan_tiles_max_k);
    sz_metal_size_t const tile_groups = {tiles, 1, 1}, one = {1, 1, 1};
    sz_status_t status;
    if (!count) return sz_success_k;
    arguments->scan_values = (sz_u64_t)values, arguments->scan_count = count, arguments->scan_tiles = tiles;
    arguments->scan_elements_per_tile = sz_size_divide_round_up(count, tiles);
    status = sz_substrings_simt_encode_(device, "sz_substrings_scan_reduce_metal_kernel_", arguments, tile_groups);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_scan_carry_metal_kernel_", arguments, one);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_scan_apply_metal_kernel_", arguments, tile_groups);
    return status;
}

/** Whether @p stream is the device the engine lives on and @p haystacks is a view array there, the
 *  two things every verb checks first. */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_resident_(sz_substrings_engine_t const *engine,
                                                            sz_metal_device_t const *device,
                                                            sz_sequence_t const *haystacks) {
    if (!device || !sz_memory_reaches_metal(device, engine->memory)) return sz_device_memory_mismatch_k;
    return sz_metal_views_(device, haystacks);
}

/** Whether @p count entries of @p bytes each from @p first all lie in @p device 's arena. */
STRINGZILLA_INLINE sz_bool_t sz_substrings_simt_reaches_(sz_metal_device_t const *device, void const *first,
                                                         sz_size_t count, sz_size_t bytes) {
    return sz_memory_reaches_metal(device, first) &&
                   sz_memory_reaches_metal(device, (char const *)first + (count - 1) * bytes)
               ? sz_true_k
               : sz_false_k;
}

/** Zeroes @p arguments and fills what every dispatch of a round reads: the arena's base, the
 *  automaton and the haystacks. */
STRINGZILLA_INLINE void sz_substrings_simt_arguments_(sz_substrings_engine_t const *engine,
                                                      sz_metal_device_t const *device, sz_sequence_t const *haystacks,
                                                      sz_substrings_simt_arguments_t *arguments) {
    sz_fill_serial_((char *)arguments, sizeof(*arguments), 0);
    arguments->host_base = (sz_u64_t)device->arena_host;
    arguments->hot_rows = (sz_u64_t)engine->hot_rows, arguments->byte_to_class = (sz_u64_t)engine->byte_to_class;
    arguments->base = (sz_u64_t)engine->base, arguments->check = (sz_u64_t)engine->check;
    arguments->fail = (sz_u64_t)engine->fail, arguments->accepts_words = (sz_u64_t)engine->accepts_words;
    arguments->outputs = (sz_u64_t)engine->outputs, arguments->outputs_counts = (sz_u64_t)engine->outputs_counts;
    arguments->outputs_offsets = (sz_u64_t)engine->outputs_offsets;
    arguments->hot_count = engine->hot_count, arguments->classes_count = engine->classes_count;
    arguments->root = engine->root, arguments->needles_count = engine->needles_count;
    arguments->max_source_match_bytes = engine->max_source_match_bytes;
    arguments->overlap_policy = (sz_u64_t)engine->overlap_policy;
    arguments->haystacks = (sz_u64_t)haystacks->handle, arguments->haystacks_count = haystacks->count;
}

/**
 *  @brief Walks every chunk and settles the cover, leaving @p arguments holding what the verbs'
 *      own dispatches read, all in the call @p device has open. @p arguments comes from
 *      @ref sz_substrings_simt_arguments_.
 *
 *  @param[in] emitting Whether the caller reads the matches, since an overlapping count answers
 *      from the boundaries its sizing walk already scanned. A cover always emits.
 *  @param[out] haystack_offsets Where the per-haystack boundaries land, or @c STRINGZILLA_NULL to
 *      leave them in the arena.
 */
STRINGZILLA_INLINE sz_status_t sz_substrings_simt_walk_(sz_substrings_engine_t const *engine, sz_metal_device_t *device,
                                                        sz_sequence_t const *haystacks, sz_bool_t emitting,
                                                        sz_size_t *haystack_offsets,
                                                        sz_substrings_simt_arguments_t *arguments) {
    sz_substrings_simt_arena_t const arena = sz_substrings_simt_arena_(engine, haystacks->count);
    char *const block = (char *)engine->scratch;
    sz_bool_t const covering = (sz_bool_t)(engine->overlap_policy != sz_substrings_overlapping_k);
    sz_size_t const boundaries = haystacks->count + 1;
    sz_metal_size_t const one_group = {1, 1, 1};
    sz_status_t status;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;

    arguments->report = (sz_u64_t)(block + arena.report);
    arguments->chunk_bytes = (sz_u64_t)(block + arena.chunk_bytes);
    arguments->chunk_offsets = (sz_u64_t)(block + arena.chunk_offsets);
    arguments->haystack_offsets = (sz_u64_t)(haystack_offsets ? (char *)haystack_offsets
                                                              : block + arena.haystack_offsets);
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

    // The head of the arena, and under a cover every keep flag: the cover writes only those below
    // the emitted count, and its scan reads them all.
    status = sz_substrings_simt_clear_(device, arguments, block + arena.report, arena.tile_sums - arena.report);
    if (status == sz_success_k && covering)
        status = sz_substrings_simt_clear_(device, arguments, block + arena.keep_offsets,
                                           (engine->matches_budget + 1) * sizeof(sz_size_t));

    // The chunk width and each haystack's chunk range, both device-side, so the host never
    // reads the corpus.
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_chunk_bytes_metal_kernel_", arguments, one_group);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_chunk_counts_metal_kernel_", arguments,
                                            sz_substrings_simt_groups_(haystacks->count));
    if (status == sz_success_k)
        status = sz_substrings_simt_scan_(device, arguments, block + arena.chunk_offsets, boundaries);

    // The sizing walk, over the chunk budget rather than a discovered chunk count, and
    // what it found.
    arguments->pass = sz_substrings_simt_sizing_k;
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_walk_metal_kernel_", arguments,
                                            sz_substrings_simt_groups_(arena.slots_count));
    if (status == sz_success_k)
        status = sz_substrings_simt_scan_(device, arguments, block + arena.chunk_slots, arena.slots_count);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_one_(device, "sz_substrings_sized_metal_kernel_", arguments);

    // The writing walk, the cover over what it wrote, and the boundaries both feed, each retiring
    // at its first instruction when the sizing walk outran the budget.
    arguments->pass = sz_substrings_simt_writing_k;
    if (status == sz_success_k && arguments->emitting)
        status = sz_substrings_simt_encode_(device, "sz_substrings_walk_metal_kernel_", arguments,
                                            sz_substrings_simt_groups_(arena.slots_count));
    if (status == sz_success_k && covering) {
        status = sz_substrings_simt_encode_(device, "sz_substrings_cover_metal_kernel_", arguments,
                                            sz_substrings_simt_groups_(engine->matches_budget));
        if (status == sz_success_k)
            status = sz_substrings_simt_scan_(device, arguments, block + arena.keep_offsets,
                                              engine->matches_budget + 1);
        if (status == sz_success_k)
            status = sz_substrings_simt_encode_(device, "sz_substrings_compact_metal_kernel_", arguments,
                                                sz_substrings_simt_groups_(engine->matches_budget));
        if (status == sz_success_k)
            status = sz_substrings_simt_encode_one_(device, "sz_substrings_covered_metal_kernel_", arguments);
    }
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_haystack_offsets_metal_kernel_", arguments,
                                            sz_substrings_simt_groups_(boundaries));
    return status;
}

/**
 *  @brief Compiles @p needles into one block of @p stream 's arena, reserves the round arena its
 *      budgets size, and builds every kernel.
 *
 *  Beside the refusals of the host compile, it draws @c sz_device_memory_mismatch_k for memory
 *  outside the arena from @p allocator or a device other than @p ordinal, @c sz_missing_gpu_k
 *  without a device, and @c sz_device_code_mismatch_k for a case-insensitive vocabulary or a
 *  kernel that fails to build.
 *
 *  @param[in] ordinal The device @p stream was opened on.
 *  @param[in] allocator From @ref sz_memory_allocator_init_metal over the same device, or
 *      @c STRINGZILLA_NULL to derive one from @p stream.
 *  @param[in] stream The @ref sz_metal_device_t every round encodes into.
 *  @return @c sz_success_k, or one of the refusals above.
 *  @sa sz_substrings_engine_init
 */
STRINGZILLA_API sz_status_t sz_substrings_engine_init_metal(
    sz_substrings_engine_t *engine, sz_sequence_t const *needles, sz_substrings_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_size_t ordinal, sz_memory_allocator_t *allocator, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    sz_memory_allocator_t arena;
    sz_status_t status;
    if (!device || !device->device) return sz_missing_gpu_k;
    if (device->ordinal != ordinal) return sz_device_memory_mismatch_k;
    if (case_sensitivity == sz_substrings_uncased_k) return sz_device_code_mismatch_k;
    if (allocator) arena = *allocator;
    else sz_memory_allocator_init_metal(&arena, device);
    if (!matches_budget) matches_budget = (sz_size_t)sz_substrings_simt_matches_budget_default_k;
    if (!haystacks_budget) haystacks_budget = (sz_size_t)sz_substrings_simt_haystacks_budget_default_k;
    status = sz_substrings_engine_compile_(needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                           sz_cap_metal_k, &arena, engine);
    if (status != sz_success_k) return status;
    engine->chunk_budget = sz_substrings_simt_chunk_budget_k;
    engine->haystacks_budget = haystacks_budget;

    sz_size_t const scratch_bytes = sz_substrings_simt_arena_(engine, haystacks_budget).total;
    engine->scratch = arena.allocate(scratch_bytes, arena.handle);
    engine->scratch_bytes = engine->scratch ? scratch_bytes : 0;
    if (!engine->scratch) status = sz_bad_alloc_k;
    else if (!sz_memory_reaches_metal(device, engine->memory) || !sz_memory_reaches_metal(device, engine->scratch))
        status = sz_device_memory_mismatch_k;
    // Building every pipeline here keeps the compiles out of the rounds.
    for (sz_size_t index = 0; status == sz_success_k &&
                              index != sizeof(sz_substrings_simt_kernels_) / sizeof(sz_substrings_simt_kernels_[0]);
         ++index)
        if (!sz_metal_pipeline_(device, sz_substrings_simt_source_, sz_substrings_simt_kernels_[index]))
            status = sz_device_code_mismatch_k;
    if (status != sz_success_k) {
        sz_substrings_engine_free_(engine);
        return status;
    }
    engine->report = (sz_substrings_report_t *)engine->scratch;
    engine->ordinal = ordinal;
    return sz_success_k;
}

/** The Metal kernel of @ref sz_substrings_counts. */
STRINGZILLA_API sz_status_t sz_substrings_counts_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_size_t *counts, sz_size_t counts_stride, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    sz_substrings_simt_arguments_t arguments;
    sz_status_t status;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_substrings_simt_resident_(engine, device, haystacks);
    if (status != sz_success_k) return status;
    if (!sz_substrings_simt_reaches_(device, counts, haystacks->count, counts_stride * sizeof(sz_size_t)))
        return sz_device_memory_mismatch_k;

    sz_substrings_simt_arguments_(engine, device, haystacks, &arguments);
    status = sz_substrings_simt_walk_(engine, device, haystacks, sz_false_k, STRINGZILLA_NULL, &arguments);
    arguments.counts = (sz_u64_t)counts, arguments.counts_stride = counts_stride;
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_counts_metal_kernel_", &arguments,
                                            sz_substrings_simt_groups_(haystacks->count));
    if (status == sz_success_k) sz_metal_commit_(device);
    return status;
}

/** The Metal kernel of @ref sz_substrings_find. */
STRINGZILLA_API sz_status_t sz_substrings_find_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    sz_substrings_simt_arguments_t arguments;
    sz_status_t status;
    if (!haystacks->count) return sz_success_k;
    status = sz_substrings_simt_resident_(engine, device, haystacks);
    if (status != sz_success_k) return status;
    if (!sz_substrings_simt_reaches_(device, matches_offsets, haystacks->count + 1, sizeof(sz_size_t)))
        return sz_device_memory_mismatch_k;
    if (matches_capacity &&
        !sz_substrings_simt_reaches_(device, matches, matches_capacity, sizeof(sz_substrings_match_t)))
        return sz_device_memory_mismatch_k;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;

    // The boundaries kernel retires when the matches did not fit, so the caller's array
    // is zeroed first.
    sz_substrings_simt_arguments_(engine, device, haystacks, &arguments);
    status = sz_substrings_simt_clear_(device, &arguments, matches_offsets, (haystacks->count + 1) * sizeof(sz_size_t));
    if (status == sz_success_k)
        status = sz_substrings_simt_walk_(engine, device, haystacks, sz_true_k, matches_offsets, &arguments);
    arguments.matches = (sz_u64_t)matches, arguments.matches_capacity = matches_capacity;
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_store_matches_metal_kernel_", &arguments,
                                            sz_substrings_simt_groups_(matches_capacity));
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_one_(device, "sz_substrings_stored_metal_kernel_", &arguments);
    if (status == sz_success_k) sz_metal_commit_(device);
    return status;
}

/** The Metal kernel of @ref sz_substrings_replace. */
STRINGZILLA_API sz_status_t sz_substrings_replace_metal(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                        sz_sequence_t const *replacements, sz_ptr_t target,
                                                        sz_size_t target_capacity, sz_size_t *offsets, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    sz_size_t const boundaries = haystacks->count + 1;
    sz_substrings_simt_arguments_t arguments;
    sz_metal_size_t copy_groups = {1, 1, 1};
    sz_status_t status;
    // A substitution over matches that share bytes is not a function, so there is no
    // cover to apply.
    if (engine->overlap_policy == sz_substrings_overlapping_k) return sz_status_unknown_k;
    if (replacements->count != engine->needles_count) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_substrings_simt_resident_(engine, device, haystacks);
    if (status == sz_success_k) status = sz_metal_views_(device, replacements);
    if (status != sz_success_k) return status;
    if (!sz_substrings_simt_reaches_(device, offsets, boundaries, sizeof(sz_size_t)))
        return sz_device_memory_mismatch_k;
    if (target_capacity && !sz_substrings_simt_reaches_(device, target, target_capacity, 1))
        return sz_device_memory_mismatch_k;

    sz_substrings_simt_arguments_(engine, device, haystacks, &arguments);
    status = sz_substrings_simt_walk_(engine, device, haystacks, sz_true_k, STRINGZILLA_NULL, &arguments);
    arguments.replacements = (sz_u64_t)replacements->handle, arguments.output_offsets = (sz_u64_t)offsets;
    arguments.target = (sz_u64_t)target, arguments.target_ceiling = target_capacity;
    // The offsets kernels retire when the matches did not fit, so the caller's array
    // is zeroed first.
    if (status == sz_success_k)
        status = sz_substrings_simt_clear_(device, &arguments, offsets, boundaries * sizeof(sz_size_t));
    // One threadgroup per haystack, so the drift scan a rewrite needs stays inside
    // one threadgroup's carry.
    if (status == sz_success_k) {
        sz_metal_size_t const haystack_groups = {
            sz_min_of_two(haystacks->count, (sz_size_t)sz_substrings_simt_groups_max_k), 1, 1};
        status = sz_substrings_simt_encode_(device, "sz_substrings_rewrite_offsets_metal_kernel_", &arguments,
                                            haystack_groups);
    }
    if (status == sz_success_k) status = sz_substrings_simt_scan_(device, &arguments, offsets, boundaries);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_one_(device, "sz_substrings_target_metal_kernel_", &arguments);
    // The copy's grid comes from the caller's own capacity, the last host number a rewrite needs.
    copy_groups.width = sz_min_of_two(
        sz_max_of_two(sz_size_divide_round_up(target_capacity, sz_substrings_simt_rewrite_tile_bytes_k), (sz_size_t)1),
        (sz_size_t)sz_substrings_simt_groups_max_k);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_rewrite_copy_metal_kernel_", &arguments,
                                            copy_groups);
    if (status == sz_success_k) sz_metal_commit_(device);
    return status;
}

/** The Metal kernel of @ref sz_substrings_bm25_scores. */
STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_metal(sz_substrings_engine_t *engine,
                                                            sz_sequence_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    sz_substrings_simt_arena_t const arena = sz_substrings_simt_arena_(engine, haystacks->count);
    sz_bool_t const hashed = (sz_bool_t)(engine->needles_count > sz_substrings_simt_tally_slots_k);
    sz_substrings_simt_arguments_t arguments;
    sz_metal_size_t groups = {1, 1, 1};
    sz_status_t status = sz_substrings_bm25_check(parameters, needle_weights);
    if (status != sz_success_k) return status;
    if (!scores_stride) return sz_unexpected_dimensions_k;
    if (!haystacks->count) return sz_success_k;
    status = sz_substrings_simt_resident_(engine, device, haystacks);
    if (status != sz_success_k) return status;
    if (!sz_substrings_simt_reaches_(device, scores, haystacks->count, scores_stride * sizeof(sz_f32_t)))
        return sz_device_memory_mismatch_k;
    if (engine->needles_count &&
        !sz_substrings_simt_reaches_(device, needle_weights, engine->needles_count, sizeof(sz_f32_t)))
        return sz_device_memory_mismatch_k;
    if (document_lengths && !sz_substrings_simt_reaches_(device, document_lengths, haystacks->count, sizeof(sz_f32_t)))
        return sz_device_memory_mismatch_k;
    if (haystacks->count > engine->haystacks_budget) return sz_unexpected_dimensions_k;

    sz_substrings_simt_arguments_(engine, device, haystacks, &arguments);
    arguments.overflow_rows = (sz_u64_t)((char *)engine->scratch + arena.overflow_rows);
    arguments.document_lengths = (sz_u64_t)document_lengths, arguments.needle_weights = (sz_u64_t)needle_weights;
    arguments.scores = (sz_u64_t)scores, arguments.scores_stride = scores_stride, arguments.tally_hashed = hashed;
    arguments.saturation = parameters->term_frequency_saturation;
    arguments.normalization = parameters->length_normalization;
    arguments.average_length = parameters->average_document_length;

    // A hashed tally spills into one row per threadgroup, which the arena was sized for at init.
    status = sz_substrings_simt_clear_(device, &arguments, engine->report, sizeof(sz_substrings_report_t));
    if (status == sz_success_k && hashed)
        status = sz_substrings_simt_clear_(device, &arguments, (char *)engine->scratch + arena.overflow_rows,
                                           arena.total - arena.overflow_rows);
    groups.width = sz_min_of_two(haystacks->count, (sz_size_t)sz_substrings_simt_bm25_groups_k);
    if (status == sz_success_k)
        status = sz_substrings_simt_encode_(device, "sz_substrings_bm25_metal_kernel_", &arguments, groups);
    if (status == sz_success_k) sz_metal_commit_(device);
    return status;
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_SUBSTRINGS_SIMT_H_
