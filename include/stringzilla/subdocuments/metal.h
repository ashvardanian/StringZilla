/**
 *  @file include/stringzilla/subdocuments/metal.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Metal backend for path queries on Apple7 and newer: the paths compiled on the host into
 *      one shared block, and the CUDA tier's launches as dispatches.
 *
 *  @sa include/stringzilla/subdocuments/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/subdocuments/cuda.cuh, the CUDA sibling
 *
 *  The verbs keep the serial tier's shape, with an @c id<MTLCommandQueue> as the stream. Every
 *  block the kernels read must come from @ref sz_allocator_init_unified_metal on that stream's
 *  device, and documents must be tapes. Both verbs wait for their counting walks, as a malformed
 *  document fails the call and the documents needing more frames than the first walk's stacks hold
 *  are counted again; a find then scans the counts on the host, every Metal block being shared, and
 *  leaves its storing walks running. The large documents the counting walks list walk over chunks
 *  in one pipeline past both walks' stacks, as the CUDA tier walks them, waiting once all their
 *  chunks are counted; storing reruns only its last walk.
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_METAL_H_
#define STRINGZILLA_SUBDOCUMENTS_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/subdocuments.h"

#if STRINGZILLA_ARCH_METAL_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_METAL

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the walks, compiled once per device behind the shared prelude. */
static char const sz_subdocuments_source_metal_[] = {
#embed "stringzilla/subdocuments/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Threads every threadgroup runs. */
enum { sz_subdocuments_threads_per_group_metal_k = 256 };

/** Threads a walk runs at most, which with their stacks bounds the round block. */
enum { sz_subdocuments_threads_metal_k = 32768 };

/** Frames a thread's stack holds in the first walk, past which a document waits for the second. */
enum { sz_subdocuments_frames_metal_k = 16 };

/** Bytes the second walk's stacks may grow to, so the listed documents walk side by side. */
enum { sz_subdocuments_deep_bytes_metal_k = 1 << 30 };

/** Bytes past which a JSON document under paths without filters walks over chunks rather
 *  than on one thread. */
enum { sz_subdocuments_large_bytes_metal_k = 1 << 20 };

/** Entries of one level of depth minima each entry of the level above summarizes. */
enum { sz_subdocuments_minima_fanout_metal_k = 32 };

/** Tiles a cooperative scan cuts its values into at most, one threadgroup each. */
enum { sz_subdocuments_scan_tiles_metal_k = 1024 };

/** The buffers a call binds, in the kernels' numbering, the launch record after the last. */
enum {
    sz_subdocuments_memory_buffer_metal_k,
    sz_subdocuments_scratch_buffer_metal_k,
    sz_subdocuments_documents_buffer_metal_k,
    sz_subdocuments_documents_offsets_buffer_metal_k,
    sz_subdocuments_matches_buffer_metal_k,
    sz_subdocuments_matches_offsets_buffer_metal_k,
    sz_subdocuments_counts_buffer_metal_k,
    sz_subdocuments_regex_buffer_metal_k,
    sz_subdocuments_buffers_metal_k,
};

/** The cooperative walk of the listed large documents as host addresses inside the round block and
 *  its shape, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t firsts, malformed, transfers, tokens, opens, depths, extremes, sums, matches, tiles, scanned;
    sz_u64_t starts, positions, openers, closers, minima, parents, elements, jumps;
    sz_u64_t keys, kinds, offsets, selectors, lives, carried;
    sz_u64_t listed, bound, chunks, containers_count, threads, key_bytes, levels;
    sz_u64_t minima_offsets[8];
    sz_u64_t scanned_count, scan_tiles, round;
    sz_u32_t compose, storing, carried_buffer, descendants;
} sz_subdocuments_large_metal_t;

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host, scratch_host, regex_host;
    sz_u64_t names_data, names_offsets, names_slots, names_steps, wildcard_steps, descendant_steps, relative_steps;
    sz_u64_t attribute_steps, text_steps, filter_steps, filters_roots, filters, instructions, strings_data, final_steps;
    sz_u64_t root_steps, slices, slices_offsets, steps_paths, byte_to_class, forward_rows;
    sz_u64_t header, counted, deep, keys, stacks, large_list;
    sz_u64_t documents_count, documents_layout, counts_stride, matches_capacity, items, threads, frames, frame_words;
    sz_u64_t key_bytes, large_bytes;
    sz_u32_t words_count, names_slots_count, filters_count, format, listed, storing;
    sz_u32_t forward_start, forward_accepts_begin, classes_count;
    sz_subdocuments_large_metal_t large;
} sz_subdocuments_arguments_metal_t;

/** Grows the round block to @p bytes, keeping what it holds; the move waits for the stream. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_reserve_metal_(sz_subdocuments_engine_t *engine, sz_size_t bytes,
                                                              sz_stream_t stream) {
    void *grown;
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    grown = engine->allocator.reallocate(engine->scratch, engine->scratch_bytes, bytes, engine->allocator.handle,
                                         stream);
    if (!grown) return sz_bad_alloc_k;
    engine->scratch = grown, engine->scratch_bytes = bytes;
    return sz_success_k;
}

/** Opens one call on @p stream with every buffer the kernels name bound into @p buffers,
 *  standing the round block in for the outputs a verb has not got; the caller commits the
 *  call whatever this returns. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_open_metal_(sz_subdocuments_engine_t const *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_bytes, sz_match_t *matches,
                                                           sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                           sz_metal_call_t *call, sz_metal_bound_t *buffers,
                                                           sz_stream_t stream) {
    sz_status_t status = sz_device_enter_metal_(stream, call);
    for (sz_size_t index = 0; index != sz_subdocuments_buffers_metal_k; ++index)
        buffers[index].buffer = STRINGZILLA_NULL, buffers[index].offset = 0;
    if (status != sz_success_k) return status;
    if (!sz_metal_resolve_call_(call, engine->memory, engine->memory_bytes,
                                buffers + sz_subdocuments_memory_buffer_metal_k) ||
        !sz_metal_resolve_call_(call, engine->scratch, engine->scratch_bytes,
                                buffers + sz_subdocuments_scratch_buffer_metal_k) ||
        (counts &&
         !sz_metal_resolve_call_(call, counts, counts_bytes, buffers + sz_subdocuments_counts_buffer_metal_k)) ||
        (matches_offsets && !sz_metal_resolve_call_(call, matches_offsets, (documents->count + 1) * sizeof(sz_size_t),
                                                    buffers + sz_subdocuments_matches_offsets_buffer_metal_k)) ||
        (matches_capacity && !sz_metal_resolve_call_(call, matches, matches_capacity * sizeof(sz_match_t),
                                                     buffers + sz_subdocuments_matches_buffer_metal_k)) ||
        (engine->patterns_count && !sz_metal_resolve_call_(call, engine->regex.memory, engine->regex.memory_bytes,
                                                           buffers + sz_subdocuments_regex_buffer_metal_k)))
        return sz_device_memory_mismatch_k;
    status = sz_metal_tape_call_(call, documents, buffers + sz_subdocuments_documents_buffer_metal_k,
                                 buffers + sz_subdocuments_documents_offsets_buffer_metal_k);
    for (sz_size_t index = sz_subdocuments_matches_buffer_metal_k; index != sz_subdocuments_buffers_metal_k; ++index)
        if (!buffers[index].buffer) buffers[index] = buffers[sz_subdocuments_scratch_buffer_metal_k];
    return status;
}

/** Encodes kernel @p name over @p groups threadgroups with @p arguments into @p call. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_dispatch_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                               char const *name,
                                                               sz_subdocuments_arguments_metal_t const *arguments,
                                                               sz_size_t groups) {
    sz_metal_size_t const grid = {groups, 1, 1}, threads = {sz_subdocuments_threads_per_group_metal_k, 1, 1};
    return sz_metal_encode_(call, sz_subdocuments_source_metal_, name, buffers, sz_subdocuments_buffers_metal_k,
                            arguments, sizeof(*arguments), grid, threads);
}

/** Encodes one walk dispatch per record of @p arguments into one call, then the cooperative walk of
 *  the first when it stores, and commits it. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_call_metal_(sz_subdocuments_engine_t const *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_bytes, sz_match_t *matches,
                                                           sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                           sz_subdocuments_arguments_metal_t const *arguments,
                                                           sz_size_t dispatches, sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_subdocuments_buffers_metal_k];
    sz_metal_call_t call;
    sz_status_t status = sz_subdocuments_open_metal_(engine, documents, counts, counts_bytes, matches, matches_capacity,
                                                     matches_offsets, &call, buffers, stream);
    for (sz_size_t index = 0; status == sz_success_k && index != dispatches; ++index)
        status = sz_subdocuments_dispatch_metal_(
            &call, buffers, "sz_subdocuments_walk_metal_kernel_", arguments + index,
            sz_size_divide_round_up((sz_size_t)arguments[index].threads, sz_subdocuments_threads_per_group_metal_k));
    if (status == sz_success_k && arguments->large.storing)
        status = sz_subdocuments_dispatch_metal_(
            &call, buffers, "sz_subdocuments_large_walk_metal_kernel_", arguments,
            (sz_size_t)sz_subdocuments_threads_metal_k / sz_subdocuments_threads_per_group_metal_k);
    return sz_metal_commit_(&call, status);
}

/** Moves every address of @p arguments inside the round block along with it, which growing
 *  it may have moved. */
STRINGZILLA_INLINE void sz_subdocuments_rebase_metal_(sz_subdocuments_engine_t const *engine,
                                                      sz_subdocuments_arguments_metal_t *arguments) {
    sz_u64_t const moved = (sz_u64_t)engine->scratch - arguments->scratch_host;
    arguments->scratch_host += moved, arguments->header += moved;
    arguments->counted += arguments->counted ? moved : 0;
    arguments->deep += moved, arguments->keys += moved, arguments->stacks += moved, arguments->large_list += moved;
}

/** Takes @p bytes past @p offset, each array 64-byte aligned, and returns where they start. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_take_metal_(sz_size_t *offset, sz_size_t bytes) {
    sz_size_t const taken = *offset;
    *offset += sz_size_divide_round_up(bytes, 64) * 64;
    return taken;
}

/**
 *  @brief Points @p large at host addresses from @p region on for @p listed documents over
 *      @p chunks chunks: the documents' and the chunks' values, and, once their scans counted
 *      them, @p tokens tokens and @p containers containers.
 *
 *  The layout is the one @c sz_subdocuments_large_layout_simt_ gives the CUDA walk.
 *
 *  @return Bytes the region spans.
 */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_large_layout_metal_(sz_subdocuments_engine_t const *engine,
                                                                 sz_u64_t region, sz_size_t listed, sz_size_t chunks,
                                                                 sz_size_t threads, sz_size_t key_bytes,
                                                                 sz_size_t tokens, sz_size_t containers,
                                                                 sz_subdocuments_large_metal_t *large) {
    sz_size_t const values = (chunks + 1) * sizeof(sz_u64_t), words = engine->words_count;
    sz_size_t offset = 0, level = 1, size = containers, minima = containers, word;
    sz_u64_t descendants = 0;
    for (word = 0; word != words; ++word) descendants |= engine->descendant_steps[word];
    large->firsts = region + sz_subdocuments_take_metal_(&offset, (listed + 1) * sizeof(sz_u64_t));
    large->malformed = region + sz_subdocuments_take_metal_(&offset, listed * sizeof(sz_u32_t));
    large->transfers = region + sz_subdocuments_take_metal_(&offset, values);
    large->tokens = region + sz_subdocuments_take_metal_(&offset, values);
    large->opens = region + sz_subdocuments_take_metal_(&offset, values);
    large->depths = region + sz_subdocuments_take_metal_(&offset, values);
    large->extremes = region + sz_subdocuments_take_metal_(&offset, values);
    large->sums = region + sz_subdocuments_take_metal_(&offset, values);
    large->matches = region + sz_subdocuments_take_metal_(&offset, values);
    large->tiles = region + sz_subdocuments_take_metal_(&offset, sz_subdocuments_scan_tiles_metal_k * sizeof(sz_u64_t));
    large->starts = region + sz_subdocuments_take_metal_(&offset, chunks * sizeof(sz_u32_t));
    large->keys = region + sz_subdocuments_take_metal_(&offset, threads * key_bytes);
    large->positions = region + sz_subdocuments_take_metal_(&offset, tokens * sizeof(sz_u32_t));
    large->kinds = region + sz_subdocuments_take_metal_(&offset, tokens);
    large->openers = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u32_t));
    large->closers = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u32_t));
    large->parents = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u32_t));
    large->elements = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u32_t));
    large->minima_offsets[0] = 0;
    for (; size > sz_subdocuments_minima_fanout_metal_k; ++level) {
        size = sz_size_divide_round_up(size, sz_subdocuments_minima_fanout_metal_k);
        large->minima_offsets[level] = minima, minima += size;
    }
    large->minima_offsets[level] = minima, large->levels = level;
    large->minima = region + sz_subdocuments_take_metal_(&offset, minima * sizeof(sz_u32_t));
    large->offsets = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u64_t));
    large->selectors = region + sz_subdocuments_take_metal_(&offset, containers * sizeof(sz_u64_t));
    large->lives = region + sz_subdocuments_take_metal_(&offset, 2 * containers * words * sizeof(sz_u64_t));
    // Without descendant segments every live set is its parent's advanced, and nothing is gathered.
    large->descendants = descendants != 0;
    large->jumps = region + sz_subdocuments_take_metal_(&offset, descendants ? 2 * containers * sizeof(sz_u32_t) : 0);
    large->carried = region +
                     sz_subdocuments_take_metal_(&offset, descendants ? 4 * containers * words * sizeof(sz_u64_t) : 0);
    large->listed = listed, large->chunks = chunks, large->containers_count = containers, large->threads = threads;
    large->key_bytes = key_bytes;
    return offset;
}

/** Steps of the longest path, which bounds how many times live sets advance before they settle. */
STRINGZILLA_INLINE sz_size_t sz_subdocuments_longest_path_metal_(sz_subdocuments_engine_t const *engine) {
    sz_size_t longest = 0, first = 0;
    for (sz_size_t step = 0; step != engine->steps_count; ++step) {
        if (engine->root_steps[step >> 6] >> (step & 63) & 1) first = step;
        if (engine->final_steps[step >> 6] >> (step & 63) & 1) longest = sz_max_of_two(longest, step - first + 1);
    }
    return longest;
}

/** Encodes an exclusive scan of the @p count values at host address @p values in three dispatches:
 *  summing them, or composing the chunks' effects on the carries when @p compose is set. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_scan_metal_(sz_metal_call_t *call, sz_metal_bound_t const *buffers,
                                                                 sz_subdocuments_arguments_metal_t *arguments,
                                                                 sz_u64_t values, sz_size_t count, sz_u32_t compose) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_subdocuments_threads_per_group_metal_k),
                                          (sz_size_t)sz_subdocuments_scan_tiles_metal_k);
    sz_status_t status;
    arguments->large.scanned = values, arguments->large.scanned_count = count, arguments->large.compose = compose;
    arguments->large.scan_tiles = tiles;
    status = sz_subdocuments_dispatch_metal_(call, buffers, "sz_subdocuments_large_reduce_metal_kernel_", arguments,
                                             tiles);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(call, buffers, "sz_subdocuments_large_carry_metal_kernel_", arguments,
                                                 1);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(call, buffers, "sz_subdocuments_large_apply_metal_kernel_", arguments,
                                                 tiles);
    return status;
}

/** Walks every document the round block's header lists for chunks, below its first malformed one,
 *  as one pipeline from @p region on in the round block, as @c sz_subdocuments_large_cuda_ does:
 *  counts their matches into their counts, refusing each where the serial walk would, and leaves
 *  the first record of @p arguments for a storing walk to rerun. Waits once their chunks are
 *  counted, and moves both records of @p arguments along with the round block. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_metal_(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_bytes,
                                                            sz_subdocuments_arguments_metal_t *arguments,
                                                            sz_size_t region, sz_stream_t stream) {
    sz_size_t const threads = sz_subdocuments_threads_metal_k;
    sz_size_t const groups = threads / sz_subdocuments_threads_per_group_metal_k;
    sz_size_t const iterations = sz_subdocuments_longest_path_metal_(engine);
    sz_u32_t const *header = (sz_u32_t const *)engine->scratch;
    sz_size_t const listed = header[3], chunks = header[4];
    sz_subdocuments_large_metal_t *const large = &arguments->large;
    sz_size_t bytes = sz_subdocuments_large_layout_metal_(engine, 0, listed, chunks, threads, arguments->key_bytes, 0,
                                                          0, large);
    sz_size_t level, iteration, jump, rounds = 0, tokens, containers;
    sz_metal_bound_t buffers[sz_subdocuments_buffers_metal_k];
    sz_metal_call_t call;
    sz_status_t status = sz_subdocuments_reserve_metal_(engine, region + bytes, stream);
    if (status != sz_success_k) return status;
    sz_subdocuments_rebase_metal_(engine, arguments), sz_subdocuments_rebase_metal_(engine, arguments + 1);
    header = (sz_u32_t const *)engine->scratch;
    sz_subdocuments_large_layout_metal_(engine, arguments->scratch_host + region, listed, chunks, threads,
                                        arguments->key_bytes, 0, 0, large);
    large->bound = header[0], large->storing = 0, large->carried_buffer = 0;

    // Each document's first chunk and the chunks' carries, then their tokens, containers and
    // depths, numbered by scans past the documents their depths or quotes refuse.
    status = sz_subdocuments_open_metal_(engine, documents, counts, counts_bytes, STRINGZILLA_NULL, 0, STRINGZILLA_NULL,
                                         &call, buffers, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_lengths_metal_kernel_",
                                                 arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->firsts, listed + 1, 0);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_summary_metal_kernel_",
                                                 arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->transfers, chunks + 1, 1);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_count_metal_kernel_", arguments,
                                                 groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->depths, chunks + 1, 0);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_check_metal_kernel_", arguments,
                                                 groups);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_drop_metal_kernel_", arguments,
                                                 groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->tokens, chunks + 1, 0);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->opens, chunks + 1, 0);
    status = sz_metal_commit_(&call, status);
    if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);
    if (status != sz_success_k) return status;
    sz_u64_t const *const token_starts = (sz_u64_t const *)large->tokens;
    sz_u64_t const *const container_starts = (sz_u64_t const *)large->opens;
    tokens = (sz_size_t)token_starts[chunks];
    containers = (sz_size_t)container_starts[chunks];

    bytes = sz_subdocuments_large_layout_metal_(engine, 0, listed, chunks, threads, arguments->key_bytes, tokens,
                                                containers, large);
    status = sz_subdocuments_reserve_metal_(engine, region + bytes, stream);
    if (status != sz_success_k) return status;
    sz_subdocuments_rebase_metal_(engine, arguments), sz_subdocuments_rebase_metal_(engine, arguments + 1);
    header = (sz_u32_t const *)engine->scratch;
    sz_subdocuments_large_layout_metal_(engine, arguments->scratch_host + region, listed, chunks, threads,
                                        arguments->key_bytes, tokens, containers, large);

    // Tokens and containers, the parents the pyramid of minima finds, closes,
    // element counts and positions.
    status = sz_subdocuments_open_metal_(engine, documents, counts, counts_bytes, STRINGZILLA_NULL, 0, STRINGZILLA_NULL,
                                         &call, buffers, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_tokens_metal_kernel_",
                                                 arguments, groups);
    for (level = 1; status == sz_success_k && level != large->levels; ++level)
        large->round = level, status = sz_subdocuments_dispatch_metal_(
                                  &call, buffers, "sz_subdocuments_large_minima_metal_kernel_", arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_parents_metal_kernel_",
                                                 arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_closes_metal_kernel_",
                                                 arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_sums_metal_kernel_", arguments,
                                                 groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->sums, chunks + 1, 0);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_offsets_metal_kernel_",
                                                 arguments, groups);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_selectors_metal_kernel_",
                                                 arguments, groups);

    // Live sets advance once per step of the longest path, each time after
    // gathering what ancestors carry.
    while ((sz_size_t)1 << rounds < header[5]) ++rounds;
    for (iteration = 0; status == sz_success_k && iteration <= iterations; ++iteration) {
        for (jump = 0; large->descendants && status == sz_success_k && jump <= rounds; ++jump)
            large->round = jump, status = sz_subdocuments_dispatch_metal_(
                                     &call, buffers, "sz_subdocuments_large_jump_metal_kernel_", arguments, groups);
        large->carried_buffer = rounds & 1;
        if (status == sz_success_k && iteration != iterations)
            status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_advance_metal_kernel_",
                                                     arguments, groups);
    }

    // Counting checks every token and scans the matches, from which storing walks again.
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_walk_metal_kernel_", arguments,
                                                 groups);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_metal_(&call, buffers, arguments, large->matches, chunks + 1, 0);
    if (status == sz_success_k)
        status = sz_subdocuments_dispatch_metal_(&call, buffers, "sz_subdocuments_large_finish_metal_kernel_",
                                                 arguments, groups);
    return sz_metal_commit_(&call, status);
}

/** Counts every document into @p counts, or, with @p counts null, finds every match into @p matches
 *  and @p matches_offsets; waits for the counting walks, and fails as the serial tier does on the
 *  first malformed document. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_walks_metal_(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_match_t *matches,
                                                            sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                            sz_size_t *malformed_index, sz_stream_t stream) {
    sz_size_t const count = documents->count, words = engine->words_count;
    // An XML frame's positional counters follow its live set, two to a word, as the serial
    // markup walk keeps them.
    sz_size_t const counter_words = engine->format == sz_subdocuments_xml_k
                                        ? (engine->slices_offsets[engine->steps_count] + 1) / 2
                                        : 0;
    sz_size_t const frame_words = sz_subdocuments_frame_header_words_() + words + counter_words;
    sz_size_t const frame_bytes = frame_words * sizeof(sz_u64_t), key_bytes = sz_subdocuments_key_bytes_(engine);
    sz_size_t const threads = sz_min_of_two(count, (sz_size_t)sz_subdocuments_threads_metal_k);
    sz_size_t const counts_bytes = counts ? ((count - 1) * counts_stride + 1) * sizeof(sz_size_t) : 0;
    // The header's six 32-bit words, the counts, the deep and the large documents, the
    // key buffers, the stacks.
    sz_size_t const counted = 6 * sizeof(sz_u32_t), deep = counted + (count + 1) * sizeof(sz_u64_t);
    sz_size_t const large = deep + sz_size_divide_round_up(count, 2) * sizeof(sz_u64_t);
    sz_size_t const keys = large + 2 * count * sizeof(sz_u64_t);
    sz_size_t const stacks = keys + threads * key_bytes;
    sz_size_t const stacks_bytes = threads * sz_subdocuments_frames_metal_k * frame_bytes;
    sz_subdocuments_arguments_metal_t arguments[2];
    sz_u32_t *header;
    sz_size_t larges = 0, deep_stacks_bytes = 0;
    sz_status_t status;
    if (count >> 32) return sz_unexpected_dimensions_k;
    status = sz_subdocuments_reserve_metal_(engine, stacks + stacks_bytes, stream);
    if (status != sz_success_k) return status;

    memset(arguments, 0, sizeof(arguments));
    arguments[0].engine_host = (sz_u64_t)engine->memory, arguments[0].scratch_host = (sz_u64_t)engine->scratch;
    arguments[0].names_data = (sz_u64_t)engine->names_data,
    arguments[0].names_offsets = (sz_u64_t)engine->names_offsets;
    arguments[0].names_slots = (sz_u64_t)engine->names_slots, arguments[0].names_steps = (sz_u64_t)engine->names_steps;
    arguments[0].wildcard_steps = (sz_u64_t)engine->wildcard_steps;
    arguments[0].descendant_steps = (sz_u64_t)engine->descendant_steps;
    arguments[0].relative_steps = (sz_u64_t)engine->relative_steps;
    arguments[0].attribute_steps = (sz_u64_t)engine->attribute_steps;
    arguments[0].text_steps = (sz_u64_t)engine->text_steps, arguments[0].filter_steps = (sz_u64_t)engine->filter_steps;
    arguments[0].filters_roots = (sz_u64_t)engine->filters_roots, arguments[0].filters = (sz_u64_t)engine->filters;
    arguments[0].instructions = (sz_u64_t)engine->instructions;
    arguments[0].strings_data = (sz_u64_t)engine->strings_data;
    arguments[0].final_steps = (sz_u64_t)engine->final_steps, arguments[0].root_steps = (sz_u64_t)engine->root_steps;
    arguments[0].slices = (sz_u64_t)engine->slices, arguments[0].slices_offsets = (sz_u64_t)engine->slices_offsets;
    arguments[0].steps_paths = (sz_u64_t)engine->steps_paths;
    if (engine->patterns_count) {
        arguments[0].regex_host = (sz_u64_t)engine->regex.memory;
        arguments[0].byte_to_class = (sz_u64_t)engine->regex.byte_to_class;
        arguments[0].forward_rows = (sz_u64_t)engine->regex.forward_rows;
        arguments[0].forward_start = engine->regex.forward_starts[0];
        arguments[0].forward_accepts_begin = engine->regex.forward_accepts_begin;
        arguments[0].classes_count = engine->regex.classes_count;
    }
    arguments[0].header = (sz_u64_t)engine->scratch;
    arguments[0].counted = counts ? 0 : (sz_u64_t)engine->scratch + counted;
    arguments[0].deep = (sz_u64_t)engine->scratch + deep, arguments[0].keys = (sz_u64_t)engine->scratch + keys;
    arguments[0].stacks = (sz_u64_t)engine->scratch + stacks,
    arguments[0].large_list = (sz_u64_t)engine->scratch + large;
    // Filters and XML keep every document on one thread.
    arguments[0].large_bytes = engine->format == sz_subdocuments_json_k && !engine->filters_count
                                   ? (sz_u64_t)sz_subdocuments_large_bytes_metal_k
                                   : (sz_u64_t)STRINGZILLA_SIZE_MAX;
    arguments[0].documents_count = count, arguments[0].documents_layout = (sz_u64_t)documents->layout;
    arguments[0].counts_stride = counts_stride, arguments[0].matches_capacity = matches_capacity;
    arguments[0].items = count, arguments[0].threads = threads, arguments[0].frames = sz_subdocuments_frames_metal_k;
    arguments[0].frame_words = frame_words, arguments[0].key_bytes = key_bytes;
    arguments[0].words_count = (sz_u32_t)words, arguments[0].names_slots_count = engine->names_slots_count;
    arguments[0].filters_count = engine->filters_count, arguments[0].format = (sz_u32_t)engine->format;
    arguments[1] = arguments[0], arguments[1].listed = 1;

    // The stream is joined after every counting walk, so no queued dispatch still reads the header.
    header = (sz_u32_t *)engine->scratch;
    header[0] = (sz_u32_t)count, header[1] = 0, header[2] = 0, header[3] = 0, header[4] = 0, header[5] = 0;
    status = sz_subdocuments_call_metal_(engine, documents, counts, counts_bytes, STRINGZILLA_NULL, 0, STRINGZILLA_NULL,
                                         arguments, 1, stream);
    if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);

    // The listed documents walk again side by side, on stacks as deep as the deepest needs, which
    // grow with how many are listed up to a cap past which they take turns on fewer threads.
    if (status == sz_success_k && header[1]) {
        sz_size_t const frames = (sz_size_t)header[2], needed = (sz_size_t)header[1] * frames * frame_bytes;
        sz_size_t const budget = sz_max_of_two(stacks_bytes,
                                               sz_min_of_two(needed, (sz_size_t)sz_subdocuments_deep_bytes_metal_k));
        sz_size_t const deep_threads = sz_min_of_two(sz_min_of_two((sz_size_t)header[1], threads),
                                                     sz_max_of_two(budget / (frames * frame_bytes), (sz_size_t)1));
        arguments[1].items = header[1], arguments[1].threads = deep_threads, arguments[1].frames = frames;
        deep_stacks_bytes = deep_threads * frames * frame_bytes;
        status = sz_subdocuments_reserve_metal_(engine, stacks + deep_stacks_bytes, stream);
        // A grown block moved, and every address inside it with it.
        if (status == sz_success_k)
            sz_subdocuments_rebase_metal_(engine, arguments), sz_subdocuments_rebase_metal_(engine, arguments + 1);
        header = (sz_u32_t *)engine->scratch;
        if (status == sz_success_k)
            status = sz_subdocuments_call_metal_(engine, documents, counts, counts_bytes, STRINGZILLA_NULL, 0,
                                                 STRINGZILLA_NULL, arguments + 1, 1, stream);
        if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);
    }

    // The large documents walk over chunks together, past none already known malformed, after the
    // stacks of both walks, so storing reruns those while keeping their arrays.
    larges = header[3];
    if (status == sz_success_k && larges) {
        sz_size_t const stacked = stacks + sz_max_of_two(stacks_bytes, deep_stacks_bytes);
        sz_size_t const region = sz_size_divide_round_up(stacked, 64) * 64;
        status = sz_subdocuments_large_metal_(engine, documents, counts, counts_bytes, arguments, region, stream);
        if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);
        header = (sz_u32_t *)engine->scratch;
    }
    if (status != sz_success_k) return status;
    if (header[0] != count) {
        *malformed_index = header[0];
        return sz_invalid_syntax_k;
    }
    if (counts) return sz_success_k;

    // Each document's count becomes its first match, the total after the last, and every walk
    // stores in one call.
    {
        sz_size_t *const values = (sz_size_t *)((sz_ptr_t)engine->scratch + counted);
        sz_size_t running = 0;
        for (sz_size_t index = 0; index != count; ++index) {
            sz_size_t const value = values[index];
            values[index] = running, running += value;
        }
        values[count] = running;
    }
    arguments[0].storing = 1, arguments[1].storing = 1, arguments[0].large.storing = larges != 0;
    return sz_subdocuments_call_metal_(engine, documents, STRINGZILLA_NULL, 0, matches, matches_capacity,
                                       matches_offsets, arguments, header[1] ? 2 : 1, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_metal(sz_subdocuments_engine_t *engine,
                                                              sz_strings_view_t const *paths,
                                                              sz_subdocuments_format_t format,
                                                              sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                              sz_stream_t stream) {
    sz_metal_context_t *const context = sz_metal_context_(stream);
    sz_allocator_t unified;
    sz_metal_bound_t bound;
    sz_status_t status;
    if (!context) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    status = sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_metal_k, &unified, stream);
    if (status != sz_success_k) return status;
    // `match()` and `search()` walk the forward DFA, which a pattern set past
    // its state budget lacks.
    if (engine->patterns_count && !engine->regex.forward_rows) status = sz_device_code_mismatch_k;
    else if (!sz_metal_resolve_(context, engine->memory, engine->memory_bytes, &bound))
        status = sz_device_memory_mismatch_k;
    // Building the pipeline here keeps the compile out of the calls.
    else if (!sz_metal_pipeline_(context, sz_subdocuments_source_metal_, "sz_subdocuments_walk_metal_kernel_"))
        status = sz_device_code_mismatch_k;
    if (status != sz_success_k) sz_subdocuments_engine_free_(engine, stream);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_metal(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_size_t *counts,
                                                         sz_size_t counts_stride, sz_size_t *malformed_index,
                                                         sz_stream_t stream) {
    if (!engine->steps_count) return sz_missing_kernel_k;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!documents->count) return sz_success_k;
    if (documents->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    return sz_subdocuments_walks_metal_(engine, documents, counts, counts_stride, STRINGZILLA_NULL, 0, STRINGZILLA_NULL,
                                        malformed_index, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_metal(sz_subdocuments_engine_t *engine,
                                                       sz_strings_view_t const *documents, sz_match_t *matches,
                                                       sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                       sz_size_t *malformed_index, sz_stream_t stream) {
    if (!engine->steps_count) return sz_missing_kernel_k;
    if (documents->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    if (!documents->count) {
        matches_offsets[0] = 0;
        return sz_success_k;
    }
    return sz_subdocuments_walks_metal_(engine, documents, STRINGZILLA_NULL, 0, matches, matches_capacity,
                                        matches_offsets, malformed_index, stream);
}

#pragma endregion Metal

#endif // STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_METAL_
#endif // STRINGZILLA_SUBDOCUMENTS_METAL_H_
