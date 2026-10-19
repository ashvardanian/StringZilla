/**
 *  @file include/stringzilla/overlap/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal backend for window overlap on Apple7 and newer: the engine's forest built on the
 *      host into one shared block, and one call of three passes per round - a segment scan, the
 *      scoring, and the matches.
 *
 *  @sa include/stringzilla/overlap/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/overlap/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @c id<MTLCommandQueue> as the stream. Every block
 *  the kernels read, the forest, the round block, the candidates' tape and the matches, must come
 *  from @ref sz_allocator_init_unified_metal on that stream's device; memory outside the device's
 *  registry draws @c sz_device_memory_mismatch_k.
 *
 *  The kernels read the tape itself, since Metal has no device function pointers to call an
 *  accessor through, so a round's candidates must be a tape, as @ref sz_strings_to_tape_best writes
 *  one with that allocator; an accessor draws @c sz_device_memory_mismatch_k.
 */
#ifndef STRINGZILLA_OVERLAP_METAL_H_
#define STRINGZILLA_OVERLAP_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/overlap/serial.h"

#if STRINGZILLA_ARCH_METAL_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_METAL

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the overlap kernels, compiled once per device behind the shared prelude. */
static char const sz_overlap_source_metal_[] = {
#embed "stringzilla/overlap/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Segments one threadgroup scores, the CUDA tier's flat-range ceiling. */
enum { sz_overlap_candidates_per_group_metal_k = 512 };

/** Bytes of a candidate one thread scores, and the scan's threads, as in the kernels. */
enum { sz_overlap_segment_bytes_metal_k = 128, sz_overlap_scan_threads_metal_k = 1024 };

/** The share of the 32 KB of threadgroup memory a staged tree may take: up to half (16 KB). */
enum { sz_overlap_staged_tree_share_metal_k = 2 };

/** Threads a scoring grid carries before further queries fold into each thread's own loop, which
 *  finds its segment's candidate once for all of them: several times what any Apple GPU keeps
 *  resident, which Metal does not report. */
enum { sz_overlap_grid_threads_metal_k = 1 << 20 };

/** Candidates a round carries when the caller names no budget. */
enum { sz_overlap_candidates_budget_default_metal_k = 4096 };

/** Bytes of the segment offsets heading the round block of a @p candidates_budget, a cache line
 *  multiple so the counters behind them start on one whatever the round's own count. */
STRINGZILLA_INLINE sz_size_t sz_overlap_offsets_bytes_metal_(sz_size_t candidates_budget) {
    return sz_size_divide_round_up((candidates_budget + 1) * sizeof(sz_u32_t), 64) * 64;
}

/** What @ref sz_overlap_engine_init_metal resolved once, kept at the head of the engine's block. */
typedef struct sz_overlap_geometry_metal_t {

    /** Threads one threadgroup runs. */
    sz_size_t candidates_per_group;

    /** @c u32 entries a threadgroup stages, sized by the widest tree, or zero for none. */
    sz_size_t staged_nodes_count;

    /** Candidates one round may carry, which sized the round block once. */
    sz_size_t candidates_budget;
} sz_overlap_geometry_metal_t;

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host, scratch_host;
    sz_u64_t nodes, nodes_offsets, keys_counts, widths, powers, lengths;
    sz_u64_t segments, counts;
    sz_u64_t queries_count, widths_count, candidates_count, segments_count;
    sz_u64_t matches_query_stride, matches_candidate_stride, staged_nodes_count, widest_width, candidates_layout,
        misses_bound;
} sz_overlap_arguments_metal_t;

/** Encodes kernel @p name over @p groups of @p threads into @p call, with the round's @p buffers. */
STRINGZILLA_INLINE sz_status_t sz_overlap_enqueue_metal_(sz_metal_call_t *call, char const *name,
                                                         sz_metal_bound_t const *buffers,
                                                         sz_overlap_arguments_metal_t const *arguments,
                                                         sz_size_t staged_bytes, sz_metal_size_t groups,
                                                         sz_metal_size_t threads) {
    void *const pipeline = sz_metal_pipeline_(call->context, sz_overlap_source_metal_, name);
    void *const encoder = pipeline ? call->encoder : STRINGZILLA_NULL;
    if (!encoder) return sz_device_code_mismatch_k;
    if (staged_bytes) sz_metal_threadgroup_memory_(encoder, staged_bytes, 0);
    sz_metal_enqueue_(encoder, pipeline, buffers, 5, arguments, sizeof(*arguments), groups, threads);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_metal(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    sz_metal_call_t call;
    sz_status_t const entered = sz_device_enter_metal_(stream, &call);
    if (entered != sz_success_k) return sz_metal_commit_(&call, entered);

    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count,
                                                       sizeof(sz_overlap_geometry_metal_t), &unified, stream, engine);
    if (opened != sz_success_k) return sz_metal_commit_(&call, opened);
    sz_metal_bound_t bound;
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &bound)) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    }

    // The chain is host working space no kernel reads, so it comes from the host, not the device.
    sz_size_t longest_query = 0, widest_nodes = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
        sz_size_t const entries = engine->nodes_offsets[index + 1] - engine->nodes_offsets[index];
        if (entries > widest_nodes) widest_nodes = entries;
    }
    sz_allocator_t host;
    sz_status_t const status = sz_allocator_init_heap(&host);
    if (status != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, status);
    }
    sz_size_t const chain_bytes = (longest_query + 1) * sizeof(sz_f64_t);
    sz_f64_t *const chain = (sz_f64_t *)host.allocate(chain_bytes, host.handle, stream);
    if (!chain) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_bad_alloc_k);
    }
    sz_overlap_engine_fill_serial_(engine, queries, chain);
    host.free(chain, chain_bytes, host.handle, stream);

    // The round block holds the segment offsets, then one counter per query, candidate and width,
    // which start at zero here and which the matches pass clears again after reading them.
    if (!candidates_budget) candidates_budget = sz_overlap_candidates_budget_default_metal_k;
    sz_size_t const round_bytes = sz_overlap_offsets_bytes_metal_(candidates_budget) +
                                  engine->count * candidates_budget * engine->widths_count * sizeof(sz_u32_t);
    if (sz_overlap_engine_grow_(engine, round_bytes, stream) != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_bad_alloc_k);
    }
    if (!sz_metal_resolve_call_(&call, engine->scratch, engine->scratch_bytes, &bound)) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    }
    memset(engine->scratch, 0, engine->scratch_bytes);

    // Building every pipeline here keeps the compiles out of the rounds and names their ceilings.
    void *const segments = sz_metal_pipeline_(call.context, sz_overlap_source_metal_,
                                              "sz_overlap_segments_metal_kernel_");
    void *const pipeline = sz_metal_pipeline_(call.context, sz_overlap_source_metal_,
                                              "sz_overlap_scores_metal_kernel_");
    void *const counts = sz_metal_pipeline_(call.context, sz_overlap_source_metal_, "sz_overlap_matches_metal_kernel_");
    if (!segments || !pipeline || !counts ||
        sz_metal_count_(segments, "maxTotalThreadsPerThreadgroup") < sz_overlap_scan_threads_metal_k) {
        sz_overlap_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    }
    sz_size_t const ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup");
    sz_size_t const threadgroup_bytes = sz_metal_count_(call.context->device, "maxThreadgroupMemoryLength");
    sz_overlap_geometry_metal_t *const geometry = (sz_overlap_geometry_metal_t *)sz_overlap_engine_head_(engine);
    geometry->candidates_per_group = ceiling < sz_overlap_candidates_per_group_metal_k
                                         ? ceiling
                                         : sz_overlap_candidates_per_group_metal_k;
    geometry->staged_nodes_count = sz_min_of_two(
        widest_nodes, threadgroup_bytes / (sz_overlap_staged_tree_share_metal_k * sizeof(sz_u32_t)));
    geometry->candidates_budget = candidates_budget;
    engine->capability = sz_cap_metal_k;
    return sz_metal_commit_(&call, sz_success_k);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_metal(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream) {
    sz_status_t status = sz_overlap_engine_strides_(engine, candidates->count, matches_query_stride,
                                                    matches_candidate_stride);
    if (status != sz_success_k) return status;
    if (!candidates->count || !engine->count) return sz_success_k;

    // The forest, the round block, the candidates' payload and offsets, and the matches, as the
    // kernels number them.
    sz_metal_bound_t buffers[5];
    sz_metal_call_t call;
    status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    sz_size_t const matches_count = (engine->count - 1) * matches_query_stride +
                                    (candidates->count - 1) * matches_candidate_stride + engine->widths_count;
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &buffers[0]) ||
        !sz_metal_resolve_call_(&call, engine->scratch, engine->scratch_bytes, &buffers[1]) ||
        !sz_metal_resolve_call_(&call, matches, matches_count * sizeof(sz_u32_t), &buffers[4]))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    status = sz_metal_tape_call_(&call, candidates, &buffers[2], &buffers[3]);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);

    // The host sizes the scoring grid from the tape, which no kernel writes; the offsets are
    // scanned on the device, so a round still in flight never sees them change under it.
    sz_size_t segments_count = 0;
    sz_string_view_t views[64];
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        if (index % 64 == 0) sz_strings_get_block_(candidates, index, views);
        segments_count += sz_size_divide_round_up(views[index % 64].length, sz_overlap_segment_bytes_metal_k);
    }
    if (segments_count >> 32) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);

    // The counters sit behind offsets sized for the budget rather than for this round, so a smaller
    // round never reads an earlier round's offsets as counts.
    sz_overlap_geometry_metal_t const *const geometry = (sz_overlap_geometry_metal_t const *)sz_overlap_engine_head_(
        engine);
    if (candidates->count > geometry->candidates_budget) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    sz_size_t const offsets_bytes = sz_overlap_offsets_bytes_metal_(geometry->candidates_budget);

    sz_overlap_arguments_metal_t arguments;
    arguments.engine_host = (sz_u64_t)engine->memory, arguments.scratch_host = (sz_u64_t)engine->scratch;
    arguments.nodes = (sz_u64_t)engine->nodes, arguments.nodes_offsets = (sz_u64_t)engine->nodes_offsets;
    arguments.keys_counts = (sz_u64_t)engine->keys_counts, arguments.widths = (sz_u64_t)engine->widths;
    arguments.powers = (sz_u64_t)engine->powers, arguments.lengths = (sz_u64_t)engine->lengths;
    arguments.segments = (sz_u64_t)engine->scratch;
    arguments.counts = (sz_u64_t)((char *)engine->scratch + offsets_bytes);
    arguments.queries_count = engine->count, arguments.widths_count = engine->widths_count;
    arguments.candidates_count = candidates->count, arguments.segments_count = segments_count;
    arguments.matches_query_stride = matches_query_stride;
    arguments.matches_candidate_stride = matches_candidate_stride;
    arguments.staged_nodes_count = geometry->staged_nodes_count;
    arguments.candidates_layout = (sz_u64_t)candidates->layout;
    arguments.misses_bound = sz_overlap_misses_bound_(radius);
    arguments.widest_width = 0;
    for (sz_size_t index = 0; index != engine->widths_count; ++index)
        if (engine->widths[index] > arguments.widest_width) arguments.widest_width = engine->widths[index];

    sz_metal_size_t const one = {1, 1, 1}, scan_threads = {sz_overlap_scan_threads_metal_k, 1, 1};
    status = sz_overlap_enqueue_metal_(&call, "sz_overlap_segments_metal_kernel_", buffers, &arguments, 0, one,
                                       scan_threads);

    // Threadgroup memory is sized in 16-byte steps, and is bound even when nothing is staged.
    if (status == sz_success_k && segments_count) {
        sz_size_t const staged_nodes_bytes = geometry->staged_nodes_count * sizeof(sz_u32_t);
        sz_size_t const staged_bytes = sz_size_divide_round_up(staged_nodes_bytes, 16) * 16;
        sz_size_t const per_group = geometry->candidates_per_group;
        sz_size_t const groups = sz_size_divide_round_up(segments_count, per_group);
        sz_size_t const rows = sz_min_of_two(
            engine->count, sz_size_divide_round_up(sz_overlap_grid_threads_metal_k, groups * per_group));
        sz_metal_size_t const grid = {groups, rows, 1};
        sz_metal_size_t const threads = {per_group, 1, 1};
        status = sz_overlap_enqueue_metal_(&call, "sz_overlap_scores_metal_kernel_", buffers, &arguments,
                                           staged_bytes ? staged_bytes : 16, grid, threads);
    }

    sz_metal_size_t const counts_grid = {sz_size_divide_round_up(candidates->count, 256), engine->count, 1},
                          counts_threads = {256, 1, 1};
    if (status == sz_success_k)
        status = sz_overlap_enqueue_metal_(&call, "sz_overlap_matches_metal_kernel_", buffers, &arguments, 0,
                                           counts_grid, counts_threads);
    return sz_metal_commit_(&call, status);
}

#pragma endregion Metal

#endif // STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_METAL_
#endif // STRINGZILLA_OVERLAP_METAL_H_
