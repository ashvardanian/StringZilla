/**
 *  @file include/stringzilla/overlap/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal backend for window overlap on Apple7 and newer: the engine's forest built on the
 *      host into one shared block, and one call of three passes per round - a segment scan, the
 *      scoring, and the shares.
 *
 *  @sa include/stringzilla/overlap/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/overlap/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @c id<MTLCommandQueue> as the stream. Every block
 *  the kernels read, the forest, the round block, the candidates' tape and the scores, must come
 *  from @ref sz_allocator_init_unified_metal on that stream's device; memory outside the device's
 *  registry draws @c sz_device_memory_mismatch_k.
 *
 *  The kernels read the tape itself instead of calling a sequence's accessors, since Metal has no
 *  device function pointers to call cheaply, so a round's candidates must come from
 *  @ref sz_sequence_realloc_metal; any other sequence draws @c sz_device_code_mismatch_k.
 */
#ifndef STRINGZILLA_OVERLAP_METAL_H_
#define STRINGZILLA_OVERLAP_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/memory/serial.h" // `sz_fill_serial_`
#include "stringzilla/overlap/serial.h"

#if STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the overlap kernels, compiled once per device behind the shared prelude. */
static char const sz_overlap_source_metal_[] = {
#embed "metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Segments one threadgroup scores, the CUDA tier's flat-range ceiling. */
enum { sz_overlap_candidates_per_group_metal_k = 512 };

/** Bytes of a candidate one thread scores, and the scan's threads, as in the kernels. */
enum { sz_overlap_segment_bytes_metal_k = 128, sz_overlap_scan_threads_metal_k = 1024 };

/** The share of the 32 KB of threadgroup memory a staged tree may take, as on CUDA. */
enum { sz_overlap_staged_tree_share_metal_k = 3 };

/** Bytes of the per-query membership filter every threadgroup keeps beside the staged tree. */
enum { sz_overlap_filter_bytes_metal_k = 16384 };

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
    sz_u64_t scores_query_stride, scores_candidate_stride, staged_nodes_count, widest_width;
} sz_overlap_arguments_metal_t;

/** Encodes kernel @p name over @p groups of @p threads into @p call, with the round's @p buffers. */
STRINGZILLA_INLINE sz_status_t sz_overlap_enqueue_metal_(sz_metal_call_t *call, char const *name,
                                                         sz_metal_bound_t const *buffers,
                                                         sz_overlap_arguments_metal_t const *arguments,
                                                         sz_size_t staged_bytes, sz_metal_size_t groups,
                                                         sz_metal_size_t threads) {
    void *const pipeline = sz_metal_pipeline_(call->context, sz_overlap_source_metal_, name);
    void *const encoder = pipeline ? sz_metal_encoder_(call) : STRINGZILLA_NULL;
    if (!encoder) return sz_device_code_mismatch_k;
    if (staged_bytes) {
        sz_metal_threadgroup_memory_(encoder, staged_bytes, 0);
        sz_metal_threadgroup_memory_(encoder, sz_overlap_filter_bytes_metal_k, 1);
    }
    sz_metal_enqueue_(encoder, pipeline, buffers, 4, arguments, sizeof(*arguments), groups, threads);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_metal(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    sz_metal_call_t call;
    sz_status_t const entered = sz_device_enter_metal_(stream, &call);
    if (entered != sz_success_k) return entered;
    if (!window_widths_count || window_widths_count > sz_overlap_gpu_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] > sz_overlap_gpu_widest_window_k) return sz_unexpected_dimensions_k;

    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count,
                                                       sizeof(sz_overlap_geometry_metal_t), &unified, stream, engine);
    if (opened != sz_success_k) return opened;
    sz_metal_bound_t bound;
    if (!sz_metal_resolve_(call.context, engine->memory, engine->memory_bytes, &bound)) {
        sz_overlap_engine_close_(engine, stream);
        return sz_device_memory_mismatch_k;
    }

    // The chain is host working space no kernel reads, so it comes from the host, not the device.
    sz_size_t longest_query = 0, widest_nodes = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
        sz_size_t const entries = engine->nodes_offsets[index + 1] - engine->nodes_offsets[index];
        if (entries > widest_nodes) widest_nodes = entries;
    }
    sz_allocator_t host;
    sz_allocator_init_default(&host);
    sz_size_t const chain_bytes = (longest_query + 1) * sizeof(sz_f64_t);
    sz_f64_t *const chain = (sz_f64_t *)host.allocate(chain_bytes, host.handle, stream);
    if (!chain) {
        sz_overlap_engine_close_(engine, stream);
        return sz_bad_alloc_k;
    }
    sz_overlap_engine_fill_serial_(engine, queries, chain);
    host.free(chain, chain_bytes, host.handle, stream);

    // The round block holds the segment offsets, then one counter per query, candidate and width,
    // which start at zero here and which the shares pass clears again after reading them.
    if (!candidates_budget) candidates_budget = sz_overlap_candidates_budget_default_metal_k;
    sz_size_t const round_bytes = sz_overlap_offsets_bytes_metal_(candidates_budget) +
                                  engine->count * candidates_budget * engine->widths_count * sizeof(sz_u32_t);
    if (sz_overlap_engine_grow_(engine, round_bytes, stream) != sz_success_k) {
        sz_overlap_engine_close_(engine, stream);
        return sz_bad_alloc_k;
    }
    if (!sz_metal_resolve_(call.context, engine->scratch, engine->scratch_bytes, &bound)) {
        sz_overlap_engine_close_(engine, stream);
        return sz_device_memory_mismatch_k;
    }
    sz_fill_serial_((char *)engine->scratch, engine->scratch_bytes, 0);

    // Building every pipeline here keeps the compiles out of the rounds and names their ceilings.
    void *const segments = sz_metal_pipeline_(call.context, sz_overlap_source_metal_,
                                              "sz_overlap_segments_metal_kernel_");
    void *const pipeline = sz_metal_pipeline_(call.context, sz_overlap_source_metal_,
                                              "sz_overlap_scores_metal_kernel_");
    void *const shares = sz_metal_pipeline_(call.context, sz_overlap_source_metal_, "sz_overlap_shares_metal_kernel_");
    if (!segments || !pipeline || !shares ||
        sz_metal_count_(segments, "maxTotalThreadsPerThreadgroup") < sz_overlap_scan_threads_metal_k) {
        sz_overlap_engine_close_(engine, stream);
        return sz_device_code_mismatch_k;
    }
    sz_size_t const ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup");
    sz_size_t const threadgroup_bytes = sz_metal_count_(call.context->device, "maxThreadgroupMemoryLength");
    sz_overlap_geometry_metal_t *const geometry = (sz_overlap_geometry_metal_t *)sz_overlap_engine_head_(engine);
    geometry->candidates_per_group = ceiling < sz_overlap_candidates_per_group_metal_k
                                         ? ceiling
                                         : sz_overlap_candidates_per_group_metal_k;
    geometry->staged_nodes_count =
        widest_nodes * sizeof(sz_u32_t) * sz_overlap_staged_tree_share_metal_k <= threadgroup_bytes ? widest_nodes : 0;
    geometry->candidates_budget = candidates_budget;
    engine->capability = sz_cap_metal_k;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_overlap_scores_metal(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                    sz_f32_t *scores, sz_size_t scores_query_stride,
                                                    sz_size_t scores_candidate_stride, sz_stream_t stream) {
    sz_status_t status = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                    scores_candidate_stride);
    if (status != sz_success_k) return status;
    if (!candidates->count || !engine->count) return sz_success_k;

    // The forest, the round block, the candidates' tape and the scores, as the kernels number them.
    sz_metal_bound_t buffers[4];
    sz_metal_call_t call;
    status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return status;
    sz_size_t const scores_count = (engine->count - 1) * scores_query_stride +
                                   (candidates->count - 1) * scores_candidate_stride + engine->widths_count;
    if (!sz_metal_resolve_(call.context, engine->memory, engine->memory_bytes, &buffers[0]) ||
        !sz_metal_resolve_(call.context, engine->scratch, engine->scratch_bytes, &buffers[1]) ||
        !sz_metal_resolve_(call.context, scores, scores_count * sizeof(sz_f32_t), &buffers[3]))
        return sz_device_memory_mismatch_k;
    status = sz_metal_tape_(call.context, candidates, &buffers[2]);
    if (status != sz_success_k) return status;

    // The host sizes the scoring grid from the tape, which no kernel writes; the offsets are
    // scanned on the device, so a round still in flight never sees them change under it.
    sz_size_t segments_count = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index)
        segments_count += sz_size_divide_round_up(candidates->get_length(candidates->handle, index),
                                                  sz_overlap_segment_bytes_metal_k);
    if (segments_count >> 32) return sz_unexpected_dimensions_k;

    // The counters sit behind offsets sized for the budget rather than for this round, so a smaller
    // round never reads an earlier round's offsets as counts.
    sz_overlap_geometry_metal_t const *const geometry = (sz_overlap_geometry_metal_t const *)sz_overlap_engine_head_(
        engine);
    if (candidates->count > geometry->candidates_budget) return sz_unexpected_dimensions_k;
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
    arguments.scores_query_stride = scores_query_stride;
    arguments.scores_candidate_stride = scores_candidate_stride;
    arguments.staged_nodes_count = geometry->staged_nodes_count;
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
        sz_metal_size_t const grid = {sz_size_divide_round_up(segments_count, per_group), engine->count, 1};
        sz_metal_size_t const threads = {per_group, 1, 1};
        status = sz_overlap_enqueue_metal_(&call, "sz_overlap_scores_metal_kernel_", buffers, &arguments,
                                           staged_bytes ? staged_bytes : 16, grid, threads);
    }

    sz_metal_size_t const shares_grid = {sz_size_divide_round_up(candidates->count, 256), engine->count, 1},
                          shares_threads = {256, 1, 1};
    if (status == sz_success_k)
        status = sz_overlap_enqueue_metal_(&call, "sz_overlap_shares_metal_kernel_", buffers, &arguments, 0,
                                           shares_grid, shares_threads);
    return sz_metal_commit_(&call, status);
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_OVERLAP_METAL_H_
