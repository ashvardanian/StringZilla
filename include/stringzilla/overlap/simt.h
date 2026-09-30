/**
 *  @file include/stringzilla/overlap/simt.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal backend for window overlap on Apple7 and newer: the engine's forest built on the
 *      host into the device's arena, and one call of three passes per round - a segment scan, the
 *      scoring, and the shares.
 *
 *  @sa include/stringzilla/overlap/simt.metal, the kernels this embeds
 *  @sa include/stringzilla/overlap/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @ref sz_metal_device_t where the stream goes.
 *  Every block the kernels read - the forest, the candidates' views, their texts, the scores - must
 *  come from that device's arena, which @ref sz_memory_allocator_init_metal hands out.
 *
 *  The kernels read the view array itself instead of calling a sequence's accessors, since Metal
 *  has no device function pointers to call cheaply, so a round's candidates must be bound by
 *  @ref sz_sequence_from_string_views; any other sequence is refused.
 */
#ifndef STRINGZILLA_OVERLAP_SIMT_H_
#define STRINGZILLA_OVERLAP_SIMT_H_

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

/** The MSL source of the overlap kernel, compiled once per device. */
static char const sz_overlap_simt_source_[] = {
#embed "simt.metal"
    , 0};
#pragma clang diagnostic pop

/** Segments one threadgroup scores, the CUDA tier's flat-range ceiling. */
enum { sz_overlap_simt_candidates_per_group_k = 512 };

/** Bytes of a candidate one thread scores, and the scan's threads, as in the kernels. */
enum { sz_overlap_simt_segment_bytes_k = 128, sz_overlap_simt_scan_threads_k = 1024 };

/** The share of the 32 KB of threadgroup memory a staged tree may take, as on CUDA. */
enum { sz_overlap_simt_staged_tree_share_k = 3 };

/** Bytes of the per-query membership filter every threadgroup keeps beside the staged tree. */
enum { sz_overlap_simt_filter_bytes_k = 16384 };

/** Candidates a round carries when the caller names no budget. */
enum { sz_overlap_simt_candidates_budget_default_k = 4096 };

/** Bytes of the segment offsets heading the round block of a @p candidates_budget, a cache line
 *  multiple so the counters behind them start on one whatever the round's own count. */
STRINGZILLA_INLINE sz_size_t sz_overlap_simt_offsets_bytes_(sz_size_t candidates_budget) {
    return ((candidates_budget + 1) * sizeof(sz_u32_t) + 63) / 64 * 64;
}

/** What @ref sz_overlap_engine_init_metal resolved once, kept at the head of the engine's block. */
typedef struct sz_overlap_simt_geometry_t {

    /** The device every round encodes into. */
    sz_metal_device_t *device;

    /** Threads one threadgroup runs. */
    sz_size_t candidates_per_group;

    /** @c u32 entries a threadgroup stages, sized by the widest tree, or zero for none. */
    sz_size_t staged_nodes_count;

    /** Candidates one round may carry, which sized the round block once. */
    sz_size_t candidates_budget;
} sz_overlap_simt_geometry_t;

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t host_base;
    sz_u64_t nodes, nodes_offsets, keys_counts, widths, powers, lengths;
    sz_u64_t views, scores, segments, counts;
    sz_u64_t queries_count, widths_count, candidates_count, segments_count;
    sz_u64_t scores_query_stride, scores_candidate_stride, staged_nodes_count, widest_width;
} sz_overlap_simt_arguments_t;

/** Binds the arena and @p arguments, then encodes kernel @p name over @p groups of @p threads. */
STRINGZILLA_INLINE sz_status_t sz_overlap_simt_enqueue_(sz_metal_device_t *device, char const *name,
                                                        sz_overlap_simt_arguments_t const *arguments,
                                                        sz_size_t staged_bytes, sz_metal_size_t groups,
                                                        sz_metal_size_t threads) {
    void *const pipeline = sz_metal_pipeline_(device, sz_overlap_simt_source_, name);
    void *const encoder = pipeline ? sz_metal_encoder_(device) : STRINGZILLA_NULL;
    if (!encoder) return device->status;
    ((void (*)(void *, SEL, void *, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setBuffer:offset:atIndex:"), device->arena, 0, 0);
    ((void (*)(void *, SEL, void const *, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setBytes:length:atIndex:"), arguments, sizeof(*arguments), 1);
    if (staged_bytes) {
        ((void (*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
            encoder, sel_registerName("setThreadgroupMemoryLength:atIndex:"), staged_bytes, 0);
        ((void (*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
            encoder, sel_registerName("setThreadgroupMemoryLength:atIndex:"), sz_overlap_simt_filter_bytes_k, 1);
    }
    sz_metal_enqueue_(device, pipeline, groups, threads);
    return sz_success_k;
}

/**
 *  @brief Prepares every query of @p queries into one block of @p stream 's arena and resolves the
 *      launch geometry, compiling the kernel on first use.
 *
 *  @param[in] queries Read on the @b host, so its accessors must be host-callable.
 *  @param[in] candidates_budget Candidates one round may carry, zero asking for
 *      @ref sz_overlap_simt_candidates_budget_default_k. The round block is sized for it here, so
 *      no round allocates or joins, and a round carrying more is refused.
 *  @param[in] ordinal The device @p stream was opened on.
 *  @param[in] allocator From @ref sz_memory_allocator_init_metal over the same device, or
 *      @c STRINGZILLA_NULL to derive one from @p stream.
 *  @param[in] stream The @ref sz_metal_device_t every round encodes into.
 *  @return @c sz_success_k; @c sz_unexpected_dimensions_k for zero widths, more than
 *      @ref sz_overlap_simt_widths_max_k, or a width past @ref sz_overlap_simt_widest_window_k;
 *      @c sz_device_memory_mismatch_k for memory outside the arena from @p allocator or a foreign
 *      @p stream; @c sz_bad_alloc_k when the block cannot be taken; @c sz_missing_gpu_k without a
 *      device; or @c sz_device_code_mismatch_k when the kernel fails to build.
 *  @sa sz_overlap_engine_init
 */
STRINGZILLA_API sz_status_t sz_overlap_engine_init_metal(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_size_t ordinal,
                                                         sz_memory_allocator_t *allocator, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    if (!device || !device->device) return sz_missing_gpu_k;
    if (device->ordinal != ordinal) return sz_device_memory_mismatch_k;
    if (!window_widths_count || window_widths_count > sz_overlap_simt_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] > sz_overlap_simt_widest_window_k) return sz_unexpected_dimensions_k;

    sz_memory_allocator_t arena;
    if (allocator) arena = *allocator;
    else sz_memory_allocator_init_metal(&arena, device);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count,
                                                       sizeof(sz_overlap_simt_geometry_t), &arena, engine);
    if (opened != sz_success_k) return opened;
    if (!sz_memory_reaches_metal(device, engine->memory)) {
        sz_overlap_engine_close_(engine);
        return sz_device_memory_mismatch_k;
    }

    // The chain is host working space no kernel reads, so it comes from the host, not the arena.
    sz_size_t longest_query = 0, widest_nodes = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
        sz_size_t const entries = engine->nodes_offsets[index + 1] - engine->nodes_offsets[index];
        if (entries > widest_nodes) widest_nodes = entries;
    }
    sz_memory_allocator_t host;
    sz_memory_allocator_init_default(&host);
    sz_size_t const chain_bytes = (longest_query + 1) * sizeof(sz_f64_t);
    sz_f64_t *const chain = (sz_f64_t *)host.allocate(chain_bytes, host.handle);
    if (!chain) {
        sz_overlap_engine_close_(engine);
        return sz_bad_alloc_k;
    }
    sz_overlap_engine_fill_serial_(engine, queries, chain);
    host.free(chain, chain_bytes, host.handle);

    // The round block holds the segment offsets, then one counter per query, candidate and width,
    // which start at zero here and which the shares pass clears again after reading them.
    if (!candidates_budget) candidates_budget = sz_overlap_simt_candidates_budget_default_k;
    sz_size_t const round_bytes = sz_overlap_simt_offsets_bytes_(candidates_budget) +
                                  engine->count * candidates_budget * engine->widths_count * sizeof(sz_u32_t);
    if (sz_overlap_engine_grow_(engine, round_bytes) != sz_success_k) {
        sz_overlap_engine_close_(engine);
        return sz_bad_alloc_k;
    }
    if (!sz_memory_reaches_metal(device, engine->scratch)) {
        sz_overlap_engine_close_(engine);
        return sz_device_memory_mismatch_k;
    }
    sz_fill_serial_((char *)engine->scratch, engine->scratch_bytes, 0);

    // Building every pipeline here keeps the compiles out of the rounds and names their ceilings.
    void *const segments = sz_metal_pipeline_(device, sz_overlap_simt_source_, "sz_overlap_segments_metal_kernel_");
    void *const pipeline = sz_metal_pipeline_(device, sz_overlap_simt_source_, "sz_overlap_scores_metal_kernel_");
    void *const shares = sz_metal_pipeline_(device, sz_overlap_simt_source_, "sz_overlap_shares_metal_kernel_");
    if (!segments || !pipeline || !shares ||
        sz_metal_count_(segments, "maxTotalThreadsPerThreadgroup") < sz_overlap_simt_scan_threads_k) {
        sz_overlap_engine_close_(engine);
        return sz_device_code_mismatch_k;
    }
    sz_size_t const ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup");
    sz_size_t const threadgroup_bytes = sz_metal_count_(device->device, "maxThreadgroupMemoryLength");
    sz_overlap_simt_geometry_t *const geometry = (sz_overlap_simt_geometry_t *)sz_overlap_engine_head_(engine);
    geometry->device = device;
    geometry->candidates_per_group = ceiling < sz_overlap_simt_candidates_per_group_k
                                         ? ceiling
                                         : sz_overlap_simt_candidates_per_group_k;
    geometry->staged_nodes_count =
        widest_nodes * sizeof(sz_u32_t) * sz_overlap_simt_staged_tree_share_k <= threadgroup_bytes ? widest_nodes : 0;
    geometry->candidates_budget = candidates_budget;
    engine->capability = sz_cap_metal_k, engine->ordinal = ordinal;
    return sz_success_k;
}

/**
 *  @brief The Metal kernel of @ref sz_overlap_scores.
 *  @pre @p candidates is bound by @ref sz_sequence_from_string_views, its views and texts all in
 *      the arena of the device the engine was built on.
 *  @return @c sz_success_k; @c sz_unexpected_dimensions_k when either stride is under its axis or
 *      the round carries more candidates than the engine's budget; @c sz_device_memory_mismatch_k
 *      when the scores, the views or @p stream lie outside the engine's device; or
 *      @c sz_device_code_mismatch_k when the sequence is not a view array or encoding fails.
 *  @note Encodes into @p stream, the engine's own device, and returns; the caller synchronizes it
 *      before reading @p scores.
 */
STRINGZILLA_API sz_status_t sz_overlap_scores_metal(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                    sz_f32_t *scores, sz_size_t scores_query_stride,
                                                    sz_size_t scores_candidate_stride, void *stream) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;

    sz_overlap_simt_geometry_t const *const geometry = (sz_overlap_simt_geometry_t const *)sz_overlap_engine_head_(
        engine);
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    if (device != geometry->device) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_metal(device, scores)) return sz_device_memory_mismatch_k;
    sz_status_t status = sz_metal_views_(device, candidates);
    if (status != sz_success_k) return status;
    sz_string_view_t const *const views = (sz_string_view_t const *)candidates->handle;

    // The host sizes the scoring grid from the views, which no kernel writes; the offsets are
    // scanned on the device, so a round still in flight never sees them change under it.
    sz_size_t segments_count = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index)
        segments_count += sz_size_divide_round_up(views[index].length, sz_overlap_simt_segment_bytes_k);
    if (segments_count >> 32) return sz_unexpected_dimensions_k;

    // The counters sit behind offsets sized for the budget rather than for this round, so a smaller
    // round never reads an earlier round's offsets as counts.
    if (candidates->count > geometry->candidates_budget) return sz_unexpected_dimensions_k;
    sz_size_t const offsets_bytes = sz_overlap_simt_offsets_bytes_(geometry->candidates_budget);

    sz_overlap_simt_arguments_t arguments;
    arguments.host_base = (sz_u64_t)device->arena_host;
    arguments.nodes = (sz_u64_t)engine->nodes, arguments.nodes_offsets = (sz_u64_t)engine->nodes_offsets;
    arguments.keys_counts = (sz_u64_t)engine->keys_counts, arguments.widths = (sz_u64_t)engine->widths;
    arguments.powers = (sz_u64_t)engine->powers, arguments.lengths = (sz_u64_t)engine->lengths;
    arguments.views = (sz_u64_t)candidates->handle, arguments.scores = (sz_u64_t)scores;
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

    sz_metal_size_t const one = {1, 1, 1}, scan_threads = {sz_overlap_simt_scan_threads_k, 1, 1};
    status = sz_overlap_simt_enqueue_(device, "sz_overlap_segments_metal_kernel_", &arguments, 0, one, scan_threads);
    if (status != sz_success_k) return status;

    // Threadgroup memory is sized in 16-byte steps, and is bound even when nothing is staged.
    if (segments_count) {
        sz_size_t const staged_nodes_bytes = geometry->staged_nodes_count * sizeof(sz_u32_t);
        sz_size_t const staged_bytes = sz_size_divide_round_up(staged_nodes_bytes, 16) * 16;
        sz_size_t const per_group = geometry->candidates_per_group;
        sz_metal_size_t const grid = {sz_size_divide_round_up(segments_count, per_group), engine->count, 1};
        sz_metal_size_t const threads = {per_group, 1, 1};
        status = sz_overlap_simt_enqueue_(device, "sz_overlap_scores_metal_kernel_", &arguments,
                                          staged_bytes ? staged_bytes : 16, grid, threads);
        if (status != sz_success_k) return status;
    }

    sz_metal_size_t const shares_grid = {(candidates->count + 255) / 256, engine->count, 1},
                          shares_threads = {256, 1, 1};
    status = sz_overlap_simt_enqueue_(device, "sz_overlap_shares_metal_kernel_", &arguments, 0, shares_grid,
                                      shares_threads);
    if (status == sz_success_k) sz_metal_commit_(device);
    return status;
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_OVERLAP_SIMT_H_
