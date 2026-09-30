/**
 *  @file include/stringzilla/levenshtein/simt.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Metal backend for Levenshtein distances on Apple7 and newer: the batch prepared on the
 *      host into the device's arena, and one call per round, a dispatch per rung the batch reaches.
 *
 *  @sa include/stringzilla/levenshtein/simt.metal, the kernels this embeds
 *  @sa include/stringzilla/levenshtein/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @ref sz_metal_device_t where the stream goes.
 *  Every block the kernels read - the planes, the candidates' views, their texts, the distances -
 *  must come from that device's arena, which @ref sz_memory_allocator_init_metal hands out, and a
 *  round's candidates must be bound by @ref sz_sequence_from_string_views, since the kernels read
 *  the view array itself.
 *
 *  The planes are built on the host, bytes and runes alike, since the arena is memory both sides
 *  address. The CUDA tier's narrow byte lanes are a throughput rung only, so their queries take the
 *  one-word threaded rung here, and the tiled wavefront has no Metal arm: Apple GPUs promise no
 *  forward progress between threadgroups, which its cross-block wait needs.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_SIMT_H_
#define STRINGZILLA_LEVENSHTEIN_SIMT_H_

#include "stringzilla/metal.h"
#include "stringzilla/levenshtein/serial.h"

#if STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the Levenshtein kernels, compiled once per device. */
static char const sz_levenshtein_simt_source_[] = {
#embed "simt.metal"
    , 0};
#pragma clang diagnostic pop

/** Lanes a warped candidate is spread across, which is a simdgroup, and the query words one lane
 *  keeps verticals for, so a simdgroup at its widest reaches the family's word limit. */
enum {
    sz_levenshtein_simt_warp_lanes_k = 32,
    sz_levenshtein_simt_warp_words_per_lane_max_k = sz_levenshtein_simt_words_max_k / sz_levenshtein_simt_warp_lanes_k,
};

/** Query words from which the warped rung is launched rather than the threaded one, as on CUDA. */
enum { sz_levenshtein_simt_warp_words_min_k = 16 };

/** Threads a threadgroup runs at most, so a group schedules as finely as the CUDA tier's blocks. */
enum { sz_levenshtein_simt_threads_per_group_max_k = 256 };

/** Rows one dispatch's query axis spans, past which a rung is cut into several dispatches. */
enum { sz_levenshtein_simt_grid_rows_max_k = 65535 };

/** Entry points by word count and words per lane, over bytes and runes, as `simt.metal` names. */
static char const *const sz_levenshtein_simt_threaded_names_[2][sz_levenshtein_simt_warp_words_min_k - 1] = {
    {"sz_levenshtein_distances_w1_metal_kernel_", "sz_levenshtein_distances_w2_metal_kernel_",
     "sz_levenshtein_distances_w3_metal_kernel_", "sz_levenshtein_distances_w4_metal_kernel_",
     "sz_levenshtein_distances_w5_metal_kernel_", "sz_levenshtein_distances_w6_metal_kernel_",
     "sz_levenshtein_distances_w7_metal_kernel_", "sz_levenshtein_distances_w8_metal_kernel_",
     "sz_levenshtein_distances_w9_metal_kernel_", "sz_levenshtein_distances_w10_metal_kernel_",
     "sz_levenshtein_distances_w11_metal_kernel_", "sz_levenshtein_distances_w12_metal_kernel_",
     "sz_levenshtein_distances_w13_metal_kernel_", "sz_levenshtein_distances_w14_metal_kernel_",
     "sz_levenshtein_distances_w15_metal_kernel_"},
    {"sz_levenshtein_distances_utf8_w1_metal_kernel_", "sz_levenshtein_distances_utf8_w2_metal_kernel_",
     "sz_levenshtein_distances_utf8_w3_metal_kernel_", "sz_levenshtein_distances_utf8_w4_metal_kernel_",
     "sz_levenshtein_distances_utf8_w5_metal_kernel_", "sz_levenshtein_distances_utf8_w6_metal_kernel_",
     "sz_levenshtein_distances_utf8_w7_metal_kernel_", "sz_levenshtein_distances_utf8_w8_metal_kernel_",
     "sz_levenshtein_distances_utf8_w9_metal_kernel_", "sz_levenshtein_distances_utf8_w10_metal_kernel_",
     "sz_levenshtein_distances_utf8_w11_metal_kernel_", "sz_levenshtein_distances_utf8_w12_metal_kernel_",
     "sz_levenshtein_distances_utf8_w13_metal_kernel_", "sz_levenshtein_distances_utf8_w14_metal_kernel_",
     "sz_levenshtein_distances_utf8_w15_metal_kernel_"},
};
static char const *const sz_levenshtein_simt_warped_names_[2][sz_levenshtein_simt_warp_words_per_lane_max_k] = {
    {"sz_levenshtein_distances_k1_metal_kernel_", "sz_levenshtein_distances_k2_metal_kernel_",
     "sz_levenshtein_distances_k3_metal_kernel_", "sz_levenshtein_distances_k4_metal_kernel_",
     "sz_levenshtein_distances_k5_metal_kernel_", "sz_levenshtein_distances_k6_metal_kernel_",
     "sz_levenshtein_distances_k7_metal_kernel_", "sz_levenshtein_distances_k8_metal_kernel_"},
    {"sz_levenshtein_distances_utf8_k1_metal_kernel_", "sz_levenshtein_distances_utf8_k2_metal_kernel_",
     "sz_levenshtein_distances_utf8_k3_metal_kernel_", "sz_levenshtein_distances_utf8_k4_metal_kernel_",
     "sz_levenshtein_distances_utf8_k5_metal_kernel_", "sz_levenshtein_distances_utf8_k6_metal_kernel_",
     "sz_levenshtein_distances_utf8_k7_metal_kernel_", "sz_levenshtein_distances_utf8_k8_metal_kernel_"},
};

/** The entry point a query of @p words words launches from: one candidate per thread, or one per
 *  simdgroup past the crossing. */
STRINGZILLA_INLINE char const *sz_levenshtein_simt_kernel_(sz_levenshtein_symbol_t symbol, sz_size_t words) {
    sz_size_t const alphabet = symbol == sz_levenshtein_runes_k ? 1 : 0;
    if (words < sz_levenshtein_simt_warp_words_min_k) return sz_levenshtein_simt_threaded_names_[alphabet][words - 1];
    return sz_levenshtein_simt_warped_names_[alphabet]
                                            [sz_size_divide_round_up(words, sz_levenshtein_simt_warp_lanes_k) - 1];
}

/** What a device round needs that the batch already fixed: the device, and the queries bucketed by
 *  word count with the threads each bucket's entry point takes. */
typedef struct sz_levenshtein_simt_head_t {

    /** The device every round encodes into. */
    sz_metal_device_t *device;

    /** Word counts the batch spans, one bucket each, the widest query's own last. */
    sz_size_t buckets;

    /** The @b [buckets+1] first position of each word count inside @c order. */
    sz_size_t *bucket_offsets;

    /** The @b [buckets] threads a threadgroup of that bucket's entry point runs, or zero. */
    sz_size_t *per_group;

    /** The @b [count] query indices, the buckets' runs back to back. */
    sz_u32_t *order;
} sz_levenshtein_simt_head_t;

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning @p buckets
 *  word counts. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_simt_head_bytes_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_simt_head_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t host_base;
    sz_u64_t masks, masks_offsets, symbol_to_class, lengths, order;
    sz_u64_t views, distances;
    sz_u64_t candidates_count, distances_stride;
} sz_levenshtein_simt_arguments_t;

/**
 *  @brief Points the head's tables into the block, buckets the batch by word count, and builds the
 *      entry point every bucket launches.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when a kernel fails to build.
 */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_simt_bind_head_(sz_levenshtein_engine_t *engine,
                                                              sz_metal_device_t *device, sz_size_t buckets_bound) {
    sz_levenshtein_simt_head_t *const head = (sz_levenshtein_simt_head_t *)engine->memory;
    sz_size_t *const tables = (sz_size_t *)((char *)engine->memory + sizeof(sz_levenshtein_simt_head_t));
    sz_size_t cursors[sz_levenshtein_simt_words_max_k];
    head->device = device;
    head->buckets = sz_levenshtein_engine_words_max_(engine);
    head->bucket_offsets = tables;
    head->per_group = tables + buckets_bound + 1;
    head->order = (sz_u32_t *)(tables + 2 * buckets_bound + 1);

    // A counting sort by word count, so one dispatch only carries queries that share
    // an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) head->bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        ++head->bucket_offsets[sz_levenshtein_query_words(engine->lengths[index])];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        head->bucket_offsets[bucket] += head->bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = head->bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        head->order[cursors[sz_levenshtein_query_words(engine->lengths[index]) - 1]++] = (sz_u32_t)index;

    // Building every pipeline here keeps the compiles out of the rounds and names their ceilings.
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        head->per_group[bucket] = 0;
        if (head->bucket_offsets[bucket] == head->bucket_offsets[bucket + 1]) continue;
        void *const pipeline = sz_metal_pipeline_(device, sz_levenshtein_simt_source_,
                                                  sz_levenshtein_simt_kernel_(engine->symbol, bucket + 1));
        if (!pipeline) return sz_device_code_mismatch_k;
        sz_size_t const ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup");
        sz_size_t const width = sz_metal_count_(pipeline, "threadExecutionWidth");
        if (width != sz_levenshtein_simt_warp_lanes_k || ceiling < width) return sz_device_code_mismatch_k;
        sz_size_t const threads = sz_min_of_two(ceiling, (sz_size_t)sz_levenshtein_simt_threads_per_group_max_k);
        head->per_group[bucket] = threads / width * width;
    }
    return sz_success_k;
}

/**
 *  @brief Prepares every query of @p queries into one block of @p stream 's arena and builds the
 *      entry points its word counts reach.
 *
 *  An empty batch, an empty query, or one past @ref sz_levenshtein_simt_words_max_k words draws
 *  @c sz_unexpected_dimensions_k; memory outside the arena from @p allocator, or a device other
 *  than @p ordinal, draws @c sz_device_memory_mismatch_k; a block that cannot be taken draws
 *  @c sz_bad_alloc_k; no device draws @c sz_missing_gpu_k; and a kernel that fails to build draws
 *  @c sz_device_code_mismatch_k.
 *
 *  @param[in] queries Read on the @b host, so its accessors must be host-callable.
 *  @param[in] ordinal The device @p stream was opened on.
 *  @param[in] allocator From @ref sz_memory_allocator_init_metal over the same device, or
 *      @c STRINGZILLA_NULL to derive one from @p stream.
 *  @param[in] stream The @ref sz_metal_device_t every round encodes into.
 *  @return @c sz_success_k, or one of the refusals above.
 *  @sa sz_levenshtein_engine_init
 */
STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_metal(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                                             sz_memory_allocator_t *allocator, void *stream) {
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    if (!device || !device->device) return sz_missing_gpu_k;
    if (device->ordinal != ordinal) return sz_device_memory_mismatch_k;
    if (queries->count == 0) return sz_unexpected_dimensions_k;

    // Every query seeds its score from its own last word, so an empty one has no word to read it
    // off, and the ceiling is checked twice: on the bytes here, which bound the runes, and on the
    // symbols once measured.
    sz_size_t longest = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (bytes == 0) return sz_unexpected_dimensions_k;
        longest = sz_max_of_two(longest, bytes);
    }
    if (symbol == sz_levenshtein_bytes_k && longest > sz_levenshtein_simt_words_max_k * 64)
        return sz_unexpected_dimensions_k;

    sz_memory_allocator_t arena;
    if (allocator) arena = *allocator;
    else sz_memory_allocator_init_metal(&arena, device);
    sz_size_t const buckets_bound = sz_levenshtein_query_words(longest);
    sz_status_t status = sz_levenshtein_engine_build_(
        queries, symbol, sz_levenshtein_simt_head_bytes_(queries->count, buckets_bound), &arena, engine);
    if (status != sz_success_k) return status;
    if (!sz_memory_reaches_metal(device, engine->memory)) status = sz_device_memory_mismatch_k;
    else if (sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_simt_words_max_k)
        status = sz_unexpected_dimensions_k;
    else status = sz_levenshtein_simt_bind_head_(engine, device, buckets_bound);
    if (status != sz_success_k) {
        sz_levenshtein_engine_free_(engine);
        return status;
    }
    sz_levenshtein_engine_fill_(engine, queries);
    engine->capability = sz_cap_metal_k, engine->ordinal = ordinal;
    return sz_success_k;
}

/**
 *  @brief The Metal kernel of @ref sz_levenshtein_distances.
 *  @pre @p candidates is bound by @ref sz_sequence_from_string_views, its views and texts all in
 *      the arena of the device the engine was built on.
 *  @return @c sz_success_k; @c sz_unexpected_dimensions_k when the stride is under the candidates;
 *      @c sz_device_memory_mismatch_k when the distances, the views or @p stream lie outside the
 *      engine's device; or @c sz_device_code_mismatch_k for a sequence other than a view array.
 *  @note Encodes into @p stream, the engine's own device, and returns; the caller synchronizes it
 *      before reading @p distances.
 */
STRINGZILLA_API sz_status_t sz_levenshtein_distances_metal(sz_levenshtein_engine_t *engine,
                                                           sz_sequence_t const *candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, void *stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (!candidates->count) return sz_success_k;

    sz_levenshtein_simt_head_t const *const head = (sz_levenshtein_simt_head_t const *)engine->memory;
    sz_metal_device_t *const device = (sz_metal_device_t *)stream;
    if (device != head->device) return sz_device_memory_mismatch_k;
    sz_size_t const last = candidates->count - 1;
    if (!sz_memory_reaches_metal(device, distances) ||
        !sz_memory_reaches_metal(device, distances + (engine->count - 1) * distances_stride + last))
        return sz_device_memory_mismatch_k;
    sz_status_t status = sz_metal_views_(device, candidates);
    if (status != sz_success_k) return status;

    sz_levenshtein_simt_arguments_t arguments;
    arguments.host_base = (sz_u64_t)device->arena_host;
    arguments.masks = (sz_u64_t)engine->masks, arguments.masks_offsets = (sz_u64_t)engine->masks_offsets;
    arguments.symbol_to_class = (sz_u64_t)engine->symbol_to_class, arguments.lengths = (sz_u64_t)engine->lengths;
    arguments.views = (sz_u64_t)candidates->handle, arguments.distances = (sz_u64_t)distances;
    arguments.candidates_count = candidates->count, arguments.distances_stride = distances_stride;

    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        sz_size_t const first = head->bucket_offsets[bucket], end = head->bucket_offsets[bucket + 1];
        if (first == end) continue;
        char const *const kernel = sz_levenshtein_simt_kernel_(engine->symbol, bucket + 1);
        sz_size_t const threads = head->per_group[bucket];
        sz_size_t const per_group = bucket + 1 < sz_levenshtein_simt_warp_words_min_k
                                        ? threads
                                        : threads / sz_levenshtein_simt_warp_lanes_k;
        for (sz_size_t row = first; row < end; row += sz_levenshtein_simt_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(end - row, (sz_size_t)sz_levenshtein_simt_grid_rows_max_k);
            sz_metal_size_t const groups = {sz_size_divide_round_up(candidates->count, per_group), rows, 1};
            sz_metal_size_t const group = {threads, 1, 1};
            arguments.order = (sz_u64_t)(head->order + row);
            status = sz_metal_encode_(device, sz_levenshtein_simt_source_, kernel, &arguments, sizeof(arguments),
                                      groups, group);
            if (status != sz_success_k) return status;
        }
    }
    sz_metal_commit_(device);
    return sz_success_k;
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_LEVENSHTEIN_SIMT_H_
