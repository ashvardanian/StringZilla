/**
 *  @file include/stringzilla/levenshtein/metal.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Metal backend for Levenshtein distances on Apple7 and newer: the batch prepared on the
 *      host into one shared block, and one call per round, a dispatch per rung the batch reaches.
 *
 *  @sa include/stringzilla/levenshtein/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/levenshtein/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @c id<MTLCommandQueue> as the stream. Every block
 *  the kernels read, the engine's, the candidates' tape and the distances, must come from
 *  @ref sz_memory_allocator_init_unified_metal on that stream's device, and a round's candidates
 *  must be a tape from @ref sz_sequence_copy_metal, since the kernels read the tape itself. Memory
 *  outside the device's registry draws @c sz_device_memory_mismatch_k, and accessors other than a
 *  tape's draw @c sz_device_code_mismatch_k.
 *
 *  The planes are built on the host, bytes and runes alike, since the block is memory both sides
 *  address. The CUDA tier's narrow byte lanes are a throughput rung only, so their queries take the
 *  one-word threaded rung here, and the tiled wavefront has no Metal arm: Apple GPUs promise no
 *  forward progress between threadgroups, which its cross-block wait needs.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_METAL_H_
#define STRINGZILLA_LEVENSHTEIN_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/levenshtein/serial.h"

#if STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the Levenshtein kernels, compiled once per device behind the shared prelude. */
static char const sz_levenshtein_source_metal_[] = {
#embed "metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Threads a threadgroup runs at most, so a group schedules as finely as the CUDA tier's blocks. */
enum { sz_levenshtein_threads_per_group_max_metal_k = 256 };

/** Entry points by word count and words per lane, over bytes and runes, as `metal.metal` names. */
static char const *const sz_levenshtein_threaded_names_metal_[2][sz_levenshtein_gpu_warp_words_min_k - 1] = {
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
static char const *const sz_levenshtein_warped_names_metal_[2][sz_levenshtein_gpu_warp_words_per_lane_max_k] = {
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
STRINGZILLA_INLINE char const *sz_levenshtein_entry_metal_(sz_levenshtein_symbol_t symbol, sz_size_t words) {
    sz_size_t const alphabet = symbol == sz_levenshtein_runes_k ? 1 : 0;
    if (words < sz_levenshtein_gpu_warp_words_min_k) return sz_levenshtein_threaded_names_metal_[alphabet][words - 1];
    return sz_levenshtein_warped_names_metal_[alphabet]
                                             [sz_size_divide_round_up(words, sz_levenshtein_gpu_warp_lanes_k) - 1];
}

/** What a round needs that the batch already fixed: the queries bucketed by word count with the
 *  threads each bucket's entry point takes, its tables addressed by byte offsets inside the block. */
typedef struct sz_levenshtein_head_metal_t {

    /** Word counts the batch spans, one bucket each, the widest query's own last. */
    sz_size_t buckets;

    /** Where the @b [buckets+1] first position of each word count inside @c order sits. */
    sz_size_t bucket_offsets;

    /** Where the @b [buckets] threads a threadgroup of that bucket's entry point runs sit. */
    sz_size_t per_group;

    /** Where the @b [count] query indices sit, the buckets' runs back to back. */
    sz_size_t order;
} sz_levenshtein_head_metal_t;

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning @p buckets
 *  word counts. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_head_bytes_metal_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_head_metal_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host;
    sz_u64_t masks, masks_offsets, symbol_to_class, lengths, order;
    sz_u64_t candidates_count, distances_stride;
} sz_levenshtein_arguments_metal_t;

/**
 *  @brief Lays the head's tables out in the block, buckets the batch by word count, and builds the
 *      entry point every bucket launches.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when a kernel fails to build.
 */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_bind_head_metal_(sz_levenshtein_engine_t *engine,
                                                               sz_metal_context_t *context, sz_size_t buckets_bound) {
    char *const block = (char *)engine->memory;
    sz_levenshtein_head_metal_t *const head = (sz_levenshtein_head_metal_t *)block;
    head->buckets = sz_levenshtein_engine_words_max_(engine);
    head->bucket_offsets = sizeof(sz_levenshtein_head_metal_t);
    head->per_group = head->bucket_offsets + (buckets_bound + 1) * sizeof(sz_size_t);
    head->order = head->per_group + buckets_bound * sizeof(sz_size_t);
    sz_size_t *const bucket_offsets = (sz_size_t *)(block + head->bucket_offsets);
    sz_size_t *const per_group = (sz_size_t *)(block + head->per_group);
    sz_u32_t *const order = (sz_u32_t *)(block + head->order);
    sz_size_t cursors[sz_levenshtein_gpu_words_max_k];

    // A counting sort by word count, so one dispatch only carries queries that share
    // an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        ++bucket_offsets[sz_levenshtein_query_words(engine->lengths[index])];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        bucket_offsets[bucket] += bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        order[cursors[sz_levenshtein_query_words(engine->lengths[index]) - 1]++] = (sz_u32_t)index;

    // Building every pipeline here keeps the compiles out of the rounds and names their ceilings.
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        per_group[bucket] = 0;
        if (bucket_offsets[bucket] == bucket_offsets[bucket + 1]) continue;
        void *const pipeline = sz_metal_pipeline_(context, sz_levenshtein_source_metal_,
                                                  sz_levenshtein_entry_metal_(engine->symbol, bucket + 1));
        if (!pipeline) return sz_device_code_mismatch_k;
        sz_size_t const ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup");
        sz_size_t const width = sz_metal_count_(pipeline, "threadExecutionWidth");
        if (width != sz_levenshtein_gpu_warp_lanes_k || ceiling < width) return sz_device_code_mismatch_k;
        sz_size_t const threads = sz_min_of_two(ceiling, (sz_size_t)sz_levenshtein_threads_per_group_max_metal_k);
        per_group[bucket] = threads / width * width;
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_metal(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *queries,
                                                             sz_levenshtein_symbol_t symbol,
                                                             sz_memory_allocator_t *allocator, void *stream) {
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return status;
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
    if (symbol == sz_levenshtein_bytes_k && longest > sz_levenshtein_gpu_words_max_k * 64)
        return sz_unexpected_dimensions_k;

    sz_memory_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_memory_allocator_init_unified_metal(&unified);
    sz_size_t const buckets_bound = sz_levenshtein_query_words(longest);
    status = sz_levenshtein_engine_build_(
        queries, symbol, sz_levenshtein_head_bytes_metal_(queries->count, buckets_bound), &unified, stream, engine);
    if (status != sz_success_k) return status;
    sz_metal_bound_t memory;
    if (!sz_metal_resolve_(call.context, engine->memory, engine->memory_bytes, &memory))
        status = sz_device_memory_mismatch_k;
    else if (sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_gpu_words_max_k)
        status = sz_unexpected_dimensions_k;
    else status = sz_levenshtein_bind_head_metal_(engine, call.context, buckets_bound);
    if (status != sz_success_k) {
        sz_levenshtein_engine_free_(engine, stream);
        return status;
    }
    sz_levenshtein_engine_fill_(engine, queries);
    engine->capability = sz_cap_metal_k;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_metal(sz_levenshtein_engine_t *engine,
                                                           sz_sequence_t const *candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, void *stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (!candidates->count) return sz_success_k;

    // The engine's block, the candidates' tape and the distances, as the kernels number them.
    sz_metal_bound_t buffers[3];
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return status;
    sz_size_t const distances_count = (engine->count - 1) * distances_stride + candidates->count;
    if (!sz_metal_resolve_(call.context, engine->memory, engine->memory_bytes, &buffers[0]) ||
        !sz_metal_resolve_(call.context, distances, distances_count * sizeof(sz_size_t), &buffers[2]))
        return sz_device_memory_mismatch_k;
    status = sz_metal_tape_(call.context, candidates, &buffers[1]);
    if (status != sz_success_k) return status;

    char const *const block = (char const *)engine->memory;
    sz_levenshtein_head_metal_t const *const head = (sz_levenshtein_head_metal_t const *)block;
    sz_size_t const *const bucket_offsets = (sz_size_t const *)(block + head->bucket_offsets);
    sz_size_t const *const per_group = (sz_size_t const *)(block + head->per_group);
    sz_u32_t const *const order = (sz_u32_t const *)(block + head->order);
    sz_levenshtein_arguments_metal_t arguments;
    arguments.engine_host = (sz_u64_t)engine->memory;
    arguments.masks = (sz_u64_t)engine->masks, arguments.masks_offsets = (sz_u64_t)engine->masks_offsets;
    arguments.symbol_to_class = (sz_u64_t)engine->symbol_to_class, arguments.lengths = (sz_u64_t)engine->lengths;
    arguments.candidates_count = candidates->count, arguments.distances_stride = distances_stride;

    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        sz_size_t const first = bucket_offsets[bucket], end = bucket_offsets[bucket + 1];
        if (first == end) continue;
        char const *const kernel = sz_levenshtein_entry_metal_(engine->symbol, bucket + 1);
        sz_size_t const threads = per_group[bucket];
        sz_size_t const candidates_per_group = bucket + 1 < sz_levenshtein_gpu_warp_words_min_k
                                                   ? threads
                                                   : threads / sz_levenshtein_gpu_warp_lanes_k;
        for (sz_size_t row = first; row < end; row += sz_levenshtein_gpu_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(end - row, (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k);
            sz_metal_size_t const groups = {sz_size_divide_round_up(candidates->count, candidates_per_group), rows, 1};
            sz_metal_size_t const group = {threads, 1, 1};
            arguments.order = (sz_u64_t)(order + row);
            status = sz_metal_encode_(&call, sz_levenshtein_source_metal_, kernel, buffers, 3, &arguments,
                                      sizeof(arguments), groups, group);
            if (status != sz_success_k) return sz_metal_commit_(&call, status);
        }
    }
    return sz_metal_commit_(&call, sz_success_k);
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_LEVENSHTEIN_METAL_H_
