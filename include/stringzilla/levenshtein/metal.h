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
 *  the kernels read, the engine's, the candidates' tape and the distances, must come from @ref
 *  sz_allocator_init_unified_metal on that stream's device, and a round's candidates must be a tape
 *  from @ref sz_sequence_realloc_metal, since the kernels read the tape itself. Memory outside the
 *  device's registry draws @c sz_device_memory_mismatch_k, and accessors other than a tape's draw
 *  @c sz_device_code_mismatch_k.
 *
 *  The planes are built on the host, bytes and runes alike, since the block is memory both sides
 *  address. The CUDA tier's narrow byte lanes are a throughput rung only, so their queries take the
 *  one-word threaded rung here. The tiled wavefront uses ordered diagonal dispatches, since Apple
 *  GPUs promise no forward progress between threadgroups.
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

/** Levenshtein MSL source, compiled once per device. */
static char const sz_levenshtein_source_metal_[] = {
#embed "stringzilla/levenshtein/metal.metal"
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

/** Query buckets and launch geometry at the start of the engine's allocation. */
typedef struct sz_levenshtein_head_metal_t {

    /** Number of query word-count buckets. */
    sz_size_t buckets;

    /** Byte offset of @b [buckets+1] bucket boundaries in @c order. */
    sz_size_t bucket_offsets;

    /** Byte offset of @b [buckets] thread counts per threadgroup. */
    sz_size_t per_group;

    /** Byte offset of @b [count] query indices grouped by word count. */
    sz_size_t order;

    /** Byte offset of @b [count+1] query boundaries relative to @c query_text. */
    sz_size_t query_offsets;

    /** Byte offset of packed long byte queries used by the tiled kernel. */
    sz_size_t query_text;

    /** Empty and long queries, grouped by length in the tail of @c order. */
    sz_levenshtein_length_bucket_t long_buckets[sz_levenshtein_length_buckets_k];
} sz_levenshtein_head_metal_t;

/** Bytes required for the header and bucket tables of @p count queries in @p buckets buckets. */
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
 *  @brief Groups queries by word count and prepares their kernel pipelines.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k if a pipeline is unusable.
 */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_bind_head_metal_(sz_levenshtein_engine_t *engine,
                                                               sz_metal_context_t *context, sz_size_t buckets_bound) {
    char *const block = (char *)engine->memory;
    sz_levenshtein_head_metal_t *const head = (sz_levenshtein_head_metal_t *)block;
    head->buckets = sz_min_of_two(sz_levenshtein_engine_words_max_(engine), (sz_size_t)sz_levenshtein_gpu_words_max_k);
    head->bucket_offsets = sizeof(sz_levenshtein_head_metal_t);
    head->per_group = head->bucket_offsets + (buckets_bound + 1) * sizeof(sz_size_t);
    head->order = head->per_group + buckets_bound * sizeof(sz_size_t);
    sz_size_t *const bucket_offsets = (sz_size_t *)(block + head->bucket_offsets);
    sz_size_t *const per_group = (sz_size_t *)(block + head->per_group);
    sz_u32_t *const order = (sz_u32_t *)(block + head->order);
    sz_size_t cursors[sz_levenshtein_gpu_words_max_k];

    // Group queries by word count so each dispatch uses one kernel.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const words = sz_levenshtein_query_words(engine->lengths[index]);
        if (words && words <= head->buckets) ++bucket_offsets[words];
    }
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        bucket_offsets[bucket] += bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const words = sz_levenshtein_query_words(engine->lengths[index]);
        if (words && words <= head->buckets) order[cursors[words - 1]++] = (sz_u32_t)index;
    }

    for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket) {
        head->long_buckets[bucket].offset = 0;
        head->long_buckets[bucket].count = 0;
        head->long_buckets[bucket].length_max = 0;
    }
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const length = engine->lengths[index];
        sz_size_t const words = sz_levenshtein_query_words(length);
        if (words && words <= head->buckets) continue;
        sz_levenshtein_length_bucket_t *const bucket = head->long_buckets + sz_levenshtein_length_bucket_(length);
        ++bucket->count;
        bucket->length_max = sz_max_of_two(bucket->length_max, length);
    }
    sz_size_t cursor = bucket_offsets[head->buckets];
    for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket) {
        head->long_buckets[bucket].offset = cursor;
        cursors[bucket] = cursor;
        cursor += head->long_buckets[bucket].count;
    }
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const length = engine->lengths[index];
        sz_size_t const words = sz_levenshtein_query_words(length);
        if (!words || words > head->buckets) order[cursors[sz_levenshtein_length_bucket_(length)]++] = (sz_u32_t)index;
    }

    // Compile pipelines once during initialization and cache their threadgroup sizes.
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
    if (bucket_offsets[head->buckets] != engine->count) {
        char const *const name = engine->symbol == sz_levenshtein_runes_k ? "sz_levenshtein_long_utf8_metal_kernel_"
                                                                          : "sz_levenshtein_long_metal_kernel_";
        void *const pipeline = sz_metal_pipeline_(context, sz_levenshtein_source_metal_, name);
        if (!pipeline || sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") < 32)
            return sz_device_code_mismatch_k;
    }
    if (engine->symbol == sz_levenshtein_bytes_k &&
        sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_gpu_words_max_k) {
        char const *const names[] = {"sz_levenshtein_tiled_metal_kernel_",
                                     "sz_levenshtein_tiled_identity_metal_kernel_"};
        for (sz_size_t index = 0; index != 2; ++index) {
            void *const pipeline = sz_metal_pipeline_(context, sz_levenshtein_source_metal_, names[index]);
            if (!pipeline || sz_metal_count_(pipeline, "threadExecutionWidth") != 32 ||
                sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") < 32)
                return sz_device_code_mismatch_k;
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_metal(sz_levenshtein_engine_t *engine,
                                                             sz_sequence_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (queries->count == 0 || queries->count > 0xFFFFFFFFu) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);

    sz_size_t longest = 0, text_bytes = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        longest = sz_max_of_two(longest, bytes);
        if (symbol == sz_levenshtein_bytes_k && bytes > sz_levenshtein_gpu_words_max_k * 64) {
            if (bytes > 0xFFFFFF00u || text_bytes > STRINGZILLA_SIZE_MAX - bytes)
                return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
            text_bytes += bytes;
        }
    }
    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    sz_size_t const buckets_bound = sz_min_of_two(sz_levenshtein_query_words(longest),
                                                  (sz_size_t)sz_levenshtein_gpu_words_max_k);
    sz_size_t const head_bytes = sz_levenshtein_head_bytes_metal_(queries->count, buckets_bound);
    sz_size_t const offsets_bytes = (queries->count + 1) * sizeof(sz_size_t);
    sz_size_t const aligned_head_bytes = sz_size_divide_round_up(head_bytes, sizeof(sz_size_t)) * sizeof(sz_size_t);
    if (aligned_head_bytes > STRINGZILLA_SIZE_MAX - offsets_bytes ||
        text_bytes > STRINGZILLA_SIZE_MAX - offsets_bytes - aligned_head_bytes)
        return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    status = sz_levenshtein_engine_build_(queries, symbol, aligned_head_bytes + offsets_bytes + text_bytes, &unified,
                                          stream, engine);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    sz_metal_bound_t memory;
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &memory))
        status = sz_device_memory_mismatch_k;
    else status = sz_levenshtein_bind_head_metal_(engine, call.context, buckets_bound);
    if (status != sz_success_k) {
        sz_levenshtein_engine_free_(engine, stream);
        return sz_metal_commit_(&call, status);
    }
    sz_levenshtein_head_metal_t *const head = (sz_levenshtein_head_metal_t *)engine->memory;
    head->query_offsets = aligned_head_bytes;
    head->query_text = head->query_offsets + offsets_bytes;
    sz_size_t *const offsets = (sz_size_t *)((char *)engine->memory + head->query_offsets);
    char *const text = (char *)engine->memory + head->query_text;
    sz_size_t cursor = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        offsets[index] = cursor;
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (symbol == sz_levenshtein_bytes_k && bytes > sz_levenshtein_gpu_words_max_k * 64) {
            sz_cptr_t const source = queries->get_start(queries->handle, index);
            for (sz_size_t byte = 0; byte != bytes; ++byte) text[cursor + byte] = source[byte];
            cursor += bytes;
        }
    }
    offsets[queries->count] = cursor;
    sz_levenshtein_engine_fill_(engine, queries);
    engine->capability = sz_cap_metal_k;
    return sz_metal_commit_(&call, sz_success_k);
}

/** Launch parameters for the dynamic Myers and tiled kernels, shared with the Metal source. */
typedef struct sz_levenshtein_long_arguments_metal_t {
    sz_levenshtein_arguments_metal_t common;
    sz_u64_t query_first, candidate_first, candidate_count, words_stride;
    sz_u64_t query_offsets, query_text, scratch_stride;
    sz_u32_t diagonal, row_first;
} sz_levenshtein_long_arguments_metal_t;

/** Scratch stride and wavefront geometry for one cross-product of length buckets. */
typedef struct sz_levenshtein_long_layout_metal_t {
    sz_size_t scratch_stride, rows, columns;
    sz_bool_t tiled;
} sz_levenshtein_long_layout_metal_t;

STRINGZILLA_INLINE sz_levenshtein_long_layout_metal_t sz_levenshtein_long_layout_metal_(sz_levenshtein_symbol_t symbol,
                                                                                        sz_size_t query_max,
                                                                                        sz_size_t candidate_max) {
    sz_levenshtein_long_layout_metal_t layout = {0};
    layout.tiled = symbol == sz_levenshtein_bytes_k && query_max > sz_levenshtein_gpu_words_max_k * 64 &&
                           candidate_max > sz_levenshtein_gpu_words_max_k * 64 && candidate_max <= 0xFFFFFF00u
                       ? sz_true_k
                       : sz_false_k;
    if (layout.tiled) {
        layout.rows = sz_size_divide_round_up(sz_min_of_two(query_max, candidate_max), 128);
        layout.columns = sz_size_divide_round_up(sz_max_of_two(query_max, candidate_max), 128);
        layout.scratch_stride = layout.rows * 396;
    }
    else layout.scratch_stride = sz_levenshtein_query_words(query_max) * 2 * sizeof(sz_u64_t);
    return layout;
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_metal(sz_levenshtein_engine_t *engine,
                                                           sz_sequence_t const *candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_stream_t stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (!candidates->count) return sz_success_k;

    // The engine's block, the candidates' tape and the distances, as the kernels number them.
    sz_metal_bound_t buffers[3];
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (engine->count > 1 && distances_stride > (STRINGZILLA_SIZE_MAX - candidates->count) / (engine->count - 1))
        return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    sz_size_t const distances_count = (engine->count - 1) * distances_stride + candidates->count;
    if (distances_count > STRINGZILLA_SIZE_MAX / sizeof(sz_size_t))
        return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &buffers[0]) ||
        !sz_metal_resolve_call_(&call, distances, distances_count * sizeof(sz_size_t), &buffers[2]))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    status = sz_metal_tape_call_(&call, candidates, &buffers[1]);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);

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
    if (bucket_offsets[head->buckets] == engine->count) return sz_metal_commit_(&call, status);
    char const *const myers_kernel = engine->symbol == sz_levenshtein_runes_k ? "sz_levenshtein_long_utf8_metal_kernel_"
                                                                              : "sz_levenshtein_long_metal_kernel_";
    void *const myers_pipeline = sz_metal_pipeline_(call.context, sz_levenshtein_source_metal_, myers_kernel);
    if (!myers_pipeline || sz_metal_count_(myers_pipeline, "maxTotalThreadsPerThreadgroup") < 32)
        return sz_metal_commit_(&call, sz_device_code_mismatch_k);

    sz_levenshtein_length_bucket_t candidate_buckets[sz_levenshtein_length_buckets_k] = {0};
    sz_size_t cursors[sz_levenshtein_length_buckets_k];
    for (sz_size_t candidate = 0; candidate != candidates->count; ++candidate) {
        sz_size_t const bytes = candidates->get_length(candidates->handle, candidate);
        sz_levenshtein_length_bucket_t *const bucket = candidate_buckets + sz_levenshtein_length_bucket_(bytes);
        ++bucket->count;
        bucket->length_max = sz_max_of_two(bucket->length_max, bytes);
    }
    sz_size_t cursor = 0;
    for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket) {
        candidate_buckets[bucket].offset = cursor;
        cursors[bucket] = cursor;
        cursor += candidate_buckets[bucket].count;
    }
    void *const candidate_order = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        call.context->device, sel_registerName("newBufferWithLength:options:"), candidates->count * sizeof(sz_u64_t),
        (sz_size_t)0);
    if (!candidate_order) return sz_metal_commit_(&call, sz_bad_alloc_k);
    sz_u64_t *const indices = (sz_u64_t *)sz_metal_get_(candidate_order, "contents");
    for (sz_size_t candidate = 0; candidate != candidates->count; ++candidate) {
        sz_size_t const bytes = candidates->get_length(candidates->handle, candidate);
        indices[cursors[sz_levenshtein_length_bucket_(bytes)]++] = candidate;
    }
    sz_size_t const working_set = sz_metal_count_(call.context->device, "recommendedMaxWorkingSetSize");
    sz_size_t const budget = sz_max_of_two((sz_size_t)4 * 1024 * 1024,
                                           sz_min_of_two(working_set / 128, (sz_size_t)64 * 1024 * 1024));
    sz_size_t scratch_bytes = 1;
    for (sz_size_t query_bucket = 0; query_bucket != sz_levenshtein_length_buckets_k; ++query_bucket) {
        sz_levenshtein_length_bucket_t const *const queries = head->long_buckets + query_bucket;
        if (!queries->count) continue;
        for (sz_size_t candidate_bucket = 0; candidate_bucket != sz_levenshtein_length_buckets_k; ++candidate_bucket) {
            sz_levenshtein_length_bucket_t const *const texts = candidate_buckets + candidate_bucket;
            if (!texts->count) continue;
            sz_levenshtein_long_layout_metal_t const layout = sz_levenshtein_long_layout_metal_(
                engine->symbol, queries->length_max, texts->length_max);
            sz_size_t const stride = layout.scratch_stride;
            if (!stride) continue;
            sz_size_t const pairs = sz_min_of_two(queries->count * texts->count,
                                                  sz_min_of_two((sz_size_t)sz_levenshtein_gpu_grid_rows_max_k,
                                                                sz_max_of_two(budget / stride, (sz_size_t)1)));
            scratch_bytes = sz_max_of_two(scratch_bytes, pairs * stride);
        }
    }
    void *const scratch = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        call.context->device, sel_registerName("newBufferWithLength:options:"), scratch_bytes, (sz_size_t)32);
    if (!scratch) {
        sz_metal_do_(candidate_order, "release");
        return sz_metal_commit_(&call, sz_bad_alloc_k);
    }
    sz_metal_bound_t long_buffers[5] = {buffers[0], buffers[1], buffers[2], {scratch, 0}, {candidate_order, 0}};
    sz_levenshtein_long_arguments_metal_t long_arguments = {0};
    long_arguments.common = arguments;
    long_arguments.query_offsets = head->query_offsets, long_arguments.query_text = head->query_text;
    sz_metal_size_t const threads = {32, 1, 1};
    for (sz_size_t query_bucket = 0; status == sz_success_k && query_bucket != sz_levenshtein_length_buckets_k;
         ++query_bucket) {
        sz_levenshtein_length_bucket_t const *const queries = head->long_buckets + query_bucket;
        if (!queries->count) continue;
        long_arguments.words_stride = sz_levenshtein_query_words(queries->length_max);
        for (sz_size_t candidate_bucket = 0; candidate_bucket != sz_levenshtein_length_buckets_k; ++candidate_bucket) {
            sz_levenshtein_length_bucket_t const *const texts = candidate_buckets + candidate_bucket;
            if (!texts->count) continue;
            sz_levenshtein_long_layout_metal_t const layout = sz_levenshtein_long_layout_metal_(
                engine->symbol, queries->length_max, texts->length_max);
            char const *const tiled_kernel = queries->count == engine->count && texts->count == candidates->count
                                                 ? "sz_levenshtein_tiled_identity_metal_kernel_"
                                                 : "sz_levenshtein_tiled_metal_kernel_";
            void *const pipeline = layout.tiled
                                       ? sz_metal_pipeline_(call.context, sz_levenshtein_source_metal_, tiled_kernel)
                                       : myers_pipeline;
            if (!pipeline) {
                status = sz_device_code_mismatch_k;
                break;
            }
            long_arguments.scratch_stride = layout.scratch_stride;
            sz_size_t const pair_capacity = long_arguments.scratch_stride
                                                ? sz_min_of_two(scratch_bytes / long_arguments.scratch_stride,
                                                                (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k)
                                                : (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k;
            sz_size_t const candidate_chunk = sz_min_of_two(texts->count, pair_capacity);
            sz_size_t const query_chunk = sz_min_of_two(queries->count, pair_capacity / candidate_chunk);
            for (sz_size_t query = 0; query < queries->count; query += query_chunk) {
                sz_size_t const query_count = sz_min_of_two(queries->count - query, query_chunk);
                long_arguments.common.order = (sz_u64_t)(order + queries->offset + query);
                long_arguments.query_first = query;
                for (sz_size_t candidate = 0; candidate < texts->count; candidate += candidate_chunk) {
                    sz_size_t const candidate_count = sz_min_of_two(texts->count - candidate, candidate_chunk);
                    long_arguments.candidate_first = texts->offset + candidate;
                    long_arguments.candidate_count = candidate_count;
                    if (!layout.tiled) {
                        sz_metal_size_t const groups = {sz_size_divide_round_up(candidate_count, 32), query_count, 1};
                        sz_metal_enqueue_(call.encoder, pipeline, long_buffers, 5, &long_arguments,
                                          sizeof(long_arguments), groups, threads);
                        continue;
                    }
                    for (sz_size_t diagonal = 0; diagonal < layout.rows + layout.columns - 1; ++diagonal) {
                        sz_size_t const first = diagonal >= layout.columns ? diagonal - layout.columns + 1 : 0;
                        sz_size_t const end = sz_min_of_two(layout.rows, diagonal + 1);
                        sz_metal_size_t const groups = {end - first, query_count * candidate_count, 1};
                        long_arguments.diagonal = (sz_u32_t)diagonal, long_arguments.row_first = (sz_u32_t)first;
                        sz_metal_enqueue_(call.encoder, pipeline, long_buffers, 5, &long_arguments,
                                          sizeof(long_arguments), groups, threads);
                    }
                }
            }
        }
    }
    sz_metal_do_(scratch, "release");
    sz_metal_do_(candidate_order, "release");
    return sz_metal_commit_(&call, status);
}

#pragma endregion Metal

#ifdef __cplusplus
}
#endif
#endif
#endif
