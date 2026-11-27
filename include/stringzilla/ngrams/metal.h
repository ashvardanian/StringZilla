/**
 *  @file include/stringzilla/ngrams/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Metal backend for N-grams on Apple7 and newer: the engine's tables filled on the host
 *      into one shared block, one dispatch per overlaps round and one per sketches round.
 *
 *  @sa include/stringzilla/ngrams/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/ngrams/simt.cuh, the CUDA sibling
 *
 *  The verbs keep the CUDA tier's shape, with an @c id<MTLCommandQueue> as the stream. Every block
 *  the kernels read, the tables, the texts' tapes, the shares and the sketches, must come from
 *  @c sz_allocator_init_unified_metal on that stream's device; memory outside the device's registry
 *  draws @c sz_device_memory_mismatch_k.
 *
 *  The kernels read the tape itself, since Metal has no device function pointers to call an
 *  accessor through, so a round's texts must be a tape, as @ref sz_strings_to_tape_best writes one
 *  with that allocator; an accessor draws @c sz_device_memory_mismatch_k.
 *
 *  A kit composing its own kernels compiles @ref sz_ngrams_source_metal behind the shared prelude
 *  and calls its @c sz_ngrams_*_metal steps.
 */
#ifndef STRINGZILLA_NGRAMS_METAL_H_
#define STRINGZILLA_NGRAMS_METAL_H_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_METAL_

#include "stringzilla/metal.h"
#include "stringzilla/ngrams/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_METAL

#pragma region Metal

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the N-grams steps and kernels, compiled once per device behind the prelude. */
static char const sz_ngrams_source_metal[] = {
#embed "stringzilla/ngrams/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** Candidates one threadgroup's tile holds, a simdgroup scoring one at a time, and the threads of
 *  the threadgroup at most, as the CUDA tier's blocks. */
enum { sz_ngrams_candidates_per_group_metal_k = 128 };

/** The share of the 32 KB of threadgroup memory a staged table may take: up to half (16 KB), so the
 *  hit words keep the rest. */
enum { sz_ngrams_staged_table_share_metal_k = 2 };

/** Threads an overlaps grid carries before further queries fold into each threadgroup's own loop,
 *  which keeps its simdgroups on one tile for all of them: several times what any Apple GPU keeps
 *  resident, which Metal does not report. */
enum { sz_ngrams_grid_threads_metal_k = 1 << 20 };

/** Threads one sketching threadgroup runs at most, a simdgroup sketching one text at a time over
 *  its own ranks; fewer where the ranks of four simdgroups outgrow the threadgroup memory. */
enum { sz_ngrams_sketch_threads_per_group_metal_k = 128 };

/** Threadgroups one sketching grid carries at most; a batch past them is covered by each
 *  simdgroup's own loop. */
enum { sz_ngrams_sketch_groups_per_grid_metal_k = 0x7FFFFFFF };

/** What @ref sz_ngrams_engine_init_metal resolved once, kept at the head of the engine's block. */
typedef struct sz_ngrams_geometry_metal_t {

    /** Threads one threadgroup runs, a multiple of the simdgroup width. */
    sz_size_t candidates_per_group;

    /** Slots a threadgroup stages, sized by the widest table, or zero for none. */
    sz_size_t staged_slots_count;

    /** Hit words each simdgroup keeps, the widest query's N-grams in bits. */
    sz_size_t hit_words_per_simdgroup;

    /** Threadgroup memory the device hands one threadgroup, which bounds a sketch's ranks. */
    sz_size_t threadgroup_bytes;
} sz_ngrams_geometry_metal_t;

/** The overlaps launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t engine_host;
    sz_u64_t slots, slots_offsets, filters, distinct_counts, widths, powers, lengths;
    sz_u64_t queries_count, widths_count, candidates_count;
    sz_u64_t shares_query_stride, shares_candidate_stride, staged_slots_count, hit_words_per_simdgroup,
        candidates_layout, misses_bound;
} sz_ngrams_overlaps_arguments_metal_t;

/** The sketches launch record, laid out as the kernels' record of the same name. */
typedef struct {
    sz_u64_t texts_count, slots, sketches_stride, texts_layout, widths_count;
    sz_u32_t widths[sz_ngrams_widths_max_k];
} sz_ngrams_sketches_arguments_metal_t;

/** Threadgroup memory one threadgroup of @p threads takes: the staged table, then each simdgroup's
 *  hit words, in the 16-byte steps Metal sizes it in and bound even when it holds neither. */
STRINGZILLA_INLINE sz_size_t sz_ngrams_threadgroup_bytes_metal_(sz_size_t staged_slots_count, sz_size_t threads,
                                                                sz_size_t hit_words_per_simdgroup) {
    sz_size_t const bytes = staged_slots_count * sizeof(sz_u64_t) +
                            sz_size_divide_round_up(threads, 32) * hit_words_per_simdgroup * sizeof(sz_u32_t);
    return bytes ? sz_size_divide_round_up(bytes, 16) * 16 : 16;
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    sz_unused_(candidates_budget);
    sz_metal_call_t call;
    sz_status_t const entered = sz_device_enter_metal_(stream, &call);
    if (entered != sz_success_k) return sz_metal_commit_(&call, entered);
    // A gather committed earlier on this queue may still be writing the queries read from here on.
    sz_status_t const drained = sz_stream_synchronize_metal_(call.queue);
    if (drained != sz_success_k) return sz_metal_commit_(&call, drained);

    sz_allocator_t unified;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_metal(&unified);
    sz_status_t const opened = sz_ngrams_engine_open_(queries, window_widths, window_widths_count,
                                                      sizeof(sz_ngrams_geometry_metal_t), &unified, stream, engine);
    if (opened != sz_success_k) return sz_metal_commit_(&call, opened);
    sz_metal_bound_t bound;
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &bound)) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    }

    // The prefixes and the N-gram residues are host working space that no kernel reads, so they
    // come from the host.
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        longest_query = sz_max_of_two(longest_query, (sz_size_t)engine->lengths[index]);
    sz_allocator_t host;
    sz_status_t const status = sz_allocator_init_heap(&host);
    if (status != sz_success_k) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_metal_commit_(&call, status);
    }
    sz_size_t const prefixes_bytes = (longest_query + 1) * sizeof(sz_f64_t) +
                                     (sz_ngrams_count(longest_query, engine->widths, engine->widths_count) + 1) *
                                         sizeof(sz_u32_t);
    sz_f64_t *const prefixes = (sz_f64_t *)host.allocate(prefixes_bytes, host.handle, stream);
    if (!prefixes) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_bad_alloc_k);
    }
    sz_ngrams_engine_fill_serial_(engine, queries, prefixes, (sz_u32_t *)(prefixes + longest_query + 1));
    host.free(prefixes, prefixes_bytes, host.handle, stream);
    sz_size_t widest_slots = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        widest_slots = sz_max_of_two(widest_slots, engine->slots_offsets[index + 1] - engine->slots_offsets[index]);

    // Building the pipeline here keeps the compile out of the rounds and names its ceiling. Every
    // threadgroup stages one table and every simdgroup keeps hit words for the widest query, so a
    // query whose words outgrow one simdgroup's share of the threadgroup is refused here rather
    // than at the first round.
    void *const pipeline = sz_metal_pipeline_(call.context, sz_ngrams_source_metal, "sz_ngrams_overlaps_metal_kernel_");
    if (!pipeline) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    }
    sz_size_t const threads_ceiling = sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") / 32 * 32;
    sz_size_t const threadgroup_ceiling = sz_metal_count_(call.context->device, "maxThreadgroupMemoryLength");
    sz_size_t const hit_words = sz_size_divide_round_up(sz_ngrams_engine_widest_ngrams_(engine), 32);
    if (!threads_ceiling || sz_ngrams_threadgroup_bytes_metal_(0, 32, hit_words) > threadgroup_ceiling) {
        sz_ngrams_engine_close_(engine, stream);
        return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    }
    sz_ngrams_geometry_metal_t *const geometry = (sz_ngrams_geometry_metal_t *)sz_ngrams_engine_head_(engine);
    sz_size_t widest_group = sz_min_of_two(threads_ceiling, (sz_size_t)sz_ngrams_candidates_per_group_metal_k);
    geometry->threadgroup_bytes = threadgroup_ceiling;
    geometry->hit_words_per_simdgroup = hit_words;
    geometry->staged_slots_count = 0;
    if (widest_slots * sizeof(sz_u64_t) * sz_ngrams_staged_table_share_metal_k <= threadgroup_ceiling &&
        sz_ngrams_threadgroup_bytes_metal_(widest_slots, widest_group, hit_words) <= threadgroup_ceiling)
        geometry->staged_slots_count = widest_slots;
    while (widest_group > 32 && sz_ngrams_threadgroup_bytes_metal_(geometry->staged_slots_count, widest_group,
                                                                   hit_words) > threadgroup_ceiling)
        widest_group /= 2;
    geometry->candidates_per_group = widest_group;
    engine->capability = sz_cap_metal_k;
    return sz_metal_commit_(&call, sz_success_k);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_f32_t *shares, sz_size_t shares_query_stride,
                                                     sz_size_t shares_candidate_stride, sz_stream_t stream) {
    sz_size_t query_stride_entries, candidate_stride_entries;
    sz_status_t status = sz_ngrams_engine_strides_(engine, candidates->count, shares_query_stride,
                                                   shares_candidate_stride, &query_stride_entries,
                                                   &candidate_stride_entries);
    if (status != sz_success_k) return status;
    if (!candidates->count || !engine->count) return sz_success_k;

    // The tables, the candidates' payload and offsets, and the shares, as the kernel numbers them.
    sz_metal_bound_t buffers[4];
    sz_metal_call_t call;
    status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    sz_size_t const shares_count = (engine->count - 1) * query_stride_entries +
                                   (candidates->count - 1) * candidate_stride_entries + engine->widths_count;
    if (!sz_metal_resolve_call_(&call, engine->memory, engine->memory_bytes, &buffers[0]) ||
        !sz_metal_resolve_call_(&call, shares, shares_count * sizeof(sz_f32_t), &buffers[3]))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    status = sz_metal_tape_call_(&call, candidates, &buffers[1], &buffers[2]);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);

    sz_ngrams_geometry_metal_t const *const geometry = (sz_ngrams_geometry_metal_t const *)sz_ngrams_engine_head_(
        engine);
    sz_ngrams_overlaps_arguments_metal_t arguments;
    arguments.engine_host = (sz_u64_t)engine->memory;
    arguments.slots = (sz_u64_t)engine->slots, arguments.slots_offsets = (sz_u64_t)engine->slots_offsets;
    arguments.filters = (sz_u64_t)engine->filters;
    arguments.distinct_counts = (sz_u64_t)engine->distinct_counts, arguments.widths = (sz_u64_t)engine->widths;
    arguments.powers = (sz_u64_t)engine->powers, arguments.lengths = (sz_u64_t)engine->lengths;
    arguments.queries_count = engine->count, arguments.widths_count = engine->widths_count;
    arguments.candidates_count = candidates->count;
    arguments.shares_query_stride = query_stride_entries;
    arguments.shares_candidate_stride = candidate_stride_entries;
    arguments.staged_slots_count = geometry->staged_slots_count;
    arguments.hit_words_per_simdgroup = geometry->hit_words_per_simdgroup;
    arguments.candidates_layout = (sz_u64_t)candidates->layout;
    arguments.misses_bound = sz_ngrams_misses_bound_(radius);

    sz_size_t const per_group = geometry->candidates_per_group;
    sz_size_t const groups = sz_size_divide_round_up(candidates->count, per_group);
    sz_size_t const rows = sz_min_of_two(engine->count,
                                         sz_size_divide_round_up(sz_ngrams_grid_threads_metal_k, groups * per_group));
    sz_metal_size_t const grid = {groups, rows, 1}, threads = {per_group, 1, 1};
    void *const pipeline = sz_metal_pipeline_(call.context, sz_ngrams_source_metal, "sz_ngrams_overlaps_metal_kernel_");
    if (!pipeline || !call.encoder) return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    sz_metal_threadgroup_memory_(
        call.encoder,
        sz_ngrams_threadgroup_bytes_metal_(geometry->staged_slots_count, per_group, geometry->hit_words_per_simdgroup),
        0);
    sz_metal_enqueue_(call.encoder, pipeline, buffers, 4, &arguments, sizeof(arguments), grid, threads);
    return sz_metal_commit_(&call, sz_success_k);
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_metal(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream) {
    sz_status_t status = sz_ngrams_sketch_dimensions_(engine->widths, engine->widths_count, slots);
    if (status != sz_success_k) return status;
    if (sketches_stride < slots) return sz_unexpected_dimensions_k;
    // A simdgroup's ranks live in threadgroup memory, so a sketch outgrowing it is refused.
    sz_ngrams_geometry_metal_t const *const geometry = (sz_ngrams_geometry_metal_t const *)sz_ngrams_engine_head_(
        engine);
    sz_size_t const simdgroup_bytes = slots * sizeof(sz_u32_t);
    if (simdgroup_bytes > geometry->threadgroup_bytes) return sz_unexpected_dimensions_k;
    if (!texts->count) return sz_success_k;

    // The texts' payload and offsets, and the sketches, in the kernel's order.
    sz_metal_bound_t buffers[3];
    sz_metal_call_t call;
    status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    sz_size_t const sketches_bytes = (texts->count - 1) * sketches_stride + slots;
    if (!sz_metal_resolve_call_(&call, sketches, sketches_bytes, &buffers[2]))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    status = sz_metal_tape_call_(&call, texts, &buffers[0], &buffers[1]);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);

    sz_ngrams_sketches_arguments_metal_t arguments;
    arguments.texts_count = texts->count, arguments.slots = slots, arguments.sketches_stride = sketches_stride;
    arguments.texts_layout = (sz_u64_t)texts->layout, arguments.widths_count = engine->widths_count;
    for (sz_size_t width_index = 0; width_index != sz_ngrams_widths_max_k; ++width_index)
        arguments.widths[width_index] = width_index < engine->widths_count ? (sz_u32_t)engine->widths[width_index] : 0;

    void *const pipeline = sz_metal_pipeline_(call.context, sz_ngrams_source_metal, "sz_ngrams_sketches_metal_kernel_");
    if (!pipeline || !call.encoder) return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    sz_size_t const simdgroups = sz_min_of_two(sz_min_of_two(sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup"),
                                                             (sz_size_t)sz_ngrams_sketch_threads_per_group_metal_k) /
                                                   32,
                                               geometry->threadgroup_bytes / simdgroup_bytes);
    if (!simdgroups) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    sz_size_t const groups = sz_min_of_two(sz_size_divide_round_up(texts->count, simdgroups),
                                           (sz_size_t)sz_ngrams_sketch_groups_per_grid_metal_k);
    sz_metal_size_t const grid = {groups, 1, 1}, threads = {simdgroups * 32, 1, 1};
    sz_metal_threadgroup_memory_(call.encoder, sz_size_divide_round_up(simdgroups * simdgroup_bytes, 16) * 16, 0);
    sz_metal_enqueue_(call.encoder, pipeline, buffers, 3, &arguments, sizeof(arguments), grid, threads);
    return sz_metal_commit_(&call, sz_success_k);
}

#pragma endregion Metal

#endif // STRINGZILLA_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_METAL_
#endif // STRINGZILLA_NGRAMS_METAL_H_
