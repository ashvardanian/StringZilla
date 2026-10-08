/**
 *  @file include/stringzilla/levenshtein/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of Levenshtein distances: the rung buckets, the launch geometry and the
 *      @c _rocm exports, over the walks and kernels of `levenshtein/simt.cuh`.
 *
 *  A batch is bucketed by rung key at @ref sz_levenshtein_engine_init_scoped_rocm_, so one launch
 *  carries only queries that share an entry point, and the occupancy walk each of those entry
 *  points needs is paid there rather than once per round. The byte planes are built by a kernel
 *  from the queries staged into device-reachable memory; the rune planes are built on the host, an
 *  init being allowed to join where a round is not.
 *
 *  AMD blocks never take over unstarted ones, so a tile always holds several rounds for its
 *  block's threads to share, and the wavefront's warps sit a whole 64-lane wavefront apart.
 *
 *  @sa include/stringzilla/levenshtein/simt.cuh
 *  @sa include/stringzilla/levenshtein/cuda.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_ROCM_CUH_
#define STRINGZILLA_LEVENSHTEIN_ROCM_CUH_

#include "stringzilla/rocm.cuh"
#include "stringzilla/levenshtein/simt.cuh"

#if STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Entry Points

/** Threads a block runs when the device cannot be asked; the register budget differs per word
 *  count, so the launcher takes what the occupancy calculator answers for the entry point it is
 *  about to launch instead. */
enum { sz_levenshtein_candidates_per_block_rocm_k = 128 };

/** Widest block the launcher considers: past this a block schedules too coarsely for
 *  what residency returns. */
enum { sz_levenshtein_candidates_per_block_max_rocm_k = 256 };

/** Rounds of candidates one block's tile holds, as its blocks cannot take over unstarted ones. */
enum { sz_levenshtein_tile_rounds_rocm_k = 4 };

/** Query symbols one byte lane of a Myers word holds, which is the longest query the byte rung
 *  takes, and one sixteen-bit lane holds, which is the longest query either narrow rung takes. */
enum {
    sz_levenshtein_byte_lanes_symbols_max_rocm_k = 8,
    sz_levenshtein_short_lanes_symbols_max_rocm_k = 16,
};

/** Words one lane of a warped candidate owns, which is what selects the rung's entry point. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_warp_words_per_lane_rocm_(sz_size_t words) {
    return sz_size_divide_round_up(words, sz_levenshtein_gpu_warp_lanes_k);
}

#if STRINGZILLA_TARGET_ROCM

/*  Warped byte rung: one entry point per words-per-lane. */
static __global__ void sz_levenshtein_distances_k1_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 1, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k2_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 2, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k3_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 3, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k4_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 4, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k5_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 5, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k6_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 6, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k7_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 7, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_k8_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 8, sz_shuffle_up_rocm_,
                                    sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_draw_simt_);
}

/*  Warped rune rung: one entry point per words-per-lane. */
static __global__ void sz_levenshtein_distances_utf8_k1_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 1, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k2_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 2, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k3_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 3, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k4_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 4, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k5_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 5, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k6_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 6, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k7_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 7, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void sz_levenshtein_distances_utf8_k8_rocm_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 8, sz_shuffle_up_rocm_,
                                         sz_shuffle_down_rocm_, sz_lanes_broadcast_rocm_, sz_lanes_any_rocm_,
                                         sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

#endif // STRINGZILLA_TARGET_ROCM

/** Threads from one tile-column's first lane to the next one's, a whole 64-wide wavefront, whose
 *  halves would march two tile-columns in lockstep, one spinning on the other forever; and the
 *  threads one block of the device-spanning wavefront runs. */
enum {
    sz_levenshtein_tiled_warp_stride_rocm_k = 64,
    sz_levenshtein_tiled_threads_per_block_rocm_k = sz_levenshtein_tiled_warps_per_block_simt_k *
                                                    sz_levenshtein_tiled_warp_stride_rocm_k,
};

/** Publishes that this tile-column finished @p tile_row, releasing the frontier writes before it
 *  at agent scope, as the CUDA twin does in PTX. */
STRINGZILLA_DEVICE void sz_levenshtein_publish_rocm_(sz_u32_t *counter, sz_u32_t tile_row) {
    __hip_atomic_store(counter, tile_row + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

/** Spins until the left tile-column published past @p tile_row, every lane acquiring its
 *  frontier writes, as the CUDA twin does. */
STRINGZILLA_DEVICE void sz_levenshtein_await_rocm_(sz_u32_t const *counter, sz_u32_t tile_row) {
    sz_u32_t observed;
    do observed = __hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    while (observed <= tile_row);
}

static __global__
__launch_bounds__(sz_levenshtein_tiled_threads_per_block_rocm_k) void sz_levenshtein_tiled_batch_rocm_kernel_(
    sz_levenshtein_engine_t engine, sz_sequence_t candidates, sz_size_t *distances, sz_ptr_t workspace,
    sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_tiled_batch_simt_(engine, candidates, distances, workspace, batch,
                                     sz_levenshtein_tiled_warp_stride_rocm_k, sz_shuffle_up_rocm_,
                                     sz_levenshtein_publish_rocm_, sz_levenshtein_await_rocm_);
}

/**
 *  @brief One ROCm tier's entry points into the walks, which its exports hand the host side.
 *
 *  @c hipLaunchKernel takes the host-side symbol of a @c __global__, so each table is what a
 *  chevron launch would have selected, spelled as data, and addressed by word count or by the
 *  words each lane owns.
 */
typedef struct sz_levenshtein_entry_points_rocm_t {

    /** The threaded byte rung, one entry point per word count. */
    void const *threaded[sz_levenshtein_thread_words_max_simt_k];

    /** The threaded rune rung, one entry point per word count. */
    void const *threaded_utf8[sz_levenshtein_thread_words_max_simt_k];

    /** The warped byte rung, one entry point per words-per-lane. */
    void const *warped[sz_levenshtein_gpu_warp_words_per_lane_max_k];

    /** The warped rune rung, one entry point per words-per-lane. */
    void const *warped_utf8[sz_levenshtein_gpu_warp_words_per_lane_max_k];

    /** The capability the tier's engines record. */
    sz_capability_t capability;
} sz_levenshtein_entry_points_rocm_t;

/** Threads a sweep gives a block on this device, for the entry point it is about to launch, the
 *  walk stopping at @c sz_levenshtein_candidates_per_block_max_rocm_k. */
static sz_size_t sz_levenshtein_per_block_rocm_(void const *entry_point) {
    return sz_block_size_rocm_(entry_point, 0, sz_levenshtein_candidates_per_block_max_rocm_k,
                               sz_levenshtein_candidates_per_block_rocm_k);
}

/** Candidates one thread takes, which is the lanes a query of @p length symbols leaves
 *  in one register. */
static sz_size_t sz_levenshtein_lanes_per_thread_rocm_(sz_size_t length) {
    return length <= sz_levenshtein_byte_lanes_symbols_max_rocm_k ? 4 : 2;
}

/** Candidates a batch must carry before a narrow rung is launched at all, which the device's own
 *  residency scales. @c STRINGZILLA_SIZE_MAX where the device cannot be asked, so the threaded rung
 *  keeps every batch. */
static sz_size_t sz_levenshtein_lanes_candidates_min_rocm_(void) {
    sz_size_t const resident_threads = sz_device_multiprocessors_rocm_() * sz_device_threads_per_multiprocessor_rocm_();
    return resident_threads ? resident_threads * sz_levenshtein_gpu_lanes_waves_min_k : STRINGZILLA_SIZE_MAX;
}

#pragma endregion Entry Points

#pragma region Myers Engine

/** Query buckets and launch geometry at the start of the engine's allocation. */
typedef struct sz_levenshtein_head_rocm_t {

    /** Number of word-count buckets plus the two narrow-kernel buckets. */
    sz_size_t buckets;

    /** Minimum candidate count for the narrow kernels. */
    sz_size_t candidates_min;

    /** Threads per block for the byte-lane and short-lane kernels. */
    sz_size_t narrow_per_block[2];

    /** @b [buckets+1] bucket boundaries in @c order. */
    sz_size_t *bucket_offsets;

    /** @b [buckets] thread counts per block for the wide kernels. */
    sz_size_t *per_block;

    /** @b [count] query indices grouped by bucket. */
    sz_u32_t *order;

    /** @b [count+1] query byte offsets into @c query_text. */
    sz_size_t *query_offsets;

    /** Empty and long query indices, grouped by length after the short-query buckets. */
    sz_levenshtein_length_bucket_t long_buckets[sz_levenshtein_length_buckets_k];

    /** Packed long byte queries used by the tiled kernel. */
    sz_ptr_t query_text;
} sz_levenshtein_head_rocm_t;

/** The rung key a query of @p length symbols launches from: the two narrow lanes, then one
 *  key per word. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_bucket_rocm_(sz_size_t length) {
    if (length <= sz_levenshtein_byte_lanes_symbols_max_rocm_k) return 0;
    if (length <= sz_levenshtein_short_lanes_symbols_max_rocm_k) return 1;
    return 1 + sz_levenshtein_query_words(length);
}

/** Rung keys a batch whose widest query spans @p words words can reach. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_buckets_rocm_(sz_size_t words) { return words + 2; }

/** Query words the wide rung of @p bucket steps, the two narrow keys sharing the
 *  one-word entry points. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_bucket_words_rocm_(sz_size_t bucket) {
    return bucket <= 1 ? 1 : bucket - 1;
}

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning
 *  @p buckets rung keys. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_head_bytes_rocm_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_head_rocm_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/** The entry point @p bucket 's wide rung reaches: one candidate per thread, or one per warp
 *  past the crossing. */
static void const *sz_levenshtein_entry_point_rocm_(sz_levenshtein_entry_points_rocm_t const *entry_points,
                                                    sz_levenshtein_symbol_t symbol, sz_size_t bucket) {
    sz_size_t const words = sz_levenshtein_bucket_words_rocm_(bucket);
    if (words < sz_levenshtein_gpu_warp_words_min_k)
        return symbol == sz_levenshtein_bytes_k ? entry_points->threaded[words - 1]
                                                : entry_points->threaded_utf8[words - 1];
    sz_size_t const per_lane = sz_levenshtein_warp_words_per_lane_rocm_(words);
    return symbol == sz_levenshtein_bytes_k ? entry_points->warped[per_lane - 1]
                                            : entry_points->warped_utf8[per_lane - 1];
}

/** Points the head's tables into the block, buckets the batch by rung key, and asks the device
 *  its geometry once, for the tier whose @p entry_points the rounds will launch. */
static void sz_levenshtein_bind_head_rocm_(sz_levenshtein_engine_t *engine, sz_size_t buckets_bound,
                                           sz_levenshtein_entry_points_rocm_t const *entry_points) {
    sz_levenshtein_head_rocm_t *const head = (sz_levenshtein_head_rocm_t *)engine->memory;
    sz_size_t *const tables = (sz_size_t *)((sz_ptr_t)engine->memory + sizeof(sz_levenshtein_head_rocm_t));
    sz_size_t cursors[sz_levenshtein_gpu_words_max_k + 2];
    head->buckets = sz_levenshtein_buckets_rocm_(
        sz_min_of_two(sz_levenshtein_engine_words_max_(engine), (sz_size_t)sz_levenshtein_gpu_words_max_k));
    head->bucket_offsets = tables;
    head->per_block = tables + buckets_bound + 1;
    head->order = (sz_u32_t *)(tables + 2 * buckets_bound + 1);

    // A counting sort by rung key, so one launch only ever carries queries that share an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) head->bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] &&
            sz_levenshtein_query_words(engine->lengths[index]) <= sz_levenshtein_gpu_words_max_k)
            ++head->bucket_offsets[sz_levenshtein_bucket_rocm_(engine->lengths[index]) + 1];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        head->bucket_offsets[bucket] += head->bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = head->bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] &&
            sz_levenshtein_query_words(engine->lengths[index]) <= sz_levenshtein_gpu_words_max_k)
            head->order[cursors[sz_levenshtein_bucket_rocm_(engine->lengths[index])]++] = (sz_u32_t)index;

    // Every driver round trip a round would otherwise pay: the residency floor once, and one occupancy walk
    // per entry point the batch can reach, which the buckets fixed here and no later call can widen.
    for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket)
        head->long_buckets[bucket] = (sz_levenshtein_length_bucket_t) {0, 0, 0};
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const length = engine->lengths[index];
        if (length && sz_levenshtein_query_words(length) <= sz_levenshtein_gpu_words_max_k) continue;
        sz_levenshtein_length_bucket_t *const bucket = head->long_buckets + sz_levenshtein_length_bucket_(length);
        ++bucket->count;
        bucket->length_max = sz_max_of_two(bucket->length_max, length);
    }
    for (sz_size_t bucket = 0, offset = head->bucket_offsets[head->buckets]; bucket != sz_levenshtein_length_buckets_k;
         ++bucket) {
        head->long_buckets[bucket].offset = offset;
        cursors[bucket] = offset;
        offset += head->long_buckets[bucket].count;
    }
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_size_t const length = engine->lengths[index];
        if (length && sz_levenshtein_query_words(length) <= sz_levenshtein_gpu_words_max_k) continue;
        head->order[cursors[sz_levenshtein_length_bucket_(length)]++] = (sz_u32_t)index;
    }

    head->candidates_min = sz_levenshtein_lanes_candidates_min_rocm_();
    head->narrow_per_block[0] = sz_levenshtein_per_block_rocm_(
        (void const *)sz_levenshtein_distances_u8x4_simt_kernel_);
    head->narrow_per_block[1] = sz_levenshtein_per_block_rocm_(
        (void const *)sz_levenshtein_distances_u16x2_simt_kernel_);
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket)
        head->per_block[bucket] = head->bucket_offsets[bucket] == head->bucket_offsets[bucket + 1]
                                      ? 0
                                      : sz_levenshtein_per_block_rocm_(
                                            sz_levenshtein_entry_point_rocm_(entry_points, engine->symbol, bucket));
}

/** Threads the mask builder runs: one per byte value, so a thread owns one class flag for
 *  the whole build. */
enum { sz_levenshtein_masks_threads_rocm_k = sz_levenshtein_byte_classes_k };

/**
 *  @brief Stages the batch's query texts where the device reads them and builds every
 *      byte plane there.
 *  @note Joins @p stream, which is what lets the staging be released; only an init is allowed to.
 */
static sz_status_t sz_levenshtein_build_masks_rocm_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                                    sz_stream_t stream) {
    sz_size_t texts_bytes = 0;
    for (sz_size_t index = 0; index != queries->count; ++index)
        texts_bytes += queries->get_length(queries->handle, index);
    sz_size_t const views_bytes = queries->count * sizeof(sz_string_view_t);
    sz_size_t const staged_bytes = views_bytes + texts_bytes;
    sz_ptr_t const staged = (sz_ptr_t)engine->allocator.allocate(staged_bytes, engine->allocator.handle, stream);
    if (!staged) return sz_bad_alloc_k;
    sz_string_view_t *const views = (sz_string_view_t *)staged;
    sz_ptr_t const arena = staged + views_bytes;
    for (sz_size_t index = 0, written = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        for (sz_size_t byte = 0; byte != bytes; ++byte) arena[written + byte] = text[byte];
        views[index].start = arena + written, views[index].length = bytes;
        written += bytes;
    }

    sz_size_t const masks_bytes = engine->masks_offsets[engine->count] * sizeof(sz_u64_t);
    sz_status_t status = sz_fill_rocm_((void *)engine->masks, masks_bytes, 0, stream);
    sz_prefetch_rocm_(engine->memory, engine->memory_bytes, stream);
    sz_prefetch_rocm_(staged, staged_bytes, stream);
    for (sz_size_t first = 0; first < queries->count && status == sz_success_k;
         first += sz_levenshtein_gpu_grid_rows_max_k) {
        sz_levenshtein_engine_t launch_engine = *engine;
        sz_string_view_t const *launch_views = views;
        sz_size_t launch_first = first;
        void *arguments[3];
        arguments[0] = &launch_engine, arguments[1] = &launch_views, arguments[2] = &launch_first;
        dim3 grid, block;
        grid.x = 1, grid.z = 1;
        grid.y = (unsigned)sz_min_of_two(queries->count - first, (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k);
        block.x = sz_levenshtein_masks_threads_rocm_k, block.y = 1, block.z = 1;
        status = sz_launch_rocm_((void const *)sz_levenshtein_masks_simt_kernel_, grid, block, arguments, 0, stream);
    }
    // The staging is the host's, so it outlives the builder only as long as the join below takes.
    if (status == sz_success_k) status = sz_stream_synchronize_rocm_(stream);
    engine->allocator.free(staged, staged_bytes, engine->allocator.handle, stream);
    return status;
}

/** Prepares @p queries on the device the caller already made current, for the tier whose
 *  @p entry_points the rounds will launch, which is every step of
 *  @ref sz_levenshtein_engine_init_scoped_rocm_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_rocm_(sz_levenshtein_engine_t *engine,
                                                                sz_sequence_t const *queries,
                                                                sz_levenshtein_symbol_t symbol,
                                                                sz_levenshtein_entry_points_rocm_t const *entry_points,
                                                                sz_allocator_t *allocator, sz_stream_t stream) {
    sz_allocator_t unified;
    if (!sz_device_multiprocessors_rocm_()) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_rocm_(&unified);
    if (queries->count == 0 || queries->count > 0xFFFFFFFFu) return sz_unexpected_dimensions_k;

    sz_size_t longest = 0, text_bytes = 0, all_text_bytes = 0;
    if (queries->count > (STRINGZILLA_SIZE_MAX / sizeof(sz_string_view_t)) - 1) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (bytes > 0xFFFFFF00u) return sz_unexpected_dimensions_k;
        if (bytes > STRINGZILLA_SIZE_MAX - all_text_bytes) return sz_unexpected_dimensions_k;
        all_text_bytes += bytes;
        longest = sz_max_of_two(longest, bytes);
        if (symbol == sz_levenshtein_bytes_k && bytes > sz_levenshtein_gpu_words_max_k * 64) text_bytes += bytes;
    }
    if (all_text_bytes > STRINGZILLA_SIZE_MAX - queries->count * sizeof(sz_string_view_t))
        return sz_unexpected_dimensions_k;
    sz_size_t const buckets_bound = sz_levenshtein_buckets_rocm_(
        sz_min_of_two(longest / 64 + (longest % 64 != 0), (sz_size_t)sz_levenshtein_gpu_words_max_k));
    sz_size_t const head_bytes = (sz_levenshtein_head_bytes_rocm_(queries->count, buckets_bound) + 7) & ~(sz_size_t)7;
    sz_size_t const offsets_bytes = (queries->count + 1) * sizeof(sz_size_t);
    if (head_bytes > STRINGZILLA_SIZE_MAX - offsets_bytes ||
        text_bytes > STRINGZILLA_SIZE_MAX - head_bytes - offsets_bytes)
        return sz_unexpected_dimensions_k;
    sz_status_t status = sz_levenshtein_engine_build_(queries, symbol, head_bytes + offsets_bytes + text_bytes,
                                                      &unified, stream, engine);
    if (status != sz_success_k) return status;
    engine->capability = entry_points->capability;
    sz_levenshtein_bind_head_rocm_(engine, buckets_bound, entry_points);
    sz_levenshtein_head_rocm_t *const head = (sz_levenshtein_head_rocm_t *)engine->memory;
    head->query_offsets = (sz_size_t *)((sz_ptr_t)engine->memory + head_bytes);
    head->query_text = (sz_ptr_t)head->query_offsets + offsets_bytes;
    for (sz_size_t index = 0, written = 0; index != queries->count; ++index) {
        head->query_offsets[index] = written;
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (symbol == sz_levenshtein_bytes_k && bytes > sz_levenshtein_gpu_words_max_k * 64) {
            sz_copy_serial_(head->query_text + written, queries->get_start(queries->handle, index), bytes);
            written += bytes;
        }
        head->query_offsets[index + 1] = written;
    }

    // The rune planes are the host's: a page table is a scan with no counterpart here, and an init may join
    // where a round may not, so the one crossing is a bulk migration rather than a fault per page.
    if (symbol == sz_levenshtein_runes_k) {
        sz_levenshtein_engine_fill_(engine, queries);
        sz_prefetch_rocm_(engine->memory, engine->memory_bytes, stream);
        return sz_success_k;
    }
    status = sz_levenshtein_build_masks_rocm_(engine, queries, stream);
    if (status != sz_success_k) sz_levenshtein_engine_free_(engine, stream);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_scoped_rocm_(
    sz_levenshtein_engine_t *engine, sz_sequence_t const *queries, sz_levenshtein_symbol_t symbol,
    sz_levenshtein_entry_points_rocm_t const *entry_points, sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_engine_init_rocm_(engine, queries, symbol, entry_points, allocator, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

/** Launches whichever rung's entry point the caller chose, over a grid of candidate tiles
 *  by prepared queries. */
static sz_status_t sz_levenshtein_launch_rocm_(void const *entry_point, sz_size_t blocks, sz_size_t queries,
                                               sz_size_t per_block, sz_stream_t stream, sz_levenshtein_engine_t engine,
                                               sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
                                               sz_size_t distances_stride, sz_size_t const *candidate_order) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)queries, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    // Only the threaded rungs declare the last argument; a launch reads what its kernel declares.
    void *arguments[6];
    arguments[0] = &engine, arguments[1] = &order, arguments[2] = &candidates, arguments[3] = &distances;
    arguments[4] = &distances_stride, arguments[5] = &candidate_order;
    return sz_launch_rocm_(entry_point, grid, block, arguments, 0, stream);
}

/** Whether @p bucket launches a narrow rung, which packs short queries' candidates into lanes and
 *  so never draws through a candidate order. */
static sz_bool_t sz_levenshtein_narrow_rocm_(sz_levenshtein_engine_t const *engine, sz_size_t bucket,
                                             sz_size_t candidates_count) {
    sz_levenshtein_head_rocm_t const *const head = (sz_levenshtein_head_rocm_t const *)engine->memory;
    // A query short enough to leave a word's bits unused shares that word between candidates, which
    // pays once the batch is wide enough to keep the narrower grid resident; the lanes read a byte
    // map, so a rune batch keeps the threaded rung whatever its queries are worth.
    return bucket <= 1 && engine->symbol == sz_levenshtein_bytes_k && candidates_count >= head->candidates_min
               ? sz_true_k
               : sz_false_k;
}

/** Enqueues the longest-first order of @p candidates into @p storage: the cursors of every length
 *  key, then the order itself. */
static sz_status_t sz_levenshtein_order_longest_rocm_(sz_sequence_t candidates, sz_size_t *storage,
                                                      sz_stream_t stream) {
    sz_size_t *candidate_order = storage + sz_levenshtein_length_keys_simt_k;
    // One resident wave of blocks walks the batch, as wider grids would only add tallies to merge.
    sz_size_t const wave = sz_device_multiprocessors_rocm_() * sz_device_threads_per_multiprocessor_rocm_() /
                           sz_levenshtein_length_keys_simt_k;
    dim3 const grid((unsigned)sz_max_of_two(
        (sz_size_t)1,
        sz_min_of_two(sz_size_divide_round_up(candidates.count, sz_levenshtein_length_keys_simt_k), wave))),
        block(sz_levenshtein_length_keys_simt_k);
    void *counts_arguments[] = {&candidates, &storage};
    void *offsets_arguments[] = {&storage};
    void *scatter_arguments[] = {&candidates, &storage, &candidate_order};
    sz_status_t status = sz_fill_rocm_(storage, sz_levenshtein_length_keys_simt_k * sizeof(sz_size_t), 0, stream);
    if (status == sz_success_k)
        status = sz_launch_rocm_((void const *)sz_levenshtein_length_counts_simt_kernel_, grid, block, counts_arguments,
                                 0, stream);
    if (status == sz_success_k)
        status = sz_launch_rocm_((void const *)sz_levenshtein_length_offsets_simt_kernel_, dim3(1), block,
                                 offsets_arguments, 0, stream);
    if (status == sz_success_k)
        status = sz_launch_rocm_((void const *)sz_levenshtein_length_scatter_simt_kernel_, grid, block,
                                 scatter_arguments, 0, stream);
    return status;
}

/** Enqueues candidate ordering and bucket cross-products into per-call workspace. */
static sz_status_t sz_levenshtein_enqueue_long_rocm_(sz_levenshtein_engine_t engine, sz_sequence_t candidates,
                                                     sz_levenshtein_length_bucket_t const *candidate_buckets,
                                                     sz_size_t resident, sz_bool_t cooperative, sz_size_t *distances,
                                                     sz_size_t distances_stride, sz_ptr_t workspace,
                                                     sz_size_t *candidate_storage, sz_stream_t stream) {
    sz_levenshtein_head_rocm_t const *const head = (sz_levenshtein_head_rocm_t const *)engine.memory;
    sz_size_t *candidate_order = STRINGZILLA_NULL;
    sz_status_t status = sz_success_k;
    if (candidate_storage) {
        candidate_order = candidate_storage + sz_levenshtein_length_buckets_k;
        status = sz_fill_rocm_(candidate_storage, sz_levenshtein_length_buckets_k * sizeof(sz_size_t), 0, stream);
        if (status != sz_success_k) return status;
        sz_levenshtein_candidate_buckets_simt_t buckets;
        for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket)
            buckets.offsets[bucket] = candidate_buckets[bucket].offset;
        void *arguments[] = {&candidates, &buckets, &candidate_storage, &candidate_order};
        sz_size_t const blocks = sz_min_of_two(sz_size_divide_round_up(candidates.count, 128), resident * 4);
        status = sz_launch_rocm_((void const *)sz_levenshtein_order_simt_kernel_, dim3((unsigned)blocks, 1, 1),
                                 dim3(128, 1, 1), arguments, 0, stream);
        if (status != sz_success_k) return status;
    }
    sz_levenshtein_long_arguments_simt_t batch = {};
    batch.distances_stride = distances_stride;
    batch.query_order = head->order, batch.candidate_order = candidate_order;
    batch.query_offsets = head->query_offsets, batch.query_text = head->query_text;
    for (sz_size_t query_bucket = 0; query_bucket != sz_levenshtein_length_buckets_k; ++query_bucket) {
        sz_levenshtein_length_bucket_t const queries = head->long_buckets[query_bucket];
        if (!queries.count) continue;
        for (sz_size_t candidate_bucket = 0; candidate_bucket != sz_levenshtein_length_buckets_k; ++candidate_bucket) {
            sz_levenshtein_length_bucket_t const group = candidate_buckets[candidate_bucket];
            if (!group.count) continue;
            sz_levenshtein_long_layout_simt_t const layout = sz_levenshtein_long_layout_simt_(engine.symbol, queries,
                                                                                              group, resident);
            batch.scratch_stride = layout.scratch_stride, batch.frontier_cells = layout.frontier_cells;
            sz_size_t const query_chunk = sz_min_of_two(queries.count, layout.pairs);
            for (sz_size_t query = 0; query < queries.count;) {
                sz_size_t const query_count = sz_min_of_two(queries.count - query, query_chunk);
                batch.query_first = queries.offset + query;
                for (sz_size_t candidate = 0; candidate < group.count;) {
                    batch.candidate_first = group.offset + candidate;
                    batch.candidate_count = sz_min_of_two(group.count - candidate, layout.pairs / query_count);
                    batch.pair_count = query_count * batch.candidate_count;
                    void *arguments[] = {&engine, &candidates, &distances, &workspace, &batch};
                    if (layout.tiled) {
                        status = sz_fill_rocm_(workspace, layout.scratch_stride * batch.pair_count, 0, stream);
                        if (status != sz_success_k) return status;
                        sz_size_t const wanted = sz_size_divide_round_up(layout.progress_cells,
                                                                         sz_levenshtein_tiled_warps_per_block_simt_k);
                        sz_size_t const blocks = cooperative && batch.pair_count < resident
                                                     ? sz_min_of_two(wanted, resident / batch.pair_count)
                                                     : 1;
                        dim3 const grid((unsigned)blocks, (unsigned)batch.pair_count, 1);
                        dim3 const block(sz_levenshtein_tiled_threads_per_block_rocm_k, 1, 1);
                        if (blocks > 1) {
                            // Cooperative admission keeps every producer resident.
                            if (hipLaunchCooperativeKernel((void const *)sz_levenshtein_tiled_batch_rocm_kernel_, grid,
                                                           block, arguments, 0, (hipStream_t)stream) != hipSuccess)
                                return sz_device_code_mismatch_k;
                        }
                        else
                            status = sz_launch_rocm_((void const *)sz_levenshtein_tiled_batch_rocm_kernel_, grid, block,
                                                     arguments, 0, stream);
                    }
                    else {
                        sz_size_t const blocks = sz_size_divide_round_up(batch.pair_count, 128);
                        status = sz_launch_rocm_((void const *)sz_levenshtein_long_simt_kernel_,
                                                 dim3((unsigned)blocks, 1, 1), dim3(128, 1, 1), arguments, 0, stream);
                    }
                    if (status != sz_success_k) return status;
                    candidate += batch.candidate_count;
                }
                query += query_count;
            }
        }
    }
    return sz_success_k;
}

static sz_status_t sz_levenshtein_long_rocm_(sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates,
                                             sz_size_t *distances, sz_size_t distances_stride, sz_stream_t stream) {
    sz_levenshtein_head_rocm_t const *const head = (sz_levenshtein_head_rocm_t const *)engine->memory;
    if (head->bucket_offsets[head->buckets] == engine->count) return sz_success_k;
    sz_levenshtein_length_bucket_t candidate_buckets[sz_levenshtein_length_buckets_k] = {{0, 0, 0}};
    hipPointerAttribute_t attributes;
    sz_bool_t const host_readable = hipPointerGetAttributes(&attributes, candidates->handle) == hipSuccess &&
                                            attributes.type == hipMemoryTypeManaged
                                        ? sz_true_k
                                        : sz_false_k;
    sz_size_t populated = 0;
    if (host_readable) {
        for (sz_size_t index = 0; index != candidates->count; ++index) {
            sz_size_t const length = candidates->get_length(candidates->handle, index);
            sz_levenshtein_length_bucket_t *const bucket = candidate_buckets + sz_levenshtein_length_bucket_(length);
            ++bucket->count;
            bucket->length_max = sz_max_of_two(bucket->length_max, length);
        }
        for (sz_size_t bucket = 0, offset = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket) {
            candidate_buckets[bucket].offset = offset;
            offset += candidate_buckets[bucket].count;
            populated += candidate_buckets[bucket].count != 0;
        }
    }
    else {
        // Device-only offsets keep the generic kernel to avoid a synchronous metadata readback.
        candidate_buckets[0].count = candidates->count;
        populated = 1;
    }

    sz_bool_t has_tiled = sz_false_k;
    if (engine->symbol == sz_levenshtein_bytes_k &&
        sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_gpu_words_max_k)
        for (sz_size_t bucket = 0; bucket != sz_levenshtein_length_buckets_k; ++bucket)
            if (candidate_buckets[bucket].length_max > sz_levenshtein_gpu_words_max_k * 64 &&
                candidate_buckets[bucket].length_max <= 0xFFFFFF00u)
                has_tiled = sz_true_k;
    sz_size_t const multiprocessors = sz_device_multiprocessors_rocm_();
    void const *const entry_point = has_tiled ? (void const *)sz_levenshtein_tiled_batch_rocm_kernel_
                                              : (void const *)sz_levenshtein_long_simt_kernel_;
    sz_size_t const per_block = has_tiled ? sz_levenshtein_tiled_threads_per_block_rocm_k : 128;
    sz_size_t const resident = multiprocessors * sz_resident_blocks_rocm_(entry_point, per_block, 0);
    if (!resident) return sz_device_code_mismatch_k;
    sz_bool_t const cooperative = has_tiled && sz_device_attribute_rocm_(hipDeviceAttributeCooperativeLaunch)
                                      ? sz_true_k
                                      : sz_false_k;
    sz_size_t scratch_bytes = sizeof(sz_size_t);
    for (sz_size_t query = 0; query != sz_levenshtein_length_buckets_k; ++query) {
        if (!head->long_buckets[query].count) continue;
        for (sz_size_t candidate = 0; candidate != sz_levenshtein_length_buckets_k; ++candidate) {
            if (!candidate_buckets[candidate].count) continue;
            sz_levenshtein_long_layout_simt_t const layout = sz_levenshtein_long_layout_simt_(
                engine->symbol, head->long_buckets[query], candidate_buckets[candidate], resident);
            scratch_bytes = sz_max_of_two(scratch_bytes, layout.scratch_stride * layout.pairs);
        }
    }
    sz_size_t candidate_bytes = 0;
    if (populated > 1) {
        if (candidates->count > STRINGZILLA_SIZE_MAX / sizeof(sz_size_t) - sz_levenshtein_length_buckets_k)
            return sz_unexpected_dimensions_k;
        candidate_bytes = (candidates->count + sz_levenshtein_length_buckets_k) * sizeof(sz_size_t);
    }
    if (candidate_bytes > STRINGZILLA_SIZE_MAX - scratch_bytes) return sz_unexpected_dimensions_k;
    sz_ptr_t workspace = STRINGZILLA_NULL;
    if (hipMallocAsync((void **)&workspace, scratch_bytes + candidate_bytes, (hipStream_t)stream) != hipSuccess)
        return sz_bad_alloc_k;
    sz_size_t *const candidate_storage = candidate_bytes ? (sz_size_t *)(workspace + scratch_bytes) : STRINGZILLA_NULL;
    sz_status_t status = sz_levenshtein_enqueue_long_rocm_(*engine, *candidates, candidate_buckets, resident,
                                                           cooperative, distances, distances_stride, workspace,
                                                           candidate_storage, stream);
    if (hipFreeAsync(workspace, (hipStream_t)stream) != hipSuccess && status == sz_success_k)
        status = sz_device_code_mismatch_k;
    return status;
}

/** Launches every short-query bucket's rung over @p candidates, the threaded rungs drawing them in
 *  @p candidate_order, or in the batch's own order where it is null. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_rounds_rocm_(sz_levenshtein_engine_t *engine,
                                                           sz_sequence_t const *candidates, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_size_t const *candidate_order,
                                                           sz_levenshtein_entry_points_rocm_t const *entry_points,
                                                           sz_stream_t stream) {
    sz_levenshtein_head_rocm_t const *const head = (sz_levenshtein_head_rocm_t const *)engine->memory;
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        sz_size_t const first = head->bucket_offsets[bucket], last = head->bucket_offsets[bucket + 1];
        if (first == last) continue;

        sz_bool_t const narrow = sz_levenshtein_narrow_rocm_(engine, bucket, candidates->count);
        void const *entry_point = sz_levenshtein_entry_point_rocm_(entry_points, engine->symbol, bucket);
        sz_size_t per_block = head->per_block[bucket];
        sz_size_t const rounds = sz_levenshtein_tile_rounds_rocm_k;
        sz_size_t candidates_per_block = per_block * rounds;
        if (narrow) {
            entry_point = bucket == 0 ? (void const *)sz_levenshtein_distances_u8x4_simt_kernel_
                                      : (void const *)sz_levenshtein_distances_u16x2_simt_kernel_;
            per_block = head->narrow_per_block[bucket];
            candidates_per_block = per_block * rounds *
                                   sz_levenshtein_lanes_per_thread_rocm_(
                                       bucket == 0 ? sz_levenshtein_byte_lanes_symbols_max_rocm_k
                                                   : sz_levenshtein_short_lanes_symbols_max_rocm_k);
        }
        else if (sz_levenshtein_bucket_words_rocm_(bucket) >= sz_levenshtein_gpu_warp_words_min_k)
            candidates_per_block = per_block / sz_levenshtein_gpu_warp_lanes_k * rounds;
        sz_size_t const blocks = sz_size_divide_round_up(candidates->count, candidates_per_block);

        for (sz_size_t row = first; row < last; row += sz_levenshtein_gpu_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(last - row, (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k);
            sz_status_t const launched = sz_levenshtein_launch_rocm_(entry_point, blocks, rows, per_block, stream,
                                                                     *engine, head->order + row, *candidates, distances,
                                                                     distances_stride, candidate_order);
            if (launched != sz_success_k) return launched;
        }
    }
    return sz_success_k;
}

/** Enqueues one round over the tier's @p entry_points on the device the caller already made
 *  current, which is every step of @ref sz_levenshtein_distances_scoped_rocm_ without its scope:
 *  the longest-first order where a threaded rung will draw through it, the short-query rungs,
 *  then the long queries. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_rocm_(sz_levenshtein_engine_t *engine,
                                                              sz_sequence_t const *candidates, sz_size_t *distances,
                                                              sz_size_t distances_stride,
                                                              sz_levenshtein_entry_points_rocm_t const *entry_points,
                                                              sz_stream_t stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (candidates->count == 0) return sz_success_k;
    if (engine->count > 1 && distances_stride > (STRINGZILLA_SIZE_MAX - candidates->count) / (engine->count - 1))
        return sz_unexpected_dimensions_k;
    if ((engine->count - 1) * distances_stride + candidates->count > STRINGZILLA_SIZE_MAX / sizeof(sz_size_t))
        return sz_unexpected_dimensions_k;

    if (!sz_memory_accessible_rocm_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_rocm_(distances)) return sz_device_memory_mismatch_k;
    if (candidates->get_start != sz_sequence_tape_start || candidates->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_rocm_(candidates->handle))
        return sz_device_memory_mismatch_k;

    sz_levenshtein_head_rocm_t const *const head = (sz_levenshtein_head_rocm_t const *)engine->memory;
    sz_bool_t threaded = sz_false_k;
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket)
        if (head->bucket_offsets[bucket] != head->bucket_offsets[bucket + 1] &&
            sz_levenshtein_bucket_words_rocm_(bucket) < sz_levenshtein_gpu_warp_words_min_k &&
            !sz_levenshtein_narrow_rocm_(engine, bucket, candidates->count))
            threaded = sz_true_k;

    sz_status_t status = sz_success_k;
    sz_size_t *storage = STRINGZILLA_NULL, *candidate_order = STRINGZILLA_NULL;
    if (threaded && candidates->count > 1) {
        if (candidates->count > STRINGZILLA_SIZE_MAX / sizeof(sz_size_t) - sz_levenshtein_length_keys_simt_k)
            return sz_unexpected_dimensions_k;
        sz_size_t const storage_bytes = (sz_levenshtein_length_keys_simt_k + candidates->count) * sizeof(sz_size_t);
        if (hipMallocAsync((void **)&storage, storage_bytes, (hipStream_t)stream) != hipSuccess) return sz_bad_alloc_k;
        status = sz_levenshtein_order_longest_rocm_(*candidates, storage, stream);
        candidate_order = storage + sz_levenshtein_length_keys_simt_k;
    }
    if (status == sz_success_k)
        status = sz_levenshtein_rounds_rocm_(engine, candidates, distances, distances_stride, candidate_order,
                                             entry_points, stream);
    if (storage && hipFreeAsync(storage, (hipStream_t)stream) != hipSuccess && status == sz_success_k)
        status = sz_device_code_mismatch_k;
    if (status != sz_success_k) return status;
    return sz_levenshtein_long_rocm_(engine, candidates, distances, distances_stride, stream);
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_scoped_rocm_(
    sz_levenshtein_engine_t *engine, sz_sequence_t const *candidates, sz_size_t *distances, sz_size_t distances_stride,
    sz_levenshtein_entry_points_rocm_t const *entry_points, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_distances_rocm_(engine, candidates, distances, distances_stride, entry_points, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

#pragma endregion Myers Engine

#if STRINGZILLA_TARGET_ROCM

/** The ROCm tier's entry points. */
static sz_levenshtein_entry_points_rocm_t const sz_levenshtein_entry_points_rocm_ = {
    {
        (void const *)sz_levenshtein_distances_w1_simt_kernel_,
        (void const *)sz_levenshtein_distances_w2_simt_kernel_,
        (void const *)sz_levenshtein_distances_w3_simt_kernel_,
        (void const *)sz_levenshtein_distances_w4_simt_kernel_,
        (void const *)sz_levenshtein_distances_w5_simt_kernel_,
        (void const *)sz_levenshtein_distances_w6_simt_kernel_,
        (void const *)sz_levenshtein_distances_w7_simt_kernel_,
        (void const *)sz_levenshtein_distances_w8_simt_kernel_,
        (void const *)sz_levenshtein_distances_w9_simt_kernel_,
        (void const *)sz_levenshtein_distances_w10_simt_kernel_,
        (void const *)sz_levenshtein_distances_w11_simt_kernel_,
        (void const *)sz_levenshtein_distances_w12_simt_kernel_,
        (void const *)sz_levenshtein_distances_w13_simt_kernel_,
        (void const *)sz_levenshtein_distances_w14_simt_kernel_,
        (void const *)sz_levenshtein_distances_w15_simt_kernel_,
        (void const *)sz_levenshtein_distances_w16_simt_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_w1_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w2_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w3_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w4_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w5_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w6_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w7_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w8_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w9_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w10_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w11_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w12_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w13_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w14_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w15_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w16_simt_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_k1_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k2_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k3_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k4_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k5_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k6_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k7_rocm_kernel_,
        (void const *)sz_levenshtein_distances_k8_rocm_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_k1_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k2_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k3_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k4_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k5_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k6_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k7_rocm_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k8_rocm_kernel_,
    },
    sz_cap_rocm_k,
};

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_rocm(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_rocm_(engine, queries, symbol, &sz_levenshtein_entry_points_rocm_,
                                                   allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_rocm(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream) {
    return sz_levenshtein_distances_scoped_rocm_(engine, candidates, distances, distances_stride,
                                                 &sz_levenshtein_entry_points_rocm_, stream);
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_LEVENSHTEIN_ROCM_CUH_
