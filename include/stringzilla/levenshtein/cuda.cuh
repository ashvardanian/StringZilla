/**
 *  @file include/stringzilla/levenshtein/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of Levenshtein distances: the rung buckets, the launch geometry and the
 *      @c _cuda exports, over the kernels of `levenshtein/simt.cuh`.
 *
 *  A batch is bucketed by rung key at @ref sz_levenshtein_engine_init_scoped_cuda_, so one launch
 *  carries only queries that share an entry point, and the occupancy walk each of those entry
 *  points needs is paid there rather than once per round. The byte planes are built by a kernel
 *  from the queries staged into device-reachable memory; the rune planes are built on the host, an
 *  init being allowed to join where a round is not.
 *
 *  On Blackwell a block done with its tile takes over a block not yet started, so the grid is as
 *  wide as the batch and a tile holds one round; elsewhere a tile holds several.
 *
 *  @sa include/stringzilla/levenshtein/simt.cuh
 *  @sa include/stringzilla/levenshtein/rocm.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_CUDA_CUH_
#define STRINGZILLA_LEVENSHTEIN_CUDA_CUH_

#include "stringzilla/cuda.cuh"
#include "stringzilla/levenshtein/simt.cuh"

#if STRINGZILLA_TARGET_CUDA && defined(__CUDACC__) && !defined(__HIP__)

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Entry Points

/** Threads a block runs when the device cannot be asked; the register budget differs per word
 *  count, so the launcher takes what the occupancy calculator answers for the entry point it is
 *  about to launch instead. */
enum { sz_levenshtein_candidates_per_block_cuda_k = 128 };

/** Widest block the launcher considers: past this a block schedules too coarsely for
 *  what residency returns. */
enum { sz_levenshtein_candidates_per_block_max_cuda_k = 256 };

/** Rounds of candidates one block's tile holds where its blocks cannot take over unstarted ones.
 *  Measured on one wave of XLSum lines, four rounds at a quarter of the blocks beat one, two and
 *  eight, the balance being worth more than the residency it costs. */
enum { sz_levenshtein_tile_rounds_cuda_k = 4 };

/** Query symbols one byte lane of a Myers word holds, which is the longest query the byte rung
 *  takes, and one sixteen-bit lane holds, which is the longest query either narrow rung takes. */
enum {
    sz_levenshtein_byte_lanes_symbols_max_cuda_k = 8,
    sz_levenshtein_short_lanes_symbols_max_cuda_k = 16,
};

/** Words one lane of a warped candidate owns, which is what selects the rung's entry point. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_warp_words_per_lane_cuda_(sz_size_t words) {
    return sz_size_divide_round_up(words, sz_levenshtein_gpu_warp_lanes_k);
}

/*  One entry point per word count, addressed by it. @c cudaLaunchKernel takes the host-side
 *  symbol of a @c __global__, so the table is what a chevron launch would have selected,
 *  spelled as data. */
static void const *const sz_levenshtein_entry_points_cuda_[sz_levenshtein_thread_words_max_simt_k] = {
    (void const *)sz_levenshtein_distances_w1_simt_kernel_,  (void const *)sz_levenshtein_distances_w2_simt_kernel_,
    (void const *)sz_levenshtein_distances_w3_simt_kernel_,  (void const *)sz_levenshtein_distances_w4_simt_kernel_,
    (void const *)sz_levenshtein_distances_w5_simt_kernel_,  (void const *)sz_levenshtein_distances_w6_simt_kernel_,
    (void const *)sz_levenshtein_distances_w7_simt_kernel_,  (void const *)sz_levenshtein_distances_w8_simt_kernel_,
    (void const *)sz_levenshtein_distances_w9_simt_kernel_,  (void const *)sz_levenshtein_distances_w10_simt_kernel_,
    (void const *)sz_levenshtein_distances_w11_simt_kernel_, (void const *)sz_levenshtein_distances_w12_simt_kernel_,
    (void const *)sz_levenshtein_distances_w13_simt_kernel_, (void const *)sz_levenshtein_distances_w14_simt_kernel_,
    (void const *)sz_levenshtein_distances_w15_simt_kernel_, (void const *)sz_levenshtein_distances_w16_simt_kernel_,
};

/*  One entry point per words-per-lane, addressed by it, beside the threaded rung's own table. */
static void const *const sz_levenshtein_entry_points_warp_cuda_[sz_levenshtein_gpu_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_distances_k1_simt_kernel_, (void const *)sz_levenshtein_distances_k2_simt_kernel_,
    (void const *)sz_levenshtein_distances_k3_simt_kernel_, (void const *)sz_levenshtein_distances_k4_simt_kernel_,
    (void const *)sz_levenshtein_distances_k5_simt_kernel_, (void const *)sz_levenshtein_distances_k6_simt_kernel_,
    (void const *)sz_levenshtein_distances_k7_simt_kernel_, (void const *)sz_levenshtein_distances_k8_simt_kernel_,
};

/*  One rune entry point per word count, addressed by it, beside the byte tier's own table. */
static void const *const sz_levenshtein_entry_points_utf8_cuda_[sz_levenshtein_thread_words_max_simt_k] = {
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
};

/*  One warped rune entry point per words-per-lane, addressed by it, beside the threaded rune
 *  tier's own table. */
static void const *const sz_levenshtein_entry_points_utf8_warp_cuda_[sz_levenshtein_gpu_warp_words_per_lane_max_k] = {
    (void const *)sz_levenshtein_distances_utf8_k1_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k2_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k3_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k4_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k5_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k6_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k7_simt_kernel_,
    (void const *)sz_levenshtein_distances_utf8_k8_simt_kernel_,
};

/**
 *  @brief Threads a sweep gives a block on this device, for the entry point it is about to launch.
 *
 *  Register pressure moves the residency ceiling from one word count to the next, so the block
 *  landing the most warps per multiprocessor is the device's answer and not a constant's. The walk
 *  stops at @c sz_levenshtein_candidates_per_block_max_cuda_k rather than at what the entry point
 *  allows, since past it a block schedules too coarsely for what its residency returns; ties below
 *  it go to the wider block.
 */
static sz_size_t sz_levenshtein_per_block_cuda_(void const *entry_point) {
    return sz_block_size_cuda_(entry_point, 0, sz_levenshtein_candidates_per_block_max_cuda_k,
                               sz_levenshtein_candidates_per_block_cuda_k);
}

/** Candidates one thread takes, which is the lanes a query of @p length symbols leaves
 *  in one register. */
static sz_size_t sz_levenshtein_lanes_per_thread_cuda_(sz_size_t length) {
    return length <= sz_levenshtein_byte_lanes_symbols_max_cuda_k ? 4 : 2;
}

/** Candidates a batch must carry before a narrow rung is launched at all, which the device's own
 *  residency scales. @c STRINGZILLA_SIZE_MAX where the device cannot be asked, so the threaded rung
 *  keeps every batch. */
static sz_size_t sz_levenshtein_lanes_candidates_min_cuda_(void) {
    sz_size_t const resident_threads = sz_device_multiprocessors_cuda_() * sz_device_threads_per_multiprocessor_cuda_();
    return resident_threads ? resident_threads * sz_levenshtein_gpu_lanes_waves_min_k : STRINGZILLA_SIZE_MAX;
}

#pragma endregion Entry Points

#pragma region Myers Engine

/** What a device round needs that the batch already fixed: its rung buckets and the geometry each
 *  of them takes. */
typedef struct sz_levenshtein_head_cuda_t {

    /** Rung keys the batch spans: @c words_max plus the two narrow ones. */
    sz_size_t buckets;

    /** Candidates before a narrow rung is launched, asked of the device once. */
    sz_size_t candidates_min;

    /** Whether the wide rungs' blocks take over unstarted ones, so a tile holds one round. */
    sz_bool_t steals;

    /** Threads the byte-lane and the short-lane entry points take. */
    sz_size_t narrow_per_block[2];

    /** The @b [buckets+1] first position of each key inside @c order. */
    sz_size_t *bucket_offsets;

    /** The @b [buckets] threads that key's own wide entry point takes. */
    sz_size_t *per_block;

    /** The @b [count] query indices, the keys' runs back to back. */
    sz_u32_t *order;
} sz_levenshtein_head_cuda_t;

/** The rung key a query of @p length symbols launches from: the two narrow lanes, then one
 *  key per word. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_bucket_cuda_(sz_size_t length) {
    if (length <= sz_levenshtein_byte_lanes_symbols_max_cuda_k) return 0;
    if (length <= sz_levenshtein_short_lanes_symbols_max_cuda_k) return 1;
    return 1 + sz_levenshtein_query_words(length);
}

/** Rung keys a batch whose widest query spans @p words words can reach. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_buckets_cuda_(sz_size_t words) { return words + 2; }

/** Query words the wide rung of @p bucket steps, the two narrow keys sharing the
 *  one-word entry points. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_bucket_words_cuda_(sz_size_t bucket) {
    return bucket <= 1 ? 1 : bucket - 1;
}

/** Bytes the tier-private head takes ahead of a batch of @p count queries spanning
 *  @p buckets rung keys. */
STRINGZILLA_CONSTEXPR sz_size_t sz_levenshtein_head_bytes_cuda_(sz_size_t count, sz_size_t buckets) {
    return sizeof(sz_levenshtein_head_cuda_t) + (2 * buckets + 1) * sizeof(sz_size_t) + count * sizeof(sz_u32_t);
}

/**
 *  @brief The entry point @p bucket 's wide rung reaches: one candidate per thread, or one per warp
 *      past the crossing.
 *
 *  A narrow query's verticals fit one thread's registers, and nothing beats keeping the recurrence
 *  there. A wider one spreads them across a warp, where the skew turns the carry between words
 *  into one shuffle.
 */
static void const *sz_levenshtein_entry_point_cuda_(sz_levenshtein_symbol_t symbol, sz_size_t bucket) {
    sz_size_t const words = sz_levenshtein_bucket_words_cuda_(bucket);
    if (words < sz_levenshtein_gpu_warp_words_min_k)
        return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_entry_points_cuda_[words - 1]
                                                : sz_levenshtein_entry_points_utf8_cuda_[words - 1];
    sz_size_t const per_lane = sz_levenshtein_warp_words_per_lane_cuda_(words);
    return symbol == sz_levenshtein_bytes_k ? sz_levenshtein_entry_points_warp_cuda_[per_lane - 1]
                                            : sz_levenshtein_entry_points_utf8_warp_cuda_[per_lane - 1];
}

/** Points the head's tables into the block, buckets the batch by rung key, and asks the device
 *  its geometry once. */
static void sz_levenshtein_bind_head_cuda_(sz_levenshtein_engine_t *engine, sz_size_t buckets_bound) {
    sz_levenshtein_head_cuda_t *const head = (sz_levenshtein_head_cuda_t *)engine->memory;
    sz_size_t *const tables = (sz_size_t *)((sz_ptr_t)engine->memory + sizeof(sz_levenshtein_head_cuda_t));
    sz_size_t cursors[sz_levenshtein_gpu_words_max_k + 2];
    head->buckets = sz_levenshtein_buckets_cuda_(sz_levenshtein_engine_words_max_(engine));
    head->bucket_offsets = tables;
    head->per_block = tables + buckets_bound + 1;
    head->order = (sz_u32_t *)(tables + 2 * buckets_bound + 1);

    // A counting sort by rung key, so one launch only ever carries queries that share an entry point.
    for (sz_size_t bucket = 0; bucket != head->buckets + 1; ++bucket) head->bucket_offsets[bucket] = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        ++head->bucket_offsets[sz_levenshtein_bucket_cuda_(engine->lengths[index]) + 1];
    for (sz_size_t bucket = 1; bucket != head->buckets + 1; ++bucket)
        head->bucket_offsets[bucket] += head->bucket_offsets[bucket - 1];
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) cursors[bucket] = head->bucket_offsets[bucket];
    for (sz_size_t index = 0; index != engine->count; ++index)
        head->order[cursors[sz_levenshtein_bucket_cuda_(engine->lengths[index])]++] = (sz_u32_t)index;

    // Every driver round trip a round would otherwise pay: the residency floor once, and one occupancy walk
    // per entry point the batch can reach, which the buckets fixed here and no later call can widen.
    head->candidates_min = sz_levenshtein_lanes_candidates_min_cuda_();
    head->steals = sz_kernel_steals_cuda_((void const *)sz_levenshtein_distances_w1_simt_kernel_);
    head->narrow_per_block[0] = sz_levenshtein_per_block_cuda_(
        (void const *)sz_levenshtein_distances_u8x4_simt_kernel_);
    head->narrow_per_block[1] = sz_levenshtein_per_block_cuda_(
        (void const *)sz_levenshtein_distances_u16x2_simt_kernel_);
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket)
        head->per_block[bucket] = head->bucket_offsets[bucket] == head->bucket_offsets[bucket + 1]
                                      ? 0
                                      : sz_levenshtein_per_block_cuda_(
                                            sz_levenshtein_entry_point_cuda_(engine->symbol, bucket));
}

/** Threads the mask builder runs: one per byte value, so a thread owns one class flag for
 *  the whole build. */
enum { sz_levenshtein_masks_threads_cuda_k = sz_levenshtein_byte_classes_k };

/**
 *  @brief Stages the batch's query texts where the device reads them and builds every
 *      byte plane there.
 *  @note Joins @p stream, which is what lets the staging be released; only an init is allowed to.
 */
static sz_status_t sz_levenshtein_build_masks_cuda_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
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
    sz_status_t status = sz_fill_cuda_((void *)engine->masks, masks_bytes, 0, stream);
    sz_prefetch_cuda_(engine->memory, engine->memory_bytes, stream);
    sz_prefetch_cuda_(staged, staged_bytes, stream);
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
        block.x = sz_levenshtein_masks_threads_cuda_k, block.y = 1, block.z = 1;
        status = sz_launch_cuda_((void const *)sz_levenshtein_masks_simt_kernel_, grid, block, arguments, 0, stream);
    }
    // The staging is the host's, so it outlives the builder only as long as the join below takes.
    if (status == sz_success_k) status = sz_stream_synchronize_cuda_(stream);
    engine->allocator.free(staged, staged_bytes, engine->allocator.handle, stream);
    return status;
}

/** Prepares @p queries on the device the caller already made current, which is every step of
 *  @ref sz_levenshtein_engine_init_scoped_cuda_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_cuda_(sz_levenshtein_engine_t *engine,
                                                                sz_sequence_t const *queries,
                                                                sz_levenshtein_symbol_t symbol,
                                                                sz_allocator_t *allocator, sz_stream_t stream) {
    sz_allocator_t unified;
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_cuda_(&unified);
    if (queries->count == 0) return sz_unexpected_dimensions_k;

    // Every query seeds its score from its own last word, so an empty one has no word to read it off, and the
    // rung ceiling is checked twice: on the bytes here, which bound the runes, and on the symbols once measured.
    sz_size_t longest = 0;
    for (sz_size_t index = 0; index != queries->count; ++index) {
        sz_size_t const bytes = queries->get_length(queries->handle, index);
        if (bytes == 0) return sz_unexpected_dimensions_k;
        longest = sz_max_of_two(longest, bytes);
    }
    if (symbol == sz_levenshtein_bytes_k && longest > sz_levenshtein_gpu_words_max_k * 64)
        return sz_unexpected_dimensions_k;

    sz_size_t const buckets_bound = sz_levenshtein_buckets_cuda_(sz_size_divide_round_up(longest, 64));
    sz_size_t const head_bytes = sz_levenshtein_head_bytes_cuda_(queries->count, buckets_bound);
    sz_status_t status = sz_levenshtein_engine_build_(queries, symbol, head_bytes, &unified, stream, engine);
    if (status != sz_success_k) return status;
    if (sz_levenshtein_engine_words_max_(engine) > sz_levenshtein_gpu_words_max_k) {
        sz_levenshtein_engine_free_(engine, stream);
        return sz_unexpected_dimensions_k;
    }
    engine->capability = sz_cap_cuda_k;
    sz_levenshtein_bind_head_cuda_(engine, buckets_bound);

    // The rune planes are the host's: a page table is a scan with no counterpart here, and an init may join
    // where a round may not, so the one crossing is a bulk migration rather than a fault per page.
    if (symbol == sz_levenshtein_runes_k) {
        sz_levenshtein_engine_fill_(engine, queries);
        sz_prefetch_cuda_(engine->memory, engine->memory_bytes, stream);
        return sz_success_k;
    }
    status = sz_levenshtein_build_masks_cuda_(engine, queries, stream);
    if (status != sz_success_k) sz_levenshtein_engine_free_(engine, stream);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_engine_init_scoped_cuda_(sz_levenshtein_engine_t *engine,
                                                                       sz_sequence_t const *queries,
                                                                       sz_levenshtein_symbol_t symbol,
                                                                       sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_engine_init_cuda_(engine, queries, symbol, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

/** Launches whichever rung's entry point the caller chose, over a grid of candidate tiles
 *  by prepared queries. */
static sz_status_t sz_levenshtein_launch_cuda_(void const *entry_point, sz_size_t blocks, sz_size_t queries,
                                               sz_size_t per_block, sz_stream_t stream, sz_levenshtein_engine_t engine,
                                               sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
                                               sz_size_t distances_stride) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = (unsigned)queries, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[5];
    arguments[0] = &engine, arguments[1] = &order, arguments[2] = &candidates, arguments[3] = &distances;
    arguments[4] = &distances_stride;
    return sz_launch_cuda_(entry_point, grid, block, arguments, 0, stream);
}

/** Enqueues one round on the device the caller already made current, which is every step of
 *  @ref sz_levenshtein_distances_scoped_cuda_ but the device scope. */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_cuda_(sz_levenshtein_engine_t *engine,
                                                              sz_sequence_t const *candidates, sz_size_t *distances,
                                                              sz_size_t distances_stride, sz_stream_t stream) {
    if (distances_stride < candidates->count) return sz_unexpected_dimensions_k;
    if (candidates->count == 0) return sz_success_k;

    if (!sz_memory_accessible_cuda_(engine->memory)) return sz_device_memory_mismatch_k;
    if (!sz_memory_accessible_cuda_(distances)) return sz_device_memory_mismatch_k;
    if (candidates->get_start != sz_sequence_tape_start || candidates->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_cuda_(candidates->handle))
        return sz_device_memory_mismatch_k;

    sz_levenshtein_head_cuda_t const *const head = (sz_levenshtein_head_cuda_t const *)engine->memory;
    for (sz_size_t bucket = 0; bucket != head->buckets; ++bucket) {
        sz_size_t const first = head->bucket_offsets[bucket], last = head->bucket_offsets[bucket + 1];
        if (first == last) continue;

        // A query short enough to leave a word's bits unused shares that word between candidates, which pays
        // once the batch is wide enough to keep the narrower grid resident; the lanes read a byte map, so a
        // rune batch keeps the threaded rung whatever its queries are worth.
        sz_bool_t const narrow = bucket <= 1 && engine->symbol == sz_levenshtein_bytes_k &&
                                         candidates->count >= head->candidates_min
                                     ? sz_true_k
                                     : sz_false_k;
        void const *entry_point = sz_levenshtein_entry_point_cuda_(engine->symbol, bucket);
        sz_size_t per_block = head->per_block[bucket];
        // A block that takes over unstarted ones needs no more than one round of its own, and the
        // grid is then as wide as the batch; elsewhere a tile holds several rounds for its block's
        // threads to share.
        sz_size_t const rounds = head->steals && !narrow ? 1 : sz_levenshtein_tile_rounds_cuda_k;
        sz_size_t candidates_per_block = per_block * rounds;
        if (narrow) {
            entry_point = bucket == 0 ? (void const *)sz_levenshtein_distances_u8x4_simt_kernel_
                                      : (void const *)sz_levenshtein_distances_u16x2_simt_kernel_;
            per_block = head->narrow_per_block[bucket];
            candidates_per_block = per_block * rounds *
                                   sz_levenshtein_lanes_per_thread_cuda_(
                                       bucket == 0 ? sz_levenshtein_byte_lanes_symbols_max_cuda_k
                                                   : sz_levenshtein_short_lanes_symbols_max_cuda_k);
        }
        else if (sz_levenshtein_bucket_words_cuda_(bucket) >= sz_levenshtein_gpu_warp_words_min_k)
            candidates_per_block = per_block / sz_levenshtein_gpu_warp_lanes_k * rounds;
        sz_size_t const blocks = sz_size_divide_round_up(candidates->count, candidates_per_block);

        for (sz_size_t row = first; row < last; row += sz_levenshtein_gpu_grid_rows_max_k) {
            sz_size_t const rows = sz_min_of_two(last - row, (sz_size_t)sz_levenshtein_gpu_grid_rows_max_k);
            sz_status_t const launched = sz_levenshtein_launch_cuda_(entry_point, blocks, rows, per_block, stream,
                                                                     *engine, head->order + row, *candidates, distances,
                                                                     distances_stride);
            if (launched != sz_success_k) return launched;
        }
    }
    return sz_success_k;
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_distances_scoped_cuda_(sz_levenshtein_engine_t *engine,
                                                                     sz_sequence_t const *candidates,
                                                                     sz_size_t *distances, sz_size_t distances_stride,
                                                                     sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_distances_cuda_(engine, candidates, distances, distances_stride, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#pragma endregion Myers Engine

#pragma region Tiled

/** The wavefront's block on CUDA: its warps a whole warp of 32 lanes apart. */
enum {
    sz_levenshtein_tiled_warp_stride_cuda_k = 32,
    sz_levenshtein_tiled_threads_per_block_cuda_k = sz_levenshtein_tiled_warps_per_block_simt_k *
                                                    sz_levenshtein_tiled_warp_stride_cuda_k,
};

/*  Tile-rows a pair needs before the wavefront is worth reaching for. Its makespan is
 *  tile-rows plus tile-columns tile latencies, so tile-rows is how many tile-columns it ever
 *  runs at once: under three of them the wavefront is a serial chain down the long axis and
 *  one Myers lane's bit-parallel sweep is quicker, while from three up it is measured ahead at
 *  every long-axis length. */
enum { sz_levenshtein_tiled_rows_min_cuda_k = 3 };

/**
 *  @brief One pair's Levenshtein distance through the tiled wavefront, on texts the
 *      device already reaches, on the caller's current device.
 *  @param[in] a First text, device-reachable.
 *  @param[in] a_length Its length in bytes.
 *  @param[in] b Second text, device-reachable.
 *  @param[in] b_length Its length in bytes.
 *  @param[in] scratch The frontier, device-reachable, at least
 *      @ref sz_levenshtein_distance_tiled_scratch_bytes bytes, and untouched by anything else until
 *      the caller joins @p stream.
 *  @param[out] distance Device-reachable slot the distance lands in once @p stream is joined.
 *  @param[in] stream The @c cudaStream_t to schedule on, or @c STRINGZILLA_NULL for the default.
 *  @return @c sz_success_k, @c sz_unexpected_dimensions_k when either text is longer than the
 *      kernel indexes, or @c sz_device_memory_mismatch_k when a text, the scratch or the distance
 *      is not memory the device reaches.
 *  @note Enqueues and returns, allocating nothing and joining nothing.
 *  @sa sz_levenshtein_distances_scoped_cuda_, which scores a whole batch through the
 *      bit-parallel rungs instead.
 */
STRINGZILLA_INLINE sz_status_t sz_levenshtein_distance_tiled_cuda_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                   sz_size_t b_length, void *scratch,
                                                                   sz_size_t *distance, sz_stream_t stream) {
    if (a_length + sz_levenshtein_tiled_padding_simt_k > ((sz_size_t)1 << 32) ||
        b_length + sz_levenshtein_tiled_padding_simt_k > ((sz_size_t)1 << 32))
        return sz_unexpected_dimensions_k;
    if (!sz_memory_accessible_cuda_(distance)) return sz_device_memory_mismatch_k;

    // The recurrence is symmetric, and putting the shorter text on the row axis keeps the frontier small and
    // makes the parallel axis the long one.
    sz_cptr_t const shorter_text = a_length <= b_length ? a : b;
    sz_cptr_t const longer_text = a_length <= b_length ? b : a;
    sz_u32_t const shorter_length = (sz_u32_t)(a_length <= b_length ? a_length : b_length);
    sz_u32_t const longer_length = (sz_u32_t)(a_length <= b_length ? b_length : a_length);
    if (shorter_length == 0) {
        // A copy out of a host local may join the stream, and a launch captures its arguments instead.
        sz_size_t *launch_distance = distance;
        sz_size_t launch_value = longer_length;
        void *arguments[2];
        dim3 one;
        one.x = 1, one.y = 1, one.z = 1;
        arguments[0] = &launch_distance, arguments[1] = &launch_value;
        return sz_launch_cuda_((void const *)sz_levenshtein_store_simt_kernel_, one, one, arguments, 0, stream);
    }
    if (!sz_memory_accessible_cuda_(shorter_text) || !sz_memory_accessible_cuda_(longer_text) ||
        !sz_memory_accessible_cuda_(scratch))
        return sz_device_memory_mismatch_k;

    sz_size_t const tile_grid_columns = sz_size_divide_round_up(longer_length, sz_levenshtein_tile_side_simt_k);
    sz_size_t const row_frontier_cells =
        sz_size_divide_round_up(shorter_length, sz_levenshtein_tile_side_simt_k) * sz_levenshtein_tile_side_simt_k + 1u;
    sz_u32_t *const row_frontier = (sz_u32_t *)scratch;
    sz_u32_t *const progress = row_frontier + row_frontier_cells;

    // Every launched block has to be resident, since a block spins on a tile-column another block owns.
    sz_size_t const resident_per_multiprocessor = sz_resident_blocks_cuda_(
        (void const *)sz_levenshtein_tiled_simt_kernel_, sz_levenshtein_tiled_threads_per_block_cuda_k, 0);
    sz_size_t const resident_blocks = sz_device_multiprocessors_cuda_() *
                                      sz_max_of_two(resident_per_multiprocessor, (sz_size_t)1);
    if (!resident_blocks) return sz_missing_gpu_k;
    sz_status_t const cleared = sz_fill_cuda_(progress, tile_grid_columns * sizeof(sz_u32_t), 0, stream);
    if (cleared != sz_success_k) return cleared;

    // A warp waits on the tile-column to its left, so a block that never gets scheduled is a block its
    // neighbour spins on forever; the grid is capped at what the occupancy answer says stays resident.
    sz_size_t const wanted_blocks = sz_size_divide_round_up(tile_grid_columns,
                                                            sz_levenshtein_tiled_warps_per_block_simt_k);
    dim3 grid, block;
    grid.x = (unsigned)(wanted_blocks < resident_blocks ? wanted_blocks : resident_blocks), grid.y = 1, grid.z = 1;
    block.x = sz_levenshtein_tiled_threads_per_block_cuda_k, block.y = 1, block.z = 1;

    sz_cptr_t launch_shorter_text = shorter_text, launch_longer_text = longer_text;
    sz_u32_t launch_shorter_length = shorter_length, launch_longer_length = longer_length;
    sz_u32_t *launch_row_frontier = row_frontier, *launch_progress = progress;
    sz_size_t *launch_distance = distance;
    void *arguments[7];
    arguments[0] = &launch_shorter_text, arguments[1] = &launch_shorter_length;
    arguments[2] = &launch_longer_text, arguments[3] = &launch_longer_length;
    arguments[4] = &launch_row_frontier, arguments[5] = &launch_progress, arguments[6] = &launch_distance;
    return sz_launch_cuda_((void const *)sz_levenshtein_tiled_simt_kernel_, grid, block, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_levenshtein_distance_tiled_scoped_cuda_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                                          sz_size_t b_length, void *scratch,
                                                                          sz_size_t *distance, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_levenshtein_distance_tiled_cuda_(a, a_length, b, b_length, scratch, distance, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#pragma endregion Tiled

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cuda(sz_levenshtein_engine_t *engine,
                                                            sz_sequence_t const *queries,
                                                            sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                            sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_cuda_(engine, queries, symbol, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_cuda(sz_levenshtein_engine_t *engine,
                                                          sz_sequence_t const *candidates, sz_size_t *distances,
                                                          sz_size_t distances_stride, sz_stream_t stream) {
    return sz_levenshtein_distances_scoped_cuda_(engine, candidates, distances, distances_stride, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distance_tiled_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length, void *scratch, sz_size_t *distance,
                                                               sz_stream_t stream) {
    return sz_levenshtein_distance_tiled_scoped_cuda_(a, a_length, b, b_length, scratch, distance, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA && defined(__CUDACC__) && !defined(__HIP__)
#endif // STRINGZILLA_LEVENSHTEIN_CUDA_CUH_
