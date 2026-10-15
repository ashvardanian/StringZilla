/**
 *  @file include/stringzilla/subdocuments/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief CUDA host side of path queries: the warp walk, the round block the walks run out of, the
 *      launches, and the @c _cuda exports, over the kernels of `subdocuments/simt.cuh`.
 *
 *  A routing launch settles which documents of the batch are large. Both verbs read back once their
 *  counting walks end, which joins the stream: a malformed document fails the call as it does on
 *  the CPU, and the documents needing more frames than the first launch's stacks hold are counted
 *  again before anything is stored. The scan and the storing walks then run on the stream; a find
 *  stages each warp's matches while counting, so its first storing launch moves them and walks
 *  again only the documents whose matches outgrew the staging. The large documents the counting
 *  walks list then walk over chunks in one pipeline past both launches' stacks, reading back once
 *  all their chunks are counted; storing reruns only its last walk over the arrays it keeps.
 *
 *  @sa include/stringzilla/subdocuments/simt.cuh
 *  @sa include/stringzilla/subdocuments/rocm.cuh
 *  @sa include/stringzilla/subdocuments/metal.h, whose calls this mirrors launch for dispatch
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_CUDA_CUH_
#define STRINGZILLA_SUBDOCUMENTS_CUDA_CUH_

#include "stringzilla/cuda.cuh"
#include "stringzilla/subdocuments/simt.cuh"

#if STRINGZILLA_ARCH_CUDA_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Warp Walks

/** @ref sz_subdocuments_load_window_simt_ over this warp. */
STRINGZILLA_DEVICE void sz_subdocuments_load_warp_cuda_(sz_subdocuments_cursor_simt_t *cursor, sz_size_t offset) {
    sz_subdocuments_load_window_simt_(cursor, offset, sz_lanes_ballot_cuda_, sz_shuffle_index_cuda_);
}

/** @ref sz_subdocuments_walk_documents_simt_ with a warp walking each JSON document. */
static __global__ void sz_subdocuments_warp_walk_cuda_kernel_(sz_subdocuments_engine_t engine,
                                                              sz_strings_view_t documents,
                                                              sz_subdocuments_round_simt_t round) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, thread / 32, thread & 31, 32,
                                         sz_subdocuments_load_warp_cuda_);
}

/** Blocks of the dense warp walk each multiprocessor keeps resident: once every walker has a
 *  document, more warps hiding the serial walk's latency outweigh the registers they give up. */
enum { sz_subdocuments_walk_blocks_cuda_k = 3 };

/** @ref sz_subdocuments_warp_walk_cuda_kernel_ for launches of every walker the device holds. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k, sz_subdocuments_walk_blocks_cuda_k)
    sz_subdocuments_warp_walk_dense_cuda_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                                 sz_subdocuments_round_simt_t round) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, thread / 32, thread & 31, 32,
                                         sz_subdocuments_load_warp_cuda_);
}

#pragma endregion Warp Walks

#pragma region Host Plumbing

/** Blocks every launch of the cooperative walk runs. */
enum { sz_subdocuments_blocks_cuda_k = 256 };

/** Grows the round block to @p bytes, keeping what it holds. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_reserve_cuda_(sz_subdocuments_engine_t *engine, sz_size_t bytes,
                                                             sz_stream_t stream) {
    void *grown;
    if (engine->scratch_bytes >= bytes) return sz_success_k;
    grown = engine->allocator.reallocate(engine->scratch, engine->scratch_bytes, bytes, engine->allocator.handle,
                                         stream);
    if (!grown) return sz_bad_alloc_k;
    engine->scratch = grown, engine->scratch_bytes = bytes;
    return sz_success_k;
}

/** Points @p round into the round block: the header, the counts, the list of deep documents, the
 *  list of large ones and where each one's matches are staged among @p count, then the key buffers
 *  of the threads, their staged matches from @p staging on, and their stacks from @p stacks on. */
STRINGZILLA_INLINE void sz_subdocuments_round_cuda_(sz_subdocuments_engine_t const *engine, sz_size_t count,
                                                    sz_size_t staging, sz_size_t stacks,
                                                    sz_subdocuments_round_simt_t *round) {
    sz_ptr_t const block = (sz_ptr_t)engine->scratch;
    sz_size_t const counted = sizeof(sz_subdocuments_header_simt_t), deep = counted + (count + 1) * sizeof(sz_size_t);
    sz_size_t const large = deep + count * sizeof(sz_u64_t);
    round->header = (sz_subdocuments_header_simt_t *)block;
    round->counted = round->counts ? (sz_size_t *)STRINGZILLA_NULL : (sz_size_t *)(block + counted);
    round->deep = (sz_u64_t *)(block + deep), round->large = (sz_u64_t *)(block + large);
    round->staged = (sz_u64_t *)(block + large + 2 * count * sizeof(sz_u64_t));
    round->keys = (sz_u64_t *)(block + large + 3 * count * sizeof(sz_u64_t));
    round->staging = (sz_match_t *)(block + staging), round->stacks = (sz_u64_t *)(block + stacks);
}

/** Launches one walk over @p round 's items, a warp each for JSON and a thread each for XML; the
 *  dense warp walk once there are as many walkers as the device holds warps. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_launch_cuda_(sz_subdocuments_engine_t const *engine,
                                                            sz_strings_view_t const *documents,
                                                            sz_subdocuments_round_simt_t const *round,
                                                            sz_stream_t stream) {
    sz_subdocuments_engine_t launched_engine = *engine;
    sz_strings_view_t launched_documents = *documents;
    sz_subdocuments_round_simt_t launched_round = *round;
    sz_bool_t const warps = (sz_bool_t)(engine->format == sz_subdocuments_json_k);
    void *arguments[3];
    dim3 grid, block;
    arguments[0] = &launched_engine, arguments[1] = &launched_documents, arguments[2] = &launched_round;
    grid.x = (unsigned)sz_size_divide_round_up(round->threads * (warps ? 32 : 1),
                                               sz_subdocuments_threads_per_block_simt_k);
    grid.y = 1, grid.z = 1;
    block.x = sz_subdocuments_threads_per_block_simt_k, block.y = 1, block.z = 1;
    sz_bool_t const dense = (sz_bool_t)(round->threads * 32 >= sz_device_multiprocessors_cuda_() *
                                                                   sz_device_threads_per_multiprocessor_cuda_());
    return sz_launch_cuda_(!warps  ? (void const *)sz_subdocuments_walk_simt_kernel_
                           : dense ? (void const *)sz_subdocuments_warp_walk_dense_cuda_kernel_
                                   : (void const *)sz_subdocuments_warp_walk_cuda_kernel_,
                           grid, block, arguments, 0, stream);
}

/** Routes @p documents: settles the header's @c large_bytes, which every walk then reads. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_route_cuda_(sz_strings_view_t const *documents,
                                                           sz_subdocuments_round_simt_t const *round,
                                                           sz_stream_t stream) {
    sz_strings_view_t launched_documents = *documents;
    sz_subdocuments_round_simt_t launched_round = *round;
    void *arguments[2];
    dim3 grid, block;
    arguments[0] = &launched_documents, arguments[1] = &launched_round;
    grid.x = (unsigned)sz_min_of_two(
        sz_size_divide_round_up(documents->count, sz_subdocuments_threads_per_block_simt_k),
        (sz_size_t)sz_subdocuments_blocks_cuda_k);
    grid.y = 1, grid.z = 1;
    block.x = sz_subdocuments_threads_per_block_simt_k, block.y = 1, block.z = 1;
    return sz_launch_cuda_((void const *)sz_subdocuments_route_simt_kernel_, grid, block, arguments, 0, stream);
}

/** Copies the header back to @p header, which joins @p stream. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_read_cuda_(sz_subdocuments_round_simt_t const *round,
                                                          sz_subdocuments_header_simt_t *header, sz_stream_t stream) {
    sz_status_t const status = sz_copy_cuda_(header, round->header, sizeof(*header), stream);
    return status == sz_success_k ? sz_stream_synchronize_cuda_(stream) : status;
}

/** Launches cooperative @p kernel over @p blocks blocks with @p arguments. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_launch_cuda_(void const *kernel, sz_size_t blocks,
                                                                  void **arguments, sz_stream_t stream) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = sz_subdocuments_threads_per_block_simt_k, block.y = 1, block.z = 1;
    return sz_launch_cuda_(kernel, grid, block, arguments, 0, stream);
}

/** Scans the @p count values at @p values in place, exclusively, in three launches: summing them,
 *  or composing the chunks' effects on the carries when @p compose is set. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_scan_cuda_(sz_subdocuments_large_simt_t *large, sz_u64_t *values,
                                                                sz_size_t count, sz_u32_t compose, sz_stream_t stream) {
    void *arguments[1];
    sz_status_t status;
    large->scanned = values, large->scanned_count = count, large->compose = compose;
    large->scan_tiles = sz_min_of_two(sz_size_divide_round_up(count, sz_subdocuments_threads_per_block_simt_k),
                                      (sz_size_t)sz_subdocuments_scan_tiles_simt_k);
    arguments[0] = large;
    status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_reduce_simt_kernel_,
                                                large->scan_tiles, arguments, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_carry_simt_kernel_, 1,
                                                    arguments, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_apply_simt_kernel_,
                                                    large->scan_tiles, arguments, stream);
    return status;
}

/** Launches the walk of @p large over @p round 's documents, counting or storing as @p large is. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_walk_cuda_(sz_subdocuments_engine_t const *engine,
                                                                sz_strings_view_t const *documents,
                                                                sz_subdocuments_round_simt_t const *round,
                                                                sz_subdocuments_large_simt_t const *large,
                                                                sz_stream_t stream) {
    sz_subdocuments_engine_t launched_engine = *engine;
    sz_strings_view_t launched_documents = *documents;
    sz_subdocuments_round_simt_t launched_round = *round;
    sz_subdocuments_large_simt_t launched_large = *large;
    void *arguments[4];
    arguments[0] = &launched_engine, arguments[1] = &launched_documents, arguments[2] = &launched_round;
    arguments[3] = &launched_large;
    return sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_walk_simt_kernel_,
                                              sz_subdocuments_blocks_cuda_k, arguments, stream);
}

/** Walks every document @p header lists for chunks, below its first malformed one, as one
 *  pipeline from @p region on in the round block: counts their matches into their counts,
 *  refusing each where the serial walk would, and leaves @p large for a storing walk to rerun.
 *  Reads @p header back once their chunks are counted, which joins the stream, and repoints
 *  @p round as the block grows. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_large_cuda_(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents,
                                                           sz_subdocuments_round_simt_t *round, sz_size_t count,
                                                           sz_size_t staging, sz_size_t stacks, sz_size_t region,
                                                           sz_subdocuments_header_simt_t *header,
                                                           sz_subdocuments_large_simt_t *large, sz_stream_t stream) {
    sz_size_t const blocks = sz_subdocuments_blocks_cuda_k, threads = blocks * sz_subdocuments_threads_per_block_simt_k;
    sz_size_t const iterations = sz_subdocuments_longest_path_simt_(engine);
    sz_size_t const listed = (sz_size_t)header->large_count, chunks = (sz_size_t)header->large_chunks;
    sz_subdocuments_engine_t launched_engine = *engine;
    sz_strings_view_t launched_documents = *documents;
    sz_subdocuments_round_simt_t launched_round;
    void *with_documents[2], *with_round[2], *with_engine[2], *with_text[3], *alone[1];
    sz_size_t bytes = sz_subdocuments_large_layout_simt_(engine, STRINGZILLA_NULL, listed, chunks, threads,
                                                         round->key_bytes, 0, 0, large);
    sz_size_t level, iteration, jump, rounds = 0;
    sz_u64_t tokens = 0, containers = 0;
    // The caller has just joined the stream, so the block may move.
    sz_status_t status = sz_subdocuments_reserve_cuda_(engine, region + bytes, stream);
    if (status != sz_success_k) return status;
    sz_subdocuments_round_cuda_(engine, count, staging, stacks, round);
    sz_subdocuments_large_layout_simt_(engine, (sz_ptr_t)engine->scratch + region, listed, chunks, threads,
                                       round->key_bytes, 0, 0, large);
    large->list = round->large, large->bound = (sz_size_t)header->first_malformed;
    large->storing = 0, large->carried_buffer = 0, launched_round = *round;
    with_documents[0] = &launched_documents, with_documents[1] = large;
    with_round[0] = &launched_round, with_round[1] = large;
    with_engine[0] = &launched_engine, with_engine[1] = large;
    with_text[0] = &launched_engine, with_text[1] = &launched_documents, with_text[2] = large;
    alone[0] = large;

    // Each document's first chunk and the chunks' carries, then their tokens, containers and
    // depths, numbered by scans past the documents their depths or quotes refuse.
    status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_lengths_simt_kernel_, blocks, alone,
                                                stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->firsts, listed + 1, 0, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_summary_simt_kernel_, blocks,
                                                    with_documents, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_scan_cuda_(large, large->transfers, chunks + 1, 1, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_count_simt_kernel_, blocks,
                                                    with_documents, stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->depths, chunks + 1, 0, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_check_simt_kernel_, blocks,
                                                    with_round, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_drop_simt_kernel_, blocks,
                                                    with_round, stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->tokens, chunks + 1, 0, stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->opens, chunks + 1, 0, stream);
    if (status == sz_success_k) status = sz_copy_cuda_(&tokens, large->tokens + chunks, sizeof(tokens), stream);
    if (status == sz_success_k) status = sz_copy_cuda_(&containers, large->opens + chunks, sizeof(containers), stream);
    if (status == sz_success_k) status = sz_subdocuments_read_cuda_(round, header, stream);
    if (status != sz_success_k) return status;

    bytes = sz_subdocuments_large_layout_simt_(engine, STRINGZILLA_NULL, listed, chunks, threads, round->key_bytes,
                                               (sz_size_t)tokens, (sz_size_t)containers, large);
    status = sz_subdocuments_reserve_cuda_(engine, region + bytes, stream);
    if (status != sz_success_k) return status;
    sz_subdocuments_round_cuda_(engine, count, staging, stacks, round);
    sz_subdocuments_large_layout_simt_(engine, (sz_ptr_t)engine->scratch + region, listed, chunks, threads,
                                       round->key_bytes, (sz_size_t)tokens, (sz_size_t)containers, large);
    large->list = round->large, launched_round = *round;

    // Tokens and containers, the parents the pyramid of minima finds, closes,
    // element counts and positions.
    status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_tokens_simt_kernel_, blocks,
                                                with_documents, stream);
    for (level = 1; status == sz_success_k && level != large->levels; ++level)
        large->round = level, status = sz_subdocuments_large_launch_cuda_(
                                  (void const *)sz_subdocuments_large_minima_simt_kernel_, blocks, alone, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_parents_simt_kernel_, blocks,
                                                    alone, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_closes_simt_kernel_, blocks,
                                                    alone, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_sums_simt_kernel_, blocks,
                                                    alone, stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->sums, chunks + 1, 0, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_offsets_simt_kernel_, blocks,
                                                    alone, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_selectors_simt_kernel_, blocks,
                                                    with_text, stream);

    // Live sets advance once per step of the longest path, each time after
    // gathering what ancestors carry.
    while ((sz_u64_t)1 << rounds < header->deepest) ++rounds;
    for (iteration = 0; status == sz_success_k && iteration <= iterations; ++iteration) {
        for (jump = 0; large->descendants && status == sz_success_k && jump <= rounds; ++jump)
            large->round = jump,
            status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_jump_simt_kernel_, blocks,
                                                        with_engine, stream);
        large->carried_buffer = rounds & 1;
        if (status == sz_success_k && iteration != iterations)
            status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_advance_simt_kernel_,
                                                        blocks, with_engine, stream);
    }

    // Counting checks every token and scans the matches, from which storing walks again.
    if (status == sz_success_k) status = sz_subdocuments_large_walk_cuda_(engine, documents, round, large, stream);
    if (status == sz_success_k) status = sz_subdocuments_large_scan_cuda_(large, large->matches, chunks + 1, 0, stream);
    if (status == sz_success_k)
        status = sz_subdocuments_large_launch_cuda_((void const *)sz_subdocuments_large_finish_simt_kernel_, blocks,
                                                    with_round, stream);
    return status;
}

/** Counts every document into @p counts, or, with @p counts null, finds every match into @p matches
 *  and @p matches_offsets; reads back once the counting walks end, and fails as the serial tier
 *  does on the first malformed document. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_walks_cuda_(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_stride, sz_match_t *matches,
                                                           sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                           sz_size_t *malformed_index, sz_stream_t stream) {
    sz_size_t const count = documents->count, frame_words = sz_subdocuments_frame_words_simt_(engine);
    sz_size_t const frame_bytes = frame_words * sizeof(sz_u64_t), key_bytes = sz_subdocuments_key_bytes_(engine);
    // Walkers, each with its own stack: a warp each in JSON, a thread each in XML.
    sz_size_t const walkers = sz_device_multiprocessors_cuda_() * sz_device_threads_per_multiprocessor_cuda_() /
                              (engine->format == sz_subdocuments_json_k ? 32 : 1);
    sz_size_t const threads = sz_min_of_two(count, walkers);
    // A find stages each warp's matches while counting, so storing moves them rather than walking.
    sz_size_t const staging_capacity = !counts && engine->format == sz_subdocuments_json_k
                                           ? (sz_size_t)sz_subdocuments_staged_matches_simt_k
                                           : 0;
    sz_size_t const staging = sizeof(sz_subdocuments_header_simt_t) + (5 * count + 1) * sizeof(sz_u64_t) +
                              threads * key_bytes;
    sz_size_t const stacks = staging + threads * staging_capacity * sizeof(sz_match_t);
    sz_size_t const stacks_bytes = threads * sz_subdocuments_frames_simt_k * frame_bytes;
    sz_size_t deep_threads = 0, deep_frames = 0;
    sz_subdocuments_header_simt_t header;
    sz_subdocuments_round_simt_t round;
    sz_subdocuments_large_simt_t large;
    sz_status_t status = sz_subdocuments_reserve_cuda_(engine, stacks + stacks_bytes, stream);
    if (status != sz_success_k) return status;

    round.counts = counts, round.counts_stride = counts_stride, round.matches_offsets = matches_offsets;
    round.matches = matches, round.matches_capacity = matches_capacity, round.key_bytes = key_bytes;
    round.items = count, round.threads = threads, round.frames = sz_subdocuments_frames_simt_k;
    round.frame_words = frame_words, round.listed = 0, round.storing = 0;
    // Filters and XML keep every document on one thread.
    round.large_bytes = engine->format == sz_subdocuments_json_k && !engine->filters_count
                            ? (sz_size_t)sz_subdocuments_large_bytes_simt_k
                            : STRINGZILLA_SIZE_MAX;
    round.large_shares = sz_max_of_two(walkers / sz_subdocuments_large_walkers_simt_k, (sz_size_t)1);
    round.staging_capacity = staging_capacity;
    sz_subdocuments_round_cuda_(engine, count, staging, stacks, &round);
    header.first_malformed = count, header.deep_count = 0, header.frames = 0, header.chain = 0;
    header.large_count = 0, header.large_chunks = 0, header.deepest = 0, header.large_bytes = STRINGZILLA_SIZE_MAX;
    status = sz_copy_cuda_(round.header, &header, sizeof(header), stream);
    if (status == sz_success_k && round.large_bytes != STRINGZILLA_SIZE_MAX)
        status = sz_subdocuments_route_cuda_(documents, &round, stream);
    if (status == sz_success_k) status = sz_subdocuments_launch_cuda_(engine, documents, &round, stream);
    if (status == sz_success_k) status = sz_subdocuments_read_cuda_(&round, &header, stream);

    // The listed documents walk again side by side, on stacks as deep as the deepest needs, which
    // grow with how many are listed up to a cap past which they take turns on fewer threads.
    if (status == sz_success_k && header.deep_count) {
        sz_size_t needed, budget;
        deep_frames = (sz_size_t)header.frames;
        needed = (sz_size_t)header.deep_count * deep_frames * frame_bytes;
        budget = sz_max_of_two(stacks_bytes, sz_min_of_two(needed, (sz_size_t)sz_subdocuments_deep_bytes_simt_k));
        deep_threads = sz_min_of_two(sz_min_of_two((sz_size_t)header.deep_count, threads),
                                     sz_max_of_two(budget / (deep_frames * frame_bytes), (sz_size_t)1));
        status = sz_subdocuments_reserve_cuda_(engine, stacks + deep_threads * deep_frames * frame_bytes, stream);
        if (status == sz_success_k) {
            sz_subdocuments_round_cuda_(engine, count, staging, stacks, &round);
            round.listed = 1, round.items = (sz_size_t)header.deep_count, round.threads = deep_threads;
            round.frames = deep_frames;
            status = sz_subdocuments_launch_cuda_(engine, documents, &round, stream);
        }
        if (status == sz_success_k) status = sz_subdocuments_read_cuda_(&round, &header, stream);
    }
    // The large documents walk over chunks together, past none already known malformed, after the
    // stacks of both launches, so storing reruns those while keeping their arrays.
    if (status == sz_success_k && header.large_count) {
        sz_size_t const stacked = stacks + sz_max_of_two(stacks_bytes, deep_threads * deep_frames * frame_bytes);
        sz_size_t const region = sz_size_divide_round_up(stacked, 64) * 64;
        status = sz_subdocuments_large_cuda_(engine, documents, &round, count, staging, stacks, region, &header, &large,
                                             stream);
        if (status == sz_success_k) status = sz_subdocuments_read_cuda_(&round, &header, stream);
    }
    if (status != sz_success_k) return status;
    if (header.first_malformed < count) {
        *malformed_index = (sz_size_t)header.first_malformed;
        return sz_invalid_syntax_k;
    }
    if (counts) return sz_success_k;

    // Counts scan into each document's first match, the total after them, and every walk stores.
    {
        sz_size_t const tiles = sz_min_of_two(
            sz_size_divide_round_up(count + 1, sz_subdocuments_threads_per_block_simt_k),
            (sz_size_t)sz_chain_tiles_max_k);
        sz_size_t values_count = count + 1, elements_per_tile = sz_size_divide_round_up(count + 1, tiles);
        sz_size_t *values = round.counted, *chain = (sz_size_t *)&round.header->chain;
        void *arguments[4];
        dim3 grid, block;
        arguments[0] = &values, arguments[1] = &values_count, arguments[2] = &elements_per_tile, arguments[3] = &chain;
        grid.x = (unsigned)tiles, grid.y = 1, grid.z = 1;
        block.x = sz_subdocuments_threads_per_block_simt_k, block.y = 1, block.z = 1;
        status = sz_fill_cuda_(round.counted + count, sizeof(sz_size_t), 0, stream);
        if (status == sz_success_k)
            status = sz_launch_cuda_((void const *)sz_subdocuments_scan_simt_kernel_, grid, block, arguments, 0,
                                     stream);
    }
    round.storing = 1, round.listed = 0, round.items = count, round.threads = threads;
    round.frames = sz_subdocuments_frames_simt_k;
    if (status == sz_success_k) status = sz_subdocuments_launch_cuda_(engine, documents, &round, stream);
    round.listed = 1, round.items = (sz_size_t)header.deep_count, round.threads = deep_threads,
    round.frames = deep_frames;
    if (status == sz_success_k && header.deep_count)
        status = sz_subdocuments_launch_cuda_(engine, documents, &round, stream);
    if (status == sz_success_k && header.large_count)
        large.storing = 1, status = sz_subdocuments_large_walk_cuda_(engine, documents, &round, &large, stream);
    return status;
}

#pragma endregion Host Plumbing

#pragma region Construction

/** Compiles @p paths on the device the caller made current, as
 *  @ref sz_subdocuments_engine_init_cuda does. */
STRINGZILLA_INLINE sz_status_t sz_subdocuments_engine_init_scoped_cuda_(sz_subdocuments_engine_t *engine,
                                                                        sz_strings_view_t const *paths,
                                                                        sz_subdocuments_format_t format,
                                                                        sz_size_t *malformed_index,
                                                                        sz_allocator_t *allocator, sz_stream_t stream) {
    cudaPointerAttributes attributes;
    sz_allocator_t unified;
    sz_status_t status;
    if (!sz_device_multiprocessors_cuda_()) return sz_missing_gpu_k;
    if (allocator) unified = *allocator;
    else sz_allocator_init_unified_cuda_(&unified);
    status = sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_cuda_k, &unified, stream);
    if (status != sz_success_k) return status;
    // `match()` and `search()` walk the forward DFA, which a pattern set past
    // its state budget lacks.
    if (!sz_memory_accessible_cuda_(engine->memory) || (engine->patterns_count && !engine->regex.forward_rows)) {
        status = sz_memory_accessible_cuda_(engine->memory) ? sz_device_code_mismatch_k : sz_device_memory_mismatch_k;
        sz_subdocuments_engine_free_(engine, stream);
        return status;
    }
    // Every call reads the engine on the host too, which would pull a managed engine's pages off
    // the device for the next kernel to fault back; read-mostly pages keep a copy on each side.
    if (cudaPointerGetAttributes(&attributes, engine->memory) == cudaSuccess &&
        attributes.type == cudaMemoryTypeManaged) {
        cudaMemLocation where;
        where.type = cudaMemLocationTypeDevice, where.id = attributes.device;
        sz_unused_(cudaMemAdvise(engine->memory, engine->memory_bytes, cudaMemAdviseSetReadMostly, where));
    }
    return sz_success_k;
}

#pragma endregion Construction

#pragma region Verbs

STRINGZILLA_INLINE sz_status_t sz_subdocuments_counts_scoped_cuda_(sz_subdocuments_engine_t *engine,
                                                                   sz_strings_view_t const *documents,
                                                                   sz_size_t *counts, sz_size_t counts_stride,
                                                                   sz_size_t *malformed_index, sz_stream_t stream) {
    if (!engine->steps_count) return sz_missing_kernel_k;
    if (!counts_stride) return sz_unexpected_dimensions_k;
    if (!documents->count) return sz_success_k;
    if (!sz_strings_accessible_cuda_(documents) || !sz_memory_accessible_cuda_(counts))
        return sz_device_memory_mismatch_k;
    return sz_subdocuments_walks_cuda_(engine, documents, counts, counts_stride, STRINGZILLA_NULL, 0, STRINGZILLA_NULL,
                                       malformed_index, stream);
}

STRINGZILLA_INLINE sz_status_t sz_subdocuments_find_scoped_cuda_(sz_subdocuments_engine_t *engine,
                                                                 sz_strings_view_t const *documents,
                                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                                 sz_size_t *matches_offsets, sz_size_t *malformed_index,
                                                                 sz_stream_t stream) {
    if (!engine->steps_count) return sz_missing_kernel_k;
    if (!sz_strings_accessible_cuda_(documents) || !sz_memory_accessible_cuda_(matches_offsets) ||
        (matches_capacity && !sz_memory_accessible_cuda_(matches)))
        return sz_device_memory_mismatch_k;
    if (!documents->count) return sz_fill_cuda_(matches_offsets, sizeof(sz_size_t), 0, stream);
    return sz_subdocuments_walks_cuda_(engine, documents, STRINGZILLA_NULL, 0, matches, matches_capacity,
                                       matches_offsets, malformed_index, stream);
}

#pragma endregion Verbs

/*  Each export makes the stream's device current around its helper, and restores the caller's. */
#if STRINGZILLA_TARGET_CUDA

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_cuda(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *paths,
                                                             sz_subdocuments_format_t format,
                                                             sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_engine_init_scoped_cuda_(engine, paths, format, malformed_index, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_cuda(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_counts_scoped_cuda_(engine, documents, counts, counts_stride, malformed_index, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_cuda(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_find_scoped_cuda_(engine, documents, matches, matches_capacity, matches_offsets,
                                               malformed_index, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBDOCUMENTS_CUDA_CUH_
