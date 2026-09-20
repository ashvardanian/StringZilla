/**
 *  @brief CUDA backend for Levenshtein distances: one candidate per thread, streaming Myers' bit-parallel
 *      recurrence against a query prepared once into match masks the whole grid shares.
 *  @file include/stringzilla/levenshtein/cuda.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzilla/levenshtein.h
 *
 *  The step is the serial tier's, reached from the device through `--expt-relaxed-constexpr`, so the distances
 *  are the same integers rather than merely close ones. Myers is add-with-carry and bitwise operations over
 *  @c u64 words, all of which the device runs at its integer rate; there is nothing here for the DPX three-way
 *  maxima to accelerate, since those serve the score-based recurrences of Needleman-Wunsch and Smith-Waterman
 *  rather than this one. That is why this family has one GPU tier and not a ladder of them.
 *
 *  A candidate's recurrence is a dependency chain, so it stays on one thread and the parallelism comes from the
 *  candidates. The verticals live in the thread's own registers or local memory, @c words of them, while the
 *  match masks are read-only and shared: every thread indexes the same @c words × 256 table by the byte it is
 *  stepping, so the rows stay hot in cache instead of being rebuilt per candidate.
 */
#ifndef STRINGZILLA_LEVENSHTEIN_CUDA_CUH_
#define STRINGZILLA_LEVENSHTEIN_CUDA_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/levenshtein/serial.h"

#if SZ_USE_CUDA

#pragma region CUDA

/** Query words one thread keeps verticals for; a query past this is refused rather than silently truncated. */
enum { sz_levenshtein_cuda_words_max_k = 16 };

/** Candidates one block scores when the device cannot be asked; the register budget differs per word count,
 *  so the launcher prefers what the occupancy calculator answers for the entry point it is about to launch. */
enum { sz_levenshtein_cuda_candidates_per_block_k = 128 };

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief One candidate's Myers sweep at a compile-time @p words, which is what keeps the verticals in registers.
 *
 *  A word count the compiler cannot see makes @c verticals a dynamically indexed array, and the only place it can
 *  live is local memory - a 256-byte stack frame and four local accesses per word-step, which caps the kernel near
 *  thirty percent of the device's integer issue rate. Reached from entry points that each pass a literal, the same
 *  body is worth 2.2x at a one-word query and 3.3x at sixteen.
 */
SZ_DEVICE_INLINE void sz_levenshtein_cuda_sweep_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                 sz_size_t *distances, sz_size_t words) {
    sz_size_t const candidate = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidates.count) return;

    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
    sz_size_t const length = candidates.get_length(candidates.handle, candidate);

    sz_levenshtein_u64x1_state_serial_t state;
    sz_levenshtein_u64x1_vertical_serial_t verticals[sz_levenshtein_cuda_words_max_k];
    sz_levenshtein_u64x1_init_serial(&state, verticals, words, &query);
    for (sz_size_t position = 0; position != length; ++position)
        sz_levenshtein_u64x1_step_serial(&state, verticals, words, &query, (sz_u32_t)(sz_u8_t)text[position]);
    distances[candidate] = sz_levenshtein_u64x1_score_serial(&state, candidate);
}

/*  One entry point per word count, each passing its own literal, so every query length gets its own register
 *  budget - thirty-eight registers at one word against ninety-six at sixteen, which one shared kernel would have
 *  to spend on every launch. `sz_levenshtein_cuda_distances_` picks between them.
 */
static __global__ void sz_levenshtein_u64x1_distances_cuda_w1_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 1);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w2_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 2);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w3_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 3);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w4_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 4);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w5_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 5);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w6_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 6);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w7_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 7);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w8_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 8);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w9_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                               sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 9);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w10_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 10);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w11_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 11);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w12_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 12);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w13_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 13);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w14_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 14);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w15_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 15);
}

static __global__ void sz_levenshtein_u64x1_distances_cuda_w16_(sz_levenshtein_query_t query, sz_sequence_t candidates,
                                                                sz_size_t *distances) {
    sz_levenshtein_cuda_sweep_(query, candidates, distances, 16);
}

/*  One entry point per word count, addressed by it. `cudaLaunchKernel` takes the host-side symbol of a
 *  `__global__`, so the table is what the `<<< >>>` operator would have selected, spelled as data. */
static void const *const sz_levenshtein_cuda_entry_points_[sz_levenshtein_cuda_words_max_k] = {
    (void const *)sz_levenshtein_u64x1_distances_cuda_w1_,  (void const *)sz_levenshtein_u64x1_distances_cuda_w2_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w3_,  (void const *)sz_levenshtein_u64x1_distances_cuda_w4_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w5_,  (void const *)sz_levenshtein_u64x1_distances_cuda_w6_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w7_,  (void const *)sz_levenshtein_u64x1_distances_cuda_w8_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w9_,  (void const *)sz_levenshtein_u64x1_distances_cuda_w10_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w11_, (void const *)sz_levenshtein_u64x1_distances_cuda_w12_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w13_, (void const *)sz_levenshtein_u64x1_distances_cuda_w14_,
    (void const *)sz_levenshtein_u64x1_distances_cuda_w15_, (void const *)sz_levenshtein_u64x1_distances_cuda_w16_,
};

/** Launches the entry point whose word count matches the query's; a wider query never reaches here. */
static cudaError_t sz_levenshtein_cuda_distances_(sz_size_t words, sz_size_t blocks, sz_size_t per_block,
                                                  cudaStream_t stream, sz_levenshtein_query_t query,
                                                  sz_sequence_t candidates, sz_size_t *distances) {
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[3];
    arguments[0] = &query, arguments[1] = &candidates, arguments[2] = &distances;
    return cudaLaunchKernel(sz_levenshtein_cuda_entry_points_[words - 1], grid, block, arguments, 0, stream);
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_scheduled_cuda(sz_cptr_t query_text, sz_size_t query_length,
                                                                    sz_sequence_t const *candidates,
                                                                    sz_memory_allocator_t *alloc, sz_size_t *distances,
                                                                    void *stream) {
    if (query_length > sz_levenshtein_cuda_words_max_k * 64) return sz_unexpected_dimensions_k;
    if (!candidates->count) return sz_success_k;
    if (query_length == 0) {
        // Every distance is the candidate's own length, which the accessors answer only on the device.
        return sz_unexpected_dimensions_k;
    }

    // The handle is checked, never the accessors: those are the device's to call, so the host must not, and a
    // pointer is all this side can inspect. That the texts they answer are device-reachable is the caller's word.
    if (!sz_memory_reaches_device(distances)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(candidates->handle)) return sz_device_memory_mismatch_k;

    sz_size_t const words = sz_levenshtein_query_words(query_length);
    sz_size_t const masks_bytes = words * 256 * sizeof(sz_u64_t);
    sz_u64_t *const masks = (sz_u64_t *)alloc->allocate(masks_bytes, alloc->handle);
    if (!masks) return sz_bad_alloc_k;
    if (!sz_memory_reaches_device(masks)) {
        alloc->free(masks, masks_bytes, alloc->handle);
        return sz_device_memory_mismatch_k;
    }

    sz_levenshtein_query_t query;
    sz_levenshtein_query_prepare(query_text, query_length, masks, &query);

    cudaStream_t const on = (cudaStream_t)stream;
    sz_size_t const per_block = sz_levenshtein_cuda_candidates_per_block_k;
    sz_size_t const blocks = (candidates->count + per_block - 1) / per_block;
    cudaError_t const launched =
        sz_levenshtein_cuda_distances_(words, blocks, per_block, on, query, *candidates, distances);
    if (launched != cudaSuccess) {
        alloc->free(masks, masks_bytes, alloc->handle);
        return sz_device_code_mismatch_k;
    }

    // Only this stream is waited on, so the caller's other work on the device keeps running.
    cudaError_t const finished = cudaStreamSynchronize(on);
    alloc->free(masks, masks_bytes, alloc->handle);
    return finished == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distances_cuda(sz_cptr_t query_text, sz_size_t query_length,
                                                          sz_sequence_t const *candidates, sz_memory_allocator_t *alloc,
                                                          sz_size_t *distances) {
    // The one probe is the strict verb's own: it refuses whatever the device cannot reach, and only then is there
    // anything to stage. A caller already holding its data on the device pays nothing for the attempt.
    sz_status_t const resident = sz_levenshtein_distances_scheduled_cuda(query_text, query_length, candidates, alloc,
                                                                         distances, SZ_NULL);
    if (resident != sz_device_memory_mismatch_k) return resident;

    sz_size_t texts_bytes = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index)
        texts_bytes += candidates->get_length(candidates->handle, index);

    // The texts are only ever read by the device, so they go to plain device memory and cross once; only the
    // views and the distances, which the host writes or reads, need memory both sides address.
    sz_memory_allocator_t staging;
    sz_memory_allocator_init_unified(&staging);
    sz_size_t const shared_bytes = candidates->count * (sizeof(sz_string_view_t) + sizeof(sz_size_t)) + query_length;
    sz_ptr_t const shared = (sz_ptr_t)staging.allocate(shared_bytes, staging.handle);
    sz_ptr_t const flat = (sz_ptr_t)alloc->allocate(texts_bytes ? texts_bytes : 1, alloc->handle);
    sz_ptr_t texts = SZ_NULL;
    if (!shared || !flat || cudaMalloc((void **)&texts, texts_bytes ? texts_bytes : 1) != cudaSuccess) {
        if (shared) staging.free(shared, shared_bytes, staging.handle);
        if (flat) alloc->free(flat, texts_bytes ? texts_bytes : 1, alloc->handle);
        return sz_bad_alloc_k;
    }

    sz_string_view_t *const views = (sz_string_view_t *)shared;
    sz_size_t *const staged_distances = (sz_size_t *)(views + candidates->count);
    sz_ptr_t const staged_query = (sz_ptr_t)(staged_distances + candidates->count);
    for (sz_size_t byte = 0; byte != query_length; ++byte) staged_query[byte] = query_text[byte];

    sz_size_t written = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        sz_cptr_t const start = candidates->get_start(candidates->handle, index);
        for (sz_size_t byte = 0; byte != length; ++byte) flat[written + byte] = start[byte];
        views[index].start = texts + written, views[index].length = length;
        written += length;
    }
    cudaMemcpy(texts, flat, texts_bytes, cudaMemcpyHostToDevice);
    alloc->free(flat, texts_bytes ? texts_bytes : 1, alloc->handle);

    // Bound to the device's own accessors, so the staged round reaches the kernel exactly as a caller's own
    // device-resident sequence would - one code path from here on.
    sz_sequence_t staged_candidates;
    sz_status_t const bound = sz_sequence_from_string_views_cuda(views, candidates->count,
                                                                             &staged_candidates);
    if (bound != sz_success_k) {
        cudaFree(texts);
        staging.free(shared, shared_bytes, staging.handle);
        return bound;
    }

    sz_status_t const status = sz_levenshtein_distances_scheduled_cuda(staged_query, query_length, &staged_candidates,
                                                                       &staging, staged_distances, SZ_NULL);
    if (status == sz_success_k)
        for (sz_size_t index = 0; index != candidates->count; ++index) distances[index] = staged_distances[index];
    cudaFree(texts);
    staging.free(shared, shared_bytes, staging.handle);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_levenshtein_distance_cuda(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                         sz_size_t b_length, sz_memory_allocator_t *alloc,
                                                         sz_size_t *distance) {
    sz_string_view_t view;
    view.start = b, view.length = b_length;
    sz_sequence_t candidates;
    sz_sequence_from_string_views(&view, 1, &candidates);
    return sz_levenshtein_distances_cuda(a, a_length, &candidates, alloc, distance);
}

#pragma endregion CUDA

#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_LEVENSHTEIN_CUDA_CUH_
