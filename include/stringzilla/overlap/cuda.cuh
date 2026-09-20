/**
 *  @brief CUDA backend for window overlap: one thread per candidate, its chain walked through a ring of prefix
 *      hashes so every width is scored in one pass, and the prepared-query B-tree probed per thread.
 *  @file include/stringzilla/overlap/cuda.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzilla/overlap.h
 *
 *  The arithmetic is the serial tier's, reached from the device through `--expt-relaxed-constexpr`, so the scores
 *  are bit-identical rather than merely close. Integers, not doubles: both factors are below 2^32, so the product
 *  fits one @c u64 and the remainder by a constant lowers to a multiply-high - where an @c f64 reduction would run
 *  at the device's double-precision rate, a sixty-fourth of its single-precision one on consumer parts.
 *
 *  A candidate's chain is a dependent recurrence, so it stays on one thread; parallelism comes from the candidates,
 *  which is what the device has thousands of. The thread keeps the last @ref sz_overlap_cuda_ring_span_k prefix
 *  hashes rather than the whole chain, so its footprint is constant in the candidate's length and every width is
 *  scored in the same pass over the text.
 *
 *  Nothing here crosses the bus. The candidates, the scores and the allocator's scratch are already where the
 *  device reaches them and the texts are read in place, so a round costs one launch rather than a round trip -
 *  the difference between the bus rate and the device's own.
 */
#ifndef STRINGZILLA_OVERLAP_CUDA_CUH_
#define STRINGZILLA_OVERLAP_CUDA_CUH_

#include "stringzilla/types.cuh"

#include "stringzilla/overlap/serial.h"

#if SZ_USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

#pragma region CUDA

/** Prefix hashes one thread keeps; a power of two, so the ring index is a mask rather than a division. */
enum { sz_overlap_cuda_ring_span_k = 32 };

/** Widest window this backend scores, one short of the ring so a window's start and end never alias. */
enum { sz_overlap_cuda_widest_window_k = sz_overlap_cuda_ring_span_k - 1 };

/** Widths one call may ask for; the per-thread match counters are held in registers, so the bound is small. */
enum { sz_overlap_cuda_widths_max_k = 8 };

/** Candidates one block scores when the device cannot be asked; measured flat from 32 to 512 and off a cliff
 *  at 1024, so this is the ceiling of the flat range rather than a tuned figure. */
enum { sz_overlap_cuda_candidates_per_block_k = 512 };

/** The share of a block's shared memory a staged tree may take. Staging is worth about half again while the tree
 *  is small, and stops paying past that: the descent is data-dependent, and shared memory's banks handle a scatter
 *  worse than L1 does with its sector reuse. A fraction rather than a byte count, because the ceiling itself moves
 *  - a hundred kilobytes per multiprocessor on consumer Ada against more than twice that on the datacenter parts. */
enum { sz_overlap_cuda_shared_tree_share_k = 3 };

/** The prepared query as a kernel argument: the tree with its nodes device-reachable, beside the widths. */
typedef struct sz_overlap_cuda_query_t {
    sz_overlap_btree_t btree; /**< @c nodes must be device-reachable; the level bases travel by value. */
    sz_u32_t const *widths;   /**< The @b [window_widths] window widths, in bytes. */
    sz_u32_t const *powers;   /**< The @b [window_widths] @ref sz_overlap_window_power values, one per width. */
    sz_size_t widths_count;   /**< Entries in @c widths and @c powers, at most @ref sz_overlap_cuda_widths_max_k. */
    sz_size_t query_length;   /**< Bytes of the query the tree was built from. */
    sz_size_t nodes_count;    /**< Entries to stage in shared memory, or zero to descend through @c btree.nodes. */
} sz_overlap_cuda_query_t;

/** Whether the prepared query holds one raw @p window_hash, walking the tree a level at a time. */
SZ_DEVICE_INLINE sz_size_t sz_overlap_cuda_btree_probe_(sz_overlap_btree_t const *btree, sz_u32_t window_hash) {
    sz_u32_t const key = window_hash ^ (sz_u32_t)sz_overlap_sign_flip_k;
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 != btree->levels; ++level) {
        sz_u32_t const *const separators = btree->nodes +
                                           (btree->level_bases[level] + node) * sz_overlap_keys_per_node_k;
        node = node * sz_overlap_branches_per_node_k + sz_overlap_serial_branch_step_(separators, key);
    }
    sz_u32_t const *const leaf = btree->nodes +
                                 (btree->level_bases[btree->levels - 1] + node) * sz_overlap_keys_per_node_k;
    return sz_overlap_serial_leaf_step_(leaf, key);
}

/**
 *  @brief Scores one candidate per thread against the prepared query, one score per width.
 *
 *  The ring holds @c P(k) back to @c P(k - widest), so a window of any width up to that reads its start straight
 *  out of it and the text is walked once however many widths are asked for.
 *
 *  @p candidates travels whole and its accessors run here, on the device: one call per candidate, uniform across
 *  the warp, against multi-kilobyte texts. Nothing is flattened for the launch, and a caller whose sequence is
 *  some other layout entirely - a tape, a column, an index into someone else's arena - needs no conversion.
 */
static __global__ void sz_overlap_cuda_scores_kernel_(sz_overlap_cuda_query_t query, sz_sequence_t candidates,
                                                      sz_f32_t *scores) {
    // The whole tree, when the launch asked for it: every probe in the block then descends out of shared memory.
    extern __shared__ sz_u32_t staged_nodes_[];
    if (query.nodes_count) {
        for (sz_size_t entry = threadIdx.x; entry < query.nodes_count; entry += blockDim.x)
            staged_nodes_[entry] = query.btree.nodes[entry];
        __syncthreads();
        query.btree.nodes = staged_nodes_;
    }

    sz_size_t const candidate = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidates.count) return;

    sz_cptr_t const text = candidates.get_start(candidates.handle, candidate);
    sz_size_t const length = candidates.get_length(candidates.handle, candidate);
    sz_f32_t *const candidate_scores = scores + candidate * query.widths_count;

    sz_u32_t matches[sz_overlap_cuda_widths_max_k];
    for (sz_size_t index = 0; index != query.widths_count; ++index) matches[index] = 0;

    // `ring[k & mask]` is `P(k)`, so a window ending at `k` reads its start at `(k - width) & mask`. The base is
    // `256 mod p`, which is 256 itself, the modulus being the wider of the two.
    sz_size_t const mask = sz_overlap_cuda_ring_span_k - 1;
    sz_u32_t ring[sz_overlap_cuda_ring_span_k];
    ring[0] = 0;
    sz_u64_t prior = 0;
    for (sz_size_t position = 0; position != length; ++position) {
        prior = (prior * 256 + (sz_u64_t)(sz_u8_t)text[position]) % (sz_u64_t)sz_overlap_modulus_k;
        sz_size_t const ending = position + 1;
        ring[ending & mask] = (sz_u32_t)prior;
        for (sz_size_t index = 0; index != query.widths_count; ++index) {
            sz_size_t const width = query.widths[index];
            if (!width || width > ending) continue;
            sz_u64_t const start = (sz_u64_t)ring[(ending - width) & mask];
            sz_u64_t const shifted = start * query.powers[index] % (sz_u64_t)sz_overlap_modulus_k;
            sz_u64_t const residue = prior + (sz_u64_t)sz_overlap_modulus_k - shifted;
            matches[index] += (sz_u32_t)sz_overlap_cuda_btree_probe_(
                &query.btree, (sz_u32_t)(residue % (sz_u64_t)sz_overlap_modulus_k));
        }
    }

    for (sz_size_t index = 0; index != query.widths_count; ++index) {
        sz_size_t const width = query.widths[index];
        sz_bool_t const scored = width && width <= query.query_length && width <= length ? sz_true_k : sz_false_k;
        candidate_scores[index] =
            scored ? sz_overlap_share_(matches[index], length - width + 1, query.query_length - width + 1) : 0.0f;
    }
}

SZ_API_COMPTIME sz_status_t sz_overlap_scores_scheduled_cuda(
    sz_cptr_t query, sz_size_t query_length, sz_sequence_t const *candidates, sz_size_t const *window_widths,
    sz_size_t window_widths_count, sz_memory_allocator_t *alloc, sz_f32_t *scores, void *stream) {
    if (!window_widths_count || window_widths_count > sz_overlap_cuda_widths_max_k) return sz_unexpected_dimensions_k;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] > sz_overlap_cuda_widest_window_k) return sz_unexpected_dimensions_k;
    if (!candidates->count) return sz_success_k;
    // The handle is checked, never the accessors: those are the device's to call, so the host must not, and a
    // pointer is all this side can inspect. That the texts they answer are device-reachable is the caller's word.
    if (!sz_memory_reaches_device(scores)) return sz_device_memory_mismatch_k;
    if (!sz_memory_reaches_device(candidates->handle)) return sz_device_memory_mismatch_k;

    sz_size_t query_windows_total = 0;
    for (sz_size_t index = 0; index != window_widths_count; ++index)
        if (window_widths[index] && window_widths[index] <= query_length)
            query_windows_total += query_length - window_widths[index] + 1;

    // One allocation, widest alignment first so every view lands on its own boundary without padding. It has to
    // reach the device too: the kernel reads the tree, the widths and the powers straight out of it, and only
    // the chain is host-side working space.
    sz_size_t const nodes_count = sz_overlap_btree_entries(query_windows_total);
    sz_size_t const scratch_bytes = (query_length + 1) * sizeof(sz_f64_t) +
                                    (nodes_count + 2 * window_widths_count) * sizeof(sz_u32_t);
    sz_ptr_t const scratch = (sz_ptr_t)alloc->allocate(scratch_bytes, alloc->handle);
    if (!scratch) return sz_bad_alloc_k;
    if (!sz_memory_reaches_device(scratch)) {
        alloc->free(scratch, scratch_bytes, alloc->handle);
        return sz_device_memory_mismatch_k;
    }

    sz_f64_t *const chain = (sz_f64_t *)scratch;
    sz_u32_t *const nodes = (sz_u32_t *)(chain + query_length + 1);
    sz_u32_t *const widths = nodes + nodes_count;
    sz_u32_t *const powers = widths + window_widths_count;

    chain[0] = 0.0;
    sz_f64_t prior = 0.0;
    for (sz_size_t position = 0; position != query_length; ++position)
        prior = sz_overlap_f64x1_prefix_hash_step_serial(prior, query + position, chain + position + 1);

    // Every width's query window hashes share one tree, laid out in place by the serial tier.
    sz_size_t written = 0;
    for (sz_size_t index = 0; index != window_widths_count; ++index) {
        sz_size_t const width = window_widths[index];
        sz_f64_t const power = sz_overlap_window_power(width);
        widths[index] = (sz_u32_t)width, powers[index] = (sz_u32_t)power;
        if (!width || width > query_length) continue;
        for (sz_size_t window = 0; window + width <= query_length; ++window)
            sz_overlap_f64x1_window_hash_step_serial(chain + window, chain + window + width, power,
                                                     nodes + written + window);
        written += query_length - width + 1;
    }
    sz_overlap_cuda_query_t device_query;
    sz_overlap_btree_prepare(nodes, sz_overlap_u32x1_btree_sort_serial(nodes, written), &device_query.btree);
    device_query.widths = widths;
    device_query.powers = powers;
    device_query.widths_count = window_widths_count;
    device_query.query_length = query_length;
    // What this device will hand one block, not what the part it was tuned on would have.
    int shared_limit = 0;
    sz_size_t const tree_bytes = nodes_count * sizeof(sz_u32_t);
    device_query.nodes_count = 0;
    if (cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, 0) == cudaSuccess &&
        tree_bytes * sz_overlap_cuda_shared_tree_share_k <= (sz_size_t)shared_limit)
        device_query.nodes_count = nodes_count;

    // The sequence goes to the kernel whole - its accessors are the device's to call, once per candidate.
    // The block size is clamped to what this kernel's register footprint permits on this device, rather than to
    // a literal answering for a GPU it has never seen; the measured range above that floor is flat.
    cudaStream_t const on = (cudaStream_t)stream;
    sz_size_t const staged_bytes = device_query.nodes_count * sizeof(sz_u32_t);
    sz_size_t per_block = sz_overlap_cuda_candidates_per_block_k;
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, (void const *)sz_overlap_cuda_scores_kernel_) == cudaSuccess &&
        attributes.maxThreadsPerBlock > 0 && (sz_size_t)attributes.maxThreadsPerBlock < per_block)
        per_block = (sz_size_t)attributes.maxThreadsPerBlock;
    sz_size_t const blocks = (candidates->count + per_block - 1) / per_block;

    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = (unsigned)per_block, block.y = 1, block.z = 1;
    void *arguments[3];
    sz_sequence_t launched_candidates = *candidates;
    arguments[0] = &device_query, arguments[1] = &launched_candidates, arguments[2] = &scores;
    if (cudaLaunchKernel((void const *)sz_overlap_cuda_scores_kernel_, grid, block, arguments, staged_bytes, on) !=
        cudaSuccess) {
        alloc->free(scratch, scratch_bytes, alloc->handle);
        return sz_device_code_mismatch_k;
    }

    // Only this stream is waited on, so the caller's other work on the device keeps running.
    cudaError_t const finished = cudaStreamSynchronize(on);
    alloc->free(scratch, scratch_bytes, alloc->handle);
    return finished == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

SZ_API_COMPTIME sz_status_t sz_overlap_scores_cuda(sz_cptr_t query, sz_size_t query_length,
                                                   sz_sequence_t const *candidates, sz_size_t const *window_widths,
                                                   sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                   sz_f32_t *scores) {
    // The one probe is the strict verb's own: it refuses whatever the device cannot reach, and only then is there
    // anything to stage. A caller already holding its data on the device pays nothing for the attempt.
    sz_status_t const resident = sz_overlap_scores_scheduled_cuda(query, query_length, candidates, window_widths,
                                                                  window_widths_count, alloc, scores, SZ_NULL);
    if (resident != sz_device_memory_mismatch_k) return resident;

    sz_size_t texts_bytes = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index)
        texts_bytes += candidates->get_length(candidates->handle, index);
    sz_size_t const scores_count = candidates->count * window_widths_count;

    // The texts are only ever read by the device, so they go to plain device memory and cross once; only the
    // views and the scores, which the host writes or reads, need memory both sides address.
    sz_memory_allocator_t staging;
    sz_memory_allocator_init_unified(&staging);
    sz_size_t const shared_bytes = candidates->count * sizeof(sz_string_view_t) + scores_count * sizeof(sz_f32_t);
    sz_ptr_t const shared = (sz_ptr_t)staging.allocate(shared_bytes, staging.handle);
    sz_ptr_t const flat = (sz_ptr_t)alloc->allocate(texts_bytes, alloc->handle);
    sz_ptr_t texts = SZ_NULL;
    if (!shared || !flat || cudaMalloc((void **)&texts, texts_bytes) != cudaSuccess) {
        if (shared) staging.free(shared, shared_bytes, staging.handle);
        if (flat) alloc->free(flat, texts_bytes, alloc->handle);
        return sz_bad_alloc_k;
    }

    sz_string_view_t *const views = (sz_string_view_t *)shared;
    sz_f32_t *const staged_scores = (sz_f32_t *)(views + candidates->count);
    sz_size_t written = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        sz_cptr_t const start = candidates->get_start(candidates->handle, index);
        for (sz_size_t byte = 0; byte != length; ++byte) flat[written + byte] = start[byte];
        views[index].start = texts + written, views[index].length = length;
        written += length;
    }
    cudaMemcpy(texts, flat, texts_bytes, cudaMemcpyHostToDevice);
    alloc->free(flat, texts_bytes, alloc->handle);

    // Bound to the device's own accessors, so the staged round reaches the kernel exactly as a caller's own
    // device-resident sequence would - one code path from here on.
    sz_sequence_t staged_candidates;
    sz_status_t const bound = sz_sequence_from_string_views_cuda(views, candidates->count, &staged_candidates);
    if (bound != sz_success_k) {
        cudaFree(texts);
        staging.free(shared, shared_bytes, staging.handle);
        return bound;
    }

    // The strict verb joins the default stream before returning, so the staged scores are settled by the time
    // they are read back and the staging is released.
    sz_status_t const status = sz_overlap_scores_scheduled_cuda(query, query_length, &staged_candidates, window_widths,
                                                                window_widths_count, &staging, staged_scores, SZ_NULL);
    if (status == sz_success_k)
        for (sz_size_t index = 0; index != scores_count; ++index) scores[index] = staged_scores[index];
    cudaFree(texts);
    staging.free(shared, shared_bytes, staging.handle);
    return status;
}

SZ_API_COMPTIME sz_status_t sz_overlap_score_cuda(sz_cptr_t query, sz_size_t query_length, sz_cptr_t candidate,
                                                  sz_size_t candidate_length, sz_size_t const *window_widths,
                                                  sz_size_t window_widths_count, sz_memory_allocator_t *alloc,
                                                  sz_f32_t *scores) {
    sz_string_view_t view;
    view.start = candidate, view.length = candidate_length;
    sz_sequence_t candidates;
    sz_sequence_from_string_views(&view, 1, &candidates);
    return sz_overlap_scores_cuda(query, query_length, &candidates, window_widths, window_widths_count, alloc, scores);
}

#pragma endregion CUDA

#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_OVERLAP_CUDA_CUH_
