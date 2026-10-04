/**
 *  @file include/stringzilla/utf8_uncased_fold/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of UTF-8 case folding: the tiles a round cuts a text into, its one launch,
 *      and the @c _rocm export, over the kernel of `utf8_uncased_fold/simt.cuh`.
 *
 *  @sa include/stringzilla/utf8_uncased_fold/simt.cuh
 *  @sa include/stringzilla/utf8_uncased_fold/cuda.cuh
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_ROCM_CUH_
#define STRINGZILLA_UTF8_UNCASED_FOLD_ROCM_CUH_

#include "stringzilla/rocm.cuh"
#include "stringzilla/utf8_uncased_fold/simt.cuh"

#if STRINGZILLA_TARGET_ROCM && defined(__HIP__)

#ifdef __cplusplus
extern "C" {
#endif

/** Bytes one thread folds at the least, which keeps a short text to few tiles. */
enum { sz_utf8_uncased_fold_thread_bytes_rocm_k = 64 };

/**
 *  @brief Folds @p source into @p target on the caller's current device, the length landing in
 *      @p target_length once @p stream is joined.
 *  @return @c sz_success_k once enqueued, @c sz_unexpected_dimensions_k for a text whose fold the
 *      length slot cannot count, or @c sz_device_memory_mismatch_k when a buffer or the slot is not
 *      memory the device reaches.
 *  @note Enqueues and returns, allocating nothing and joining nothing.
 */
STRINGZILLA_INLINE sz_status_t sz_utf8_uncased_fold_rocm_(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target,
                                                          sz_size_t *target_length, sz_stream_t stream) {
    sz_u8_t const *launch_source = (sz_u8_t const *)source;
    sz_u8_t *launch_target = (sz_u8_t *)target;
    sz_size_t *launch_target_length = target_length;
    sz_size_t launch_length = source_length, tile_bytes, tiles;
    void *arguments[6];
    dim3 grid, block;
    sz_status_t status;
    if ((sz_u64_t)source_length > ((sz_u64_t)1 << sz_chain_chained_shift_k) / sz_utf8_fold_max_expansion_k)
        return sz_unexpected_dimensions_k;
    if (!sz_memory_reaches_rocm_(target_length)) return sz_device_memory_mismatch_k;
    if (source_length && (!sz_memory_reaches_rocm_(source) || !sz_memory_reaches_rocm_(target)))
        return sz_device_memory_mismatch_k;
    status = sz_fill_rocm_(target_length, sizeof(sz_size_t), 0, stream);
    if (status != sz_success_k || !source_length) return status;

    tiles = sz_chain_tiles_simt_(
        source_length, (sz_size_t)sz_utf8_uncased_fold_threads_simt_k * sz_utf8_uncased_fold_thread_bytes_rocm_k,
        &tile_bytes);
    grid.x = (unsigned)tiles, grid.y = 1, grid.z = 1;
    block.x = sz_utf8_uncased_fold_threads_simt_k, block.y = 1, block.z = 1;
    arguments[0] = &launch_source, arguments[1] = &launch_length, arguments[2] = &tile_bytes;
    arguments[3] = &tiles, arguments[4] = &launch_target, arguments[5] = &launch_target_length;
    return sz_launch_rocm_((void const *)sz_utf8_uncased_fold_simt_kernel_, grid, block, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_utf8_uncased_fold_scoped_rocm_(sz_cptr_t source, sz_size_t source_length,
                                                                 sz_ptr_t target, sz_size_t *target_length,
                                                                 sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_utf8_uncased_fold_rocm_(source, source_length, target, target_length, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_rocm(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target,
                                                      sz_size_t *target_length, sz_stream_t stream) {
    return sz_utf8_uncased_fold_scoped_rocm_(source, source_length, target, target_length, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_ROCM && defined(__HIP__)
#endif // STRINGZILLA_UTF8_UNCASED_FOLD_ROCM_CUH_
