/**
 *  @file include/stringzilla/utf8_norm/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of Unicode normalization: the tables' upload, the tiles a round cuts a
 *      text into, its one launch, and the @c _rocm export, over the kernel of `utf8_norm/simt.cuh`.
 *
 *  @sa include/stringzilla/utf8_norm/simt.cuh
 *  @sa include/stringzilla/utf8_norm/cuda.cuh
 */
#ifndef STRINGZILLA_UTF8_NORM_ROCM_CUH_
#define STRINGZILLA_UTF8_NORM_ROCM_CUH_

#include "stringzilla/rocm.cuh"
#include "stringzilla/utf8_norm/simt.cuh"

#if STRINGZILLA_TARGET_ROCM && defined(__HIP__)

#ifdef __cplusplus
extern "C" {
#endif

/** Bytes one thread normalizes at the least, which keeps a short text to few tiles. */
enum { sz_utf8_norm_thread_bytes_rocm_k = 64 };

/** Copies every table the device lookups read into the device's copies, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_utf8_norm_upload_rocm_(sz_stream_t stream) {
    sz_status_t status = sz_copy_to_symbol_rocm_(sz_utf8_norm_stage1_simt_, sz_utf8_norm_stage1_,
                                                 sizeof(sz_utf8_norm_stage1_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_stage2_simt_, sz_utf8_norm_stage2_, sizeof(sz_utf8_norm_stage2_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_stage3_simt_, sz_utf8_norm_stage3_, sizeof(sz_utf8_norm_stage3_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_props_simt_, sz_utf8_norm_props_, sizeof(sz_utf8_norm_props_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_decomp_simt_, sz_utf8_norm_decomp_, sizeof(sz_utf8_norm_decomp_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_pool_simt_, sz_utf8_norm_pool_, sizeof(sz_utf8_norm_pool_),
                                         stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_pool_astral_simt_, sz_utf8_norm_pool_astral_,
                                         sizeof(sz_utf8_norm_pool_astral_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_compose_starters_simt_, sz_utf8_norm_compose_starters_,
                                         sizeof(sz_utf8_norm_compose_starters_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_compose_partner_simt_, sz_utf8_norm_compose_partner_,
                                         sizeof(sz_utf8_norm_compose_partner_), stream);
    if (status == sz_success_k)
        status = sz_copy_to_symbol_rocm_(sz_utf8_norm_compose_value_simt_, sz_utf8_norm_compose_value_,
                                         sizeof(sz_utf8_norm_compose_value_), stream);
    return status;
}

/**
 *  @brief Normalizes @p source into @p target on the caller's current device, the length landing in
 *      @p target_length once @p stream is joined.
 *  @return @c sz_success_k once enqueued, @c sz_unexpected_dimensions_k for a text whose
 *      normalization the length slot cannot count, or @c sz_device_memory_mismatch_k when a buffer
 *      or the slot is not memory the device reaches.
 *  @note Allocates nothing and joins nothing; the tables travel on @p stream ahead of the kernel.
 */
STRINGZILLA_INLINE sz_status_t sz_utf8_norm_rocm_(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                                  sz_ptr_t target, sz_size_t *target_length, sz_stream_t stream) {
    sz_u8_t const *launch_source = (sz_u8_t const *)source;
    sz_u8_t *launch_target = (sz_u8_t *)target;
    sz_size_t *launch_target_length = target_length;
    sz_size_t launch_length = source_length, tile_bytes, tiles;
    sz_normal_form_t launch_form = form;
    void *arguments[7];
    dim3 grid, block;
    sz_status_t status;
    if ((sz_u64_t)source_length > ((sz_u64_t)1 << sz_chain_chained_shift_k) / sz_utf8_norm_decomp_max_k)
        return sz_unexpected_dimensions_k;
    if (!sz_memory_accessible_rocm_(target_length)) return sz_device_memory_mismatch_k;
    if (source_length && (!sz_memory_accessible_rocm_(source) || !sz_memory_accessible_rocm_(target)))
        return sz_device_memory_mismatch_k;
    status = sz_fill_rocm_(target_length, sizeof(sz_size_t), 0, stream);
    if (status != sz_success_k || !source_length) return status;
    status = sz_utf8_norm_upload_rocm_(stream);
    if (status != sz_success_k) return status;

    tiles = sz_chain_tiles_simt_(
        source_length, (sz_size_t)sz_utf8_norm_threads_simt_k * sz_utf8_norm_thread_bytes_rocm_k, &tile_bytes);
    grid.x = (unsigned)tiles, grid.y = 1, grid.z = 1;
    block.x = sz_utf8_norm_threads_simt_k, block.y = 1, block.z = 1;
    arguments[0] = &launch_source, arguments[1] = &launch_length, arguments[2] = &launch_form;
    arguments[3] = &tile_bytes, arguments[4] = &tiles, arguments[5] = &launch_target;
    arguments[6] = &launch_target_length;
    return sz_launch_rocm_((void const *)sz_utf8_norm_simt_kernel_, grid, block, arguments, 0, stream);
}

STRINGZILLA_INLINE sz_status_t sz_utf8_norm_scoped_rocm_(sz_cptr_t source, sz_size_t source_length,
                                                         sz_normal_form_t form, sz_ptr_t target,
                                                         sz_size_t *target_length, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_utf8_norm_rocm_(source, source_length, form, target, target_length, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_utf8_norm_rocm(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                              sz_ptr_t target, sz_size_t *target_length, sz_stream_t stream) {
    return sz_utf8_norm_scoped_rocm_(source, source_length, form, target, target_length, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_ROCM && defined(__HIP__)
#endif // STRINGZILLA_UTF8_NORM_ROCM_CUH_
