/**
 *  @file include/stringzilla/strings/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief ROCm host side of the string collection kernels in `strings/simt.cuh`: the residency
 *      checks, the scratch, the one readback a verb waits for, and the @c _rocm exports.
 *
 *  Every verb reads tapes and writes tapes the device reaches. It measures, reads the payload size
 *  back, takes the output's blocks, and writes, so a call synchronizes its stream once, before it
 *  allocates. The output reuses the tape's blocks only when their sizes match and nothing reads
 *  them, and fresh blocks replace the old ones only once the write is queued.
 *
 *  @sa include/stringzilla/strings/simt.cuh
 *  @sa include/stringzilla/strings/cuda.cuh
 */
#ifndef STRINGZILLA_STRINGS_ROCM_CUH_
#define STRINGZILLA_STRINGS_ROCM_CUH_

#include "stringzilla/rocm.cuh"
#include "stringzilla/strings/simt.cuh"
#include "stringzilla/utf8_norm/rocm.cuh" // `sz_utf8_norm_upload_rocm_`

#if STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_ROCM

/** Whether the device reaches every block a call over @p count strings touches: the tape @p strings
 *  views, @p order, and the blocks a non-null @p tape already holds. */
STRINGZILLA_INLINE sz_status_t sz_strings_check_rocm_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_strings_span_t const *tape) {
    sz_strings_view_t held;
    if (strings->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    if (count && (!sz_strings_accessible_rocm_(strings) || (order && !sz_memory_accessible_rocm_(order))))
        return sz_device_memory_mismatch_k;
    if (!tape || !sz_strings_offsets_block_(tape)) return sz_success_k;
    sz_strings_from_span(tape, &held);
    return sz_strings_accessible_rocm_(&held) ? sz_success_k : sz_device_memory_mismatch_k;
}

/** Launches @p kernel over @p items, @p items_per_block to a block, capping the grid where the
 *  kernel strides past it. */
STRINGZILLA_INLINE sz_status_t sz_strings_launch_rocm_(void const *kernel, sz_size_t items, sz_size_t items_per_block,
                                                       void **arguments, sz_stream_t stream) {
    sz_size_t const blocks = sz_min_of_two(sz_size_divide_round_up(items, items_per_block),
                                           (sz_size_t)sz_strings_blocks_max_simt_k);
    return sz_launch_rocm_(kernel, dim3((unsigned)blocks), dim3(sz_strings_threads_simt_k), arguments, 0, stream);
}

/** Allocates on @p stream the scratch of @p count strings, with no malformed position found yet;
 *  none for no strings. */
STRINGZILLA_INLINE sz_status_t sz_strings_scratch_rocm_(sz_size_t count, sz_u64_t **scratch, sz_stream_t stream) {
    if (!count) return sz_success_k;
    if (count > STRINGZILLA_SIZE_MAX / (2 * sizeof(sz_u64_t))) return sz_bad_alloc_k;
    sz_size_t const words = count + 3 + sz_size_divide_round_up(count, (sz_size_t)sz_strings_threads_simt_k);
    if (hipMallocAsync((void **)scratch, words * sizeof(sz_u64_t), (hipStream_t)stream) != hipSuccess)
        return sz_bad_alloc_k;
    return sz_fill_rocm_(*scratch + count + 1, sizeof(sz_u64_t), 0xFF, stream);
}

/** Scans the lengths in @p scratch into offsets and reads back how many strings precede the first
 *  malformed one and the payload they take, synchronizing @p stream once. */
STRINGZILLA_INLINE sz_status_t sz_strings_measured_rocm_(sz_u64_t *scratch, sz_size_t count, sz_size_t *valid,
                                                         sz_size_t *payload, sz_stream_t stream) {
    sz_size_t blocks_count = sz_size_divide_round_up(count, (sz_size_t)sz_strings_threads_simt_k), target_offset = 0;
    sz_size_t *blocks = (sz_size_t *)(scratch + count + 3);
    sz_u64_t report[2] = {0, 0};
    void *local_arguments[] = {&scratch, &scratch, &count, &blocks};
    void *blocks_arguments[] = {&blocks, &blocks_count, &scratch, &count, &target_offset};
    void *offsets_arguments[] = {&scratch, &count, &blocks};
    void *payload_arguments[] = {&scratch, &count};
    dim3 const grid((unsigned)blocks_count), block(sz_strings_threads_simt_k), single(1);
    sz_status_t status = sz_success_k;
    if (count) {
        status = sz_launch_rocm_((void const *)&sz_strings_scan_local_simt_kernel_, grid, block, local_arguments, 0,
                                 stream);
        if (status == sz_success_k)
            status = sz_launch_rocm_((void const *)&sz_strings_scan_blocks_simt_kernel_, single, block,
                                     blocks_arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_launch_rocm_((void const *)&sz_strings_scan_offsets_simt_kernel_, grid, block,
                                     offsets_arguments, 0, stream);
        if (status == sz_success_k)
            status = sz_launch_rocm_((void const *)&sz_strings_payload_simt_kernel_, single,
                                     dim3(sz_strings_payload_threads_simt_k), payload_arguments, 0, stream);
        if (status == sz_success_k) status = sz_copy_rocm_(report, scratch + count + 1, sizeof(report), stream);
    }
    // The host reads the sizes of the output's old blocks next, so the stream drains even for none.
    if (status == sz_success_k)
        status = hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
    *valid = (sz_size_t)sz_min_of_two(report[0], (sz_u64_t)count), *payload = (sz_size_t)report[1];
    return status;
}

/** Reads offsets `[0]` and `[count]` of the tape @p strings views back, synchronizing @p stream. */
STRINGZILLA_INLINE sz_status_t sz_strings_bounds_rocm_(sz_strings_view_t const *strings, sz_size_t count,
                                                       sz_size_t *first, sz_size_t *end, sz_stream_t stream) {
    sz_bool_t const wide = (sz_bool_t)(strings->layout == sz_strings_u64_tape_k);
    sz_u64_t wide_bounds[2] = {0, 0};
    sz_u32_t narrow_bounds[2] = {0, 0};
    sz_status_t status = sz_success_k;
    if (count && wide) {
        status = sz_copy_rocm_(wide_bounds, strings->data.u64_tape.offsets, sizeof(sz_u64_t), stream);
        if (status == sz_success_k)
            status = sz_copy_rocm_(wide_bounds + 1, strings->data.u64_tape.offsets + count, sizeof(sz_u64_t), stream);
    }
    else if (count) {
        status = sz_copy_rocm_(narrow_bounds, strings->data.u32_tape.offsets, sizeof(sz_u32_t), stream);
        if (status == sz_success_k)
            status = sz_copy_rocm_(narrow_bounds + 1, strings->data.u32_tape.offsets + count, sizeof(sz_u32_t), stream);
    }
    if (status == sz_success_k)
        status = hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
    *first = wide ? (sz_size_t)wide_bounds[0] : narrow_bounds[0];
    *end = wide ? (sz_size_t)wide_bounds[1] : narrow_bounds[1];
    return status;
}

/** Sets the first offset of @p tape to zero, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_strings_open_rocm_(sz_strings_span_t const *tape, sz_stream_t stream) {
    return sz_fill_rocm_(sz_strings_offsets_block_(tape), sz_strings_offsets_bytes_(tape->layout, 0), 0, stream);
}

/** Rewrites @p count strings into @p tape through the @p measure and @p write kernels, which take
 *  the strings, their order and count, the escape kind or normal form @p mode points at, and a
 *  scratch block, or names the first string @p measure finds malformed in @p malformed_index. */
STRINGZILLA_INLINE sz_status_t sz_strings_rewrite_rocm_(void const *measure, void const *write,
                                                        sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, void *mode, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_strings_view_t launch_strings = *strings;
    sz_size_t launch_count = count, valid = 0, payload = 0;
    sz_u64_t *scratch = STRINGZILLA_NULL;
    sz_strings_span_t output;
    memset(&output, 0, sizeof(output));
    void *measure_arguments[] = {&launch_strings, &order, &launch_count, mode, &scratch};
    void *write_arguments[] = {&launch_strings, &order, &launch_count, mode, &scratch, &output};
    sz_status_t status = sz_strings_check_rocm_(strings, order, count, tape);
    if (status == sz_success_k) status = sz_strings_scratch_rocm_(count, &scratch, stream);
    if (status == sz_success_k && count)
        status = sz_strings_launch_rocm_(measure, count, sz_strings_threads_simt_k, measure_arguments, stream);
    if (status == sz_success_k) status = sz_strings_measured_rocm_(scratch, count, &valid, &payload, stream);
    if (status == sz_success_k && valid != count) *malformed_index = valid, status = sz_invalid_utf8_k;
    if (status == sz_success_k) status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, stream);
    if (status == sz_success_k) status = sz_strings_open_rocm_(&output, stream);
    if (status == sz_success_k && count)
        status = sz_strings_launch_rocm_(write, count, sz_strings_threads_simt_k, write_arguments, stream);
    if (status == sz_success_k) sz_strings_commit_(tape, &output, allocator, stream);
    else sz_strings_release_(tape, &output, payload, allocator, stream);
    if (scratch && hipFreeAsync(scratch, (hipStream_t)stream) != hipSuccess && status == sz_success_k)
        status = sz_device_code_mismatch_k;
    return status;
}

/** Gathers @p count strings into @p tape through @p lut: a warp to a string in @p order, or one run
 *  of bytes from @p first to @p end for a tape read in its own order. */
STRINGZILLA_INLINE sz_status_t sz_strings_map_rocm_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_lut_simt_t const *lut, sz_size_t first,
                                                    sz_size_t end, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream) {
    sz_strings_view_t launch_strings = *strings;
    sz_strings_lut_simt_t launch_lut = *lut;
    sz_size_t launch_count = count, valid = count, payload = end - first;
    sz_u64_t *scratch = STRINGZILLA_NULL;
    sz_cptr_t const data = strings->layout == sz_strings_u64_tape_k ? strings->data.u64_tape.data
                                                                    : strings->data.u32_tape.data;
    sz_u8_t const *source = (sz_u8_t const *)data + first;
    sz_u8_t *target = STRINGZILLA_NULL;
    sz_strings_span_t output;
    void *measure_arguments[] = {&launch_strings, &order, &launch_count, &scratch};
    void *gather_arguments[] = {&launch_strings, &order, &launch_count, &launch_lut, &scratch, &output};
    void *lookup_arguments[] = {&source, &payload, &launch_lut, &target};
    void *rebase_arguments[] = {&launch_strings, &launch_count, &output};
    sz_status_t status = sz_success_k;
    if (order) {
        status = sz_strings_scratch_rocm_(count, &scratch, stream);
        if (status == sz_success_k && count)
            status = sz_strings_launch_rocm_((void const *)&sz_strings_lengths_simt_kernel_, count,
                                             sz_strings_threads_simt_k, measure_arguments, stream);
        if (status == sz_success_k) status = sz_strings_measured_rocm_(scratch, count, &valid, &payload, stream);
    }
    if (status == sz_success_k) status = sz_strings_acquire_(strings, tape, count, payload, allocator, &output, stream);
    if (status == sz_success_k) {
        target = (sz_u8_t *)sz_strings_payload_(&output);
        status = sz_strings_open_rocm_(&output, stream);
        if (status == sz_success_k && order && count)
            status = sz_strings_launch_rocm_((void const *)&sz_strings_gather_simt_kernel_, count,
                                             sz_strings_threads_simt_k / sz_strings_lanes_simt_k, gather_arguments,
                                             stream);
        if (status == sz_success_k && !order && payload)
            status = sz_strings_launch_rocm_((void const *)&sz_strings_lookup_simt_kernel_, payload,
                                             sz_strings_threads_simt_k, lookup_arguments, stream);
        if (status == sz_success_k && !order && count)
            status = sz_strings_launch_rocm_((void const *)&sz_strings_rebase_simt_kernel_, count + 1,
                                             sz_strings_threads_simt_k, rebase_arguments, stream);
        if (status == sz_success_k) sz_strings_commit_(tape, &output, allocator, stream);
        else sz_strings_release_(tape, &output, payload, allocator, stream);
    }
    if (scratch && hipFreeAsync(scratch, (hipStream_t)stream) != hipSuccess && status == sz_success_k)
        status = sz_device_code_mismatch_k;
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_strings_to_tape_rocm_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_allocator_t *allocator,
                                                        sz_strings_span_t *tape, sz_stream_t stream) {
    sz_strings_lut_simt_t identity;
    sz_size_t first = 0, end = 0;
    for (sz_size_t byte = 0; byte != 256; ++byte) identity.bytes[byte] = (sz_u8_t)byte;
    sz_status_t status = sz_strings_check_rocm_(strings, order, count, tape);
    if (status == sz_success_k && !order) status = sz_strings_bounds_rocm_(strings, count, &first, &end, stream);
    if (status != sz_success_k) return status;
    if (!order && sz_strings_aliases_(strings, tape) && !first && count == tape->count) return sz_success_k;
    return sz_strings_map_rocm_(strings, order, count, &identity, first, end, allocator, tape, stream);
}

STRINGZILLA_INLINE sz_status_t sz_strings_lookup_rocm_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, char const lut[sz_at_least_(256)],
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream) {
    sz_strings_lut_simt_t table;
    sz_size_t first = 0, end = 0, length = 0;
    sz_u8_t *bytes = STRINGZILLA_NULL;
    void *arguments[] = {&bytes, &length, &table, &bytes};
    memcpy(table.bytes, lut, sizeof(table.bytes));
    sz_status_t status = sz_strings_check_rocm_(strings, order, count, tape);
    if (status == sz_success_k && !order) status = sz_strings_bounds_rocm_(strings, count, &first, &end, stream);
    if (status != sz_success_k) return status;
    if (order || !sz_strings_aliases_(strings, tape) || count != tape->count)
        return sz_strings_map_rocm_(strings, order, count, &table, first, end, allocator, tape, stream);
    bytes = (sz_u8_t *)sz_strings_payload_(tape) + first, length = end - first;
    if (length)
        status = sz_strings_launch_rocm_((void const *)&sz_strings_lookup_simt_kernel_, length,
                                         sz_strings_threads_simt_k, arguments, stream);
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_strings_to_rows_rocm_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                        sz_size_t count, sz_strings_rows_t const *rows,
                                                        sz_size_t *longest, sz_stream_t stream) {
    sz_strings_view_t launch_strings = *strings;
    sz_strings_rows_t launch_rows = *rows;
    sz_size_t launch_count = count;
    void *arguments[] = {&launch_strings, &order, &launch_count, &launch_rows, &longest};
    sz_status_t status = sz_strings_check_rocm_(strings, order, count, STRINGZILLA_NULL);
    if (status != sz_success_k) return status;
    if (!sz_memory_accessible_rocm_(longest) || (count && rows->width && !sz_memory_accessible_rocm_(rows->data)))
        return sz_device_memory_mismatch_k;
    status = sz_fill_rocm_(longest, sizeof(sz_size_t), 0, stream);
    if (status == sz_success_k && count)
        status = sz_strings_launch_rocm_((void const *)&sz_strings_to_rows_simt_kernel_, count,
                                         sz_strings_threads_simt_k / sz_strings_lanes_simt_k, arguments, stream);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_strings_to_tape_rocm_(strings, order, count, allocator, tape, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                    sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_strings_to_rows_rocm_(strings, order, count, rows, longest, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_escape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                   sz_strings_span_t *tape, sz_stream_t stream) {
    int caller = 0;
    sz_size_t malformed_index = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_strings_rewrite_rocm_((void const *)&sz_strings_escape_lengths_simt_kernel_,
                                      (void const *)&sz_strings_escape_simt_kernel_, strings, order, count, &kind,
                                      allocator, tape, &malformed_index, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_unescape_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                     sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_strings_rewrite_rocm_((void const *)&sz_strings_unescape_lengths_simt_kernel_,
                                      (void const *)&sz_strings_unescape_simt_kernel_, strings, order, count, &kind,
                                      allocator, tape, malformed_index, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_lookup_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, char const lut[sz_at_least_(256)],
                                                   sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                   sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_strings_lookup_rocm_(strings, order, count, lut, allocator, tape, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_normalize_rocm(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_normal_form_t form, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_stream_t stream) {
    int caller = 0;
    sz_size_t malformed_index = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_utf8_norm_upload_rocm_(stream);
    if (status == sz_success_k)
        status = sz_strings_rewrite_rocm_((void const *)&sz_strings_normalize_lengths_simt_kernel_,
                                          (void const *)&sz_strings_normalize_simt_kernel_, strings, order, count,
                                          &form, allocator, tape, &malformed_index, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_STRINGS_ROCM_CUH_
