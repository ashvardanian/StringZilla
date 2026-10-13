/**
 *  @file include/stringzilla/strings/metal.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Metal host side of the string collection kernels in `strings/metal.metal`: the bindings,
 *      the scratch, the one readback a verb waits for, and the @c _metal exports.
 *
 *  Every block a verb reads or keeps must come from @ref sz_allocator_init_unified_metal on the
 *  device of its stream, and the strings it reads must be tapes. A verb measures in one command
 *  buffer, waits for it, reads the payload size straight out of the shared scratch, resizes the
 *  output through the allocator, and writes in a second command buffer, which it leaves running. A
 *  tape the output shares a block with is read into fresh blocks, and the command buffer reading
 *  the old ones keeps them alive past their release.
 *
 *  @sa include/stringzilla/strings/metal.metal, the kernels this embeds
 *  @sa include/stringzilla/strings/cuda.cuh, the CUDA sibling
 */
#ifndef STRINGZILLA_STRINGS_METAL_H_
#define STRINGZILLA_STRINGS_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/strings/serial.h"

#if STRINGZILLA_TARGET_METAL

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL source of the strings kernels, compiled once per device behind the shared prelude and
 *  the normalization tables and engine its normalizing kernels call. */
static char const sz_strings_source_metal_[] = {
#embed "stringzilla/utf8_norm/tables.h"
    ,
#embed "stringzilla/utf8_norm/metal.metal"
    ,
#embed "stringzilla/strings/metal.metal"
    , 0};
#pragma clang diagnostic pop

/** The launch shape every strings kernel shares, and the buffers a dispatch binds, in the order the
 *  kernels number them, the launch record after the last. */
enum {

    /** Threads to a threadgroup, which the scans and the staged lookup table assume. */
    sz_strings_threads_metal_k = 256,

    /** Strings to a threadgroup in the kernels giving each string a simdgroup of 32. */
    sz_strings_simdgroups_metal_k = 8,

    /** Threadgroups a dispatch striding over bytes takes at most. */
    sz_strings_groups_max_metal_k = 65535,

    /** The payload of the strings read. */
    sz_strings_data_buffer_metal_k = 0,

    /** Their offsets. */
    sz_strings_offsets_buffer_metal_k = 1,

    /** The order they are read in, when one is given. */
    sz_strings_order_buffer_metal_k = 2,

    /** The scratch: the lengths, then the offsets, and a report per measuring threadgroup. */
    sz_strings_scratch_buffer_metal_k = 3,

    /** The payload or the rows written. */
    sz_strings_target_buffer_metal_k = 4,

    /** The offsets written, or the longest length. */
    sz_strings_target_offsets_buffer_metal_k = 5,

    /** Buffers a dispatch binds before its launch record. */
    sz_strings_buffers_metal_k = 6,
};

/** The launch record, laid out as the kernels' record of the same name. */
typedef struct sz_strings_arguments_metal_t {

    /** Strings a dispatch reads or writes. */
    sz_u64_t count;

    /** The host address of the order, or zero for the strings read in their own. */
    sz_u64_t order;

    /** The layout of the tape read, as @c sz_strings_layout_t numbers it. */
    sz_u64_t layout;

    /** The layout of the tape written. */
    sz_u64_t target_layout;

    /** The escape kind or the normal form, as @c sz_escape_kind_t or @c sz_normal_form_t. */
    sz_u64_t mode;

    /** Bytes one run of a lookup maps. */
    sz_u64_t length;

    /** Bytes in every row. */
    sz_u64_t width;

    /** Bytes between the starts of consecutive rows. */
    sz_u64_t stride;

    /** Threadgroups of the measuring dispatch, each reporting once. */
    sz_u64_t tiles;

    /** The table a lookup maps every byte through. */
    sz_u8_t lut[256];
} sz_strings_arguments_metal_t;

/** The launch record of the scans, laid out as the kernels' record of the same name. */
typedef struct sz_strings_scan_arguments_metal_t {

    /** Where the first offset starts. */
    sz_u64_t offset;

    /** Lengths scanned. */
    sz_u64_t count;

    /** Block sums between the two scan levels. */
    sz_u64_t blocks;
} sz_strings_scan_arguments_metal_t;

/** Encodes kernel @p name over @p groups threadgroups into @p call, refusing a pipeline that runs
 *  fewer threads to a group, or other than 32 to a simdgroup, as the kernels assume. */
STRINGZILLA_INLINE sz_status_t sz_strings_encode_metal_(sz_metal_call_t *call, char const *name,
                                                        sz_metal_bound_t const *buffers, sz_size_t buffers_count,
                                                        void const *arguments, sz_size_t arguments_bytes,
                                                        sz_size_t groups) {
    void *const pipeline = sz_metal_pipeline_(call->context, sz_strings_source_metal_, name);
    sz_metal_size_t const grid = {groups, 1, 1}, threads = {sz_strings_threads_metal_k, 1, 1};
    if (!pipeline || sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") < sz_strings_threads_metal_k ||
        sz_metal_count_(pipeline, "threadExecutionWidth") != 32)
        return sz_device_code_mismatch_k;
    sz_metal_enqueue_(call->encoder, pipeline, buffers, buffers_count, arguments, arguments_bytes, grid, threads);
    return sz_success_k;
}

/**
 *  @brief Binds the tape @p strings views, its payload at @p first bytes in, and the @p count
 *      indices of @p order into @p buffers, for a call reading @p count strings.
 *  @return Whether the device reaches every block.
 */
STRINGZILLA_INLINE sz_bool_t sz_strings_bind_metal_(sz_metal_call_t *call, sz_strings_view_t const *strings,
                                                    sz_size_t const *order, sz_size_t count, sz_size_t first,
                                                    sz_metal_bound_t *buffers) {
    sz_bool_t const wide = (sz_bool_t)(strings->layout == sz_strings_u64_tape_k);
    sz_cptr_t const data = wide ? strings->data.u64_tape.data : strings->data.u32_tape.data;
    void const *const offsets = wide ? (void const *)strings->data.u64_tape.offsets
                                     : (void const *)strings->data.u32_tape.offsets;
    if (!count) return sz_true_k;
    if (data && !sz_metal_resolve_call_(call, data + first, 0, buffers + sz_strings_data_buffer_metal_k))
        return sz_false_k;
    if (!sz_metal_resolve_call_(call, offsets, sz_strings_offsets_bytes_(strings->layout, strings->count),
                                buffers + sz_strings_offsets_buffer_metal_k))
        return sz_false_k;
    return (sz_bool_t)(!order || sz_metal_resolve_call_(call, order, count * sizeof(sz_size_t),
                                                        buffers + sz_strings_order_buffer_metal_k));
}

/** Binds the @p payload bytes and the offsets of the tape @p tape holds into @p buffers, its
 *  payload block null when empty. @return Whether the device reaches both blocks. */
STRINGZILLA_INLINE sz_bool_t sz_strings_bind_tape_metal_(sz_metal_call_t *call, sz_strings_span_t const *tape,
                                                         sz_size_t payload, sz_metal_bound_t *buffers) {
    sz_ptr_t const data = sz_strings_payload_(tape);
    return (
        sz_bool_t)((!data || sz_metal_resolve_call_(call, data, payload, buffers + sz_strings_target_buffer_metal_k)) &&
                   sz_metal_resolve_call_(call, sz_strings_offsets_block_(tape),
                                          sz_strings_offsets_bytes_(tape->layout, tape->count),
                                          buffers + sz_strings_target_offsets_buffer_metal_k));
}

/** Whether a call reading the strings @p strings views may write into @p tape: they are a tape, and
 *  the device of @p stream reaches every block @p tape already holds. */
STRINGZILLA_INLINE sz_status_t sz_strings_check_metal_(sz_strings_view_t const *strings, sz_strings_span_t const *tape,
                                                       sz_stream_t stream) {
    sz_metal_bound_t bound;
    sz_metal_context_t *const context = sz_metal_context_(stream);
    void const *const offsets = sz_strings_offsets_block_(tape);
    sz_ptr_t const payload = sz_strings_payload_(tape);
    if (!context) return sz_missing_gpu_k;
    if (strings->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    if (!offsets) return sz_success_k;
    return (!payload || sz_metal_resolve_(context, payload, 0, &bound)) &&
                   sz_metal_resolve_(context, offsets, sz_strings_offsets_bytes_(tape->layout, tape->count), &bound)
               ? sz_success_k
               : sz_device_memory_mismatch_k;
}

/** Encodes the three scans turning the @p count lengths at the front of @p scratch into offsets. */
STRINGZILLA_INLINE sz_status_t sz_strings_scan_metal_(sz_metal_call_t *call, void *scratch, sz_size_t count) {
    sz_size_t const tiles = sz_size_divide_round_up(count, (sz_size_t)sz_strings_threads_metal_k);
    sz_strings_scan_arguments_metal_t const arguments = {0, count, tiles};
    sz_metal_bound_t buffers[3];
    buffers[0].buffer = buffers[1].buffer = buffers[2].buffer = scratch;
    buffers[0].offset = buffers[1].offset = 0, buffers[2].offset = (count + 1) * sizeof(sz_u64_t);
    sz_status_t status = sz_strings_encode_metal_(call, "sz_strings_scan_local_metal_kernel_", buffers, 3, &arguments,
                                                  sizeof(arguments), tiles);
    if (status == sz_success_k)
        status = sz_strings_encode_metal_(call, "sz_strings_scan_blocks_metal_kernel_", buffers, 3, &arguments,
                                          sizeof(arguments), 1);
    if (status == sz_success_k)
        status = sz_strings_encode_metal_(call, "sz_strings_scan_offsets_metal_kernel_", buffers, 3, &arguments,
                                          sizeof(arguments), tiles);
    return status;
}

/** Measures `arguments->count` strings with kernel @p measure into a fresh shared @p scratch, scans
 *  the lengths into offsets, and reads back how many strings precede the first malformed one and
 *  the payload they take, synchronizing @p stream once. */
STRINGZILLA_INLINE sz_status_t sz_strings_measure_metal_(char const *measure,
                                                         sz_strings_arguments_metal_t const *arguments,
                                                         sz_strings_view_t const *strings, sz_size_t const *order,
                                                         void **scratch, sz_size_t *valid, sz_size_t *payload,
                                                         sz_stream_t stream) {
    sz_size_t const count = arguments->count, tiles = arguments->tiles;
    sz_metal_bound_t buffers[sz_strings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    *valid = 0, *payload = 0;
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_strings_bind_metal_(&call, strings, order, count, 0, buffers))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (count) {
        *scratch = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
            call.context->device, sel_registerName("newBufferWithLength:options:"),
            (count + 1 + 2 * tiles) * sizeof(sz_u64_t), (sz_size_t)0);
        if (!*scratch) return sz_metal_commit_(&call, sz_bad_alloc_k);
        sz_u64_t *const words = (sz_u64_t *)sz_metal_get_(*scratch, "contents");
        memset(words + count + 1 + tiles, 0xFF, tiles * sizeof(sz_u64_t));
        buffers[sz_strings_scratch_buffer_metal_k].buffer = *scratch;
        status = sz_strings_encode_metal_(&call, measure, buffers, sz_strings_buffers_metal_k, arguments,
                                          sizeof(*arguments), tiles);
        if (status == sz_success_k) status = sz_strings_scan_metal_(&call, *scratch, count);
    }
    // The host reads the sizes of the output's old blocks next, so the stream drains even for none.
    status = sz_metal_commit_(&call, status);
    if (status == sz_success_k) status = sz_stream_synchronize_metal_(stream);
    if (status != sz_success_k || !count) return status;
    sz_u64_t const *const scanned = (sz_u64_t const *)sz_metal_get_(*scratch, "contents");
    sz_u64_t const *const reports = scanned + count + 1 + tiles;
    sz_size_t tile = 0;
    while (tile != tiles && reports[tile] == STRINGZILLA_SIZE_MAX) ++tile;
    *valid = tile == tiles ? count : (sz_size_t)reports[tile];
    *payload = (sz_size_t)scanned[*valid];
    return status;
}

/** Writes `arguments->count` strings with kernel @p write over @p groups threadgroups into @p tape,
 *  a span of @p payload bytes, from the offsets in @p scratch. */
STRINGZILLA_INLINE sz_status_t sz_strings_write_metal_(char const *write, sz_size_t groups,
                                                       sz_strings_arguments_metal_t const *arguments,
                                                       sz_strings_view_t const *strings, sz_size_t const *order,
                                                       void *scratch, sz_strings_span_t const *tape, sz_size_t payload,
                                                       sz_stream_t stream) {
    sz_metal_bound_t buffers[sz_strings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_strings_bind_metal_(&call, strings, order, arguments->count, 0, buffers) ||
        !sz_strings_bind_tape_metal_(&call, tape, payload, buffers))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    buffers[sz_strings_scratch_buffer_metal_k].buffer = scratch;
    status = sz_strings_encode_metal_(&call, write, buffers, sz_strings_buffers_metal_k, arguments, sizeof(*arguments),
                                      groups);
    return sz_metal_commit_(&call, status);
}

/** Rewrites @p count strings for @p mode, an escape kind or a normal form, into @p tape through the
 *  @p measure and @p write kernels, up to the first string @p measure finds malformed, reporting
 *  how many in @p valid. */
STRINGZILLA_INLINE sz_status_t sz_strings_rewrite_metal_(char const *measure, char const *write,
                                                         sz_strings_view_t const *strings, sz_size_t const *order,
                                                         sz_size_t count, sz_u64_t mode, sz_allocator_t *allocator,
                                                         sz_strings_span_t *tape, sz_size_t *valid,
                                                         sz_stream_t stream) {
    sz_strings_arguments_metal_t arguments;
    sz_strings_span_t output;
    void *scratch = STRINGZILLA_NULL;
    sz_size_t payload = 0;
    sz_bool_t const fresh = sz_strings_shares_(strings, tape);
    sz_status_t status = sz_strings_check_metal_(strings, tape, stream);
    memset(&arguments, 0, sizeof(arguments));
    arguments.count = count, arguments.order = (sz_u64_t)order, arguments.layout = strings->layout;
    arguments.mode = mode, arguments.tiles = sz_size_divide_round_up(count, (sz_size_t)sz_strings_threads_metal_k);
    *valid = 0;
    if (status == sz_success_k)
        status = sz_strings_measure_metal_(measure, &arguments, strings, order, &scratch, valid, &payload, stream);
    if (status == sz_success_k) status = sz_strings_acquire_(tape, fresh, *valid, payload, allocator, &output, stream);
    if (status == sz_success_k) {
        sz_strings_offset_set_(&output, 0, 0);
        arguments.count = *valid, arguments.target_layout = output.layout;
        if (*valid)
            status = sz_strings_write_metal_(write,
                                             sz_size_divide_round_up(*valid, (sz_size_t)sz_strings_threads_metal_k),
                                             &arguments, strings, order, scratch, &output, payload, stream);
        sz_strings_commit_(tape, fresh, &output, allocator, stream);
    }
    if (scratch) sz_metal_do_(scratch, "release");
    return status;
}

/** Gathers @p count strings into @p tape through @p lut: a simdgroup to a string in @p order, or
 *  one run of bytes from @p first to @p end for a tape read in its own order. */
STRINGZILLA_INLINE sz_status_t sz_strings_map_metal_(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_u8_t const *lut, sz_size_t first,
                                                     sz_size_t end, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                     sz_stream_t stream) {
    sz_strings_arguments_metal_t arguments;
    sz_strings_span_t output;
    sz_metal_bound_t buffers[sz_strings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_metal_call_t call;
    void *scratch = STRINGZILLA_NULL;
    sz_size_t valid = count, payload = end - first;
    sz_bool_t const fresh = sz_strings_shares_(strings, tape);
    sz_status_t status = sz_success_k;
    memset(&arguments, 0, sizeof(arguments));
    memcpy(arguments.lut, lut, sizeof(arguments.lut));
    arguments.count = count, arguments.order = (sz_u64_t)order, arguments.layout = strings->layout;
    arguments.tiles = sz_size_divide_round_up(count, (sz_size_t)sz_strings_threads_metal_k);
    if (order)
        status = sz_strings_measure_metal_("sz_strings_lengths_metal_kernel_", &arguments, strings, order, &scratch,
                                           &valid, &payload, stream);
    if (status == sz_success_k) status = sz_strings_acquire_(tape, fresh, count, payload, allocator, &output, stream);
    if (status != sz_success_k) {
        if (scratch) sz_metal_do_(scratch, "release");
        return status;
    }
    sz_strings_offset_set_(&output, 0, 0);
    arguments.target_layout = output.layout, arguments.length = payload;
    if (order && count)
        status = sz_strings_write_metal_("sz_strings_gather_metal_kernel_",
                                         sz_size_divide_round_up(count, (sz_size_t)sz_strings_simdgroups_metal_k),
                                         &arguments, strings, order, scratch, &output, payload, stream);
    else if (count) {
        status = sz_device_enter_metal_(stream, &call);
        if (status == sz_success_k && (!sz_strings_bind_metal_(&call, strings, order, count, first, buffers) ||
                                       !sz_strings_bind_tape_metal_(&call, &output, payload, buffers)))
            status = sz_device_memory_mismatch_k;
        if (status == sz_success_k && payload)
            status = sz_strings_encode_metal_(
                &call, "sz_strings_lookup_metal_kernel_", buffers, sz_strings_buffers_metal_k, &arguments,
                sizeof(arguments),
                sz_min_of_two(sz_size_divide_round_up(payload, (sz_size_t)sz_strings_threads_metal_k),
                              (sz_size_t)sz_strings_groups_max_metal_k));
        if (status == sz_success_k)
            status = sz_strings_encode_metal_(
                &call, "sz_strings_rebase_metal_kernel_", buffers, sz_strings_buffers_metal_k, &arguments,
                sizeof(arguments), sz_size_divide_round_up(count + 1, (sz_size_t)sz_strings_threads_metal_k));
        status = sz_metal_commit_(&call, status);
    }
    sz_strings_commit_(tape, fresh, &output, allocator, stream);
    if (scratch) sz_metal_do_(scratch, "release");
    return status;
}

/** Reads offsets `[0]` and `[count]` of the tape @p strings views, once @p stream drains. */
STRINGZILLA_INLINE sz_status_t sz_strings_bounds_metal_(sz_strings_view_t const *strings, sz_size_t count,
                                                        sz_size_t *first, sz_size_t *end, sz_stream_t stream) {
    sz_bool_t const wide = (sz_bool_t)(strings->layout == sz_strings_u64_tape_k);
    sz_status_t const status = sz_stream_synchronize_metal_(stream);
    *first = *end = 0;
    if (status != sz_success_k || !count) return status;
    *first = wide ? (sz_size_t)strings->data.u64_tape.offsets[0] : strings->data.u32_tape.offsets[0];
    *end = wide ? (sz_size_t)strings->data.u64_tape.offsets[count] : strings->data.u32_tape.offsets[count];
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_to_tape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_allocator_t *allocator,
                                                     sz_strings_span_t *tape, sz_stream_t stream) {
    sz_u8_t identity[256];
    sz_size_t first = 0, end = 0;
    for (sz_size_t byte = 0; byte != 256; ++byte) identity[byte] = (sz_u8_t)byte;
    sz_status_t status = sz_strings_check_metal_(strings, tape, stream);
    if (status == sz_success_k && !order) status = sz_strings_bounds_metal_(strings, count, &first, &end, stream);
    if (status != sz_success_k) return status;
    if (!order && sz_strings_aliases_(strings, tape) && !first && count == tape->count) return sz_success_k;
    return sz_strings_map_metal_(strings, order, count, identity, first, end, allocator, tape, stream);
}

STRINGZILLA_API sz_status_t sz_strings_lookup_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, char const lut[sz_at_least_(256)],
                                                    sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                    sz_stream_t stream) {
    sz_strings_arguments_metal_t arguments;
    sz_metal_bound_t buffers[sz_strings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_metal_call_t call;
    sz_size_t first = 0, end = 0;
    sz_status_t status = sz_strings_check_metal_(strings, tape, stream);
    if (status == sz_success_k && !order) status = sz_strings_bounds_metal_(strings, count, &first, &end, stream);
    if (status != sz_success_k) return status;
    if (order || !sz_strings_aliases_(strings, tape))
        return sz_strings_map_metal_(strings, order, count, (sz_u8_t const *)lut, first, end, allocator, tape, stream);
    // Shrinking first leaves the host no offsets to read while the kernel runs.
    if (count != tape->count)
        status = sz_strings_fit_(tape, sz_strings_offset_(tape, tape->count), count, allocator, stream);
    if (status != sz_success_k || end == first) return status;
    memset(&arguments, 0, sizeof(arguments));
    memcpy(arguments.lut, lut, sizeof(arguments.lut));
    arguments.length = end - first;
    status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_metal_resolve_call_(&call, sz_strings_payload_(tape) + first, arguments.length,
                                buffers + sz_strings_data_buffer_metal_k))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    buffers[sz_strings_target_buffer_metal_k] = buffers[sz_strings_data_buffer_metal_k];
    status = sz_strings_encode_metal_(
        &call, "sz_strings_lookup_metal_kernel_", buffers, sz_strings_buffers_metal_k, &arguments, sizeof(arguments),
        sz_min_of_two(sz_size_divide_round_up(arguments.length, (sz_size_t)sz_strings_threads_metal_k),
                      (sz_size_t)sz_strings_groups_max_metal_k));
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_strings_to_rows_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                     sz_size_t count, sz_strings_rows_t const *rows, sz_size_t *longest,
                                                     sz_stream_t stream) {
    sz_strings_arguments_metal_t arguments;
    sz_metal_bound_t buffers[sz_strings_buffers_metal_k] = {{STRINGZILLA_NULL, 0}};
    sz_metal_call_t call;
    void *scratch = STRINGZILLA_NULL;
    if (strings->layout == sz_strings_accessor_k) return sz_device_memory_mismatch_k;
    memset(&arguments, 0, sizeof(arguments));
    arguments.count = count, arguments.order = (sz_u64_t)order, arguments.layout = strings->layout;
    arguments.width = rows->width, arguments.stride = rows->stride;
    arguments.tiles = sz_size_divide_round_up(count, (sz_size_t)sz_strings_simdgroups_metal_k);
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (!sz_strings_bind_metal_(&call, strings, order, count, 0, buffers) ||
        !sz_metal_resolve_call_(&call, longest, sizeof(sz_size_t),
                                buffers + sz_strings_target_offsets_buffer_metal_k) ||
        (count && rows->width &&
         !sz_metal_resolve_call_(&call, rows->data, (count - 1) * rows->stride + rows->width,
                                 buffers + sz_strings_target_buffer_metal_k)))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (count) {
        scratch = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
            call.context->device, sel_registerName("newBufferWithLength:options:"),
            (sz_size_t)arguments.tiles * sizeof(sz_u64_t), (sz_size_t)32);
        if (!scratch) return sz_metal_commit_(&call, sz_bad_alloc_k);
        buffers[sz_strings_scratch_buffer_metal_k].buffer = scratch;
        status = sz_strings_encode_metal_(&call, "sz_strings_to_rows_metal_kernel_", buffers,
                                          sz_strings_buffers_metal_k, &arguments, sizeof(arguments), arguments.tiles);
    }
    if (status == sz_success_k)
        status = sz_strings_encode_metal_(&call, "sz_strings_longest_metal_kernel_", buffers,
                                          sz_strings_buffers_metal_k, &arguments, sizeof(arguments), 1);
    if (scratch) sz_metal_do_(scratch, "release");
    return sz_metal_commit_(&call, status);
}

STRINGZILLA_API sz_status_t sz_strings_escape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                    sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                    sz_strings_span_t *tape, sz_stream_t stream) {
    sz_size_t valid = 0;
    return sz_strings_rewrite_metal_("sz_strings_escape_lengths_metal_kernel_", "sz_strings_escape_metal_kernel_",
                                     strings, order, count, kind, allocator, tape, &valid, stream);
}

STRINGZILLA_API sz_status_t sz_strings_unescape_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                      sz_size_t count, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                                      sz_strings_span_t *tape, sz_size_t *malformed_index,
                                                      sz_stream_t stream) {
    sz_size_t valid = 0;
    sz_status_t status = sz_strings_rewrite_metal_("sz_strings_unescape_lengths_metal_kernel_",
                                                   "sz_strings_unescape_metal_kernel_", strings, order, count, kind,
                                                   allocator, tape, &valid, stream);
    if (status == sz_success_k && valid != count) *malformed_index = valid, status = sz_invalid_utf8_k;
    return status;
}

STRINGZILLA_API sz_status_t sz_strings_normalize_metal(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_normal_form_t form,
                                                       sz_allocator_t *allocator, sz_strings_span_t *tape,
                                                       sz_stream_t stream) {
    sz_size_t valid = 0;
    return sz_strings_rewrite_metal_("sz_strings_normalize_lengths_metal_kernel_", "sz_strings_normalize_metal_kernel_",
                                     strings, order, count, form, allocator, tape, &valid, stream);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_STRINGS_METAL_H_
