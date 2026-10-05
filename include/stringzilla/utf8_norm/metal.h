/**
 *  @file include/stringzilla/utf8_norm/metal.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Unicode normalization in caller-owned Metal buffers.
 */
#ifndef STRINGZILLA_UTF8_NORM_METAL_H_
#define STRINGZILLA_UTF8_NORM_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/utf8_norm/serial.h"

#if STRINGZILLA_TARGET_METAL
#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
static char const sz_utf8_norm_source_metal_[] = {
#embed "stringzilla/utf8_norm/tables.h"
    , '\n',
#embed "stringzilla/utf8_norm/metal.metal"
    , 0};
#pragma clang diagnostic pop

typedef struct sz_utf8_norm_arguments_metal_t {
    sz_u64_t length, form, chunks;
} sz_utf8_norm_arguments_metal_t;

/** Enqueues normalization into caller-owned unified memory and its device length slot. */
STRINGZILLA_API sz_status_t sz_utf8_norm_metal(sz_cptr_t source, sz_size_t source_length, sz_normal_form_t form,
                                               sz_ptr_t target, sz_size_t *target_length, sz_stream_t stream) {
    if (source_length > (sz_size_t)-1 / sz_utf8_norm_decomp_max_k) return sz_unexpected_dimensions_k;
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    sz_metal_bound_t buffers[5];
    if (!sz_metal_resolve_call_(&call, target_length, sizeof(*target_length), buffers + 2))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    buffers[0] = buffers[1] = buffers[2];
    if (source_length &&
        (!sz_metal_resolve_call_(&call, source, source_length, buffers) ||
         !sz_metal_resolve_call_(&call, target, source_length * sz_utf8_norm_decomp_max_k, buffers + 1)))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    sz_size_t const chunks = sz_size_divide_round_up(source_length, (sz_size_t)64);
    sz_utf8_norm_arguments_metal_t const arguments = {source_length, (sz_u64_t)form, chunks};
    if (source_length <= 16384) {
        void *const pipeline = sz_metal_pipeline_(call.context, sz_utf8_norm_source_metal_,
                                                  "sz_utf8_norm_metal_kernel_");
        sz_size_t const ceiling = pipeline ? sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") : 0;
        if (!ceiling) return sz_metal_commit_(&call, sz_device_code_mismatch_k);
        sz_size_t const threads = sz_min_of_two(
            sz_min_of_two((sz_size_t)256, ceiling),
            sz_max_of_two((sz_size_t)1, sz_size_divide_round_up(source_length, (sz_size_t)64)));
        sz_metal_size_t const groups_size = {1, 1, 1}, threads_size = {threads, 1, 1};
        sz_metal_enqueue_(call.encoder, pipeline, buffers, 3, &arguments, sizeof(arguments), groups_size, threads_size);
        return sz_metal_commit_(&call, sz_success_k);
    }
    sz_size_t const groups = sz_size_divide_round_up(chunks, (sz_size_t)32);
    if (groups > 0xFFFFFFFFu) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    void *const count = sz_metal_pipeline_(call.context, sz_utf8_norm_source_metal_,
                                           "sz_utf8_norm_count_metal_kernel_");
    void *const offsets = sz_metal_pipeline_(call.context, sz_utf8_norm_source_metal_,
                                             "sz_utf8_norm_offsets_metal_kernel_");
    void *const write = sz_metal_pipeline_(call.context, sz_utf8_norm_source_metal_,
                                           "sz_utf8_norm_write_metal_kernel_");
    if (!count || !offsets || !write || sz_metal_count_(count, "threadExecutionWidth") != 32 ||
        sz_metal_count_(offsets, "threadExecutionWidth") != 32 ||
        sz_metal_count_(count, "maxTotalThreadsPerThreadgroup") < 32 ||
        sz_metal_count_(offsets, "maxTotalThreadsPerThreadgroup") < 256 ||
        sz_metal_count_(write, "maxTotalThreadsPerThreadgroup") < 32)
        return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    sz_size_t const spans_bytes = chunks * 3 * sizeof(sz_u64_t);
    sz_size_t const blocks_bytes = groups * sizeof(sz_u64_t);
    void *const scratch = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        call.context->device, sel_registerName("newBufferWithLength:options:"), spans_bytes + blocks_bytes,
        (sz_size_t)32);
    if (!scratch) return sz_metal_commit_(&call, sz_bad_alloc_k);
    buffers[3] = (sz_metal_bound_t) {scratch, 0};
    buffers[4] = (sz_metal_bound_t) {scratch, spans_bytes};
    sz_metal_size_t const grid = {groups, 1, 1}, threads = {32, 1, 1};
    sz_metal_enqueue_(call.encoder, count, buffers, 5, &arguments, sizeof(arguments), grid, threads);
    sz_metal_enqueue_(call.encoder, offsets, buffers, 5, &arguments, sizeof(arguments), (sz_metal_size_t) {1, 1, 1},
                      (sz_metal_size_t) {256, 1, 1});
    sz_metal_enqueue_(call.encoder, write, buffers, 5, &arguments, sizeof(arguments), grid, threads);
    sz_metal_do_(scratch, "release");
    return sz_metal_commit_(&call, sz_success_k);
}

#ifdef __cplusplus
}
#endif
#endif
#endif
