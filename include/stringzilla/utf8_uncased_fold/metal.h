/**
 *  @file include/stringzilla/utf8_uncased_fold/metal.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Parallel UTF-8 case folding through counting, prefix sums, and output passes.
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_METAL_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_METAL_H_

#include "stringzilla/metal.h"
#include "stringzilla/utf8_uncased_fold/serial.h"

#if STRINGZILLA_TARGET_METAL
#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
static char const sz_utf8_uncased_fold_source_metal_[] = {
#embed "stringzilla/utf8_uncased_fold/tables.h"
    ,
#embed "stringzilla/utf8_uncased_fold/metal.metal"
    , 0};
#pragma clang diagnostic pop

typedef struct {
    sz_u64_t length, span, groups;
} sz_utf8_uncased_fold_arguments_metal_t;

STRINGZILLA_API sz_status_t sz_utf8_uncased_fold_metal(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target,
                                                       sz_size_t *target_length, sz_stream_t stream) {
    sz_metal_call_t call;
    sz_status_t status = sz_device_enter_metal_(stream, &call);
    if (status != sz_success_k) return sz_metal_commit_(&call, status);
    if (source_length > (sz_size_t)-1 / sz_utf8_fold_max_expansion_k)
        return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    sz_metal_bound_t buffers[3] = {{0}};
    if (!sz_metal_resolve_call_(&call, target_length, sizeof(*target_length), &buffers[2]) ||
        (source_length &&
         (!sz_metal_resolve_call_(&call, source, source_length, &buffers[0]) ||
          !sz_metal_resolve_call_(&call, target, source_length * sz_utf8_fold_max_expansion_k, &buffers[1]))))
        return sz_metal_commit_(&call, sz_device_memory_mismatch_k);
    if (!source_length) {
        status = sz_metal_encode_(&call, sz_utf8_uncased_fold_source_metal_, "sz_utf8_uncased_fold_clear_metal_kernel_",
                                  &buffers[2], 1, STRINGZILLA_NULL, 0, (sz_metal_size_t) {1, 1, 1},
                                  (sz_metal_size_t) {1, 1, 1});
        return sz_metal_commit_(&call, status);
    }
    sz_metal_size_t const threads = {256, 1, 1};
    if (source_length <= 16384) {
        void *const pipeline = sz_metal_pipeline_(call.context, sz_utf8_uncased_fold_source_metal_,
                                                  "sz_utf8_uncased_fold_metal_kernel_");
        if (!pipeline || sz_metal_count_(pipeline, "threadExecutionWidth") != 32 ||
            sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") < threads.width)
            return sz_metal_commit_(&call, sz_device_code_mismatch_k);
        sz_utf8_uncased_fold_arguments_metal_t const args = {source_length,
                                                             sz_size_divide_round_up(source_length, threads.width), 1};
        sz_metal_enqueue_(call.encoder, pipeline, buffers, 3, &args, sizeof(args), (sz_metal_size_t) {1, 1, 1},
                          threads);
        return sz_metal_commit_(&call, sz_success_k);
    }
    sz_utf8_uncased_fold_arguments_metal_t const args = {source_length, 64,
                                                         sz_size_divide_round_up(source_length, (sz_size_t)16384)};
    if (args.groups > 0xFFFFFFFFu) return sz_metal_commit_(&call, sz_unexpected_dimensions_k);
    char const *const names[] = {"sz_utf8_uncased_fold_count_metal_kernel_", "sz_utf8_uncased_fold_scan_metal_kernel_",
                                 "sz_utf8_uncased_fold_write_metal_kernel_"};
    void *pipelines[3];
    for (sz_size_t index = 0; index != 3; ++index) {
        pipelines[index] = sz_metal_pipeline_(call.context, sz_utf8_uncased_fold_source_metal_, names[index]);
        if (!pipelines[index] || sz_metal_count_(pipelines[index], "threadExecutionWidth") != 32 ||
            sz_metal_count_(pipelines[index], "maxTotalThreadsPerThreadgroup") < threads.width)
            return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    }
    sz_size_t const scratch_bytes = args.groups * (sizeof(sz_u64_t) + 256 * sizeof(sz_u32_t));
    void *const scratch = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        call.context->device, sel_registerName("newBufferWithLength:options:"), scratch_bytes, (sz_size_t)32);
    if (!scratch) return sz_metal_commit_(&call, sz_bad_alloc_k);
    sz_metal_bound_t parallel_buffers[4] = {buffers[0], buffers[1], buffers[2], {scratch, 0}};
    sz_metal_size_t const groups = {(sz_size_t)args.groups, 1, 1};
    sz_metal_enqueue_(call.encoder, pipelines[0], parallel_buffers, 4, &args, sizeof(args), groups, threads);
    sz_metal_enqueue_(call.encoder, pipelines[1], parallel_buffers, 4, &args, sizeof(args), (sz_metal_size_t) {1, 1, 1},
                      threads);
    sz_metal_enqueue_(call.encoder, pipelines[2], parallel_buffers, 4, &args, sizeof(args), groups, threads);
    sz_metal_do_(scratch, "release");
    return sz_metal_commit_(&call, sz_success_k);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_UTF8_UNCASED_FOLD_METAL_H_
