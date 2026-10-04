/**
 *  @file include/stringzilla/utf8_uncased_fold/metal.h
 *  @author Ash Vardanian
 *  @date October 4, 2026
 *  @brief Scratch-free UTF-8 case folding through ordered Metal dispatches.
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
    sz_u64_t length, begin, end;
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
    status = sz_metal_encode_(&call, sz_utf8_uncased_fold_source_metal_, "sz_utf8_uncased_fold_clear_metal_kernel_",
                              &buffers[2], 1, STRINGZILLA_NULL, 0, (sz_metal_size_t) {1, 1, 1},
                              (sz_metal_size_t) {1, 1, 1});
    if (status != sz_success_k || !source_length) return sz_metal_commit_(&call, status);
    void *const pipeline = sz_metal_pipeline_(call.context, sz_utf8_uncased_fold_source_metal_,
                                              "sz_utf8_uncased_fold_metal_kernel_");
    if (!pipeline) return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    sz_size_t const width = sz_metal_count_(pipeline, "threadExecutionWidth");
    if (!width || 256 % width || sz_metal_count_(pipeline, "maxTotalThreadsPerThreadgroup") < 256)
        return sz_metal_commit_(&call, sz_device_code_mismatch_k);
    // Ordered dispatches carry their output offset in the length slot without cross-group spinning.
    for (sz_size_t begin = 0; begin < source_length; begin += 65536) {
        sz_utf8_uncased_fold_arguments_metal_t const args = {source_length, begin,
                                                             sz_min_of_two(begin + 65536, source_length)};
        sz_metal_enqueue_(call.encoder, pipeline, buffers, 3, &args, sizeof(args), (sz_metal_size_t) {1, 1, 1},
                          (sz_metal_size_t) {256, 1, 1});
    }
    return sz_metal_commit_(&call, sz_success_k);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_METAL
#endif // STRINGZILLA_UTF8_UNCASED_FOLD_METAL_H_
