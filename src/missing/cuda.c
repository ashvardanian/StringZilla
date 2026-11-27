/**
 *  @file src/missing/cuda.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The CUDA device exports of a library built without CUDA, each answering
 *      @c sz_missing_gpu_k.
 *
 *  Without @c STRINGZILLA_CUDA_ARCH, CMake compiles it in place of
 *  <tt>src/capabilities/cuda.cu</tt>, so every build exports the same names and the @c _best
 *  dispatch points call them without asking which vendors it holds.
 */
#include "stringzilla/strings.h" // `sz_allocator_init_unified_cuda`, `sz_count_devices_cuda`

STRINGZILLA_API sz_status_t sz_count_devices_cuda(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_cuda(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_cuda(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_cuda(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}
