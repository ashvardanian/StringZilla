/**
 *  @file src/missing/rocm.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The ROCm device exports of a library built without ROCm, each answering
 *      @c sz_missing_gpu_k.
 *
 *  Without @c STRINGZILLA_ROCM_ARCH, CMake compiles it in place of
 *  <tt>src/capabilities/rocm.hip</tt>, so every build exports the same names and the @c _best
 *  dispatch points call them without asking which vendors it holds.
 */
#include "stringzilla/strings.h" // `sz_allocator_init_unified_rocm`, `sz_count_devices_rocm`

STRINGZILLA_API sz_status_t sz_count_devices_rocm(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_rocm(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_rocm(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}
