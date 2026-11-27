/**
 *  @file src/missing/metal.c
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief The Metal device exports of a library built without Metal, each answering
 *      @c sz_missing_gpu_k.
 *
 *  Without @c STRINGZILLA_METAL_ARCH, CMake compiles it in place of
 *  <tt>src/capabilities/metal.c</tt>, so every build exports the same names and the @c _best
 *  dispatch points call them without asking which vendors it holds.
 */
#include "stringzilla/strings.h" // `sz_allocator_init_unified_metal`, `sz_count_devices_metal`

STRINGZILLA_API sz_status_t sz_count_devices_metal(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator) {
    sz_unused_(allocator);
    return sz_missing_gpu_k;
}
