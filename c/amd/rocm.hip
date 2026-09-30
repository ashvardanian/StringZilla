/**
 *  @file c/amd/rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The engines' @c rocm kernels and the ROCm device exports, defined once for the library.
 */
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"

#include "stringzilla/levenshtein/simt.cuh"
#include "stringzilla/overlap/simt.cuh"
#include "stringzilla/substrings/simt.cuh"

extern "C" STRINGZILLA_API sz_status_t sz_rocm_count_devices(sz_size_t *count) {
    *count = sz_rocm_count_devices_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_capabilities_detected(sz_size_t device, sz_capability_t *capabilities) {
    return sz_rocm_capabilities_detected_(device, capabilities);
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_memory_reaches_device(void const *pointer, sz_bool_t *reaches) {
    *reaches = sz_memory_reaches_device_(pointer);
    return sz_success_k;
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_unified(sz_memory_allocator_t *allocator,
                                                                             sz_size_t ordinal) {
    sz_memory_allocator_init_unified_(allocator, ordinal);
    return sz_success_k;
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_device(sz_memory_allocator_t *allocator,
                                                                            sz_size_t ordinal) {
    sz_memory_allocator_init_device_(allocator, ordinal);
    return sz_success_k;
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_pinned(sz_memory_allocator_t *allocator,
                                                                            sz_size_t ordinal) {
    sz_memory_allocator_init_pinned_(allocator, ordinal);
    return sz_success_k;
}

extern "C" STRINGZILLA_API sz_status_t sz_rocm_sequence_from_string_views(sz_string_view_t const *views,
                                                                          sz_size_t count, sz_sequence_t *sequence) {
    return sz_sequence_from_string_views_cuda_(views, count, sequence);
}
