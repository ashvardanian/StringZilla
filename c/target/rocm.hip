/**
 *  @file c/target/rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The @c rocm kernels and the ROCm device exports, defined once for the library.
 */
#include "stringzilla/memory.h"
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/utf8_uncased_fold.h"
#include "stringzilla/utf8_norm.h"

#include "stringzilla/levenshtein/rocm.cuh"
#include "stringzilla/overlap/rocm.cuh"
#include "stringzilla/substrings/rocm.cuh"
#include "stringzilla/utf8_uncased_fold/rocm.cuh"
#include "stringzilla/utf8_norm/rocm.cuh"

STRINGZILLA_API sz_status_t sz_device_count_rocm(sz_size_t *count) {
    *count = sz_device_count_rocm_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_capabilities_detected_rocm_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream) {
    return sz_stream_init_rocm_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream) { return sz_stream_free_rocm_(stream); }

STRINGZILLA_API sz_status_t sz_allocator_init_unified_rocm(sz_allocator_t *allocator) {
    sz_allocator_init_unified_rocm_(allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_rocm(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_device_rocm_;
    allocator->free = sz_free_device_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_pinned_rocm_;
    allocator->free = sz_free_pinned_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_rocm(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_stream_t stream) {
    return sz_sequence_realloc_rocm_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream) {
    return sz_stream_synchronize_rocm_(stream);
}
