/**
 *  @file c/target/cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The @c cuda kernels and the CUDA device exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_HOPPER
#define STRINGZILLA_TARGET_HOPPER 0
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#include "stringzilla/memory.h"
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/utf8_uncased_fold.h"
#include "stringzilla/utf8_norm.h"

#include "stringzilla/levenshtein/cuda.cuh"
#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/utf8_uncased_fold/cuda.cuh"
#include "stringzilla/utf8_norm/cuda.cuh"

STRINGZILLA_API sz_status_t sz_device_count_cuda(sz_size_t *count) {
    *count = sz_device_count_cuda_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_capabilities_detected_cuda_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream) {
    return sz_stream_init_cuda_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream) { return sz_stream_free_cuda_(stream); }

STRINGZILLA_API sz_status_t sz_allocator_init_unified_cuda(sz_allocator_t *allocator) {
    sz_allocator_init_unified_cuda_(allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_cuda(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_device_cuda_;
    allocator->free = sz_free_device_cuda_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_cuda(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_pinned_cuda_;
    allocator->free = sz_free_pinned_cuda_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_cuda(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     sz_stream_t stream) {
    return sz_sequence_realloc_cuda_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream) {
    return sz_stream_synchronize_cuda_(stream);
}
