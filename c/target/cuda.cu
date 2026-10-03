/**
 *  @file c/target/cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The @c cuda kernels and the CUDA device exports, defined once for the library.
 */
#include "stringzilla/memory.h"
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/utf8_norm.h"
#include "stringzilla/utf8_uncased_fold.h"

#include "stringzilla/levenshtein/cuda.cuh"
#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/utf8_norm/cuda.cuh"
#include "stringzilla/utf8_uncased_fold/cuda.cuh"

STRINGZILLA_API sz_status_t sz_cuda_count_devices(sz_size_t *count) {
    *count = sz_cuda_count_devices_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_cuda_capabilities_detected(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_cuda_capabilities_detected_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_cuda_stream_init(sz_size_t ordinal, void **stream) {
    return sz_cuda_stream_init_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_cuda_stream_free(void *stream) { return sz_stream_destroy_cuda_(stream); }

STRINGZILLA_API sz_status_t sz_memory_allocator_init_unified_cuda(sz_memory_allocator_t *allocator) {
    sz_memory_allocator_init_unified_cuda_(allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sequence_copy_cuda(sz_sequence_t *target, sz_sequence_t const *source,
                                                  sz_memory_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                  void *stream) {
    return sz_sequence_copy_cuda_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(void *stream) { return sz_stream_synchronize_cuda_(stream); }
