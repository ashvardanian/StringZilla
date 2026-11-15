/**
 *  @file src/capabilities/cuda.cu
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The @c cuda kernels and the CUDA device exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_AMPERE
#define STRINGZILLA_TARGET_AMPERE 0
#undef STRINGZILLA_TARGET_HOPPER
#define STRINGZILLA_TARGET_HOPPER 0
#undef STRINGZILLA_TARGET_BLACKWELL
#define STRINGZILLA_TARGET_BLACKWELL 0
#include "stringzilla/strings.h"
#include "stringzilla/hash.h"
#include "stringzilla/levenshtein.h"
#include "stringzilla/overlap.h"
#include "stringzilla/substrings.h"
#include "stringzilla/regex.h"
#include "stringzilla/subdocuments.h"
#include "stringzilla/utf8_uncased_fold.h"
#include "stringzilla/utf8_norm.h"

#include "stringzilla/hash/cuda.cuh"
#include "stringzilla/levenshtein/cuda.cuh"
#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/strings/cuda.cuh"
#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/regex/cuda.cuh"
#include "stringzilla/subdocuments/cuda.cuh"
#include "stringzilla/utf8_uncased_fold/cuda.cuh"
#include "stringzilla/utf8_norm/cuda.cuh"

STRINGZILLA_API sz_status_t sz_count_devices_cuda(sz_size_t *count) {
    *count = sz_count_devices_cuda_();
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
    allocator->reallocate = sz_reallocate_device_cuda_;
    allocator->free = sz_free_device_cuda_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_cuda(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_pinned_cuda_;
    allocator->reallocate = sz_reallocate_pinned_cuda_;
    allocator->free = sz_free_pinned_cuda_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream) {
    return sz_stream_synchronize_cuda_(stream);
}

/** The kernel @c sz_capabilities_compiled_cuda asks the runtime to load. */
extern "C" void const *sz_probe_cuda_(void) { return (void const *)&sz_strings_scan_local_cuda_kernel_; }

/** Whether CUDA device @p ordinal loads @p kernel, as the runtime fails where no code in the binary
 *  runs there. */
extern "C" sz_status_t sz_kernel_loads_cuda_(sz_size_t ordinal, void const *kernel, sz_bool_t *loads) {
    int previous = 0;
    cudaFuncAttributes attributes;
    *loads = sz_false_k;
    if (ordinal >= sz_count_devices_cuda_()) return sz_missing_gpu_k;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice((int)ordinal) != cudaSuccess) {
        cudaGetLastError();
        return sz_device_code_mismatch_k;
    }
    *loads = cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess ? sz_true_k : sz_false_k;
    cudaGetLastError();
    cudaSetDevice(previous);
    return sz_success_k;
}
