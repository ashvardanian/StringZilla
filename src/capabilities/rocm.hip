/**
 *  @file src/capabilities/rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief The @c rocm kernels and the ROCm device exports, defined once for the library.
 */
#undef STRINGZILLA_TARGET_CDNA3
#define STRINGZILLA_TARGET_CDNA3 0
#undef STRINGZILLA_TARGET_CDNA4
#define STRINGZILLA_TARGET_CDNA4 0
#undef STRINGZILLA_TARGET_CDNA5
#define STRINGZILLA_TARGET_CDNA5 0
#include "stringzilla/strings.h"
#include "stringzilla/hash.h"
#include "stringzilla/levenshtein.h"
#include "stringzilla/ngrams.h"
#include "stringzilla/substrings.h"
#include "stringzilla/regex.h"
#include "stringzilla/subdocuments.h"
#include "stringzilla/utf8_uncased_fold.h"
#include "stringzilla/utf8_norm.h"

#include "stringzilla/hash/rocm.cuh"
#include "stringzilla/levenshtein/rocm.cuh"
#include "stringzilla/ngrams/rocm.cuh"
#include "stringzilla/strings/rocm.cuh"
#include "stringzilla/substrings/rocm.cuh"
#include "stringzilla/regex/rocm.cuh"
#include "stringzilla/subdocuments/rocm.cuh"
#include "stringzilla/utf8_uncased_fold/rocm.cuh"
#include "stringzilla/utf8_norm/rocm.cuh"

STRINGZILLA_API sz_status_t sz_count_devices_rocm(sz_size_t *count) {
    *count = sz_count_devices_rocm_();
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
    allocator->reallocate = sz_reallocate_device_rocm_;
    allocator->free = sz_free_device_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_pinned_rocm_;
    allocator->reallocate = sz_reallocate_pinned_rocm_;
    allocator->free = sz_free_pinned_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream) {
    return sz_stream_synchronize_rocm_(stream);
}

/** The kernel @c sz_capabilities_compiled_rocm asks the runtime to load. */
extern "C" void const *sz_probe_rocm_(void) { return (void const *)&sz_strings_scan_local_rocm_kernel_; }

/** Whether ROCm device @p ordinal loads @p kernel, as the runtime fails where no code object in the
 *  binary runs there. */
extern "C" sz_status_t sz_kernel_loads_rocm_(sz_size_t ordinal, void const *kernel, sz_bool_t *loads) {
    int previous = 0;
    hipFuncAttributes attributes;
    *loads = sz_false_k;
    if (ordinal >= sz_count_devices_rocm_()) return sz_missing_gpu_k;
    if (hipGetDevice(&previous) != hipSuccess || hipSetDevice((int)ordinal) != hipSuccess) {
        [[maybe_unused]] hipError_t const cleared = hipGetLastError();
        return sz_device_code_mismatch_k;
    }
    *loads = hipFuncGetAttributes(&attributes, kernel) == hipSuccess ? sz_true_k : sz_false_k;
    [[maybe_unused]] hipError_t const cleared = hipGetLastError(), restored = hipSetDevice(previous);
    return sz_success_k;
}
