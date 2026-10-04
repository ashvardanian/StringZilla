/**
 *  @file include/stringzilla/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The HIP runtime as the library's ROCm host code drives it: the device scope an engine
 *      call runs in, the memory both sides address, the launches, the streams and the producers.
 *
 *  Every call runs on the device of the stream it is given, which @ref sz_device_enter_rocm_ makes
 *  current for its duration, and a null stream runs on the caller's current device.
 *
 *  Only hipcc sees past the guard, and nothing here is shared with CUDA, whose twins live in
 *  `cuda.cuh` under their own names. AMD has no clusters and no block stealing, so neither appears
 *  here, and a family's ROCm host code launches every grid flat.
 *
 *  @sa include/stringzilla/cuda.cuh
 *  @sa include/stringzilla/types.cuh
 */
#ifndef STRINGZILLA_ROCM_CUH_
#define STRINGZILLA_ROCM_CUH_

#include "stringzilla/types.cuh"
#include "stringzilla/capabilities.h"  // `sz_cap_rocm_k`
#include "stringzilla/memory/serial.h" // `sz_sequence_realloc_serial_`

#if STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)
#include <hip/hip_runtime.h> // `hipLaunchKernel`, `hipMallocManaged`, `hipSetDevice`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Device Memory

/**
 *  @brief Makes the device @p stream belongs to current for one call, keeping the caller's device
 *      in @p caller.
 *  @param[in] stream The stream the call enqueues on, which names its device, or
 *      @c STRINGZILLA_NULL for the default stream of the caller's current device.
 *  @return @c sz_success_k, @c sz_missing_gpu_k when the runtime has no device to make current, or
 *      @c sz_device_memory_mismatch_k for a stream the runtime cannot place, in which case the
 *      caller's device stays current.
 */
STRINGZILLA_INLINE sz_status_t sz_device_enter_rocm_(void *stream, int *caller) {
    int device = 0;
    if (hipGetDevice(caller) != hipSuccess) return sz_missing_gpu_k;
    if (!stream) return sz_success_k;
    if (hipStreamGetDevice((hipStream_t)stream, &device) != hipSuccess) return sz_device_memory_mismatch_k;
    return hipSetDevice(device) == hipSuccess ? sz_success_k : sz_missing_gpu_k;
}

/** Makes @p caller current again, closing the scope @ref sz_device_enter_rocm_ opened. */
STRINGZILLA_INLINE void sz_device_leave_rocm_(int caller) { sz_unused_(hipSetDevice(caller)); }

/** Accepts managed memory or the current device's own allocation.
 *  Host mappings require a device address and are unsupported by this check. */
STRINGZILLA_INLINE sz_bool_t sz_memory_reaches_rocm_(void const *pointer) {
    int device = 0;
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess || hipGetDevice(&device) != hipSuccess)
        return sz_false_k;
    if (attributes.type == hipMemoryTypeManaged) return sz_true_k;
    return attributes.type == hipMemoryTypeDevice && attributes.device == device ? sz_true_k : sz_false_k;
}

/*  Each allocator below is stateless, ignoring @c handle, and works on the device of its stream. */

STRINGZILLA_INLINE void *sz_memory_allocate_unified_rocm_(sz_size_t bytes, void *handle, void *stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipMallocManaged(&pointer, bytes, hipMemAttachGlobal) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_device_rocm_(sz_size_t bytes, void *handle, void *stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipMalloc(&pointer, bytes) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_pinned_rocm_(sz_size_t bytes, void *handle, void *stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipHostMalloc(&pointer, bytes, hipHostMallocDefault) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

/** Frees unified and device blocks alike, once the device is done with everything queued before. */
STRINGZILLA_INLINE void sz_memory_free_device_rocm_(void *pointer, sz_size_t bytes, void *handle, void *stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_rocm_(stream, &caller) != sz_success_k) return;
    sz_unused_(hipFree(pointer));
    sz_device_leave_rocm_(caller);
}

STRINGZILLA_INLINE void sz_memory_free_pinned_rocm_(void *pointer, sz_size_t bytes, void *handle, void *stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_rocm_(stream, &caller) != sz_success_k) return;
    sz_unused_(hipHostFree(pointer));
    sz_device_leave_rocm_(caller);
}

/** Initializes an allocator handing back memory both the host and the device address, each call
 *  on its stream's device. */
STRINGZILLA_INLINE void sz_allocator_init_unified_rocm_(sz_allocator_t *allocator) {
    allocator->allocate = &sz_memory_allocate_unified_rocm_;
    allocator->free = &sz_memory_free_device_rocm_;
    allocator->handle = STRINGZILLA_NULL;
}

#pragma endregion Device Memory

#pragma region Launches

/** One attribute of the current device, or zero when unanswered. */
STRINGZILLA_INLINE sz_size_t sz_device_attribute_rocm_(hipDeviceAttribute_t attribute) {
    int device = 0, value = 0;
    if (hipGetDevice(&device) != hipSuccess) return 0;
    if (hipDeviceGetAttribute(&value, attribute, device) != hipSuccess) return 0;
    return value > 0 ? (sz_size_t)value : 0;
}

/** Compute units of the current device, or zero when the runtime will not say. */
STRINGZILLA_INLINE sz_size_t sz_device_multiprocessors_rocm_(void) {
    return sz_device_attribute_rocm_(hipDeviceAttributeMultiprocessorCount);
}

/** Threads one compute unit of the current device keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_threads_per_multiprocessor_rocm_(void) {
    return sz_device_attribute_rocm_(hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

/** Shared memory one block of the current device gets, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_shared_bytes_per_block_rocm_(void) {
    return sz_device_attribute_rocm_(hipDeviceAttributeMaxSharedMemoryPerBlock);
}

/** Free bytes of the current device's memory, or zero when the runtime will not say. */
STRINGZILLA_INLINE sz_size_t sz_device_free_bytes_rocm_(void) {
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
    return (sz_size_t)free_bytes;
}

/** Blocks of @p threads running @p kernel with @p shared_bytes of dynamic shared memory that one
 *  compute unit keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_resident_blocks_rocm_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes) {
    int blocks = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, (int)threads, shared_bytes) != hipSuccess)
        return 0;
    return blocks > 0 ? (sz_size_t)blocks : 0;
}

/** The block size of @p kernel landing the most resident threads per compute unit, from powers of
 *  two between 64 and @p ceiling, ties going to the wider block, or @p fallback unanswered. */
STRINGZILLA_INLINE sz_size_t sz_block_size_rocm_(void const *kernel, sz_size_t shared_bytes, sz_size_t ceiling,
                                                 sz_size_t fallback) {
    sz_size_t block_size = fallback, most_threads = 0, candidate;
    hipFuncAttributes attributes;
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess) return fallback;
    if ((sz_size_t)attributes.maxThreadsPerBlock < ceiling) ceiling = (sz_size_t)attributes.maxThreadsPerBlock;
    for (candidate = 64; candidate <= ceiling; candidate *= 2) {
        sz_size_t const threads = sz_resident_blocks_rocm_(kernel, candidate, shared_bytes) * candidate;
        if (threads >= most_threads && threads != 0) most_threads = threads, block_size = candidate;
    }
    return block_size;
}

/** Launches @p kernel over @p grid blocks of @p block threads on @p stream, its arguments passed by
 *  address. The runtime's own error stays readable through @c hipGetLastError. */
STRINGZILLA_INLINE sz_status_t sz_launch_rocm_(void const *kernel, dim3 grid, dim3 block, void **arguments,
                                               sz_size_t shared_bytes, void *stream) {
    return hipLaunchKernel(kernel, grid, block, arguments, shared_bytes, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Sets @p length bytes at device-reachable @p target to @p value, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_fill_rocm_(void *target, sz_size_t length, sz_u8_t value, void *stream) {
    return hipMemsetAsync(target, value, length, (hipStream_t)stream) == hipSuccess ? sz_success_k
                                                                                    : sz_device_code_mismatch_k;
}

/** Copies @p length bytes of @p source into @p target, each in host or device memory, in order on
 *  @p stream. */
STRINGZILLA_INLINE sz_status_t sz_copy_rocm_(void *target, void const *source, sz_size_t length, void *stream) {
    return hipMemcpyAsync(target, source, length, hipMemcpyDefault, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Copies @p bytes of host @p source into the @c __device__ variable at @p symbol, in order on
 *  @p stream. */
STRINGZILLA_INLINE sz_status_t sz_copy_to_symbol_rocm_(void const *symbol, void const *source, sz_size_t bytes,
                                                       void *stream) {
    return hipMemcpyToSymbolAsync(symbol, source, bytes, 0, hipMemcpyHostToDevice, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Waits for everything enqueued on @p stream; only an engine's init may. */
STRINGZILLA_INLINE sz_status_t sz_synchronize_rocm_(void *stream) {
    return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/** Waits for @p stream on its own device, leaving the caller's current device as it found it. */
STRINGZILLA_INLINE sz_status_t sz_stream_synchronize_rocm_(void *stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_synchronize_rocm_(stream);
    sz_device_leave_rocm_(caller);
    return status;
}

/** Creates a stream on @p device, leaving the caller's current device as it was. */
STRINGZILLA_INLINE sz_status_t sz_stream_create_rocm_(int device, void **stream) {
    int caller = 0;
    sz_status_t status = sz_success_k;
    hipStream_t created = STRINGZILLA_NULL;
    if (hipGetDevice(&caller) != hipSuccess || hipSetDevice(device) != hipSuccess) return sz_missing_gpu_k;
    if (hipStreamCreate(&created) != hipSuccess) status = sz_bad_alloc_k;
    sz_device_leave_rocm_(caller);
    *stream = (void *)created;
    return status;
}

/** Destroys @p stream once the work queued on it completes; a null stream is the default one. */
STRINGZILLA_INLINE sz_status_t sz_stream_destroy_rocm_(void *stream) {
    if (!stream) return sz_success_k;
    return hipStreamDestroy((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/** Whether @p stream still has work queued or running; a failed query answers that it has none. */
STRINGZILLA_INLINE sz_bool_t sz_stream_query_rocm_(void *stream) {
    return hipStreamQuery((hipStream_t)stream) == hipErrorNotReady ? sz_true_k : sz_false_k;
}

/** Migrates managed @p pointer to the current device on @p stream, so a kernel reading what the
 *  host just filled takes one bulk move rather than a fault per page. */
STRINGZILLA_INLINE void sz_prefetch_rocm_(void const *pointer, sz_size_t bytes, void *stream) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return;
    sz_unused_(hipMemPrefetchAsync(pointer, bytes, device, (hipStream_t)stream));
}

#pragma endregion Launches

#pragma region Device Sequences

/** Copies @p source into a tape @p stream 's device reads, behind @ref sz_sequence_realloc_best. */
STRINGZILLA_INLINE sz_status_t sz_sequence_realloc_rocm_(sz_sequence_t *target, sz_sequence_t const *source,
                                                         sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                         void *stream) {
    sz_sequence_member_start_t get_start = STRINGZILLA_NULL;
    sz_sequence_member_length_t get_length = STRINGZILLA_NULL;
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    if (hipMemcpyFromSymbol(&get_start, HIP_SYMBOL(sz_sequence_tape_start_symbol_simt_), sizeof(get_start), 0,
                            hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpyFromSymbol(&get_length, HIP_SYMBOL(sz_sequence_tape_length_symbol_simt_), sizeof(get_length), 0,
                            hipMemcpyDeviceToHost) != hipSuccess)
        status = sz_device_code_mismatch_k;
    sz_bool_t const tape = source->get_start == get_start || source->get_start == sz_sequence_tape_start ? sz_true_k
                                                                                                         : sz_false_k;
    if (status == sz_success_k && tape && sz_memory_reaches_rocm_(source->handle))
        target->handle = source->handle, target->count = source->count, *allocated_bytes = 0;
    else if (status == sz_success_k &&
             (status = sz_sequence_realloc_serial_(target, source, allocator, allocated_bytes, stream)) == sz_success_k)
        sz_prefetch_rocm_(target->handle, *allocated_bytes, stream);
    if (status == sz_success_k) target->get_start = get_start, target->get_length = get_length;
    sz_device_leave_rocm_(caller);
    return status;
}

#pragma endregion Device Sequences

#pragma region Devices

/** How many ROCm devices the runtime sees, or zero. */
STRINGZILLA_INLINE sz_size_t sz_rocm_count_devices_(void) {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess ? (sz_size_t)count : 0;
}

/** The capabilities ROCm device @p ordinal runs, by the runtime's own numbering. */
STRINGZILLA_INLINE sz_status_t sz_rocm_capabilities_detected_(sz_size_t ordinal, sz_capability_t *capabilities) {
    int multiprocessors = 0;
    *capabilities = 0;
    if (ordinal >= sz_rocm_count_devices_()) return sz_missing_gpu_k;
    if (hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, (int)ordinal) != hipSuccess)
        return sz_device_code_mismatch_k;
    if (multiprocessors > 0) *capabilities = sz_cap_rocm_k;
    return sz_success_k;
}

/** Creates a stream on ROCm device @p ordinal, leaving the caller's current device as it was. */
STRINGZILLA_INLINE sz_status_t sz_rocm_stream_init_(sz_size_t ordinal, void **stream) {
    *stream = STRINGZILLA_NULL;
    if (ordinal >= sz_rocm_count_devices_()) return sz_missing_gpu_k;
    return sz_stream_create_rocm_((int)ordinal, stream);
}

#pragma endregion Devices

/*  The library defines these once, in `c/target/rocm.hip`; header-only builds define them here. */
#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_ROCM

STRINGZILLA_API sz_status_t sz_rocm_count_devices(sz_size_t *count) {
    *count = sz_rocm_count_devices_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}

STRINGZILLA_API sz_status_t sz_rocm_capabilities_detected(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_rocm_capabilities_detected_(ordinal, capabilities);
}

STRINGZILLA_API sz_status_t sz_rocm_stream_init(sz_size_t ordinal, void **stream) {
    return sz_rocm_stream_init_(ordinal, stream);
}

STRINGZILLA_API sz_status_t sz_rocm_stream_free(void *stream) { return sz_stream_destroy_rocm_(stream); }

STRINGZILLA_API sz_status_t sz_allocator_init_unified_rocm(sz_allocator_t *allocator) {
    sz_allocator_init_unified_rocm_(allocator);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_device_rocm(sz_allocator_t *allocator) {
    allocator->allocate = sz_memory_allocate_device_rocm_;
    allocator->free = sz_memory_free_device_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_allocator_init_pinned_rocm(sz_allocator_t *allocator) {
    allocator->allocate = sz_memory_allocate_pinned_rocm_;
    allocator->free = sz_memory_free_pinned_rocm_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_rocm(sz_sequence_t *target, sz_sequence_t const *source,
                                                     sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                     void *stream) {
    return sz_sequence_realloc_rocm_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(void *stream) { return sz_stream_synchronize_rocm_(stream); }

#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_ROCM

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)
#endif // STRINGZILLA_ROCM_CUH_
