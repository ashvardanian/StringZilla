/**
 *  @file include/stringzilla/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The CUDA runtime as the library's CUDA host code drives it: the device scope an engine
 *      call runs in, the memory both sides address, the launches, the streams and the producers,
 *      and the cross-lane steps its kernels hand the shared walks.
 *
 *  Every call runs on the device of the stream it is given, which @ref sz_device_enter_cuda_ makes
 *  current for its duration, and a null stream runs on the caller's current device.
 *
 *  Only nvcc sees past the guard, and nothing here is shared with ROCm, whose twins live in
 *  `rocm.cuh` under their own names, so a library holding both vendors links one body per name.
 *  Written in C, as every `.cuh` in this library is: the only construct a C compiler would not take
 *  is the `extern "C"` that lets a C dispatch unit link against it.
 *
 *  @sa include/stringzilla/rocm.cuh
 *  @sa include/stringzilla/types.cuh
 */
#ifndef STRINGZILLA_CUDA_CUH_
#define STRINGZILLA_CUDA_CUH_

#include "stringzilla/types.cuh"
#include "stringzilla/capabilities.h"  // `sz_cap_cuda_k`
#include "stringzilla/memory/serial.h" // `sz_sequence_realloc_serial_`

#if STRINGZILLA_ARCH_CUDA_
#include <cuda_runtime.h> // `cudaLaunchKernel`, `cudaMallocManaged`, `cudaSetDevice`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Device Memory

/**
 *  @brief Makes the device @p stream belongs to current for one call, keeping the caller's device
 *      in @p caller.
 *
 *  The runtime keeps each device's primary context from its first use, so switching is two runtime
 *  calls and the thread's device is restored by @ref sz_device_leave_cuda_ rather than left behind.
 *
 *  @param[in] stream The stream the call enqueues on, which names its device, or
 *      @c STRINGZILLA_NULL for the default stream of the caller's current device.
 *  @return @c sz_success_k, @c sz_missing_gpu_k when the runtime has no device to make current, or
 *      @c sz_device_memory_mismatch_k for a stream the runtime cannot place, in which case the
 *      caller's device stays current.
 */
STRINGZILLA_INLINE sz_status_t sz_device_enter_cuda_(sz_stream_t stream, int *caller) {
    int device = 0;
    if (cudaGetDevice(caller) != cudaSuccess) return sz_missing_gpu_k;
    if (!stream) return sz_success_k;
    if (cudaStreamGetDevice((cudaStream_t)stream, &device) != cudaSuccess) return sz_device_memory_mismatch_k;
    return cudaSetDevice(device) == cudaSuccess ? sz_success_k : sz_missing_gpu_k;
}

/** Makes @p caller current again, closing the scope @ref sz_device_enter_cuda_ opened. */
STRINGZILLA_INLINE void sz_device_leave_cuda_(int caller) { sz_unused_(cudaSetDevice(caller)); }

/** Accepts managed memory or the current device's own allocation.
 *  Host mappings require a device address and are unsupported by this check. */
STRINGZILLA_INLINE sz_bool_t sz_memory_accessible_cuda_(void const *pointer) {
    int device = 0;
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess || cudaGetDevice(&device) != cudaSuccess)
        return sz_false_k;
    if (attributes.type == cudaMemoryTypeManaged) return sz_true_k;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device ? sz_true_k : sz_false_k;
}

/*  Each allocator below is stateless, ignoring @c handle, and works on the device of its stream. */

STRINGZILLA_INLINE void *sz_allocate_unified_cuda_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_cuda_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal) != cudaSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_cuda_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_allocate_device_cuda_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_cuda_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_cuda_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_allocate_pinned_cuda_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_cuda_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_cuda_(caller);
    return pointer;
}

/** Frees unified and device blocks alike, once the device is done with everything queued before. */
STRINGZILLA_INLINE void sz_free_device_cuda_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_cuda_(stream, &caller) != sz_success_k) return;
    sz_unused_(cudaFree(pointer));
    sz_device_leave_cuda_(caller);
}

STRINGZILLA_INLINE void sz_free_pinned_cuda_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_cuda_(stream, &caller) != sz_success_k) return;
    sz_unused_(cudaFreeHost(pointer));
    sz_device_leave_cuda_(caller);
}

/**
 *  @brief Initializes an allocator handing back memory both the host and the device address.
 *
 *  What a family's scratch needs when the host prepares it and a kernel reads it - a prepared
 *  query's B-tree, a Myers mask table - and what the convenience verbs stage a host-resident
 *  caller's arguments into.
 *
 *  @param[out] allocator The stateless allocator to initialize, each call on its stream's device.
 *  @sa sz_allocator_init_heap
 */
STRINGZILLA_INLINE void sz_allocator_init_unified_cuda_(sz_allocator_t *allocator) {
    allocator->allocate = &sz_allocate_unified_cuda_;
    allocator->free = &sz_free_device_cuda_;
    allocator->handle = STRINGZILLA_NULL;
}

#pragma endregion Device Memory

#pragma region Launches

/** One attribute of the current device, or zero when unanswered. */
STRINGZILLA_INLINE sz_size_t sz_device_attribute_cuda_(enum cudaDeviceAttr attribute) {
    int device = 0, value = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) return 0;
    return value > 0 ? (sz_size_t)value : 0;
}

/** Multiprocessors of the current device, or zero when the runtime will not say. */
STRINGZILLA_INLINE sz_size_t sz_device_multiprocessors_cuda_(void) {
    return sz_device_attribute_cuda_(cudaDevAttrMultiProcessorCount);
}

/** Threads one multiprocessor of the current device keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_threads_per_multiprocessor_cuda_(void) {
    return sz_device_attribute_cuda_(cudaDevAttrMaxThreadsPerMultiProcessor);
}

/** Shared memory one block of the current device gets without opting in, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_shared_bytes_per_block_cuda_(void) {
    return sz_device_attribute_cuda_(cudaDevAttrMaxSharedMemoryPerBlock);
}

/** Free bytes of the current device's memory, or zero when the runtime will not say. */
STRINGZILLA_INLINE sz_size_t sz_device_free_bytes_cuda_(void) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return 0;
    return (sz_size_t)free_bytes;
}

/** Blocks of @p threads running @p kernel with @p shared_bytes of dynamic shared memory that one
 *  multiprocessor keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_resident_blocks_cuda_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes) {
    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, (int)threads, shared_bytes) != cudaSuccess)
        return 0;
    return blocks > 0 ? (sz_size_t)blocks : 0;
}

/**
 *  @brief The block size of @p kernel landing the most resident warps per multiprocessor.
 *
 *  Register pressure and staged shared memory both move the residency ceiling, so the answer is the
 *  device's and the kernel's rather than a constant's. Powers of two from 64 up to @p ceiling are
 *  tried, ties go to the wider block, and @p fallback answers when the runtime will not.
 */
STRINGZILLA_INLINE sz_size_t sz_block_size_cuda_(void const *kernel, sz_size_t shared_bytes, sz_size_t ceiling,
                                                 sz_size_t fallback) {
    sz_size_t block_size = fallback, most_warps = 0, candidate;
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) return fallback;
    if ((sz_size_t)attributes.maxThreadsPerBlock < ceiling) ceiling = (sz_size_t)attributes.maxThreadsPerBlock;
    for (candidate = 64; candidate <= ceiling; candidate *= 2) {
        sz_size_t const warps = sz_resident_blocks_cuda_(kernel, candidate, shared_bytes) * (candidate / 32);
        if (warps >= most_warps && warps != 0) most_warps = warps, block_size = candidate;
    }
    return block_size;
}

/** Launches @p kernel over @p grid blocks of @p block threads on @p stream, its arguments passed by
 *  address. The runtime's own error stays readable through @c cudaGetLastError. */
STRINGZILLA_INLINE sz_status_t sz_launch_cuda_(void const *kernel, dim3 grid, dim3 block, void **arguments,
                                               sz_size_t shared_bytes, sz_stream_t stream) {
    return cudaLaunchKernel(kernel, grid, block, arguments, shared_bytes, (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Clusters of @p cluster_blocks blocks of @p threads running @p kernel with @p shared_bytes of
 *  dynamic shared memory the current device keeps resident at once, or zero for none. */
STRINGZILLA_INLINE sz_size_t sz_resident_clusters_cuda_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes,
                                                        sz_size_t cluster_blocks) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attribute;
    int clusters = 0;
    config.gridDim = dim3((unsigned)cluster_blocks), config.blockDim = dim3((unsigned)threads);
    config.dynamicSmemBytes = shared_bytes, config.stream = 0;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = (unsigned)cluster_blocks, attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute, config.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) != cudaSuccess) return 0;
    return clusters > 0 ? (sz_size_t)clusters : 0;
}

/** Launches @p kernel like @ref sz_launch_cuda_, in clusters of @p cluster_blocks blocks along x
 *  when above one. */
STRINGZILLA_INLINE sz_status_t sz_launch_clustered_cuda_(void const *kernel, dim3 grid, dim3 block, void **arguments,
                                                         sz_size_t shared_bytes, sz_size_t cluster_blocks,
                                                         sz_stream_t stream) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attribute;
    if (cluster_blocks <= 1) return sz_launch_cuda_(kernel, grid, block, arguments, shared_bytes, stream);
    config.gridDim = grid, config.blockDim = block, config.dynamicSmemBytes = shared_bytes;
    config.stream = (cudaStream_t)stream;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = (unsigned)cluster_blocks, attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute, config.numAttrs = 1;
    return cudaLaunchKernelExC(&config, kernel, arguments) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/** Sets @p length bytes at device-reachable @p target to @p value, in order on @p stream, in the
 *  argument order of @ref sz_fill_serial. */
STRINGZILLA_INLINE sz_status_t sz_fill_cuda_(void *target, sz_size_t length, sz_u8_t value, sz_stream_t stream) {
    return cudaMemsetAsync(target, value, length, (cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                                       : sz_device_code_mismatch_k;
}

/** Copies @p length bytes of @p source into @p target, each in host or device memory, in order on
 *  @p stream, in the argument order of @ref sz_copy_serial. */
STRINGZILLA_INLINE sz_status_t sz_copy_cuda_(void *target, void const *source, sz_size_t length, sz_stream_t stream) {
    return cudaMemcpyAsync(target, source, length, cudaMemcpyDefault, (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Copies @p bytes of host @p source into the @c __device__ variable at @p symbol, in order on
 *  @p stream; from pageable memory the runtime stages the copy before returning. */
STRINGZILLA_INLINE sz_status_t sz_copy_to_symbol_cuda_(void const *symbol, void const *source, sz_size_t bytes,
                                                       sz_stream_t stream) {
    return cudaMemcpyToSymbolAsync(symbol, source, bytes, 0, cudaMemcpyHostToDevice, (cudaStream_t)stream) ==
                   cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Waits for @p stream on its own device, leaving the caller's current device as it found it. */
STRINGZILLA_INLINE sz_status_t sz_stream_synchronize_cuda_(sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
    sz_device_leave_cuda_(caller);
    return status;
}

/** Destroys @p stream once the work queued on it completes; a null stream is the default one. */
STRINGZILLA_INLINE sz_status_t sz_stream_free_cuda_(sz_stream_t stream) {
    if (!stream) return sz_success_k;
    return cudaStreamDestroy((cudaStream_t)stream) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/** Whether @p stream still has work queued or running; a failed query answers that it has none. */
STRINGZILLA_INLINE sz_bool_t sz_stream_query_cuda_(sz_stream_t stream) {
    return cudaStreamQuery((cudaStream_t)stream) == cudaErrorNotReady ? sz_true_k : sz_false_k;
}

/** Migrates managed @p pointer to the current device on @p stream, so a kernel reading what the
 *  host just filled takes one bulk move rather than a fault per page. Memory the driver does not
 *  manage reports as much, which is not an error. */
STRINGZILLA_INLINE void sz_prefetch_cuda_(void const *pointer, sz_size_t bytes, sz_stream_t stream) {
    int device = 0;
    cudaMemLocation where;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    where.type = cudaMemLocationTypeDevice;
    where.id = device;
    sz_unused_(cudaMemPrefetchAsync(pointer, bytes, where, 0, (cudaStream_t)stream));
}

#pragma endregion Launches

#pragma region Device Sequences

/** Prepares a host-readable tape on the stream's device without changing its accessor kind. */
STRINGZILLA_INLINE sz_status_t sz_sequence_realloc_cuda_(sz_sequence_t *target, sz_sequence_t const *source,
                                                         sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                         sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    sz_sequence_t next = *source;
    sz_size_t bytes = 0;
    if (source->get_start != sz_sequence_tape_start || source->get_length != sz_sequence_tape_length ||
        !sz_memory_accessible_cuda_(source->handle))
        status = sz_sequence_realloc_serial_(&next, source, allocator, &bytes, stream);
    if (status == sz_success_k && !sz_memory_accessible_cuda_(next.handle)) {
        if (bytes) allocator->free((void *)next.handle, bytes, allocator->handle, stream);
        status = sz_device_memory_mismatch_k;
    }
    if (status == sz_success_k) {
        sz_u64_t const *offsets = (sz_u64_t const *)next.handle;
        sz_prefetch_cuda_(next.handle, (sz_size_t)offsets[next.count], stream);
        *target = next, *allocated_bytes = bytes;
    }
    sz_device_leave_cuda_(caller);
    return status;
}

#pragma endregion Device Sequences

#pragma region Devices

/** How many CUDA devices the runtime sees, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_count_cuda_(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? (sz_size_t)count : 0;
}

/** The capabilities CUDA device @p ordinal runs, by the runtime's own numbering: every tier from
 *  the baseline up to its compute capability's. */
STRINGZILLA_INLINE sz_status_t sz_capabilities_detected_cuda_(sz_size_t ordinal, sz_capability_t *capabilities) {
    int multiprocessors = 0, major = 0;
    *capabilities = 0;
    if (ordinal >= sz_device_count_cuda_()) return sz_missing_gpu_k;
    if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, (int)ordinal) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, (int)ordinal) != cudaSuccess)
        return sz_device_code_mismatch_k;
    if (multiprocessors <= 0) return sz_success_k;
    *capabilities = sz_cap_cuda_k;
    if (major >= 9) *capabilities |= sz_cap_hopper_k;
    if (major >= 10) *capabilities |= sz_cap_blackwell_k;
    return sz_success_k;
}

/** Creates a stream on CUDA device @p ordinal, leaving the caller's current device as it was. */
STRINGZILLA_INLINE sz_status_t sz_stream_init_cuda_(sz_size_t ordinal, sz_stream_t *stream) {
    int caller = 0;
    sz_status_t status = sz_success_k;
    cudaStream_t created = STRINGZILLA_NULL;
    *stream = STRINGZILLA_NULL;
    if (ordinal >= sz_device_count_cuda_()) return sz_missing_gpu_k;
    if (cudaGetDevice(&caller) != cudaSuccess || cudaSetDevice((int)ordinal) != cudaSuccess) return sz_missing_gpu_k;
    if (cudaStreamCreate(&created) != cudaSuccess) status = sz_bad_alloc_k;
    sz_device_leave_cuda_(caller);
    *stream = (void *)created;
    return status;
}

#pragma endregion Devices

#pragma region Device Primitives

/** Lane `lane - delta`'s @p value, a lane's own below @p delta, with @c shfl.sync.up . */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_cuda_(sz_u32_t value, unsigned delta) {
    return __shfl_up_sync(0xFFFFFFFFu, value, delta);
}

/** Lane `lane + delta`'s @p value, a lane's own past the last, with @c shfl.sync.down . */
STRINGZILLA_DEVICE int sz_shuffle_down_cuda_(int value, unsigned delta) {
    return __shfl_down_sync(0xFFFFFFFFu, value, delta);
}

/** Lane zero's @p value across the warp, with @c shfl.sync.idx . */
STRINGZILLA_DEVICE sz_u32_t sz_lanes_broadcast_cuda_(sz_u32_t value) { return __shfl_sync(0xFFFFFFFFu, value, 0); }

/** Whether @p predicate holds on any lane of the warp, with @c vote.sync.any . */
STRINGZILLA_DEVICE int sz_lanes_any_cuda_(int predicate) { return __any_sync(0xFFFFFFFFu, predicate) != 0; }

#pragma endregion Device Primitives

/*  The library defines these once, in `c/target/cuda.cu`; header-only builds define them here. */
#if STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_CUDA

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

#endif // STRINGZILLA_HEADER_ONLY && STRINGZILLA_TARGET_CUDA

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_CUDA_CUH_
