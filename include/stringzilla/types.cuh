/**
 *  @file include/stringzilla/types.cuh
 *  @author Ash Vardanian
 *  @date September 20, 2026
 *  @brief Core types on a CUDA or ROCm device: the memory both sides address, the device scope an
 *      engine call runs in, the launches, and the sequence a kernel can call.
 *
 *  Every family with a SIMT backend needs the same three things before its own kernel is reached,
 *  and none of them is a property of the family: a way to tell device-reachable memory from host
 *  memory, allocators handing back memory of each residency, and a @ref sz_sequence_t whose
 *  accessors run on the device. They live here for the same reason
 *  @ref sz_sequence_from_string_views lives in `types.h` rather than in a family.
 *
 *  Written in C, as every `.cuh` in this library is: the only constructs here a C compiler would
 *  not take are the `extern "C"` that lets a C dispatch unit link against it, and the kernels'
 *  launches, which go through @ref sz_device_launch_ rather than the triple-chevron syntax the
 *  language reserves for C++. HIP compiles the same source: the vendor calls differ in the bodies
 *  below, a kernel's cross-lane steps go through the @b Lanes helpers, and the few PTX sites a
 *  family keeps carry their own HIP arms.
 *
 *  @sa include/stringzilla/types.h
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.h"

/*  The library's host units know the vendors it holds but include neither runtime, as the two
 *  clash; the device code below is for the units compiled as one of them. */
#if STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)
#include <hip/hip_runtime.h> // `hipLaunchKernel`, `hipMallocManaged`, `hipSetDevice`
#elif STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__)
#include <cuda_runtime.h> // `cudaLaunchKernel`, `cudaMallocManaged`, `cudaSetDevice`
#endif

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Device Exports

/**
 *  @brief Whether a CUDA device can dereference @p pointer: managed or device memory, never host,
 *      pinned or not.
 *
 *  Page-locked host memory is the case a caller is most likely to expect to work: the driver
 *  reports it as host, a kernel cannot address it, and this answers @c sz_false_k for it.
 *
 *  @param[in] pointer Any address.
 *  @param[out] reaches Whether a kernel can dereference it.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k where the vendor is not built.
 */
STRINGZILLA_API sz_status_t sz_cuda_memory_reaches_device(void const *pointer, sz_bool_t *reaches);

/**
 *  @brief Initializes an allocator handing back memory both the host and a CUDA device address.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device, as its runtime numbers them, every allocation is made on.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k where the vendor is not built.
 */
STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/**
 *  @brief Initializes an allocator handing back memory only a CUDA device addresses.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device, as its runtime numbers them, every allocation is made on.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k where the vendor is not built.
 */
STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/**
 *  @brief Initializes an allocator handing back page-locked host memory the runtime copies from at
 *      the bus rate.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device, as its runtime numbers them, every allocation is made on.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k where the vendor is not built.
 */
STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/**
 *  @brief Binds a sequence over device-resident @p views whose accessors a CUDA kernel can call.
 *  @param[in] views The @p count views, device-reachable, each pointing at device-reachable text.
 *  @param[in] count Number of views.
 *  @param[out] sequence Left untouched unless the call succeeds.
 *  @return @c sz_success_k, @c sz_device_code_mismatch_k when the accessors' addresses cannot be
 *      read off the device, or @c sz_missing_gpu_k where the vendor is not built.
 */
STRINGZILLA_API sz_status_t sz_cuda_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence);

/** @copydoc sz_cuda_memory_reaches_device, for ROCm. */
STRINGZILLA_API sz_status_t sz_rocm_memory_reaches_device(void const *pointer, sz_bool_t *reaches);

/** @copydoc sz_cuda_memory_allocator_init_unified, for ROCm. */
STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/** @copydoc sz_cuda_memory_allocator_init_device, for ROCm. */
STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/** @copydoc sz_cuda_memory_allocator_init_pinned, for ROCm. */
STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal);

/** @copydoc sz_cuda_sequence_from_string_views, for ROCm. */
STRINGZILLA_API sz_status_t sz_rocm_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence);

#pragma endregion Device Exports

#if (STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__)) || (STRINGZILLA_ARCH_ROCM_ && defined(__HIP__))

#pragma region Device Memory

/**
 *  @brief Makes device @p ordinal current for one call, keeping the caller's device in @p caller.
 *
 *  The runtime keeps each device's primary context from its first use, so switching is two runtime
 *  calls and the thread's device is restored by @ref sz_device_leave_ rather than left behind.
 *
 *  @param[in] stream The stream the call enqueues on, which has to belong to that device, or
 *      @c STRINGZILLA_NULL for its default stream.
 *  @return @c sz_success_k, @c sz_missing_gpu_k when the runtime has no such device, or
 *      @c sz_device_memory_mismatch_k for a stream of another device, leaving the caller's current.
 */
STRINGZILLA_INLINE sz_status_t sz_device_enter_(sz_size_t ordinal, void *stream, int *caller) {
    int stream_device = 0;
#if STRINGZILLA_ARCH_ROCM_
    if (hipGetDevice(caller) != hipSuccess || hipSetDevice((int)ordinal) != hipSuccess) return sz_missing_gpu_k;
    if (!stream) return sz_success_k;
    if (hipStreamGetDevice((hipStream_t)stream, &stream_device) == hipSuccess && stream_device == (int)ordinal)
        return sz_success_k;
    sz_unused_(hipSetDevice(*caller));
#else
    if (cudaGetDevice(caller) != cudaSuccess || cudaSetDevice((int)ordinal) != cudaSuccess) return sz_missing_gpu_k;
    if (!stream) return sz_success_k;
    if (cudaStreamGetDevice((cudaStream_t)stream, &stream_device) == cudaSuccess && stream_device == (int)ordinal)
        return sz_success_k;
    sz_unused_(cudaSetDevice(*caller));
#endif
    return sz_device_memory_mismatch_k;
}

/** Makes @p caller current again, closing the scope @ref sz_device_enter_ opened. */
STRINGZILLA_INLINE void sz_device_leave_(int caller) {
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(hipSetDevice(caller));
#else
    sz_unused_(cudaSetDevice(caller));
#endif
}

/**
 *  @brief Whether the device can dereference @p pointer: managed or device memory, never host,
 *      pinned or not.
 *
 *  Page-locked host memory is the case a caller is most likely to expect to work: the runtime
 *  reports it as host, a kernel cannot address it, and this answers @c sz_false_k for it.
 */
STRINGZILLA_INLINE sz_bool_t sz_memory_reaches_device_(void const *pointer) {
#if STRINGZILLA_ARCH_ROCM_
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess) return sz_false_k;
    return attributes.type == hipMemoryTypeDevice || attributes.type == hipMemoryTypeManaged ? sz_true_k : sz_false_k;
#else
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) return sz_false_k;
    if (attributes.type == cudaMemoryTypeDevice) return sz_true_k;
    return attributes.type == cudaMemoryTypeManaged ? sz_true_k : sz_false_k;
#endif
}

/*  Each allocator below carries its device's ordinal in @c handle, which it never dereferences. */

STRINGZILLA_INLINE void *sz_memory_allocate_unified_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    if (sz_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipMallocManaged(&pointer, bytes, hipMemAttachGlobal) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_device_leave_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_device_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    if (sz_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipMalloc(&pointer, bytes) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_device_leave_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_pinned_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    if (sz_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipHostMalloc(&pointer, bytes, hipHostMallocDefault) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_device_leave_(caller);
    return pointer;
}

/** Frees unified and device blocks alike. */
STRINGZILLA_INLINE void sz_memory_free_device_(void *pointer, sz_size_t bytes, void *handle) {
    int caller = 0;
    sz_unused_(bytes);
    if (!pointer || sz_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return;
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(hipFree(pointer));
#else
    sz_unused_(cudaFree(pointer));
#endif
    sz_device_leave_(caller);
}

STRINGZILLA_INLINE void sz_memory_free_pinned_(void *pointer, sz_size_t bytes, void *handle) {
    int caller = 0;
    sz_unused_(bytes);
    if (!pointer || sz_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return;
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(hipHostFree(pointer));
#else
    sz_unused_(cudaFreeHost(pointer));
#endif
    sz_device_leave_(caller);
}

/**
 *  @brief Initializes an allocator handing back memory both the host and the device address.
 *
 *  What a family's scratch needs when the host prepares it and a kernel reads it - a prepared
 *  query's B-tree, a Myers mask table - and what the convenience verbs stage a host-resident
 *  caller's arguments into.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device every allocation and free is made on.
 *  @sa sz_memory_allocator_init_default
 */
STRINGZILLA_INLINE void sz_memory_allocator_init_unified_(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    allocator->allocate = &sz_memory_allocate_unified_;
    allocator->free = &sz_memory_free_device_;
    allocator->handle = (void *)ordinal;
}

/**
 *  @brief Initializes an allocator handing back memory only the device addresses.
 *
 *  What a round's scratch needs when no host code ever reads it, and what a unified block would
 *  otherwise pay page migration for on every access from the wrong side.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device every allocation and free is made on.
 */
STRINGZILLA_INLINE void sz_memory_allocator_init_device_(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    allocator->allocate = &sz_memory_allocate_device_;
    allocator->free = &sz_memory_free_device_;
    allocator->handle = (void *)ordinal;
}

/**
 *  @brief Initializes an allocator handing back page-locked host memory the driver copies from at
 *      the bus rate.
 *
 *  A kernel cannot address what this returns - @ref sz_memory_reaches_device_ answers @c sz_false_k
 *  for it - so it is the staging side of a transfer rather than anything a launch reads.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] ordinal The device every allocation and free is made on.
 */
STRINGZILLA_INLINE void sz_memory_allocator_init_pinned_(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
    allocator->allocate = &sz_memory_allocate_pinned_;
    allocator->free = &sz_memory_free_pinned_;
    allocator->handle = (void *)ordinal;
}

#pragma endregion Device Memory

#pragma region Launches

/** One attribute of the current device, as its vendor numbers them, or zero when unanswered. */
STRINGZILLA_INLINE sz_size_t sz_device_attribute_(int attribute) {
    int device = 0, value = 0;
#if STRINGZILLA_ARCH_ROCM_
    if (hipGetDevice(&device) != hipSuccess) return 0;
    if (hipDeviceGetAttribute(&value, (hipDeviceAttribute_t)attribute, device) != hipSuccess) return 0;
#else
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&value, (enum cudaDeviceAttr)attribute, device) != cudaSuccess) return 0;
#endif
    return value > 0 ? (sz_size_t)value : 0;
}

/** Multiprocessors of the current device, or zero when the runtime will not say. */
STRINGZILLA_INLINE sz_size_t sz_device_multiprocessors_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_device_attribute_(hipDeviceAttributeMultiprocessorCount);
#else
    return sz_device_attribute_(cudaDevAttrMultiProcessorCount);
#endif
}

/** Threads one multiprocessor of the current device keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_threads_per_multiprocessor_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_device_attribute_(hipDeviceAttributeMaxThreadsPerMultiProcessor);
#else
    return sz_device_attribute_(cudaDevAttrMaxThreadsPerMultiProcessor);
#endif
}

/** Shared memory one block of the current device gets without opting in, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_shared_bytes_per_block_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_device_attribute_(hipDeviceAttributeMaxSharedMemoryPerBlock);
#else
    return sz_device_attribute_(cudaDevAttrMaxSharedMemoryPerBlock);
#endif
}

/** Blocks of @p threads running @p kernel with @p shared_bytes of dynamic shared memory that one
 *  multiprocessor keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_resident_blocks_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes) {
    int blocks = 0;
#if STRINGZILLA_ARCH_ROCM_
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, (int)threads, shared_bytes) != hipSuccess)
        return 0;
#else
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, (int)threads, shared_bytes) != cudaSuccess)
        return 0;
#endif
    return blocks > 0 ? (sz_size_t)blocks : 0;
}

/**
 *  @brief The block size of @p kernel landing the most resident warps per multiprocessor.
 *
 *  Register pressure and staged shared memory both move the residency ceiling, so the answer is the
 *  device's and the kernel's rather than a constant's. Powers of two from 64 up to @p ceiling are
 *  tried, ties go to the wider block, and @p fallback answers when the runtime will not.
 */
STRINGZILLA_INLINE sz_size_t sz_device_block_size_(void const *kernel, sz_size_t shared_bytes, sz_size_t ceiling,
                                                   sz_size_t fallback) {
    sz_size_t block_size = fallback, most_warps = 0, candidate;
#if STRINGZILLA_ARCH_ROCM_
    hipFuncAttributes attributes;
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess) return fallback;
#else
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) return fallback;
#endif
    if ((sz_size_t)attributes.maxThreadsPerBlock < ceiling) ceiling = (sz_size_t)attributes.maxThreadsPerBlock;
    for (candidate = 64; candidate <= ceiling; candidate *= 2) {
        sz_size_t const warps = sz_device_resident_blocks_(kernel, candidate, shared_bytes) * (candidate / 32);
        if (warps >= most_warps && warps != 0) most_warps = warps, block_size = candidate;
    }
    return block_size;
}

/** Launches @p kernel over @p grid blocks of @p block threads on @p stream, its arguments passed by
 *  address. The vendor's own error stays readable through its @c GetLastError. */
STRINGZILLA_INLINE sz_status_t sz_device_launch_(void const *kernel, dim3 grid, dim3 block, void **arguments,
                                                 sz_size_t shared_bytes, void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    return hipLaunchKernel(kernel, grid, block, arguments, shared_bytes, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
#else
    return cudaLaunchKernel(kernel, grid, block, arguments, shared_bytes, (cudaStream_t)stream) == cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
#endif
}

/** Whether @p kernel was compiled for Blackwell or later, whose blocks take over the tiles of
 *  blocks not yet started, so its grid may be sized by the work rather than by residency. */
STRINGZILLA_INLINE sz_bool_t sz_device_kernel_steals_(void const *kernel) {
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(kernel);
    return sz_false_k;
#else
    cudaFuncAttributes attributes;
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess) return sz_false_k;
    return attributes.ptxVersion >= 100 ? sz_true_k : sz_false_k;
#endif
}

/**
 *  @brief Clusters of @p cluster_blocks blocks of @p threads running @p kernel with @p shared_bytes
 *      of dynamic shared memory the current device keeps resident at once, or zero for none.
 *
 *  A kernel built for a device older than Hopper has no cluster to read, so it is answered zero
 *  whatever the device could do.
 */
STRINGZILLA_INLINE sz_size_t sz_device_resident_clusters_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes,
                                                          sz_size_t cluster_blocks) {
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(kernel), sz_unused_(threads), sz_unused_(shared_bytes), sz_unused_(cluster_blocks);
    return 0;
#else
    cudaFuncAttributes attributes;
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attribute;
    int clusters = 0;
    if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess || attributes.ptxVersion < 90) return 0;
    config.gridDim = dim3((unsigned)cluster_blocks), config.blockDim = dim3((unsigned)threads);
    config.dynamicSmemBytes = shared_bytes, config.stream = 0;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = (unsigned)cluster_blocks, attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute, config.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&clusters, kernel, &config) != cudaSuccess) return 0;
    return clusters > 0 ? (sz_size_t)clusters : 0;
#endif
}

/** Launches @p kernel like @ref sz_device_launch_, in clusters of @p cluster_blocks blocks along x
 *  when above one. */
STRINGZILLA_INLINE sz_status_t sz_device_launch_clustered_(void const *kernel, dim3 grid, dim3 block, void **arguments,
                                                           sz_size_t shared_bytes, sz_size_t cluster_blocks,
                                                           void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(cluster_blocks);
    return sz_device_launch_(kernel, grid, block, arguments, shared_bytes, stream);
#else
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attribute;
    if (cluster_blocks <= 1) return sz_device_launch_(kernel, grid, block, arguments, shared_bytes, stream);
    config.gridDim = grid, config.blockDim = block, config.dynamicSmemBytes = shared_bytes;
    config.stream = (cudaStream_t)stream;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = (unsigned)cluster_blocks, attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute, config.numAttrs = 1;
    return cudaLaunchKernelExC(&config, kernel, arguments) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
#endif
}

/** Sets @p bytes at device-reachable @p pointer to @p value, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_device_memset_(void *pointer, int value, sz_size_t bytes, void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    return hipMemsetAsync(pointer, value, bytes, (hipStream_t)stream) == hipSuccess ? sz_success_k
                                                                                    : sz_device_code_mismatch_k;
#else
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                                       : sz_device_code_mismatch_k;
#endif
}

/** Copies @p bytes of host @p source into the @c __device__ variable at @p symbol, in order on
 *  @p stream; from pageable memory the runtime stages the copy before returning. */
STRINGZILLA_INLINE sz_status_t sz_copy_to_symbol_simt_(void const *symbol, void const *source, sz_size_t bytes,
                                                       void *stream) {
#if defined(__HIP__)
    return hipMemcpyToSymbolAsync(symbol, source, bytes, 0, hipMemcpyHostToDevice, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
#else
    return cudaMemcpyToSymbolAsync(symbol, source, bytes, 0, cudaMemcpyHostToDevice, (cudaStream_t)stream) ==
                   cudaSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
#endif
}

/** Waits for everything enqueued on @p stream; only an engine's init may. */
STRINGZILLA_INLINE sz_status_t sz_device_synchronize_(void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
#else
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
#endif
}

/** Migrates managed @p pointer to the current device on @p stream, so a kernel reading what the
 *  host just filled takes one bulk move rather than a fault per page. Memory the driver does not
 *  manage reports as much, which is not an error. */
STRINGZILLA_INLINE void sz_device_prefetch_(void const *pointer, sz_size_t bytes, void *stream) {
    int device = 0;
#if STRINGZILLA_ARCH_ROCM_
    if (hipGetDevice(&device) != hipSuccess) return;
    hipError_t const moved = hipMemPrefetchAsync(pointer, bytes, device, (hipStream_t)stream);
#else
    cudaMemLocation where;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    where.type = cudaMemLocationTypeDevice;
    where.id = device;
    cudaError_t const moved = cudaMemPrefetchAsync(pointer, bytes, where, 0, (cudaStream_t)stream);
#endif
    sz_unused_(moved);
}

#pragma endregion Launches

/*  Kernels are written for 32 lanes. A 64-wide AMD wavefront runs two such groups side by side, so
 *  every cross-lane step stays inside its own half. */
#pragma region Lanes

/** @p value from the lane @p delta below this one, or this lane's own below lane @p delta. */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_simt_(sz_u32_t value, unsigned delta) {
#if STRINGZILLA_ARCH_ROCM_
    return __shfl_up(value, delta, 32);
#else
    return __shfl_up_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** @p value from the lane @p delta above this one, or this lane's own past the last lane. */
STRINGZILLA_DEVICE int sz_shuffle_down_simt_(int value, unsigned delta) {
#if STRINGZILLA_ARCH_ROCM_
    return __shfl_down(value, delta, 32);
#else
    return __shfl_down_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** Lane zero's @p value, on every one of this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_lanes_broadcast_simt_(sz_u32_t value) {
#if STRINGZILLA_ARCH_ROCM_
    return __shfl(value, 0, 32);
#else
    return __shfl_sync(0xFFFFFFFFu, value, 0);
#endif
}

/** Whether @p predicate holds on any of this thread's 32 lanes. */
STRINGZILLA_DEVICE int sz_lanes_any_simt_(int predicate) {
#if STRINGZILLA_ARCH_ROCM_
    return ((__ballot(predicate) >> (__lane_id() & 32u)) & 0xFFFFFFFFull) != 0;
#else
    return __any_sync(0xFFFFFFFFu, predicate) != 0;
#endif
}

/**
 *  @brief The block's exclusive prefix sum of @p value, with the block's own total left
 *      in @p total.
 *
 *  A Hillis-Steele scan over @p shared, which the caller sizes at one entry per thread. Thirty
 *  lines rather than a dependency: a block scan is the only collective these kernels need, and
 *  pulling a template library into a C tier for it would cost the property the tier exists for.
 */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_simt_(sz_size_t value, sz_size_t *shared, sz_size_t *total) {
    unsigned const lane = threadIdx.x;
    unsigned offset;
    sz_size_t inclusive;
    shared[lane] = value;
    __syncthreads();
    for (offset = 1; offset < blockDim.x; offset *= 2) {
        sz_size_t const addend = lane >= offset ? shared[lane - offset] : 0;
        __syncthreads();
        shared[lane] += addend;
        __syncthreads();
    }
    inclusive = shared[lane];
    *total = shared[blockDim.x - 1];
    __syncthreads(); // ! The caller reuses `shared` for the next tile.
    return inclusive - value;
}

#pragma endregion Lanes

/*  Hopper and later group a grid's blocks into clusters whose shared memory each of them addresses,
 *  so a few blocks can share one block's tables. Elsewhere, and in a grid launched without
 *  clusters, every block is a cluster of one. */
#pragma region Clusters

/** The block's rank inside its cluster. */
STRINGZILLA_DEVICE sz_u32_t sz_cluster_rank_simt_(void) {
    sz_u32_t rank = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
#endif
    return rank;
}

/** Blocks in the block's cluster. */
STRINGZILLA_DEVICE sz_u32_t sz_cluster_size_simt_(void) {
    sz_u32_t size = 1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(size));
#endif
    return size;
}

/** Where @p pointer, into this block's shared memory, lands in the cluster's block at @p rank. */
STRINGZILLA_DEVICE void *sz_cluster_map_simt_(void *pointer, sz_u32_t rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    void *mapped;
    asm("mapa.u64 %0, %1, %2;" : "=l"(mapped) : "l"(pointer), "r"(rank));
    return mapped;
#else
    sz_unused_(rank);
    return pointer;
#endif
}

/** A barrier across every thread of the cluster, ordering each block's shared writes before it. */
STRINGZILLA_DEVICE void sz_cluster_sync_simt_(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#else
    __syncthreads();
#endif
}

#pragma endregion Clusters

/*  A grid can be sized by its work rather than by what stays resident: on Blackwell a block done
 *  with its own tile cancels a block not yet started and takes that block's tile, so the hardware's
 *  queue of unlaunched blocks is the queue of tiles, and a round keeps no state of its own on the
 *  device. Every other target runs each block on its own tile. */
#pragma region Tile Queues

/** One block's queue of a grid's tiles, in shared memory. */
typedef struct sz_tile_queue_t {

    /** The device's answer to the block's latest request for another tile. */
    unsigned long long answer[2] __attribute__((aligned(16)));

    /** The barrier that answer's arrival completes. */
    unsigned long long arrived;

    /** The tile being drawn from - its x in the top half, its y in the next quarter - and the items
     *  drawn from it in the last quarter, so one atomic reads all three. */
    unsigned long long drawn;

    /** Answers consumed, whose parity is the phase the next one completes. */
    sz_u32_t answers;
} sz_tile_queue_t;

/** The x a drained queue holds, past every tile. */
#define STRINGZILLA_CUDA_QUEUE_DRAINED (0xFFFFFFFFull << 32)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

/** Asks the device for a block of this grid not yet started, the answer completing the barrier. */
STRINGZILLA_DEVICE void sz_tile_queue_ask_simt_(sz_tile_queue_t *queue) {
    sz_u32_t const arrived = (sz_u32_t)__cvta_generic_to_shared(&queue->arrived);
    sz_u32_t const answer = (sz_u32_t)__cvta_generic_to_shared(queue->answer);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;" ::"r"(arrived) : "memory");
    asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128" //
                 " [%0], [%1];" ::"r"(answer),
                 "r"(arrived)
                 : "memory");
}

/** Waits for the answer to the latest request, writing the cancelled block's coordinates, and
 *  whether there was one. A request answered with none must be the block's last. */
STRINGZILLA_DEVICE int sz_tile_queue_take_simt_(sz_tile_queue_t *queue, sz_u32_t *x, sz_u32_t *y) {
    sz_u32_t const arrived = (sz_u32_t)__cvta_generic_to_shared(&queue->arrived);
    sz_u32_t const answer = (sz_u32_t)__cvta_generic_to_shared(queue->answer);
    // Counted atomically, as consecutive handovers take answers from different threads.
    sz_u32_t const parity = atomicAdd(&queue->answers, 1u) & 1u;
    sz_u32_t done = 0, taken = 0, first_x = 0, first_y = 0, first_z = 0, unused = 0;
    while (!done)
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n" //
                     " selp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done)
                     : "r"(arrived), "r"(parity)
                     : "memory");
    asm volatile("{\n .reg .b128 r;\n .reg .pred p;\n ld.shared.b128 r, [%5];\n"                            //
                 " clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n selp.u32 %0, 1, 0, p;\n" //
                 " @p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, %4}, r;\n}"
                 : "=r"(taken), "+r"(first_x), "+r"(first_y), "+r"(first_z), "+r"(unused)
                 : "r"(answer)
                 : "memory");
    *x = first_x, *y = first_y;
    return (int)taken;
}

#endif

/**
 *  @brief Points @p queue at the block's own tile and, where blocks steal, asks for the next one;
 *      every thread calls it, and it ends in a barrier.
 *
 *  @param[in] seats Items of the tile handed out without a draw, one to each of the block's first
 *      seats by @ref sz_tile_queue_first_simt_, so its opening draws never contend at once.
 */
STRINGZILLA_DEVICE void sz_tile_queue_open_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                 sz_size_t seats) {
    if (threadIdx.x == 0) {
        sz_size_t const items = sz_min_of_two(tile_size, count - (sz_size_t)blockIdx.x * tile_size);
        queue->drawn = ((unsigned long long)blockIdx.x << 32) | ((unsigned long long)blockIdx.y << 16) |
                       (unsigned long long)sz_min_of_two(seats, items);
        queue->answers = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        sz_u32_t const arrived = (sz_u32_t)__cvta_generic_to_shared(&queue->arrived);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(arrived) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        sz_tile_queue_ask_simt_(queue);
#endif
    }
    __syncthreads();
}

/**
 *  @brief Draws one item of the tiles @p queue hands out, @p tile_size of @p count to a tile, from
 *      any thread at any time.
 *
 *  @param[out] y The row of the grid the item's tile sits on.
 *  @param[out] item The item's index among all @p count.
 *  @return Zero once the grid has nothing left for this block.
 *
 *  The thread that draws one past a tile's last item hands the queue over to the tile its block's
 *  latest request took, and asks for the next; any thread drawing past it meanwhile waits for the
 *  handover, which independent thread scheduling lets it do beside the thread making it.
 */
STRINGZILLA_DEVICE int sz_tile_queue_draw_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                sz_u32_t *y, sz_size_t *item) {
    for (;;) {
        unsigned long long const drawn = atomicAdd(&queue->drawn, 1ull);
        if (drawn >= STRINGZILLA_CUDA_QUEUE_DRAINED) return 0;
        sz_size_t const first = (sz_size_t)(drawn >> 32) * tile_size;
        sz_size_t const items = sz_min_of_two(tile_size, count - first);
        sz_size_t const offset = (sz_size_t)(drawn & 0xFFFFu);
        if (offset < items) {
            *y = (sz_u32_t)(drawn >> 16) & 0xFFFFu, *item = first + offset;
            return 1;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        // The last handover moved the answer count before the tile this thread saw: fence first.
        __threadfence_block();
        if (offset == items) {
            sz_u32_t next_x, next_y;
            if (!sz_tile_queue_take_simt_(queue, &next_x, &next_y)) {
                atomicExch(&queue->drawn, STRINGZILLA_CUDA_QUEUE_DRAINED);
                return 0;
            }
            sz_tile_queue_ask_simt_(queue);
            __threadfence_block();
            atomicExch(&queue->drawn, ((unsigned long long)next_x << 32) | ((unsigned long long)next_y << 16) | 1ull);
            *y = next_y, *item = (sz_size_t)next_x * tile_size;
            return 1;
        }
        while (((*(unsigned long long volatile *)&queue->drawn ^ drawn) >> 16) == 0) __nanosleep(64);
#else
        return 0;
#endif
    }
}

/** The first item of @p seat, which is handed out directly while the block's own tile holds it and
 *  drawn like any other past that. */
STRINGZILLA_DEVICE int sz_tile_queue_first_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                 sz_size_t seat, sz_u32_t *y, sz_size_t *item) {
    sz_size_t const first = (sz_size_t)blockIdx.x * tile_size;
    if (seat < sz_min_of_two(tile_size, count - first)) {
        *y = blockIdx.y, *item = first + seat;
        return 1;
    }
    return sz_tile_queue_draw_simt_(queue, tile_size, count, y, item);
}

/**
 *  @brief The block's next tile, once every thread is done with its last; every thread calls it.
 *  @return Zero once the grid has nothing left for this block.
 *
 *  Thread zero takes the answer and asks again before the barrier, so the request is in flight
 *  while the block works through the tile it returns. The caller holds another barrier between two
 *  calls, as each overwrites what the last one shared.
 */
STRINGZILLA_DEVICE int sz_tile_queue_next_simt_(sz_tile_queue_t *queue, sz_u32_t *x, sz_u32_t *y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        sz_u32_t next_x = 0, next_y = 0;
        int const taken = sz_tile_queue_take_simt_(queue, &next_x, &next_y);
        if (taken) sz_tile_queue_ask_simt_(queue);
        queue->drawn = taken ? ((unsigned long long)next_x << 32) | ((unsigned long long)next_y << 16)
                             : STRINGZILLA_CUDA_QUEUE_DRAINED;
    }
    __syncthreads();
    unsigned long long const drawn = queue->drawn;
    if (drawn >= STRINGZILLA_CUDA_QUEUE_DRAINED) return 0;
    *x = (sz_u32_t)(drawn >> 32), *y = (sz_u32_t)(drawn >> 16) & 0xFFFFu;
    return 1;
#else
    sz_unused_(queue), sz_unused_(x), sz_unused_(y);
    return 0;
#endif
}

#pragma endregion Tile Queues

/*  A kernel whose tiles only learn their output sizes by doing their work publishes them through
 *  the caller's own length slot: blocks take tickets in the order they start, and each tile waits
 *  for the one before it to chain its total, adds its own, and only then writes. A ticket is only
 *  ever held by a running block, so the wait always ends, and a round needs no scratch beyond the
 *  slot it reports into, which the last tile overwrites with the length. */
#pragma region Chained Tiles

enum {

    /** Tiles one chain carries at most, which bounds how long the last one waits. */
    sz_chain_tiles_max_k = 1024,

    /** Where the length slot keeps the tickets handed out while a round runs. */
    sz_chain_tickets_shift_k = 54,

    /** Where it keeps the tiles chained, above the bytes those produced. */
    sz_chain_chained_shift_k = 44,
};

/** Tiles of at least @p tile_bytes_min a chain cuts @p length bytes into, at most
 *  @c sz_chain_tiles_max_k of them, and the width @p tile_bytes each takes. */
STRINGZILLA_INLINE sz_size_t sz_chain_tiles_simt_(sz_size_t length, sz_size_t tile_bytes_min, sz_size_t *tile_bytes) {
    sz_size_t const tiles = sz_min_of_two(sz_size_divide_round_up(length, tile_bytes_min),
                                          (sz_size_t)sz_chain_tiles_max_k);
    *tile_bytes = sz_size_divide_round_up(length, tiles);
    return sz_size_divide_round_up(length, *tile_bytes);
}

/** This block's ticket, the tile it works on; every thread calls it, and it ends in a barrier. */
STRINGZILLA_DEVICE sz_size_t sz_chain_ticket_simt_(sz_size_t *slot) {
    __shared__ sz_size_t ticket;
    if (threadIdx.x == 0)
        ticket = (sz_size_t)(atomicAdd((unsigned long long *)slot, 1ull << sz_chain_tickets_shift_k) >>
                             sz_chain_tickets_shift_k);
    __syncthreads();
    return ticket;
}

/** Where tile @p ticket of @p tiles writes, once every tile before it chained, publishing its own
 *  @p total for the ones after; every thread calls it, and it ends in a barrier. */
STRINGZILLA_DEVICE sz_size_t sz_chain_offset_simt_(sz_size_t *slot, sz_size_t ticket, sz_size_t tiles,
                                                   sz_size_t total) {
    __shared__ sz_size_t offset;
    if (threadIdx.x == 0) {
        unsigned long long *const chain = (unsigned long long *)slot;
        unsigned long long chained;
        do chained = *(unsigned long long volatile *)chain;
        while ((chained >> sz_chain_chained_shift_k & (sz_chain_tiles_max_k - 1)) != ticket);
        offset = (sz_size_t)(chained & ((1ull << sz_chain_chained_shift_k) - 1));
        // Every ticket is taken and every earlier tile chained, so the last tile's write is final.
        if (ticket + 1 == tiles) *slot = offset + total;
        else atomicAdd(chain, (1ull << sz_chain_chained_shift_k) + total);
    }
    __syncthreads();
    return offset;
}

/** The first place at or after @p position a serial walk over UTF-8 steps on, which is where both
 *  neighbours of a cut agree to split. */
STRINGZILLA_DEVICE sz_size_t sz_chain_utf8_cut_simt_(sz_u8_t const *text, sz_size_t length, sz_size_t position) {
    // The walk starts on the text's first byte whatever it is, and three continuation bytes in a
    // row end every codepoint begun before them, so a cut never moves further.
    for (sz_size_t step = 0; step != 3 && position != 0 && position < length && (text[position] & 0xC0u) == 0x80u;
         ++step)
        ++position;
    return position;
}

#pragma endregion Chained Tiles

#pragma region Device Sequences

/** Reads one view's text out of a @ref sz_string_view_t array, from the device. */
static __device__ sz_cptr_t sz_sequence_cuda_view_start_(void const *handle, sz_size_t index) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[index].start;
}

/** Reads one view's length out of a @ref sz_string_view_t array, from the device. */
static __device__ sz_size_t sz_sequence_cuda_view_length_(void const *handle, sz_size_t index) {
    sz_string_view_t const *views = (sz_string_view_t const *)handle;
    return views[index].length;
}

/*  A device function's address is a link-time value, so the host cannot take it with `&` - it has
 *  to read it out of a device variable that already holds it. One pair per translation unit, which
 *  is what @c static buys. */
static __device__ sz_sequence_member_start_t sz_sequence_cuda_view_start_symbol_ = &sz_sequence_cuda_view_start_;
static __device__ sz_sequence_member_length_t sz_sequence_cuda_view_length_symbol_ = &sz_sequence_cuda_view_length_;

/**
 *  @brief Binds a sequence over device-resident @p views whose accessors a kernel can call.
 *  @param[in] views The @p count views, device-reachable, each pointing at device-reachable text.
 *  @param[in] count Number of views.
 *  @param[out] sequence Left untouched unless the call succeeds.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when the accessors' addresses cannot be
 *      read off the device.
 *  @sa sz_sequence_from_string_views
 */
STRINGZILLA_INLINE sz_status_t sz_sequence_from_string_views_cuda_(sz_string_view_t const *views, sz_size_t count,
                                                                   sz_sequence_t *sequence) {
    sz_sequence_member_start_t get_start = STRINGZILLA_NULL;
    sz_sequence_member_length_t get_length = STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipMemcpyFromSymbol(&get_start, HIP_SYMBOL(sz_sequence_cuda_view_start_symbol_), sizeof(get_start), 0,
                            hipMemcpyDeviceToHost) != hipSuccess)
        return sz_device_code_mismatch_k;
    if (hipMemcpyFromSymbol(&get_length, HIP_SYMBOL(sz_sequence_cuda_view_length_symbol_), sizeof(get_length), 0,
                            hipMemcpyDeviceToHost) != hipSuccess)
        return sz_device_code_mismatch_k;
#else
    if (cudaMemcpyFromSymbol(&get_start, (void const *)&sz_sequence_cuda_view_start_symbol_, sizeof(get_start), 0,
                             cudaMemcpyDeviceToHost) != cudaSuccess)
        return sz_device_code_mismatch_k;
    if (cudaMemcpyFromSymbol(&get_length, (void const *)&sz_sequence_cuda_view_length_symbol_, sizeof(get_length), 0,
                             cudaMemcpyDeviceToHost) != cudaSuccess)
        return sz_device_code_mismatch_k;
#endif
    sequence->get_start = get_start;
    sequence->get_length = get_length;
    sequence->handle = views;
    sequence->count = count;
    return sz_success_k;
}

#pragma endregion Device Sequences

#endif // (STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__)) || (STRINGZILLA_ARCH_ROCM_ && defined(__HIP__))

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_cuda_memory_reaches_device(void const *pointer, sz_bool_t *reaches) {
#if STRINGZILLA_ARCH_CUDA_
    *reaches = sz_memory_reaches_device_(pointer);
    return sz_success_k;
#else
    sz_unused_(pointer), sz_unused_(reaches);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    sz_memory_allocator_init_unified_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    sz_memory_allocator_init_device_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_cuda_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_CUDA_
    sz_memory_allocator_init_pinned_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_cuda_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence) {
#if STRINGZILLA_ARCH_CUDA_
    return sz_sequence_from_string_views_cuda_(views, count, sequence);
#else
    sz_unused_(views), sz_unused_(count), sz_unused_(sequence);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_rocm_memory_reaches_device(void const *pointer, sz_bool_t *reaches) {
#if STRINGZILLA_ARCH_ROCM_
    *reaches = sz_memory_reaches_device_(pointer);
    return sz_success_k;
#else
    sz_unused_(pointer), sz_unused_(reaches);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    sz_memory_allocator_init_unified_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    sz_memory_allocator_init_device_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_rocm_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_size_t ordinal) {
#if STRINGZILLA_ARCH_ROCM_
    sz_memory_allocator_init_pinned_(allocator, ordinal);
    return sz_success_k;
#else
    sz_unused_(allocator), sz_unused_(ordinal);
    return sz_missing_gpu_k;
#endif
}

STRINGZILLA_API sz_status_t sz_rocm_sequence_from_string_views(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_sequence_from_string_views_cuda_(views, count, sequence);
#else
    sz_unused_(views), sz_unused_(count), sz_unused_(sequence);
    return sz_missing_gpu_k;
#endif
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TYPES_CUH_
