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
 *  launches, which go through @ref sz_cuda_launch_ rather than the triple-chevron syntax the
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
 *  calls and the thread's device is restored by @ref sz_cuda_device_leave_ rather than left behind.
 *
 *  @param[in] stream The stream the call enqueues on, which has to belong to that device, or
 *      @c STRINGZILLA_NULL for its default stream.
 *  @return @c sz_success_k, @c sz_missing_gpu_k when the runtime has no such device, or
 *      @c sz_device_memory_mismatch_k for a stream of another device, leaving the caller's current.
 */
STRINGZILLA_INLINE sz_status_t sz_cuda_device_enter_(sz_size_t ordinal, void *stream, int *caller) {
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

/** Makes @p caller current again, closing the scope @ref sz_cuda_device_enter_ opened. */
STRINGZILLA_INLINE void sz_cuda_device_leave_(int caller) {
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
    if (sz_cuda_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipMallocManaged(&pointer, bytes, hipMemAttachGlobal) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_cuda_device_leave_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_device_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    if (sz_cuda_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipMalloc(&pointer, bytes) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_cuda_device_leave_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_memory_allocate_pinned_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    if (sz_cuda_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return STRINGZILLA_NULL;
#if STRINGZILLA_ARCH_ROCM_
    if (hipHostMalloc(&pointer, bytes, hipHostMallocDefault) != hipSuccess) pointer = STRINGZILLA_NULL;
#else
    if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) pointer = STRINGZILLA_NULL;
#endif
    sz_cuda_device_leave_(caller);
    return pointer;
}

/** Frees unified and device blocks alike. */
STRINGZILLA_INLINE void sz_memory_free_device_(void *pointer, sz_size_t bytes, void *handle) {
    int caller = 0;
    sz_unused_(bytes);
    if (!pointer || sz_cuda_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return;
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(hipFree(pointer));
#else
    sz_unused_(cudaFree(pointer));
#endif
    sz_cuda_device_leave_(caller);
}

STRINGZILLA_INLINE void sz_memory_free_pinned_(void *pointer, sz_size_t bytes, void *handle) {
    int caller = 0;
    sz_unused_(bytes);
    if (!pointer || sz_cuda_device_enter_((sz_size_t)handle, STRINGZILLA_NULL, &caller) != sz_success_k) return;
#if STRINGZILLA_ARCH_ROCM_
    sz_unused_(hipHostFree(pointer));
#else
    sz_unused_(cudaFreeHost(pointer));
#endif
    sz_cuda_device_leave_(caller);
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
STRINGZILLA_INLINE sz_size_t sz_cuda_attribute_(int attribute) {
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
STRINGZILLA_INLINE sz_size_t sz_cuda_multiprocessors_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_cuda_attribute_(hipDeviceAttributeMultiprocessorCount);
#else
    return sz_cuda_attribute_(cudaDevAttrMultiProcessorCount);
#endif
}

/** Threads one multiprocessor of the current device keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_cuda_threads_per_multiprocessor_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_cuda_attribute_(hipDeviceAttributeMaxThreadsPerMultiProcessor);
#else
    return sz_cuda_attribute_(cudaDevAttrMaxThreadsPerMultiProcessor);
#endif
}

/** Shared memory one block of the current device gets without opting in, or zero. */
STRINGZILLA_INLINE sz_size_t sz_cuda_shared_bytes_per_block_(void) {
#if STRINGZILLA_ARCH_ROCM_
    return sz_cuda_attribute_(hipDeviceAttributeMaxSharedMemoryPerBlock);
#else
    return sz_cuda_attribute_(cudaDevAttrMaxSharedMemoryPerBlock);
#endif
}

/** Blocks of @p threads running @p kernel with @p shared_bytes of dynamic shared memory that one
 *  multiprocessor keeps resident, or zero. */
STRINGZILLA_INLINE sz_size_t sz_cuda_resident_blocks_(void const *kernel, sz_size_t threads, sz_size_t shared_bytes) {
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
STRINGZILLA_INLINE sz_size_t sz_cuda_block_size_(void const *kernel, sz_size_t shared_bytes, sz_size_t ceiling,
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
        sz_size_t const warps = sz_cuda_resident_blocks_(kernel, candidate, shared_bytes) * (candidate / 32);
        if (warps >= most_warps && warps != 0) most_warps = warps, block_size = candidate;
    }
    return block_size;
}

/** Launches @p kernel over @p grid blocks of @p block threads on @p stream, its arguments passed by
 *  address. The vendor's own error stays readable through its @c GetLastError. */
STRINGZILLA_INLINE sz_status_t sz_cuda_launch_(void const *kernel, dim3 grid, dim3 block, void **arguments,
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

/** Sets @p bytes at device-reachable @p pointer to @p value, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_cuda_memset_(void *pointer, int value, sz_size_t bytes, void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    return hipMemsetAsync(pointer, value, bytes, (hipStream_t)stream) == hipSuccess ? sz_success_k
                                                                                    : sz_device_code_mismatch_k;
#else
    return cudaMemsetAsync(pointer, value, bytes, (cudaStream_t)stream) == cudaSuccess ? sz_success_k
                                                                                       : sz_device_code_mismatch_k;
#endif
}

/** Waits for everything enqueued on @p stream; only an engine's init may. */
STRINGZILLA_INLINE sz_status_t sz_cuda_synchronize_(void *stream) {
#if STRINGZILLA_ARCH_ROCM_
    return hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
#else
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess ? sz_success_k : sz_device_code_mismatch_k;
#endif
}

/** Migrates managed @p pointer to the current device on @p stream, so a kernel reading what the
 *  host just filled takes one bulk move rather than a fault per page. Memory the driver does not
 *  manage reports as much, which is not an error. */
STRINGZILLA_INLINE void sz_cuda_prefetch_(void const *pointer, sz_size_t bytes, void *stream) {
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
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_(sz_u32_t value, unsigned delta) {
#if STRINGZILLA_ARCH_ROCM_
    return __shfl_up(value, delta, 32);
#else
    return __shfl_up_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** @p value from the lane @p delta above this one, or this lane's own past the last lane. */
STRINGZILLA_DEVICE int sz_shuffle_down_(int value, unsigned delta) {
#if STRINGZILLA_ARCH_ROCM_
    return __shfl_down(value, delta, 32);
#else
    return __shfl_down_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** Whether @p predicate holds on any of this thread's 32 lanes. */
STRINGZILLA_DEVICE int sz_lanes_any_(int predicate) {
#if STRINGZILLA_ARCH_ROCM_
    return ((__ballot(predicate) >> (__lane_id() & 32u)) & 0xFFFFFFFFull) != 0;
#else
    return __any_sync(0xFFFFFFFFu, predicate) != 0;
#endif
}

#pragma endregion Lanes

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
    if (cudaMemcpyFromSymbol(&get_length, (void const *)&sz_sequence_cuda_view_length_symbol_,
                             sizeof(get_length), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
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
