/**
 *  @file include/stringzilla/types.cuh
 *  @author Ash Vardanian
 *  @date September 20, 2026
 *  @brief Core types on a CUDA device: the memory both sides address, the device that owns it, and
 *      the sequence a kernel can call.
 *
 *  Every family with a CUDA backend needs the same three things before its own kernel is reached,
 *  and none of them is a property of the family: a way to tell device-reachable memory from host
 *  memory, allocators handing back memory of each residency, and a @ref sz_sequence_t whose
 *  accessors run on the device. They live here for the same reason
 *  @ref sz_sequence_from_string_views lives in `types.h` rather than in a family.
 *
 *  Written in C, as every `.cuh` in this library is: the only constructs here a C compiler would
 *  not take are the `extern "C"` that lets a C dispatch unit link against it, and the kernels'
 *  launches, which go through @c cudaLaunchKernel rather than the triple-chevron syntax the
 *  language reserves for C++.
 *
 *  Only the Device Sequences region needs a device compiler. Everything above it is host code
 *  calling the driver and runtime APIs, so a plain C or C++ translation unit in a CUDA build can
 *  allocate and probe without being handed to @c nvcc.
 *
 *  @sa include/stringzilla/types.h
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_TARGET_CUDA
#include <cuda.h>         // `CUcontext`, `cuMemAllocManaged`, `cuDevicePrimaryCtxRetain`
#include <cuda_runtime.h> // `cudaPointerGetAttributes`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Device Memory

/**
 *  @brief One device and the primary context it was retained through, owned by whoever declared it.
 *
 *  @c cuDevicePrimaryCtxRetain bumps a reference count that only @ref sz_cuda_device_free releases,
 *  so the retain belongs to an object with a lifetime rather than to a cache nothing ever drains.
 */
typedef struct sz_cuda_device_t {

    /** The device this context belongs to, as @c cuDeviceGet numbers them. */
    int ordinal;

    /** The retained @c CUcontext, released by @ref sz_cuda_device_free. */
    void *context;
} sz_cuda_device_t;

/**
 *  @brief Retains the primary context of device @p ordinal into @p device, which
 *      the caller releases.
 *
 *  @param[in] ordinal The device to retain, as @c cuDeviceGet numbers them.
 *  @param[out] device Left untouched unless the call succeeds.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when the driver will not hand back
 *      that device's context.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_cuda_device_init(int ordinal, sz_cuda_device_t *device);

/** Releases the primary context @ref sz_cuda_device_init retained, and leaves @p device empty. */
STRINGZILLA_API_RUNTIME void sz_cuda_device_free(sz_cuda_device_t *device);

/**
 *  @brief Whether the device can dereference @p pointer: managed or device memory, never host,
 *      pinned or not.
 *
 *  Page-locked host memory is the case a caller is most likely to expect to work: the driver
 *  reports it as host, a kernel cannot address it, and this answers @c sz_false_k for it.
 */
STRINGZILLA_API_RUNTIME sz_bool_t sz_memory_reaches_device(void const *pointer);

/**
 *  @brief Initializes an allocator handing back memory both the host and the device address.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 */
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_unified(sz_memory_allocator_t *allocator,
                                                              sz_cuda_device_t *device);

/**
 *  @brief Initializes an allocator handing back memory only the device addresses.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 */
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_device(sz_memory_allocator_t *allocator,
                                                             sz_cuda_device_t *device);

/**
 *  @brief Initializes an allocator handing back page-locked host memory the driver copies from at
 *      the bus rate.
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 */
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_pinned(sz_memory_allocator_t *allocator,
                                                             sz_cuda_device_t *device);

/**
 *  @brief Binds a sequence over device-resident @p views whose accessors a kernel can call.
 *  @param[in] views The @p count views, device-reachable, each pointing at device-reachable text.
 *  @param[in] count Number of views.
 *  @param[out] sequence Left untouched unless the call succeeds.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when the accessors' addresses cannot be
 *      read off the device.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_sequence_from_string_views_cuda(sz_string_view_t const *views, sz_size_t count,
                                                                       sz_sequence_t *sequence);

/**
 *  @brief Retains the primary context of device @p ordinal into @p device, which
 *      the caller releases.
 *
 *  @param[in] ordinal The device to retain, as @c cuDeviceGet numbers them.
 *  @param[out] device Left untouched unless the call succeeds.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k when the driver will not hand back
 *      that device's context.
 */
STRINGZILLA_API_COMPTIME sz_status_t sz_cuda_device_init_implementation_(int ordinal, sz_cuda_device_t *device) {
    CUcontext context = STRINGZILLA_NULL;
    CUdevice handle = 0;
    if (cuInit(0) != CUDA_SUCCESS) return sz_device_code_mismatch_k;
    if (cuDeviceGet(&handle, ordinal) != CUDA_SUCCESS) return sz_device_code_mismatch_k;
    if (cuDevicePrimaryCtxRetain(&context, handle) != CUDA_SUCCESS) return sz_device_code_mismatch_k;
    device->ordinal = ordinal;
    device->context = (void *)context;
    return sz_success_k;
}

/** Releases the primary context @ref sz_cuda_device_init retained, and leaves @p device empty. */
STRINGZILLA_API_COMPTIME void sz_cuda_device_free_implementation_(sz_cuda_device_t *device) {
    if (!device->context) return;
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device->ordinal) == CUDA_SUCCESS) cuDevicePrimaryCtxRelease(handle);
    device->ordinal = 0, device->context = STRINGZILLA_NULL;
}

/** Makes the context of @p device current, or leaves the thread's own current context alone when
 *  @p device is @c STRINGZILLA_NULL. */
STRINGZILLA_API_COMPTIME sz_bool_t sz_cuda_device_bind_(sz_cuda_device_t const *device) {
    CUcontext current = STRINGZILLA_NULL;
    if (device) return cuCtxSetCurrent((CUcontext)device->context) == CUDA_SUCCESS ? sz_true_k : sz_false_k;
    return cuCtxGetCurrent(&current) == CUDA_SUCCESS && current ? sz_true_k : sz_false_k;
}

/**
 *  @brief Whether the device can dereference @p pointer: managed or device memory, never host,
 *      pinned or not.
 *
 *  Page-locked host memory is the case a caller is most likely to expect to work: the driver
 *  reports it as host, a kernel cannot address it, and this answers @c sz_false_k for it.
 */
STRINGZILLA_API_COMPTIME sz_bool_t sz_memory_reaches_device_implementation_(void const *pointer) {
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) return sz_false_k;
    if (attributes.type == cudaMemoryTypeDevice) return sz_true_k;
    return attributes.type == cudaMemoryTypeManaged ? sz_true_k : sz_false_k;
}

STRINGZILLA_API_COMPTIME void *sz_memory_allocate_unified_(sz_size_t bytes, void *handle) {
    CUdeviceptr pointer = 0;
    if (!sz_cuda_device_bind_((sz_cuda_device_t const *)handle)) return STRINGZILLA_NULL;
    if (cuMemAllocManaged(&pointer, bytes, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) return STRINGZILLA_NULL;
    return (void *)pointer;
}

STRINGZILLA_API_COMPTIME void *sz_memory_allocate_device_(sz_size_t bytes, void *handle) {
    CUdeviceptr pointer = 0;
    if (!sz_cuda_device_bind_((sz_cuda_device_t const *)handle)) return STRINGZILLA_NULL;
    if (cuMemAlloc(&pointer, bytes) != CUDA_SUCCESS) return STRINGZILLA_NULL;
    return (void *)pointer;
}

STRINGZILLA_API_COMPTIME void *sz_memory_allocate_pinned_(sz_size_t bytes, void *handle) {
    void *pointer = STRINGZILLA_NULL;
    if (!sz_cuda_device_bind_((sz_cuda_device_t const *)handle)) return STRINGZILLA_NULL;
    return cuMemHostAlloc(&pointer, bytes, 0u) == CUDA_SUCCESS ? pointer : STRINGZILLA_NULL;
}

STRINGZILLA_API_COMPTIME void sz_memory_free_driver_(void *pointer, sz_size_t bytes, void *handle) {
    sz_unused_(bytes);
    if (!pointer) return;
    sz_cuda_device_bind_((sz_cuda_device_t const *)handle);
    cuMemFree((CUdeviceptr)pointer);
}

STRINGZILLA_API_COMPTIME void sz_memory_free_pinned_(void *pointer, sz_size_t bytes, void *handle) {
    sz_unused_(bytes);
    if (!pointer) return;
    sz_cuda_device_bind_((sz_cuda_device_t const *)handle);
    cuMemFreeHost(pointer);
}

/**
 *  @brief Initializes an allocator handing back memory both the host and the device address.
 *
 *  What a family's scratch needs when the host prepares it and a kernel reads it - a prepared
 *  query's B-tree, a Myers mask table - and what the convenience verbs stage a host-resident
 *  caller's arguments into.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 *  @sa sz_memory_allocator_init_default
 */
STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_unified_implementation_(sz_memory_allocator_t *allocator,
                                                                               sz_cuda_device_t *device) {
    allocator->allocate = &sz_memory_allocate_unified_;
    allocator->free = &sz_memory_free_driver_;
    allocator->handle = device;
}

/**
 *  @brief Initializes an allocator handing back memory only the device addresses.
 *
 *  What a round's scratch needs when no host code ever reads it, and what a unified block would
 *  otherwise pay page migration for on every access from the wrong side.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 */
STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_device_implementation_(sz_memory_allocator_t *allocator,
                                                                              sz_cuda_device_t *device) {
    allocator->allocate = &sz_memory_allocate_device_;
    allocator->free = &sz_memory_free_driver_;
    allocator->handle = device;
}

/**
 *  @brief Initializes an allocator handing back page-locked host memory the driver copies from at
 *      the bus rate.
 *
 *  A kernel cannot address what this returns - @ref sz_memory_reaches_device answers @c sz_false_k
 *  for it - so it is the staging side of a transfer rather than anything a launch reads.
 *
 *  @param[out] allocator The allocator to initialize.
 *  @param[in] device Whose context every allocation binds, or @c STRINGZILLA_NULL to use the
 *      thread's own current one.
 */
STRINGZILLA_API_COMPTIME void sz_memory_allocator_init_pinned_implementation_(sz_memory_allocator_t *allocator,
                                                                              sz_cuda_device_t *device) {
    allocator->allocate = &sz_memory_allocate_pinned_;
    allocator->free = &sz_memory_free_pinned_;
    allocator->handle = device;
}

#pragma endregion Device Memory

#ifdef __CUDACC__
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
STRINGZILLA_API_COMPTIME sz_status_t sz_sequence_from_string_views_cuda_implementation_(sz_string_view_t const *views,
                                                                                        sz_size_t count,
                                                                                        sz_sequence_t *sequence) {
    sz_sequence_member_start_t get_start = STRINGZILLA_NULL;
    sz_sequence_member_length_t get_length = STRINGZILLA_NULL;
    if (cudaMemcpyFromSymbol(&get_start, (void const *)&sz_sequence_cuda_view_start_symbol_, sizeof(get_start), 0,
                             cudaMemcpyDeviceToHost) != cudaSuccess)
        return sz_device_code_mismatch_k;
    if (cudaMemcpyFromSymbol(&get_length, (void const *)&sz_sequence_cuda_view_length_symbol_,
                             sizeof(get_length), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
        return sz_device_code_mismatch_k;
    sequence->get_start = get_start;
    sequence->get_length = get_length;
    sequence->handle = views;
    sequence->count = count;
    return sz_success_k;
}

#pragma endregion Device Sequences
#endif // __CUDACC__

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_status_t sz_cuda_device_init(int ordinal, sz_cuda_device_t *device) {
    return sz_cuda_device_init_implementation_(ordinal, device);
}
STRINGZILLA_API_RUNTIME void sz_cuda_device_free(sz_cuda_device_t *device) {
    sz_cuda_device_free_implementation_(device);
}
STRINGZILLA_API_RUNTIME sz_bool_t sz_memory_reaches_device(void const *pointer) {
    return sz_memory_reaches_device_implementation_(pointer);
}
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_unified(sz_memory_allocator_t *allocator,
                                                              sz_cuda_device_t *device) {
    sz_memory_allocator_init_unified_implementation_(allocator, device);
}
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_device(sz_memory_allocator_t *allocator,
                                                             sz_cuda_device_t *device) {
    sz_memory_allocator_init_device_implementation_(allocator, device);
}
STRINGZILLA_API_RUNTIME void sz_memory_allocator_init_pinned(sz_memory_allocator_t *allocator,
                                                             sz_cuda_device_t *device) {
    sz_memory_allocator_init_pinned_implementation_(allocator, device);
}
#ifdef __CUDACC__
STRINGZILLA_API_RUNTIME sz_status_t sz_sequence_from_string_views_cuda(sz_string_view_t const *views, sz_size_t count,
                                                                       sz_sequence_t *sequence) {
    return sz_sequence_from_string_views_cuda_implementation_(views, count, sequence);
}
#endif

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_CUDA
#endif // STRINGZILLA_TYPES_CUH_
