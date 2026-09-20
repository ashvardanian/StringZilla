/**
 *  @brief Core types on a CUDA device: the sequence a kernel can call, what memory it can reach, and the
 *      allocator that hands back memory both sides address.
 *  @file include/stringzilla/types.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzilla/types.h
 *
 *  Every family with a CUDA backend needs the same three things before its own kernel is reached, and none of
 *  them is a property of the family: a @ref sz_sequence_t whose accessors run on the device, a way to tell
 *  device-reachable memory from host memory, and somewhere to stage what a caller left on the host. They live
 *  here for the same reason @ref sz_sequence_from_string_views lives in @c types.h rather than in a family.
 *
 *  Written in C, as every @c .cuh in this library is: the only constructs here a C compiler would not take are
 *  the @c extern @c "C" that lets a C dispatch unit link against it, and the kernels' launches, which go
 *  through @c cudaLaunchKernel rather than the @c <<< @c >>> the language reserves for C++.
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.h"

#if SZ_USE_CUDA
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/*  A device function's address is a link-time value, so the host cannot take it with `&` - it has to read it out
 *  of a device variable that already holds it. One pair per translation unit, which is what `static` buys. */
static __device__ sz_sequence_member_start_t sz_sequence_cuda_view_start_symbol_ = &sz_sequence_cuda_view_start_;
static __device__ sz_sequence_member_length_t sz_sequence_cuda_view_length_symbol_ = &sz_sequence_cuda_view_length_;

/**
 *  @brief Binds a sequence over device-resident @p views whose accessors a kernel can call.
 *  @param[in] views The @b [count] views, device-reachable, each pointing at device-reachable text.
 *  @param[out] sequence Left untouched unless the call succeeds.
 *  @retval sz_device_code_mismatch_k when the accessors' addresses cannot be read off the device.
 *  @sa sz_sequence_from_string_views
 */
SZ_API_COMPTIME sz_status_t sz_sequence_from_string_views_cuda(sz_string_view_t const *views, sz_size_t count,
                                                               sz_sequence_t *sequence) {
    sz_sequence_member_start_t get_start = SZ_NULL;
    sz_sequence_member_length_t get_length = SZ_NULL;
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

#pragma region Device Memory

/**
 *  @brief Whether the device can dereference @p pointer - managed or device memory, never host, pinned or not.
 *
 *  Page-locked host memory is the case a caller is most likely to expect to work: the driver reports it as host,
 *  a kernel cannot address it, and this answers @c sz_false_k for it.
 */
SZ_API_COMPTIME sz_bool_t sz_memory_reaches_device(void const *pointer) {
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) return sz_false_k;
    if (attributes.type == cudaMemoryTypeDevice) return sz_true_k;
    return attributes.type == cudaMemoryTypeManaged ? sz_true_k : sz_false_k;
}

SZ_API_COMPTIME void *sz_memory_allocate_unified_(sz_size_t bytes, void *handle) {
    void *pointer = SZ_NULL;
    sz_unused_(handle);
    return cudaMallocManaged(&pointer, bytes) == cudaSuccess ? pointer : SZ_NULL;
}

SZ_API_COMPTIME void sz_memory_free_unified_(void *pointer, sz_size_t bytes, void *handle) {
    sz_unused_(bytes), sz_unused_(handle);
    cudaFree(pointer);
}

/**
 *  @brief Initializes an allocator handing back memory both the host and the device address.
 *
 *  What a family's scratch needs when the host prepares it and a kernel reads it - a prepared query's B-tree, a
 *  Myers mask table - and what the convenience verbs stage a host-resident caller's arguments into.
 *  @sa sz_memory_allocator_init_default
 */
SZ_API_COMPTIME void sz_memory_allocator_init_unified(sz_memory_allocator_t *allocator) {
    allocator->allocate = &sz_memory_allocate_unified_;
    allocator->free = &sz_memory_free_unified_;
    allocator->handle = SZ_NULL;
}

#pragma endregion Device Memory

#ifdef __cplusplus
}
#endif
#endif // SZ_USE_CUDA
#endif // STRINGZILLA_TYPES_CUH_
