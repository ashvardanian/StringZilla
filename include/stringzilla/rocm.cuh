/**
 *  @file include/stringzilla/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief The HIP runtime as the library's ROCm host code drives it: the device scope an engine
 *      call runs in, the memory both sides address, the launches, the streams and the producers,
 *      and the cross-lane steps its kernels hand the shared walks.
 *
 *  Every call runs on the device of the stream it is given, which @c sz_device_enter_rocm_ makes
 *  current for its duration, and a null stream runs on the caller's current device.
 *
 *  Only hipcc sees past the guard, and nothing here is shared with CUDA, whose twins live in
 *  <tt>cuda.cuh</tt> under their own names. AMD has no clusters and no block stealing, so neither appears
 *  here, and a family's ROCm host code launches every grid flat.
 *
 *  @sa include/stringzilla/cuda.cuh
 *  @sa include/stringzilla/types.cuh
 */
#ifndef STRINGZILLA_ROCM_CUH_
#define STRINGZILLA_ROCM_CUH_

#include "stringzilla/types.cuh"
#include "stringzilla/capabilities.h"   // `sz_cap_rocm_k`, `sz_cap_cdna3_k`, `sz_cap_cdna4_k`, `sz_cap_cdna5_k`
#include "stringzilla/compare/serial.h" // `sz_equal_serial_`

#if STRINGZILLA_ARCH_ROCM_
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
STRINGZILLA_INLINE sz_status_t sz_device_enter_rocm_(sz_stream_t stream, int *caller) {
    int device = 0;
    if (hipGetDevice(caller) != hipSuccess) return sz_missing_gpu_k;
    if (!stream) return sz_success_k;
    if (hipStreamGetDevice((hipStream_t)stream, &device) != hipSuccess) return sz_device_memory_mismatch_k;
    return hipSetDevice(device) == hipSuccess ? sz_success_k : sz_missing_gpu_k;
}

/** Makes @p caller current again, closing the scope @c sz_device_enter_rocm_ opened. */
STRINGZILLA_INLINE void sz_device_leave_rocm_(int caller) { sz_unused_(hipSetDevice(caller)); }

/** Accepts memory the current device addresses: managed memory, its own allocation, or registered
 *  host memory it reads over the bus, whose device address the driver reports. */
STRINGZILLA_INLINE sz_bool_t sz_memory_accessible_rocm_(void const *pointer) {
    int device = 0;
    hipPointerAttribute_t attributes;
    if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess || hipGetDevice(&device) != hipSuccess)
        return sz_false_k;
    if (attributes.type == hipMemoryTypeManaged) return sz_true_k;
    if (attributes.type == hipMemoryTypeHost) return (sz_bool_t)(attributes.devicePointer != STRINGZILLA_NULL);
    return attributes.type == hipMemoryTypeDevice && attributes.device == device ? sz_true_k : sz_false_k;
}

/** Whether @p strings is a tape the device reaches, its payload null only when all are empty. */
STRINGZILLA_INLINE sz_bool_t sz_strings_accessible_rocm_(sz_strings_view_t const *strings) {
    sz_bool_t const wide = (sz_bool_t)(strings->layout == sz_strings_u64_tape_k);
    sz_cptr_t const data = wide ? strings->data.u64_tape.data : strings->data.u32_tape.data;
    void const *const offsets = wide ? (void const *)strings->data.u64_tape.offsets
                                     : (void const *)strings->data.u32_tape.offsets;
    return (sz_bool_t)(strings->layout != sz_strings_accessor_k && (!data || sz_memory_accessible_rocm_(data)) &&
                       sz_memory_accessible_rocm_(offsets));
}

/*  Each allocator below is stateless, ignoring @c handle, and works on the device of its stream. */

STRINGZILLA_INLINE void *sz_allocate_unified_rocm_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipMallocManaged(&pointer, bytes, hipMemAttachGlobal) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_allocate_device_rocm_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipMalloc(&pointer, bytes) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

STRINGZILLA_INLINE void *sz_allocate_pinned_rocm_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    void *pointer = STRINGZILLA_NULL;
    int caller = 0;
    sz_unused_(handle);
    if (sz_device_enter_rocm_(stream, &caller) != sz_success_k) return STRINGZILLA_NULL;
    if (hipHostMalloc(&pointer, bytes, hipHostMallocDefault) != hipSuccess) pointer = STRINGZILLA_NULL;
    sz_device_leave_rocm_(caller);
    return pointer;
}

/** Frees unified and device blocks alike, once the device is done with everything queued before. */
STRINGZILLA_INLINE void sz_free_device_rocm_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_rocm_(stream, &caller) != sz_success_k) return;
    sz_unused_(hipFree(pointer));
    sz_device_leave_rocm_(caller);
}

STRINGZILLA_INLINE void sz_free_pinned_rocm_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    int caller = 0;
    sz_unused_(bytes), sz_unused_(handle);
    if (!pointer || sz_device_enter_rocm_(stream, &caller) != sz_success_k) return;
    sz_unused_(hipHostFree(pointer));
    sz_device_leave_rocm_(caller);
}

/** Moves a block into a fresh one from @p allocate with a copy on @p stream, then frees the old one
 *  through @p release, which waits for the device. */
STRINGZILLA_INLINE void *sz_reallocate_rocm_(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes, void *handle,
                                             sz_stream_t stream, sz_allocate_t allocate, sz_free_t release) {
    if (!new_bytes) {
        release(pointer, old_bytes, handle, stream);
        return STRINGZILLA_NULL;
    }
    void *const moved = allocate(new_bytes, handle, stream);
    if (!moved || !pointer) return moved;
    if (hipMemcpyAsync(moved, pointer, sz_min_of_two(old_bytes, new_bytes), hipMemcpyDefault, (hipStream_t)stream) !=
        hipSuccess) {
        release(moved, new_bytes, handle, stream);
        return STRINGZILLA_NULL;
    }
    release(pointer, old_bytes, handle, stream);
    return moved;
}

STRINGZILLA_INLINE void *sz_reallocate_unified_rocm_(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes,
                                                     void *handle, sz_stream_t stream) {
    return sz_reallocate_rocm_(pointer, old_bytes, new_bytes, handle, stream, &sz_allocate_unified_rocm_,
                               &sz_free_device_rocm_);
}

STRINGZILLA_INLINE void *sz_reallocate_device_rocm_(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes,
                                                    void *handle, sz_stream_t stream) {
    return sz_reallocate_rocm_(pointer, old_bytes, new_bytes, handle, stream, &sz_allocate_device_rocm_,
                               &sz_free_device_rocm_);
}

STRINGZILLA_INLINE void *sz_reallocate_pinned_rocm_(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes,
                                                    void *handle, sz_stream_t stream) {
    return sz_reallocate_rocm_(pointer, old_bytes, new_bytes, handle, stream, &sz_allocate_pinned_rocm_,
                               &sz_free_pinned_rocm_);
}

/** Initializes an allocator handing back memory both the host and the device address, each call
 *  on its stream's device. */
STRINGZILLA_INLINE void sz_allocator_init_unified_rocm_(sz_allocator_t *allocator) {
    allocator->allocate = &sz_allocate_unified_rocm_;
    allocator->reallocate = &sz_reallocate_unified_rocm_;
    allocator->free = &sz_free_device_rocm_;
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

/** Lanes per wavefront on the current device: 64 on GFX9, 32 on GFX10 and newer, 64 unanswered. */
STRINGZILLA_INLINE sz_size_t sz_device_wavefront_lanes_rocm_(void) {
    sz_size_t const lanes = sz_device_attribute_rocm_(hipDeviceAttributeWarpSize);
    return lanes ? lanes : 64;
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
                                               sz_size_t shared_bytes, sz_stream_t stream) {
    return hipLaunchKernel(kernel, grid, block, arguments, shared_bytes, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Sets @p length bytes at device-reachable @p target to @p value, in order on @p stream. */
STRINGZILLA_INLINE sz_status_t sz_fill_rocm_(void *target, sz_size_t length, sz_u8_t value, sz_stream_t stream) {
    return hipMemsetAsync(target, value, length, (hipStream_t)stream) == hipSuccess ? sz_success_k
                                                                                    : sz_device_code_mismatch_k;
}

/** Copies @p length bytes of @p source into @p target, each in host or device memory, in order on
 *  @p stream. */
STRINGZILLA_INLINE sz_status_t sz_copy_rocm_(void *target, void const *source, sz_size_t length, sz_stream_t stream) {
    return hipMemcpyAsync(target, source, length, hipMemcpyDefault, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Copies @p bytes of host @p source into the @c __device__ variable at @p symbol, in order on
 *  @p stream. */
STRINGZILLA_INLINE sz_status_t sz_copy_to_symbol_rocm_(void const *symbol, void const *source, sz_size_t bytes,
                                                       sz_stream_t stream) {
    return hipMemcpyToSymbolAsync(symbol, source, bytes, 0, hipMemcpyHostToDevice, (hipStream_t)stream) == hipSuccess
               ? sz_success_k
               : sz_device_code_mismatch_k;
}

/** Waits for @p stream on its own device, leaving the caller's current device as it found it. */
STRINGZILLA_INLINE sz_status_t sz_stream_synchronize_rocm_(sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = hipStreamSynchronize((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
    sz_device_leave_rocm_(caller);
    return status;
}

/** Destroys @p stream once the work queued on it completes; a null stream is the default one. */
STRINGZILLA_INLINE sz_status_t sz_stream_free_rocm_(sz_stream_t stream) {
    if (!stream) return sz_success_k;
    return hipStreamDestroy((hipStream_t)stream) == hipSuccess ? sz_success_k : sz_device_code_mismatch_k;
}

/** Whether @p stream still has work queued or running; a failed query answers that it has none. */
STRINGZILLA_INLINE sz_bool_t sz_stream_query_rocm_(sz_stream_t stream) {
    return hipStreamQuery((hipStream_t)stream) == hipErrorNotReady ? sz_true_k : sz_false_k;
}

/** Migrates managed @p pointer to the current device on @p stream, so a kernel reading what the
 *  host just filled takes one bulk move rather than a fault per page. */
STRINGZILLA_INLINE void sz_prefetch_rocm_(void const *pointer, sz_size_t bytes, sz_stream_t stream) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return;
    sz_unused_(hipMemPrefetchAsync(pointer, bytes, device, (hipStream_t)stream));
}

#pragma endregion Launches

#pragma region Devices

/** How many ROCm devices the runtime sees, or zero. */
STRINGZILLA_INLINE sz_size_t sz_count_devices_rocm_(void) {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess ? (sz_size_t)count : 0;
}

/** The capabilities ROCm device @p ordinal runs, by the architecture its runtime names: @c cdna3 on
 *  gfx942, @c cdna4 on gfx950, and @c cdna5 on both gfx1250 and gfx1251. */
STRINGZILLA_INLINE sz_status_t sz_capabilities_detected_rocm_(sz_size_t ordinal, sz_capability_t *capabilities) {
    hipDeviceProp_t properties;
    *capabilities = 0;
    if (ordinal >= sz_count_devices_rocm_()) return sz_missing_gpu_k;
    if (hipGetDeviceProperties(&properties, (int)ordinal) != hipSuccess) return sz_device_code_mismatch_k;
    if (properties.multiProcessorCount <= 0) return sz_success_k;
    sz_cptr_t const name = properties.gcnArchName;
    *capabilities = sz_cap_rocm_k;
    if (sz_equal_serial_(name, "gfx942", 6)) *capabilities |= sz_cap_cdna3_k;
    if (sz_equal_serial_(name, "gfx950", 6)) *capabilities |= sz_cap_cdna4_k;
    if (sz_equal_serial_(name, "gfx1250", 7) || sz_equal_serial_(name, "gfx1251", 7)) *capabilities |= sz_cap_cdna5_k;
    return sz_success_k;
}

/** Creates a stream on ROCm device @p ordinal, leaving the caller's current device as it was. */
STRINGZILLA_INLINE sz_status_t sz_stream_init_rocm_(sz_size_t ordinal, sz_stream_t *stream) {
    int caller = 0;
    sz_status_t status = sz_success_k;
    hipStream_t created = STRINGZILLA_NULL;
    *stream = STRINGZILLA_NULL;
    if (ordinal >= sz_count_devices_rocm_()) return sz_missing_gpu_k;
    if (hipGetDevice(&caller) != hipSuccess || hipSetDevice((int)ordinal) != hipSuccess) return sz_missing_gpu_k;
    if (hipStreamCreate(&created) != hipSuccess) status = sz_bad_alloc_k;
    sz_device_leave_rocm_(caller);
    *stream = (void *)created;
    return status;
}

#pragma endregion Devices

/*  Every step stays inside its own 32 lanes, so a 64-wide wavefront runs two groups of the shared
 *  walks side by side. */
#pragma region Device Primitives

/** Lane <tt>lane - delta</tt>'s @p value among this thread's 32 lanes, a lane's own below @p delta. */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_rocm_(sz_u32_t value, unsigned delta) { return __shfl_up(value, delta, 32); }

/** Lane <tt>lane + delta</tt>'s @p value among this thread's 32 lanes, a lane's own past the last. */
STRINGZILLA_DEVICE int sz_shuffle_down_rocm_(int value, unsigned delta) { return __shfl_down(value, delta, 32); }

/** Lane @p lane 's @p value among this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_index_rocm_(sz_u32_t value, unsigned lane) {
    return __shfl(value, (int)lane, 32);
}

/** Lane <tt>lane - delta</tt>'s @p value within this lane's aligned group of @p width lanes, a lane's own
 *  below @p delta. */
STRINGZILLA_DEVICE sz_u32_t sz_group_shuffle_up_rocm_(sz_u32_t value, unsigned delta, unsigned width) {
    return __shfl_up(value, delta, (int)width);
}

/** Lane @p lane 's @p value within this lane's aligned group of @p width lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_group_shuffle_index_rocm_(sz_u32_t value, unsigned lane, unsigned width) {
    return __shfl(value, (int)lane, (int)width);
}

/** Whether @p predicate holds on any lane of this lane's aligned group of @p width lanes, at most
 *  32, from the group's bits of a ballot. */
STRINGZILLA_DEVICE int sz_group_any_rocm_(int predicate, unsigned width) {
    sz_u64_t const members = ((1ull << width) - 1ull) << (__lane_id() & ~(width - 1u));
    return (__ballot(predicate) & members) != 0;
}

/** Lane zero's @p value across this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_lanes_broadcast_rocm_(sz_u32_t value) { return __shfl(value, 0, 32); }

/** Whether @p predicate holds on any of this thread's 32 lanes, from its half of the ballot. */
STRINGZILLA_DEVICE int sz_lanes_any_rocm_(int predicate) {
    return ((__ballot(predicate) >> (__lane_id() & 32u)) & 0xFFFFFFFFull) != 0;
}

/** A bit per lane of this thread's 32, set where @p predicate holds, from its half of a ballot. */
STRINGZILLA_DEVICE sz_u32_t sz_lanes_ballot_rocm_(int predicate) {
    return (sz_u32_t)(__ballot(predicate) >> (__lane_id() & 32u));
}

/** The lowest of this thread's 32 lanes where @p predicate holds, or 32 where it holds on none,
 *  from its half of the ballot. */
STRINGZILLA_DEVICE unsigned sz_lanes_first_rocm_(int predicate) {
    sz_u32_t const half = sz_lanes_ballot_rocm_(predicate);
    return half ? (unsigned)sz_u32_ctz_simt_(half) : 32u;
}

/** How many of this thread's 32 lanes below it @p predicate holds on, from its ballot half. */
STRINGZILLA_DEVICE unsigned sz_lanes_below_rocm_(int predicate) {
    return (unsigned)sz_u32_popcount_simt_(sz_lanes_ballot_rocm_(predicate) & ((1u << (__lane_id() & 31u)) - 1u));
}

/** The sum of @p value over this lane and every lane below it among this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_prefix_sum_rocm_(sz_u32_t value) {
    unsigned const lane = __lane_id() & 31u;
    for (unsigned delta = 1; delta != 32; delta <<= 1) {
        sz_u32_t const below = __shfl_up(value, delta, 32);
        if (lane >= delta) value += below;
    }
    return value;
}

/** @ref sz_u32_lanes_prefix_sum_rocm_ over 64-bit values, each step two @c ds_bpermute . */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_prefix_sum_rocm_(sz_u64_t value) {
    unsigned const lane = __lane_id() & 31u;
    for (unsigned delta = 1; delta != 32; delta <<= 1) {
        sz_u64_t const below = __shfl_up(value, delta, 32);
        if (lane >= delta) value += below;
    }
    return value;
}

/** The sum of @p value over this thread's 32 lanes, in every one of them, two cross-lane moves per
 *  step. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_sum_rocm_(sz_u64_t value) {
    for (unsigned delta = 16; delta != 0; delta >>= 1) value += __shfl_xor(value, (int)delta, 32);
    return value;
}

/** The least of @p value over this lane and every lane below it among this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_i32_t sz_i32_lanes_prefix_min_rocm_(sz_i32_t value) {
    unsigned const lane = __lane_id() & 31u;
    for (unsigned delta = 1; delta != 32; delta <<= 1) {
        sz_i32_t const below = __shfl_up(value, delta, 32);
        if (lane >= delta && below < value) value = below;
    }
    return value;
}

/** Adds one to @p counter for this lane, returning the count before it, one atomic per lane, as no
 *  AMD wavefront matches lanes by value. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_increment_rocm_(sz_u32_t *counter) { return atomicAdd(counter, 1u); }

/** Orders this thread's lanes and their memory at this point: a wavefront runs in lockstep, so only
 *  the wavefront-scoped fences and the compiler's scheduling barrier between them are needed. */
STRINGZILLA_DEVICE void sz_group_sync_rocm_(void) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/** @ref sz_block_scan_simt_ over the 32-lane halves of a block of @p threads, scanning through
 *  @c ds_bpermute . */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_rocm_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                 unsigned threads) {
    return sz_block_scan_simt_(value, shared, total, 32, threads, sz_u64_lanes_prefix_sum_rocm_);
}

/** @ref sz_block_scan_bounded_simt_ over the 32-lane halves of a block of @p threads, @p shared
 *  taken as 32-bit entries. */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_bounded_rocm_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                         unsigned threads) {
    return sz_block_scan_bounded_simt_(value, (sz_u32_t *)shared, total, 32, threads, sz_u32_lanes_prefix_sum_rocm_);
}

/** @ref sz_block_sum_simt_ over the 32-lane halves of a block of @p threads. */
STRINGZILLA_DEVICE sz_size_t sz_block_sum_rocm_(sz_size_t value, sz_size_t *shared, unsigned threads) {
    return sz_block_sum_simt_(value, shared, 32, threads, sz_u64_lanes_prefix_sum_rocm_);
}

/** @ref sz_chain_look_back_simt_ over the block's first 32 lanes. */
STRINGZILLA_DEVICE sz_size_t sz_chain_look_back_rocm_(sz_u64_t *states, sz_size_t ticket, sz_size_t total) {
    return sz_chain_look_back_simt_(states, ticket, total, sz_lanes_ballot_rocm_, sz_u64_lanes_prefix_sum_rocm_);
}

#pragma endregion Device Primitives

/*  Scans within each row of 16 lanes that move values between lanes with DPP rather than through
 *  the LDS crossbar @c ds_bpermute takes. Every AMD generation has them; carrying the row totals
 *  across a wavefront differs by generation, so it lives in <tt>strings/cdna3.cuh</tt> and
 *  <tt>strings/cdna5.cuh</tt>. Every move names its operation's identity as the value it leaves where
 *  its row is masked off or its source lane falls outside the row, so @c update_dpp , never
 *  @c mov_dpp . Every lane of the wavefront calls a step at once. */
#pragma region Row Scans

enum {

    /** The DPP control keeping every lane's own value, the row mask alone picking rows. */
    sz_dpp_identity_rocm_k = 0xE4,

    /** The DPP control moving every lane's value up its row of 16 lanes, by the count added. */
    sz_dpp_row_shift_up_rocm_k = 0x110,
};

/** The sum of @p value over this lane and every lane below it within its row of 16 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_rows_prefix_sum_rocm_(sz_u32_t value) {
    value += __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_shift_up_rocm_k + 1, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_shift_up_rocm_k + 2, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_shift_up_rocm_k + 4, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_shift_up_rocm_k + 8, 0xF, 0xF, false);
    return value;
}

/** @ref sz_u32_rows_prefix_sum_rocm_ over 64-bit values, each move two DPP moves. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_rows_prefix_sum_rocm_(sz_u64_t value) {
    value += __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_shift_up_rocm_k + 1, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_shift_up_rocm_k + 2, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_shift_up_rocm_k + 4, 0xF, 0xF, false);
    value += __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_shift_up_rocm_k + 8, 0xF, 0xF, false);
    return value;
}

/** The least of @p value over this lane and every lane below it within its row of 16 lanes. */
STRINGZILLA_DEVICE sz_i32_t sz_i32_rows_prefix_min_rocm_(sz_i32_t value) {
    sz_i32_t below;
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_shift_up_rocm_k + 1, 0xF, 0xF, false);
    value = below < value ? below : value;
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_shift_up_rocm_k + 2, 0xF, 0xF, false);
    value = below < value ? below : value;
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_shift_up_rocm_k + 4, 0xF, 0xF, false);
    value = below < value ? below : value;
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_shift_up_rocm_k + 8, 0xF, 0xF, false);
    return below < value ? below : value;
}

#pragma endregion Row Scans

/*  The library defines these once, in `src/capabilities/rocm.hip`; header-only builds define them
 *  here. */
#if STRINGZILLA_HEADER_ONLY

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

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_ROCM_CUH_
