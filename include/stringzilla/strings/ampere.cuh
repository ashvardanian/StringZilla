/**
 *  @file include/stringzilla/strings/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The steps Ampere added, which other families' Ampere, Hopper and Blackwell tiers include:
 *      warp reductions in one @c redux.sync and copies into shared memory with @c cp.async .
 *
 *  Every lane of the warp calls a reduction at once, and any lane a copy, waiting for its own
 *  committed groups alone. The instructions are inline PTX, which only a kernel built for compute
 *  capability 8.0 or newer emits.
 *
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_STRINGS_AMPERE_CUH_
#define STRINGZILLA_STRINGS_AMPERE_CUH_

#include "stringzilla/cuda.cuh"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#ifdef __cplusplus
extern "C" {
#endif

/** The sum of @p value over the warp, in every lane, with one @c redux.sync . */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_sum_ampere_(sz_u32_t value) {
    sz_u32_t sum;
    asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(sum) : "r"(value));
    return sum;
}

/** The sum of @p value over the warp, in every lane, for a warp whose sum is below 2^59: two exact
 *  @c redux.sync of its halves split at bit 27, as 32 lanes of 27 bits sum below 2^32. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_sum_ampere_(sz_u64_t value) {
    sz_u32_t const low = sz_u32_lanes_sum_ampere_((sz_u32_t)(value & ((1u << 27) - 1u)));
    sz_u32_t const high = sz_u32_lanes_sum_ampere_((sz_u32_t)(value >> 27));
    return ((sz_u64_t)high << 27) + low;
}

/** The least of @p value over the warp, in every lane, with one @c redux.sync . */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_min_ampere_(sz_u32_t value) {
    sz_u32_t least;
    asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(least) : "r"(value));
    return least;
}

/** The greatest of @p value over the warp, in every lane, with one @c redux.sync . */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_max_ampere_(sz_u32_t value) {
    sz_u32_t greatest;
    asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(greatest) : "r"(value));
    return greatest;
}

/** @ref sz_block_sum_simt_ over the warps of a block of @p threads, each warp's total from
 *  @ref sz_u64_lanes_sum_ampere_ , for a block whose sum is below 2^59. */
STRINGZILLA_DEVICE sz_size_t sz_block_sum_ampere_(sz_size_t value, sz_size_t *shared, unsigned threads) {
    sz_u64_t const warp_total = sz_u64_lanes_sum_ampere_((sz_u64_t)value);
    sz_size_t sum = 0;
    if ((threadIdx.x & 31u) == 0) shared[threadIdx.x / 32] = (sz_size_t)warp_total;
    __syncthreads();
    for (unsigned warp = 0; warp != threads / 32; ++warp) sum += shared[warp];
    return sum;
}

/** @ref sz_block_sum_ampere_ for a block whose sum is below 2^32, each warp's total from one
 *  @ref sz_u32_lanes_sum_ampere_ and @p shared taken as 32-bit entries. */
STRINGZILLA_DEVICE sz_size_t sz_block_sum_bounded_ampere_(sz_size_t value, sz_size_t *shared, unsigned threads) {
    sz_u32_t *const totals = (sz_u32_t *)shared;
    sz_u32_t const warp_total = sz_u32_lanes_sum_ampere_((sz_u32_t)value);
    sz_u32_t sum = 0;
    if ((threadIdx.x & 31u) == 0) totals[threadIdx.x / 32] = warp_total;
    __syncthreads();
    for (unsigned warp = 0; warp != threads / 32; ++warp) sum += totals[warp];
    return (sz_size_t)sum;
}

/** The shared-memory address of @p pointer, as @c cp.async names its target. */
STRINGZILLA_DEVICE sz_u32_t sz_shared_address_ampere_(void const *pointer) {
    return (sz_u32_t)__cvta_generic_to_shared(pointer);
}

/** Copies 4 bytes from @p global to @p shared , both 4-byte aligned, zero-filling past
 *  @p valid_bytes , with @c cp.async.ca . */
STRINGZILLA_DEVICE void sz_copy_b32_async_ampere_(sz_u32_t shared, void const *global, sz_u32_t valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(shared), "l"(global), "r"(valid_bytes)
                 : "memory");
}

/** Copies 16 bytes from @p global to @p shared , both 16-byte aligned, zero-filling past
 *  @p valid_bytes , with @c cp.async.cg , which bypasses L1. */
STRINGZILLA_DEVICE void sz_copy_b128_async_ampere_(sz_u32_t shared, void const *global, sz_u32_t valid_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared), "l"(global), "r"(valid_bytes)
                 : "memory");
}

/** Closes this thread's copies issued since the last commit into one group. */
STRINGZILLA_DEVICE void sz_commit_async_ampere_(void) { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

/** Waits until at most @p pending of this thread's committed groups are in flight, capped at 2, as
 *  PTX takes an immediate; other threads' copies need a barrier after it too. */
STRINGZILLA_DEVICE void sz_wait_async_ampere_(unsigned pending) {
    switch (pending) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_STRINGS_AMPERE_CUH_
