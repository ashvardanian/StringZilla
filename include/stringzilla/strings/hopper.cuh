/**
 *  @file include/stringzilla/strings/hopper.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The steps Hopper added, which other families' Hopper and Blackwell tiers include:
 *      clusters of blocks addressing each other's shared memory, and barriers copies complete.
 *
 *  Kernels taking the cluster steps launch in clusters of @c sz_cluster_blocks_cuda_k blocks along
 *  x. The instructions are inline PTX, which only kernels built for compute capability 9.0 emit.
 *
 *  @sa include/stringzilla/strings/ampere.cuh
 */
#ifndef STRINGZILLA_STRINGS_HOPPER_CUH_
#define STRINGZILLA_STRINGS_HOPPER_CUH_

#include "stringzilla/cuda.cuh"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_HOPPER_

#ifdef __cplusplus
extern "C" {
#endif

/** This block's rank in its cluster, from Hopper's @c %cluster_ctarank . */
STRINGZILLA_DEVICE sz_u32_t sz_cluster_rank_hopper_(void) {
    sz_u32_t rank;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

/** Where @p pointer, into this block's shared memory, lands in the cluster's block at @p rank, with
 *  Hopper's @c mapa . */
STRINGZILLA_DEVICE void *sz_cluster_map_hopper_(void *pointer, sz_u32_t rank) {
    void *mapped;
    asm("mapa.u64 %0, %1, %2;" : "=l"(mapped) : "l"(pointer), "r"(rank));
    return mapped;
}

/** Arrives at the cluster's barrier, releasing this thread's writes before it, with Hopper's
 *  @c barrier.cluster.arrive ; every thread of the cluster calls it, then the wait. */
STRINGZILLA_DEVICE void sz_cluster_arrive_hopper_(void) {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

/** Waits until every thread of the cluster arrived, acquiring their writes, with Hopper's
 *  @c barrier.cluster.wait . */
STRINGZILLA_DEVICE void sz_cluster_wait_hopper_(void) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

/** A barrier across every thread of the cluster, ordering each block's shared writes before it. */
STRINGZILLA_DEVICE void sz_cluster_sync_hopper_(void) {
    sz_cluster_arrive_hopper_();
    sz_cluster_wait_hopper_();
}

/** Readies the barrier at shared address @p barrier for @p count arrivals a phase, with
 *  @c mbarrier.init . */
STRINGZILLA_DEVICE void sz_mbarrier_init_hopper_(sz_u32_t barrier, sz_u32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(barrier), "r"(count) : "memory");
}

/** Makes this thread's barrier inits visible to the cluster and to asynchronous copies. */
STRINGZILLA_DEVICE void sz_mbarrier_init_fence_hopper_(void) {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

/** Arrives at the barrier at shared address @p barrier, expecting @p bytes more from asynchronous
 *  copies before its phase completes, with @c mbarrier.arrive.expect_tx . */
STRINGZILLA_DEVICE void sz_mbarrier_expect_bytes_hopper_(sz_u32_t barrier, sz_u32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes) : "memory");
}

/** Spins until the phase of parity @p parity of the barrier at shared address @p barrier completes,
 *  with @c mbarrier.try_wait.parity . */
STRINGZILLA_DEVICE void sz_mbarrier_wait_hopper_(sz_u32_t barrier, sz_u32_t parity) {
    sz_u32_t done = 0;
    while (!done)
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n" //
                     " selp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done)
                     : "r"(barrier), "r"(parity)
                     : "memory");
}

/** This block's tile among @p tiles, as @ref sz_chain_tile_simt_ hands them out, with one ticket
 *  per cluster: its first block takes it, and each block works the tile of its rank under it.
 *  Every thread calls it, and it ends in a cluster barrier, unless the cluster's first block lies
 *  past the tiles, in which case none of its blocks takes a ticket. */
STRINGZILLA_DEVICE sz_size_t sz_chain_tile_hopper_(sz_size_t *chain, sz_size_t tiles) {
    __shared__ sz_size_t ticket;
    sz_u32_t const rank = sz_cluster_rank_hopper_();
    sz_u32_t other;
    if ((sz_size_t)blockIdx.x - rank >= tiles) return STRINGZILLA_SIZE_MAX;
    if (rank == 0 && threadIdx.x == 0) {
        sz_size_t const taken = (sz_size_t)(atomicAdd((unsigned long long *)chain, 1ull << sz_chain_tickets_shift_k) >>
                                            sz_chain_tickets_shift_k);
        for (other = 0; other != sz_cluster_blocks_cuda_k; ++other)
            *(sz_size_t *)sz_cluster_map_hopper_(&ticket, other) = taken;
    }
    sz_cluster_sync_hopper_();
    return ticket * sz_cluster_blocks_cuda_k + rank;
}

/** Where tile @p tile of @ref sz_chain_tile_hopper_ writes, with one look-back per cluster: every
 *  block hands its @p total to the first, which looks back over the cluster's aggregate under the
 *  cluster's ticket, then hands each block the sum before it. Every thread calls it, and it ends
 *  in a cluster barrier. */
STRINGZILLA_DEVICE sz_size_t sz_chain_look_back_hopper_(sz_u64_t *states, sz_size_t tile, sz_size_t total) {
    __shared__ sz_size_t totals[sz_cluster_blocks_cuda_k];
    __shared__ sz_size_t offset;
    sz_u32_t const rank = sz_cluster_rank_hopper_();
    sz_u32_t other;
    if (threadIdx.x == 0) *(sz_size_t *)sz_cluster_map_hopper_(totals + rank, 0) = total;
    sz_cluster_sync_hopper_();
    if (rank == 0) {
        sz_size_t aggregate = 0, before = 0, prefix;
        for (other = 0; other != sz_cluster_blocks_cuda_k; ++other) aggregate += totals[other];
        prefix = sz_chain_look_back_cuda_(states, tile / sz_cluster_blocks_cuda_k, aggregate);
        if (threadIdx.x < sz_cluster_blocks_cuda_k) {
            for (other = 0; other != threadIdx.x; ++other) before += totals[other];
            *(sz_size_t *)sz_cluster_map_hopper_(&offset, threadIdx.x) = prefix + before;
        }
    }
    sz_cluster_sync_hopper_();
    return offset;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_HOPPER_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_STRINGS_HOPPER_CUH_
