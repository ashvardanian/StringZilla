/**
 *  @file include/stringzilla/types.cuh
 *  @author Ash Vardanian
 *  @date September 20, 2026
 *  @brief Core device code the CUDA and ROCm kernels share: the lanes, the block scan, the
 *      clusters, the tile queues, the chained tiles and the tape a kernel reads a sequence from.
 *
 *  Every family with a SIMT backend writes its kernels once, in its `simt.cuh`, and both nvcc and
 *  hipcc compile them. What those kernels share lives here for the same reason the host tape
 *  accessors live in `types.h` rather than in a family. The host side of every vendor is its own:
 *  `cuda.cuh` and `rocm.cuh` each drive their runtime, and nothing here calls one.
 *
 *  Written in C, as every `.cuh` in this library is. A kernel's cross-lane steps go through the
 *  @b Lanes helpers, the only place the two vendors' spellings differ, and the few PTX sites carry
 *  plain fallbacks for every other target.
 *
 *  @sa include/stringzilla/types.h
 *  @sa include/stringzilla/cuda.cuh
 *  @sa include/stringzilla/rocm.cuh
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.h"

/*  A library holds one unit per vendor it was built for, and each unit is compiled for exactly one
 *  of them, which its compiler decides: a unit HIP compiles runs ROCm, any other CUDA compiles runs
 *  CUDA. The `STRINGZILLA_ARCH_*` switches only say which vendors a unit is built for, and the host
 *  units, which know every vendor the library holds but include neither runtime, as the two clash,
 *  skip the device code altogether. */
#if STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)
#include <hip/hip_runtime.h> // `__shfl_up`, `__ballot`, `atomicAdd`
#elif STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__)
#include <cuda_runtime.h> // `__shfl_up_sync`, `__any_sync`, `atomicAdd`
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if (STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)) || (STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__) && !defined(__HIP__))

/*  Kernels are written for 32 lanes. A 64-wide AMD wavefront runs two such groups side by side, so
 *  every cross-lane step stays inside its own half. */
#pragma region Lanes

/** @p value from the lane @p delta below this one, or this lane's own below lane @p delta. */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_simt_(sz_u32_t value, unsigned delta) {
#if defined(__HIP__)
    return __shfl_up(value, delta, 32);
#else
    return __shfl_up_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** @p value from the lane @p delta above this one, or this lane's own past the last lane. */
STRINGZILLA_DEVICE int sz_shuffle_down_simt_(int value, unsigned delta) {
#if defined(__HIP__)
    return __shfl_down(value, delta, 32);
#else
    return __shfl_down_sync(0xFFFFFFFFu, value, delta);
#endif
}

/** Lane zero's @p value, on every one of this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_u32_t sz_lanes_broadcast_simt_(sz_u32_t value) {
#if defined(__HIP__)
    return __shfl(value, 0, 32);
#else
    return __shfl_sync(0xFFFFFFFFu, value, 0);
#endif
}

/** Whether @p predicate holds on any of this thread's 32 lanes. */
STRINGZILLA_DEVICE int sz_lanes_any_simt_(int predicate) {
#if defined(__HIP__)
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

/** Where a draw keeps its tile's x, which a drained queue fills with ones, past every tile. */
enum { sz_tile_queue_x_shift_k = 32 };

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
        if (drawn >= (~0ull << sz_tile_queue_x_shift_k)) return 0;
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
                atomicExch(&queue->drawn, (~0ull << sz_tile_queue_x_shift_k));
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
                             : (~0ull << sz_tile_queue_x_shift_k);
    }
    __syncthreads();
    unsigned long long const drawn = queue->drawn;
    if (drawn >= (~0ull << sz_tile_queue_x_shift_k)) return 0;
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

/** Reads one string out of a tape on the device, as @ref sz_sequence_tape_start does on a host. */
static __device__ sz_cptr_t sz_sequence_tape_start_simt_(void const *handle, sz_size_t index) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_cptr_t)handle + offsets[index];
}

/** Reads one length out of a tape, from the device. */
static __device__ sz_size_t sz_sequence_tape_length_simt_(void const *handle, sz_size_t index) {
    sz_u64_t const *offsets = (sz_u64_t const *)handle;
    return (sz_size_t)(offsets[index + 1] - offsets[index]);
}

#pragma endregion Device Sequences

#endif // (STRINGZILLA_ARCH_ROCM_ && defined(__HIP__)) || (STRINGZILLA_ARCH_CUDA_ && defined(__CUDACC__) && ...

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TYPES_CUH_
