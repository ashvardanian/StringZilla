/**
 *  @file include/stringzilla/levenshtein/blackwell.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Blackwell tier of Levenshtein distances: the tile queue over cluster launch control,
 *      the entry points passing it to the shared walks, and the @c _blackwell exports over the
 *      CUDA host side.
 *
 *  From compute capability 10.0 a block done with its own tile cancels an unstarted block of the
 *  same grid and takes that block's tile, so the hardware's queue of unlaunched blocks is the queue
 *  of tiles and a round keeps no state of its own on the device. A grid is then as wide as the
 *  batch and a tile holds one round. The overlap tier includes this file for the same queue.
 *
 *  @sa include/stringzilla/levenshtein/cuda.cuh
 *  @sa include/stringzilla/overlap/blackwell.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_BLACKWELL_CUH_
#define STRINGZILLA_LEVENSHTEIN_BLACKWELL_CUH_

#include "stringzilla/levenshtein/cuda.cuh"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_TARGET_BLACKWELL

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Tile Queue

/** Asks the device for a block of this grid not yet started, with Blackwell's
 *  @c clusterlaunchcontrol.try_cancel , the answer completing the barrier. */
STRINGZILLA_DEVICE void sz_tile_queue_ask_blackwell_(sz_tile_queue_t *queue) {
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
STRINGZILLA_DEVICE int sz_tile_queue_take_blackwell_(sz_tile_queue_t *queue, sz_u32_t *x, sz_u32_t *y) {
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

/**
 *  @brief Points @p queue at the block's own tile and asks for the next one; every thread calls it,
 *      and it ends in a barrier.
 *
 *  @param[in] seats Items of the tile handed out without a draw, one to each of the block's first
 *      seats by @ref sz_tile_queue_first_simt_, so its opening draws never contend at once.
 */
STRINGZILLA_DEVICE void sz_tile_queue_open_blackwell_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                      sz_size_t seats) {
    if (threadIdx.x == 0) {
        sz_size_t const items = sz_min_of_two(tile_size, count - (sz_size_t)blockIdx.x * tile_size);
        queue->drawn = ((unsigned long long)blockIdx.x << 32) | ((unsigned long long)blockIdx.y << 16) |
                       (unsigned long long)sz_min_of_two(seats, items);
        queue->answers = 0;
        sz_u32_t const arrived = (sz_u32_t)__cvta_generic_to_shared(&queue->arrived);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(arrived) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        sz_tile_queue_ask_blackwell_(queue);
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
STRINGZILLA_DEVICE int sz_tile_queue_draw_blackwell_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
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
        // The last handover moved the answer count before the tile this thread saw: fence first.
        __threadfence_block();
        if (offset == items) {
            sz_u32_t next_x, next_y;
            if (!sz_tile_queue_take_blackwell_(queue, &next_x, &next_y)) {
                atomicExch(&queue->drawn, (~0ull << sz_tile_queue_x_shift_k));
                return 0;
            }
            sz_tile_queue_ask_blackwell_(queue);
            __threadfence_block();
            atomicExch(&queue->drawn, ((unsigned long long)next_x << 32) | ((unsigned long long)next_y << 16) | 1ull);
            *y = next_y, *item = (sz_size_t)next_x * tile_size;
            return 1;
        }
        while (((*(unsigned long long volatile *)&queue->drawn ^ drawn) >> 16) == 0) __nanosleep(64);
    }
}

/**
 *  @brief The block's next tile, once every thread is done with its last; every thread calls it.
 *  @return Zero once the grid has nothing left for this block.
 *
 *  Thread zero takes the answer and asks again before the barrier, so the request is in flight
 *  while the block works through the tile it returns. The caller holds another barrier between two
 *  calls, as each overwrites what the last one shared.
 */
STRINGZILLA_DEVICE int sz_tile_queue_next_blackwell_(sz_tile_queue_t *queue, sz_u32_t *x, sz_u32_t *y) {
    if (threadIdx.x == 0) {
        sz_u32_t next_x = 0, next_y = 0;
        int const taken = sz_tile_queue_take_blackwell_(queue, &next_x, &next_y);
        if (taken) sz_tile_queue_ask_blackwell_(queue);
        queue->drawn = taken ? ((unsigned long long)next_x << 32) | ((unsigned long long)next_y << 16)
                             : (~0ull << sz_tile_queue_x_shift_k);
    }
    __syncthreads();
    unsigned long long const drawn = queue->drawn;
    if (drawn >= (~0ull << sz_tile_queue_x_shift_k)) return 0;
    *x = (sz_u32_t)(drawn >> 32), *y = (sz_u32_t)(drawn >> 16) & 0xFFFFu;
    return 1;
}

#pragma endregion Tile Queue

#pragma region Entry Points

/*  Threaded byte rung: one entry point per word count. */
static __global__ void sz_levenshtein_distances_w1_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 1,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w2_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 2,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w3_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 3,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w4_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 4,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w5_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 5,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w6_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 6,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w7_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 7,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w8_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 8,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w9_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 9,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w10_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 10,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w11_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 11,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w12_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 12,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w13_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 13,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w14_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 14,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w15_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 15,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_w16_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 16,
                               sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

/*  Threaded rune rung: one entry point per word count. */
static __global__ void sz_levenshtein_distances_utf8_w1_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 1,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w2_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 2,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w3_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 3,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w4_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 4,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w5_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 5,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w6_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 6,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w7_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 7,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w8_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 8,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w9_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 9,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w10_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 10,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w11_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 11,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w12_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 12,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w13_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 13,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w14_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 14,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w15_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 15,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_w16_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, candidate_order, 16,
                                    sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

/*  Warped byte rung: one entry point per words-per-lane. */
static __global__ void sz_levenshtein_distances_k1_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 1, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k2_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 2, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k3_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 3, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k4_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 4, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k5_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 5, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k6_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 6, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k7_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 7, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_k8_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, 8, sz_shuffle_up_cuda_,
                                    sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_blackwell_,
                                    sz_tile_queue_draw_blackwell_);
}

/*  Warped rune rung: one entry point per words-per-lane. */
static __global__ void sz_levenshtein_distances_utf8_k1_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 1, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k2_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 2, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k3_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 3, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k4_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 4, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k5_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 5, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k6_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 6, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k7_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 7, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

static __global__ void sz_levenshtein_distances_utf8_k8_blackwell_kernel_( //
    sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_sequence_t candidates, sz_size_t *distances,
    sz_size_t distances_stride) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, 8, sz_shuffle_up_cuda_,
                                         sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_, sz_lanes_any_cuda_,
                                         sz_tile_queue_open_blackwell_, sz_tile_queue_draw_blackwell_);
}

/** The Blackwell tier's entry points, whose blocks take over unstarted ones, so a tile holds one
 *  round. */
static sz_levenshtein_entry_points_cuda_t const sz_levenshtein_entry_points_blackwell_ = {
    {
        (void const *)sz_levenshtein_distances_w1_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w2_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w3_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w4_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w5_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w6_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w7_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w8_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w9_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w10_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w11_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w12_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w13_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w14_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w15_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_w16_blackwell_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_w1_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w2_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w3_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w4_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w5_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w6_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w7_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w8_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w9_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w10_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w11_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w12_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w13_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w14_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w15_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w16_blackwell_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_k1_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k2_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k3_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k4_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k5_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k6_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k7_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_k8_blackwell_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_k1_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k2_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k3_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k4_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k5_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k6_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k7_blackwell_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k8_blackwell_kernel_,
    },
    1,
    sz_cap_blackwell_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_blackwell(sz_levenshtein_engine_t *engine,
                                                                 sz_sequence_t const *queries,
                                                                 sz_levenshtein_symbol_t symbol,
                                                                 sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_cuda_(engine, queries, symbol, &sz_levenshtein_entry_points_blackwell_,
                                                   allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_blackwell(sz_levenshtein_engine_t *engine,
                                                               sz_sequence_t const *candidates, sz_size_t *distances,
                                                               sz_size_t distances_stride, sz_stream_t stream) {
    return sz_levenshtein_distances_scoped_cuda_(engine, candidates, distances, distances_stride,
                                                 &sz_levenshtein_entry_points_blackwell_, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_BLACKWELL
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_LEVENSHTEIN_BLACKWELL_CUH_
