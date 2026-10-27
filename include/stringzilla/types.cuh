/**
 *  @file include/stringzilla/types.cuh
 *  @author Ash Vardanian
 *  @date September 20, 2026
 *  @brief Core device code the CUDA and ROCm kernels share: the lanes, the block scan, the tile
 *      queues, the chained tiles and the tape a kernel reads strings from.
 *
 *  Every family with a SIMT backend writes its walks once, in its <tt>simt.cuh</tt>, and both nvcc and
 *  hipcc compile them. What those walks share lives here for the same reason the host tape
 *  readers live in <tt>types.h</tt> rather than in a family. The host side of every vendor is its own:
 *  <tt>cuda.cuh</tt> and <tt>rocm.cuh</tt> each drive their runtime, and nothing here calls one.
 *
 *  Written in C, as every <tt>.cuh</tt> in this library is. A walk takes its cross-lane and tile-queue
 *  steps as function parameters, which each vendor's or tier's kernel binds to its own helpers, so
 *  nothing here spells one vendor's or one generation's instructions.
 *
 *  @sa include/stringzilla/types.h
 *  @sa include/stringzilla/cuda.cuh
 *  @sa include/stringzilla/rocm.cuh
 */
#ifndef STRINGZILLA_TYPES_CUH_
#define STRINGZILLA_TYPES_CUH_

#include "stringzilla/types.h"

/*  A library holds one unit per vendor it was built for, each compiled for exactly one of them,
 *  and its host units, which list every vendor's kernels, never include a `.cuh`, so the vendor's
 *  switch alone picks the runtime here. */
#if STRINGZILLA_ARCH_ROCM_
#include <hip/hip_runtime.h> // `__shfl_up`, `__ballot`, `atomicAdd`
#elif STRINGZILLA_ARCH_CUDA_
#include <cuda_runtime.h> // `__shfl_up_sync`, `__any_sync`, `atomicAdd`
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

/*  Kernels are written for 32 lanes. A 64-wide AMD wavefront runs two such groups side by side, so
 *  every cross-lane step stays inside its own half. */
#pragma region Lanes

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

/** Stores the @p ballot of the 32 lanes marking items from @p first on, as the little-endian bytes
 *  of @p bits they fall in among the @c ceil(count/8) a bitmap of @p count items holds. */
STRINGZILLA_DEVICE void sz_u1x8_store_simt_(sz_u1x8_t *bits, sz_size_t count, sz_size_t first, sz_u32_t ballot) {
    sz_size_t const bytes = sz_size_divide_round_up(count, 8);
    sz_size_t byte;
    for (byte = first / 8; byte != first / 8 + 4 && byte < bytes; ++byte)
        bits[byte] = (sz_u1x8_t)(ballot >> (byte - first / 8) * 8);
}

#pragma endregion Lanes

/*  A block's queue of the grid's tiles. Here every block runs its own tile; Blackwell's tier, in
 *  `levenshtein/blackwell.cuh`, lets a block done with its tile cancel one not yet started and take
 *  that block's tile, so a grid there can be sized by its work rather than by what stays resident.
 *  The walks take the queue's steps as parameters, so both tiers share them. */
#pragma region Tile Queues

/** One block's queue of a grid's tiles, in shared memory; only Blackwell's tier asks for more. */
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

/**
 *  @brief Points @p queue at the block's own tile; every thread calls it, and it ends in a barrier.
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
    }
    __syncthreads();
}

/**
 *  @brief Draws one item of the block's own tile, @p tile_size of @p count to a tile, from any
 *      thread at any time.
 *
 *  @param[out] y The row of the grid the item's tile sits on.
 *  @param[out] item The item's index among all @p count.
 *  @return Zero once the tile has nothing left.
 */
STRINGZILLA_DEVICE int sz_tile_queue_draw_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                sz_u32_t *y, sz_size_t *item) {
    unsigned long long const drawn = atomicAdd(&queue->drawn, 1ull);
    sz_size_t const first = (sz_size_t)(drawn >> 32) * tile_size;
    sz_size_t const offset = (sz_size_t)(drawn & 0xFFFFu);
    if (offset >= sz_min_of_two(tile_size, count - first)) return 0;
    *y = (sz_u32_t)(drawn >> 16) & 0xFFFFu, *item = first + offset;
    return 1;
}

/** The first item of @p seat, which is handed out directly while the block's own tile holds it and
 *  taken with @p draw like any other past that. */
STRINGZILLA_DEVICE int sz_tile_queue_first_simt_(sz_tile_queue_t *queue, sz_size_t tile_size, sz_size_t count,
                                                 sz_size_t seat, sz_u32_t *y, sz_size_t *item,
                                                 int (*draw)(sz_tile_queue_t *, sz_size_t, sz_size_t, sz_u32_t *,
                                                             sz_size_t *)) {
    sz_size_t const first = (sz_size_t)blockIdx.x * tile_size;
    if (seat < sz_min_of_two(tile_size, count - first)) {
        *y = blockIdx.y, *item = first + seat;
        return 1;
    }
    return draw(queue, tile_size, count, y, item);
}

/** The block's next tile, which is none, as every block here runs only its own. */
STRINGZILLA_DEVICE int sz_tile_queue_next_simt_(sz_tile_queue_t *queue, sz_u32_t *x, sz_u32_t *y) {
    sz_unused_(queue), sz_unused_(x), sz_unused_(y);
    return 0;
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

/** Where tile @p ticket writes, summing the tiles before it by decoupled look-back over @p states,
 *  one cleared word per tile, where it publishes its own @p total first; every thread calls it, and
 *  it ends in a barrier. A word carries its total in 62 bits, read back sign-extended, so totals of
 *  signed deltas that wrap below zero scan as exactly as counts do. */
STRINGZILLA_DEVICE sz_size_t sz_chain_look_back_simt_(sz_u64_t *states, sz_size_t ticket, sz_size_t total) {
    __shared__ sz_size_t offset;
    if (threadIdx.x == 0) {
        // Bit 62 marks a word holding its tile's own total, bit 63 one holding the sum up to it.
        sz_u64_t const own = 1ull << 62, inclusive = 1ull << 63, value = own - 1;
        sz_u64_t volatile *const published = (sz_u64_t volatile *)states;
        sz_size_t prefix = 0, look = ticket;
        published[ticket] = (ticket ? own : inclusive) | (total & value);
        while (look-- != 0) {
            sz_u64_t state;
            do state = published[look];
            while (!state);
            prefix += (sz_size_t)((sz_i64_t)(state << 2) >> 2);
            if (state & inclusive) break;
        }
        if (ticket) published[ticket] = inclusive | ((prefix + total) & value);
        offset = prefix;
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

#pragma region Device Tapes

/** The first byte of string @p index of a tape the device reads, and one past the last string's end
 *  for @p index equal to the count. A warp reads one layout, so the branch never diverges. */
STRINGZILLA_DEVICE sz_cptr_t sz_strings_start_simt_(sz_strings_view_t strings, sz_size_t index) {
    return strings.layout == sz_strings_u32_tape_k ? strings.data.u32_tape.data + strings.data.u32_tape.offsets[index]
                                                   : strings.data.u64_tape.data + strings.data.u64_tape.offsets[index];
}

/** The bytes string @p index of a tape the device reads spans. */
STRINGZILLA_DEVICE sz_size_t sz_strings_length_simt_(sz_strings_view_t strings, sz_size_t index) {
    return (sz_size_t)(sz_strings_start_simt_(strings, index + 1) - sz_strings_start_simt_(strings, index));
}

#pragma endregion Device Tapes

#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TYPES_CUH_
