/**
 *  @file include/stringzilla/hash/simt.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief The multi-state SHA-256 kernels CUDA and ROCm share: one thread per message, each running
 *      the ordinary rounds in its own registers, so a batch of millions of short messages keeps
 *      every multiprocessor busy.
 *
 *  States and texts stay in device-reachable memory, and texts arrive as one tape, as every device
 *  verb takes them. A thread reads its message's blocks with aligned loads funnel-shifted into
 *  place, so no byte of a message is read twice and none past its end.
 *
 *  @sa include/stringzilla/hash/serial.h
 *  @sa include/stringzilla/hash/cuda.cuh
 *  @sa include/stringzilla/hash/rocm.cuh
 */
#ifndef STRINGZILLA_HASH_SIMT_CUH_
#define STRINGZILLA_HASH_SIMT_CUH_

#include "stringzilla/types.cuh"
#include "stringzilla/hash/serial.h"

#if STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Rounds

/** The round constants of @c sz_sha256_round_constants_, in the constant bank every round reads. */
static __constant__ sz_u32_t const sz_sha256_round_constants_simt_[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

STRINGZILLA_DEVICE sz_u32_t sz_sha256_rotr_simt_(sz_u32_t value, unsigned count) {
    return __funnelshift_r(value, value, count);
}

/** Compresses one block of 16 big-endian words in @p message into @p hash, rolling @p message
 *  forward as the schedule. */
STRINGZILLA_DEVICE void sz_sha256_compress_simt_(sz_u32_t hash[8], sz_u32_t message[16]) {
    sz_u32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3], e = hash[4], f = hash[5], g = hash[6], h = hash[7];
#pragma unroll
    for (int round = 0; round != 64; ++round) {
        if (round >= 16) {
            sz_u32_t const older = message[(round - 15) & 15], newer = message[(round - 2) & 15];
            message[round & 15] += (sz_sha256_rotr_simt_(newer, 17) ^ sz_sha256_rotr_simt_(newer, 19) ^ (newer >> 10)) +
                                   message[(round - 7) & 15] +
                                   (sz_sha256_rotr_simt_(older, 7) ^ sz_sha256_rotr_simt_(older, 18) ^ (older >> 3));
        }
        sz_u32_t const first =
            h + (sz_sha256_rotr_simt_(e, 6) ^ sz_sha256_rotr_simt_(e, 11) ^ sz_sha256_rotr_simt_(e, 25)) +
            ((e & f) ^ (~e & g)) + sz_sha256_round_constants_simt_[round] + message[round & 15];
        sz_u32_t const second = (sz_sha256_rotr_simt_(a, 2) ^ sz_sha256_rotr_simt_(a, 13) ^
                                 sz_sha256_rotr_simt_(a, 22)) +
                                ((a & b) ^ (a & c) ^ (b & c));
        h = g, g = f, f = e, e = d + first, d = c, c = b, b = a, a = first + second;
    }
    hash[0] += a, hash[1] += b, hash[2] += c, hash[3] += d, hash[4] += e, hash[5] += f, hash[6] += g, hash[7] += h;
}

/** The 16 big-endian words of the block at @p bytes, at any alignment. The 17th aligned word is
 *  read only for a misaligned block, whose last bytes it holds, so no read leaves the block. */
STRINGZILLA_DEVICE void sz_sha256_load_block_simt_(sz_u8_t const *bytes, sz_u32_t message[16]) {
    unsigned const misalignment = (unsigned)((sz_size_t)bytes & 3);
    sz_u32_t const *words = (sz_u32_t const *)(bytes - misalignment);
    sz_u32_t previous = __ldg(words);
#pragma unroll
    for (int index = 0; index != 16; ++index) {
        sz_u32_t const next = index != 15 || misalignment ? __ldg(words + index + 1) : 0;
        message[index] = __byte_perm(__funnelshift_r(previous, next, misalignment * 8), 0, 0x0123);
        previous = next;
    }
}

#pragma endregion Rounds

#pragma region Kernels

static __global__ void sz_sha256_multistate_update_simt_kernel_(sz_sha256_state_t *states, void const *tape,
                                                                sz_size_t count) {
    sz_size_t const lane = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    sz_sha256_state_t *state = states + lane;
    sz_u8_t const *text = (sz_u8_t const *)sz_sequence_tape_start_simt_(tape, lane);
    sz_size_t length = sz_sequence_tape_length_simt_(tape, lane);
    sz_size_t const buffered = state->block_length;
    state->total_length += length;
    if (buffered + length < STRINGZILLA_SHA256_BLOCK_LENGTH) {
        for (sz_size_t index = 0; index != length; ++index) state->block[buffered + index] = text[index];
        state->block_length = (sz_u8_t)(buffered + length);
        return;
    }

    sz_u32_t hash[8], message[16];
#pragma unroll
    for (int index = 0; index != 8; ++index) hash[index] = state->hash[index];
    if (buffered) {
        sz_size_t const head = STRINGZILLA_SHA256_BLOCK_LENGTH - buffered;
        for (sz_size_t index = 0; index != head; ++index) state->block[buffered + index] = text[index];
        sz_sha256_load_block_simt_(state->block, message);
        sz_sha256_compress_simt_(hash, message);
        text += head, length -= head;
    }
    for (; length >= STRINGZILLA_SHA256_BLOCK_LENGTH;
         text += STRINGZILLA_SHA256_BLOCK_LENGTH, length -= STRINGZILLA_SHA256_BLOCK_LENGTH)
        sz_sha256_load_block_simt_(text, message), sz_sha256_compress_simt_(hash, message);
    for (sz_size_t index = 0; index != length; ++index) state->block[index] = text[index];
    state->block_length = (sz_u8_t)length;
#pragma unroll
    for (int index = 0; index != 8; ++index) state->hash[index] = hash[index];
}

static __global__ void sz_sha256_multistate_digest_simt_kernel_(sz_sha256_state_t const *states, sz_size_t count,
                                                                sz_u8_t *digests) {
    sz_size_t const lane = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    sz_sha256_state_t const *state = states + lane;
    sz_u32_t hash[8], message[16];
#pragma unroll
    for (int index = 0; index != 8; ++index) hash[index] = state->hash[index];

    // Padding is assembled in big-endian words, so the buffered bytes never move through memory.
    sz_size_t const buffered = state->block_length;
#pragma unroll
    for (int index = 0; index != 16; ++index) message[index] = 0;
    for (sz_size_t index = 0; index != buffered; ++index)
        message[index / 4] |= (sz_u32_t)state->block[index] << (24 - 8 * (index % 4));
    message[buffered / 4] |= 0x80u << (24 - 8 * (buffered % 4));
    if (buffered >= 56) {
        sz_sha256_compress_simt_(hash, message);
#pragma unroll
        for (int index = 0; index != 16; ++index) message[index] = 0;
    }
    sz_u64_t const bits = state->total_length * 8;
    message[14] = (sz_u32_t)(bits >> 32), message[15] = (sz_u32_t)bits;
    sz_sha256_compress_simt_(hash, message);

    sz_u32_t *digest = (sz_u32_t *)(digests + lane * STRINGZILLA_SHA256_DIGEST_LENGTH);
#pragma unroll
    for (int index = 0; index != 8; ++index) digest[index] = __byte_perm(hash[index], 0, 0x0123);
}

#pragma endregion Kernels

/** Threads per block; one per message, so occupancy is all a launch needs. */
enum { sz_sha256_threads_per_block_simt_k = 256 };

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_ARCH_CUDA_ || STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_HASH_SIMT_CUH_
