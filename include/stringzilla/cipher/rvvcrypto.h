/**
 *  @brief RISC-V vector-crypto backend for AES-256 in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/rvvcrypto.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 *
 *  Two vector-crypto extensions carry this backend. `Zvkned` supplies the AES round and the AES-256 key
 *  schedule step, so one round is one `vaesem.vv` and one whole round key is one `vaeskf2.vi`. `Zvkg`
 *  supplies the Galois hash itself: `vghsh.vv` exclusive-ors a block into the running hash and multiplies
 *  by the subkey with the field reduction already inside the instruction, while `vgmul.vv` is the bare
 *  multiply the subkey powers are built from.
 *
 *  That second extension is what makes this file shorter than its x86 and Arm siblings. Those have only a
 *  carry-less multiply, so a block costs three multiplies and a hand-written reduction, and they hide the
 *  reduction latency by aggregating eight blocks against the precomputed powers `H^1` through `H^8`. Here
 *  a block is one instruction with no reduction left to defer, so the hash is a plain serial chain over
 *  `powers[0 .. 16)`, which is `H^1`. The remaining seven powers are never read.
 *
 *  The fourteen rounds are written out rather than driven by a loop over the round index. A round loop
 *  leaves the unrolling to the optimizer, which only does it reliably at `-O3`, and `SZ_API_COMPTIME`
 *  kernels are `static inline` and compile into whichever translation unit consumes them — so a looped
 *  kernel makes its own throughput a property of the caller's build flags. See `cipher/icelake.h`, where
 *  the same change was worth 2.1x to 2.9x at `-O2` and nothing at all at `-O3`.
 *
 *  Aggregating across element groups is expressible, since a wide `vgmul.vv` multiplies every 128-bit
 *  group of a register at once, but it does not pay. Folding `N` products back into one accumulator costs
 *  a descending-power gather plus `2 log2(N)` slide and exclusive-or instructions on top of the multiply,
 *  against `N` instructions for the serial chain, so it first breaks even at sixteen blocks per group and
 *  eight is the highest power the key carries.
 *
 *  `Zvkg` also fixes the byte order question that costs the x86 backends a shuffle per block. Its element
 *  group is the sixteen hash bytes in memory order, and the bit reflection GCM is defined over lives
 *  inside the instruction, so an ordinary byte load feeds `vghsh.vv` directly.
 */
#ifndef STRINGZILLA_CIPHER_RVVCRYPTO_H_
#define STRINGZILLA_CIPHER_RVVCRYPTO_H_

#include "stringzilla/types.h"
#include "stringzilla/cipher/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVVCRYPTO

#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvkned,+zvkg"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvkned,+zvkg")
#endif

/*  `SZ_USE_RVVCRYPTO` turns on for `Zvkned` and `Zvknhb`, the pair `sz_hash` needs, and says nothing
 *  about `Zvkg`. A target carrying AES but not the Galois hash therefore reaches this file, and every
 *  hash step below is written against one pair of helpers that resolve to `vghsh.vv` and `vgmul.vv`
 *  where the extension is present and to the constant-time serial reduction where it is not. The AES
 *  side is vectorized either way. */

#pragma region Block Element Groups

/**
 *  @brief Loads one 16-byte block into element group zero of a vector register.
 *  @param bytes The 16 bytes, at any alignment.
 *  @return The block, byte zero in the least significant position of the group.
 *
 *  Byte loads rather than word loads because every buffer this backend reads from is a `sz_u8_t` array
 *  whose alignment the caller chooses, and an element-width load may fault on an implementation without
 *  the misaligned-access extension.
 */
SZ_HELPER_INLINE vuint32m1_t sz_aes256_block_load_rvvcrypto_(sz_u8_t const *bytes) {
    return __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vle8_v_u8m1(bytes, SZ_AES_BLOCK_LENGTH));
}

/**
 *  @brief Writes element group zero of a vector register out as 16 bytes.
 *  @param bytes Receives the 16 bytes, at any alignment.
 *  @param block_u32m1 The block to store.
 */
SZ_HELPER_INLINE void sz_aes256_block_store_rvvcrypto_(sz_u8_t *bytes, vuint32m1_t block_u32m1) {
    __riscv_vse8_v_u8m1(bytes, __riscv_vreinterpret_v_u32m1_u8m1(block_u32m1), SZ_AES_BLOCK_LENGTH);
}

/**
 *  @brief Extracts one 128-bit element group from a wide register into element group zero.
 *  @param blocks_u32m4 The wide register.
 *  @param block_ordinal Which element group to take, counting from zero.
 *  @param vector_length Active lanes in @p blocks_u32m4.
 *  @return The requested group, ready for a hash or a store.
 *
 *  A slide keeps the value in registers. Spilling the wide register to a staging buffer and reloading
 *  block by block would be fewer instructions on a machine with a slow slide, but the ciphertext must
 *  reach the hash without a round trip through memory the caller can alias.
 */
SZ_HELPER_INLINE vuint32m1_t sz_aes256_group_extract_rvvcrypto_(vuint32m4_t blocks_u32m4, sz_size_t block_ordinal,
                                                                sz_size_t vector_length) {
    vuint32m4_t const slid_u32m4 = __riscv_vslidedown_vx_u32m4(blocks_u32m4, block_ordinal * 4, vector_length);
    return __riscv_vget_v_u32m4_u32m1(slid_u32m4, 0);
}

/**
 *  @brief Byte offsets that replicate a 16-byte element group across every group of a wide register.
 *  @param vector_length Active lanes, always a multiple of four.
 *  @return Lane `i` holds `(i mod 4) * 4`, the byte offset of its word within a single block.
 *
 *  An indexed load through these offsets turns any 16 bytes in memory into a wide register holding the
 *  same block in every element group, which is what the `.vv` form of an AES round expects when it is
 *  transforming several counter blocks under one round key.
 */
SZ_HELPER_INLINE vuint32m4_t sz_aes256_broadcast_offsets_rvvcrypto_(sz_size_t vector_length) {
    vuint32m4_t const lane_index_u32m4 = __riscv_vid_v_u32m4(vector_length);
    return __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(lane_index_u32m4, 3, vector_length), 2, vector_length);
}

#pragma endregion // Block Element Groups

#pragma region Key Schedule

SZ_API_COMPTIME void sz_aes256_key_init_rvvcrypto(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_size_t const vector_length = 4;
    vuint32m1_t const round_0_u32m1 = sz_aes256_block_load_rvvcrypto_(secret);
    vuint32m1_t const round_1_u32m1 = sz_aes256_block_load_rvvcrypto_(secret + 16);
    vuint32m1_t const round_2_u32m1 = __riscv_vaeskf2_vi_u32m1(round_0_u32m1, round_1_u32m1, 2, vector_length);
    vuint32m1_t const round_3_u32m1 = __riscv_vaeskf2_vi_u32m1(round_1_u32m1, round_2_u32m1, 3, vector_length);
    vuint32m1_t const round_4_u32m1 = __riscv_vaeskf2_vi_u32m1(round_2_u32m1, round_3_u32m1, 4, vector_length);
    vuint32m1_t const round_5_u32m1 = __riscv_vaeskf2_vi_u32m1(round_3_u32m1, round_4_u32m1, 5, vector_length);
    vuint32m1_t const round_6_u32m1 = __riscv_vaeskf2_vi_u32m1(round_4_u32m1, round_5_u32m1, 6, vector_length);
    vuint32m1_t const round_7_u32m1 = __riscv_vaeskf2_vi_u32m1(round_5_u32m1, round_6_u32m1, 7, vector_length);
    vuint32m1_t const round_8_u32m1 = __riscv_vaeskf2_vi_u32m1(round_6_u32m1, round_7_u32m1, 8, vector_length);
    vuint32m1_t const round_9_u32m1 = __riscv_vaeskf2_vi_u32m1(round_7_u32m1, round_8_u32m1, 9, vector_length);
    vuint32m1_t const round_10_u32m1 = __riscv_vaeskf2_vi_u32m1(round_8_u32m1, round_9_u32m1, 10, vector_length);
    vuint32m1_t const round_11_u32m1 = __riscv_vaeskf2_vi_u32m1(round_9_u32m1, round_10_u32m1, 11, vector_length);
    vuint32m1_t const round_12_u32m1 = __riscv_vaeskf2_vi_u32m1(round_10_u32m1, round_11_u32m1, 12, vector_length);
    vuint32m1_t const round_13_u32m1 = __riscv_vaeskf2_vi_u32m1(round_11_u32m1, round_12_u32m1, 13, vector_length);
    vuint32m1_t const round_14_u32m1 = __riscv_vaeskf2_vi_u32m1(round_12_u32m1, round_13_u32m1, 14, vector_length);

    __riscv_vse32_v_u32m1(key->round_keys + 0, round_0_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 4, round_1_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 8, round_2_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 12, round_3_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 16, round_4_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 20, round_5_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 24, round_6_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 28, round_7_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 32, round_8_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 36, round_9_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 40, round_10_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 44, round_11_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 48, round_12_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 52, round_13_u32m1, vector_length);
    __riscv_vse32_v_u32m1(key->round_keys + 56, round_14_u32m1, vector_length);
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/*  Round keys reach a wide register through an indexed load rather than through the `.vs` forms of the
 *  AES instructions, which take element group zero of their second operand and apply it to every group
 *  of the first, and which are the natural spelling for this. GCC 14 lowers `vaesem.vs` under a
 *  `vsetvli` carrying the LMUL of the key operand instead of the LMUL of the state, so the vector length
 *  is clamped to one element group and every block past the first leaves the pipeline untransformed.
 *  Clang lowers the same intrinsic correctly. The indexed load costs one instruction per round key,
 *  exactly what the unit-stride load feeding a `.vs` form would cost, so nothing is given up by staying
 *  on the `.vv` forms and both compilers then agree. */

/**
 *  @brief Encrypts one block held in element group zero.
 *  @param key The expanded schedule.
 *  @param block_u32m1 The 16 plaintext bytes.
 *  @return The 16 ciphertext bytes.
 */
/** @brief Applies one middle round to a single block, reading its round key from the schedule. */
SZ_HELPER_INLINE vuint32m1_t sz_aes256_round_rvvcrypto_(vuint32m1_t block_u32m1, sz_u32_t const *round_key,
                                                        sz_size_t vector_length) {
    return __riscv_vaesem_vv_u32m1(block_u32m1, __riscv_vle32_v_u32m1(round_key, vector_length), vector_length);
}

/** @brief Applies one middle round to every element group, broadcasting its round key across them. */
SZ_HELPER_INLINE vuint32m4_t sz_aes256_round_wide_rvvcrypto_(vuint32m4_t blocks_u32m4, sz_u32_t const *round_key,
                                                             vuint32m4_t broadcast_offsets_u32m4,
                                                             sz_size_t vector_length) {
    return __riscv_vaesem_vv_u32m4(
        blocks_u32m4, __riscv_vluxei32_v_u32m4(round_key, broadcast_offsets_u32m4, vector_length), vector_length);
}

SZ_HELPER_INLINE vuint32m1_t sz_aes256_block_encrypt_rvvcrypto_(sz_aes256_key_t const *key, vuint32m1_t block_u32m1) {
    sz_size_t const vector_length = 4;
    block_u32m1 = __riscv_vxor_vv_u32m1(block_u32m1, __riscv_vle32_v_u32m1(key->round_keys + 0, vector_length),
                                        vector_length);
    // The thirteen middle rounds are written out rather than looped, because a loop leaves the
    // unrolling to the optimizer and the optimizer only does it at `-O3`.
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 4, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 8, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 12, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 16, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 20, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 24, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 28, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 32, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 36, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 40, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 44, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 48, vector_length);
    block_u32m1 = sz_aes256_round_rvvcrypto_(block_u32m1, key->round_keys + 52, vector_length);
    return __riscv_vaesef_vv_u32m1(block_u32m1, __riscv_vle32_v_u32m1(key->round_keys + 56, vector_length),
                                   vector_length);
}

/**
 *  @brief Encrypts every element group of a wide register under one schedule.
 *  @param key The expanded schedule.
 *  @param blocks_u32m4 One plaintext block per element group.
 *  @param broadcast_offsets_u32m4 Offsets from `sz_aes256_broadcast_offsets_rvvcrypto_`.
 *  @param vector_length Active lanes, four per block.
 *  @return One ciphertext block per element group.
 */
SZ_HELPER_INLINE vuint32m4_t sz_aes256_blocks_encrypt_rvvcrypto_(sz_aes256_key_t const *key, vuint32m4_t blocks_u32m4,
                                                                 vuint32m4_t broadcast_offsets_u32m4,
                                                                 sz_size_t vector_length) {
    blocks_u32m4 = __riscv_vxor_vv_u32m4(
        blocks_u32m4, __riscv_vluxei32_v_u32m4(key->round_keys, broadcast_offsets_u32m4, vector_length), vector_length);
    // Written out rather than looped, for the same reason as the single-block path above.
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 4, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 8, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 12, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 16, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 20, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 24, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 28, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 32, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 36, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 40, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 44, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 48, broadcast_offsets_u32m4,
                                                   vector_length);
    blocks_u32m4 = sz_aes256_round_wide_rvvcrypto_(blocks_u32m4, key->round_keys + 52, broadcast_offsets_u32m4,
                                                   vector_length);
    return __riscv_vaesef_vv_u32m4(
        blocks_u32m4, __riscv_vluxei32_v_u32m4(key->round_keys + 56, broadcast_offsets_u32m4, vector_length),
        vector_length);
}

#pragma endregion // Block Encryption

#pragma region Counter Blocks

/** @brief Reads the big-endian 32-bit block index occupying the last four bytes of a counter block. */
SZ_HELPER_INLINE sz_u32_t sz_aes256_counter_index_load_rvvcrypto_(sz_u8_t const *block) {
    return ((sz_u32_t)block[12] << 24) | ((sz_u32_t)block[13] << 16) | ((sz_u32_t)block[14] << 8) | (sz_u32_t)block[15];
}

/** @brief Writes the big-endian 32-bit block index into the last four bytes of a counter block. */
SZ_HELPER_INLINE void sz_aes256_counter_index_store_rvvcrypto_(sz_u8_t *block, sz_u32_t block_index) {
    block[12] = (sz_u8_t)(block_index >> 24);
    block[13] = (sz_u8_t)(block_index >> 16);
    block[14] = (sz_u8_t)(block_index >> 8);
    block[15] = (sz_u8_t)(block_index >> 0);
}

/** @brief Reverses the four bytes of every lane, turning a native counter into its big-endian image. */
SZ_HELPER_INLINE vuint32m4_t sz_u32m4_bytes_reverse_rvvcrypto_(vuint32m4_t value_u32m4, sz_size_t vector_length) {
    vuint32m4_t const highest_u32m4 = __riscv_vsll_vx_u32m4(value_u32m4, 24, vector_length);
    vuint32m4_t const higher_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsll_vx_u32m4(value_u32m4, 8, vector_length),
                                                           0x00FF0000u, vector_length);
    vuint32m4_t const lower_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(value_u32m4, 8, vector_length),
                                                          0x0000FF00u, vector_length);
    vuint32m4_t const lowest_u32m4 = __riscv_vsrl_vx_u32m4(value_u32m4, 24, vector_length);
    return __riscv_vor_vv_u32m4(__riscv_vor_vv_u32m4(highest_u32m4, higher_u32m4, vector_length),
                                __riscv_vor_vv_u32m4(lower_u32m4, lowest_u32m4, vector_length), vector_length);
}

/**
 *  @brief Builds a run of consecutive counter blocks, one per element group.
 *  @param counter_block Any block carrying the 12 nonce bytes; its own index bytes are ignored.
 *  @param first_index Block index of element group zero; later groups count up and wrap at `2^32`.
 *  @param broadcast_offsets_u32m4 Offsets from `sz_aes256_broadcast_offsets_rvvcrypto_`.
 *  @param vector_length Active lanes, four per block.
 *  @return The counter blocks, ready to encrypt.
 *
 *  The nonce is identical in every group, so one indexed load replicates it, and only the fourth lane of
 *  each group differs. Building the run in registers keeps the wide path free of the per-block scalar
 *  stores a staging buffer would need.
 */
SZ_HELPER_INLINE vuint32m4_t sz_aes256_counters_build_rvvcrypto_(sz_u8_t const *counter_block, sz_u32_t first_index,
                                                                 vuint32m4_t broadcast_offsets_u32m4,
                                                                 sz_size_t vector_length) {
    sz_u128_vec_t staged_block_vec;
    vuint32m4_t lane_index_u32m4, block_ordinal_u32m4, index_u32m4, blocks_u32m4;
    vbool8_t index_lane_b8;

    sz_aes256_block_store_rvvcrypto_(staged_block_vec.u8s, sz_aes256_block_load_rvvcrypto_(counter_block));
    lane_index_u32m4 = __riscv_vid_v_u32m4(vector_length);
    blocks_u32m4 = __riscv_vluxei32_v_u32m4(staged_block_vec.u32s, broadcast_offsets_u32m4, vector_length);
    block_ordinal_u32m4 = __riscv_vsrl_vx_u32m4(lane_index_u32m4, 2, vector_length);
    index_u32m4 = __riscv_vadd_vx_u32m4(block_ordinal_u32m4, first_index, vector_length);
    index_lane_b8 = __riscv_vmseq_vx_u32m4_b8(__riscv_vand_vx_u32m4(lane_index_u32m4, 3, vector_length), 3,
                                              vector_length);
    return __riscv_vmerge_vvm_u32m4(blocks_u32m4, sz_u32m4_bytes_reverse_rvvcrypto_(index_u32m4, vector_length),
                                    index_lane_b8, vector_length);
}

#pragma endregion // Counter Blocks

#pragma region Galois Hashing

/**
 *  @brief Exclusive-ors a block into the running hash and multiplies by the subkey.
 *  @param accumulator_u32m1 The running hash.
 *  @param block_u32m1 The block to absorb.
 *  @param subkey_u32m1 The hash subkey `H`.
 *  @return The updated hash.
 *
 *  `Zvkg` carries the bit reflection GCM is defined over inside the instruction, so the three operands
 *  are the sixteen hash bytes in memory order and no shuffle stands between a load and the multiply.
 *  Where the extension is absent the constant-time serial reduction stands in, and the round trip
 *  through a staged block costs nothing measurable beside its 128 iterations.
 */
SZ_HELPER_INLINE vuint32m1_t sz_ghash_absorb_rvvcrypto_(vuint32m1_t accumulator_u32m1, vuint32m1_t block_u32m1,
                                                        vuint32m1_t subkey_u32m1) {
#if defined(__riscv_zvkg)
    return __riscv_vghsh_vv_u32m1(accumulator_u32m1, subkey_u32m1, block_u32m1, 4);
#else
    sz_u128_vec_t staged_accumulator_vec, staged_block_vec, staged_subkey_vec;
    sz_aes256_block_store_rvvcrypto_(staged_accumulator_vec.u8s, accumulator_u32m1);
    sz_aes256_block_store_rvvcrypto_(staged_block_vec.u8s, block_u32m1);
    sz_aes256_block_store_rvvcrypto_(staged_subkey_vec.u8s, subkey_u32m1);
    sz_ghash_absorb_serial_(staged_accumulator_vec.u8s, staged_block_vec.u8s, staged_subkey_vec.u8s);
    return sz_aes256_block_load_rvvcrypto_(staged_accumulator_vec.u8s);
#endif
}

/**
 *  @brief Multiplies the running hash by the subkey without absorbing anything.
 *  @param accumulator_u32m1 The value to multiply.
 *  @param subkey_u32m1 The hash subkey `H`.
 *  @return The product.
 */
SZ_HELPER_INLINE vuint32m1_t sz_ghash_multiply_rvvcrypto_(vuint32m1_t accumulator_u32m1, vuint32m1_t subkey_u32m1) {
#if defined(__riscv_zvkg)
    return __riscv_vgmul_vv_u32m1(accumulator_u32m1, subkey_u32m1, 4);
#else
    sz_u128_vec_t staged_accumulator_vec, staged_subkey_vec;
    sz_aes256_block_store_rvvcrypto_(staged_accumulator_vec.u8s, accumulator_u32m1);
    sz_aes256_block_store_rvvcrypto_(staged_subkey_vec.u8s, subkey_u32m1);
    sz_ghash_multiply_serial_(staged_accumulator_vec.u8s, staged_subkey_vec.u8s);
    return sz_aes256_block_load_rvvcrypto_(staged_accumulator_vec.u8s);
#endif
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_rvvcrypto(sz_aes256_gcm_key_t *key,
                                                      sz_u8_t const secret[sz_at_least_(32)]) {
    sz_size_t const vector_length = 4;
    sz_size_t power_index;
    vuint32m1_t subkey_u32m1, power_u32m1;

    sz_aes256_key_init_rvvcrypto(&key->block, secret);
    subkey_u32m1 = sz_aes256_block_encrypt_rvvcrypto_(&key->block, __riscv_vmv_v_x_u32m1(0, vector_length));
    sz_aes256_block_store_rvvcrypto_(key->powers, subkey_u32m1);

    power_u32m1 = subkey_u32m1;
    for (power_index = 1; power_index != 8; ++power_index) {
        power_u32m1 = sz_ghash_multiply_rvvcrypto_(power_u32m1, subkey_u32m1);
        sz_aes256_block_store_rvvcrypto_(key->powers + power_index * SZ_AES_BLOCK_LENGTH, power_u32m1);
    }
}

#pragma endregion // Galois Hashing

#pragma region Counter Mode

SZ_API_COMPTIME void sz_aes256_ctr_xor_rvvcrypto(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                 sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                 sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_size_t const maximum_vector_length = __riscv_vsetvlmax_e32m4();
    sz_size_t const maximum_blocks_per_pass = maximum_vector_length / 4;
    vuint32m4_t const broadcast_offsets_u32m4 = sz_aes256_broadcast_offsets_rvvcrypto_(maximum_vector_length);
    sz_u128_vec_t counter_block_vec, keystream_block_vec;
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);
    sz_size_t produced = 0, byte_index;

    for (byte_index = 0; byte_index != 12; ++byte_index) counter_block_vec.u8s[byte_index] = nonce[byte_index];
    sz_aes256_counter_index_store_rvvcrypto_(counter_block_vec.u8s, block_index);

    while (produced != length) {
        sz_size_t const remaining = length - produced;
        if (within_block == 0 && remaining >= SZ_AES_BLOCK_LENGTH) {
            sz_size_t blocks = remaining / SZ_AES_BLOCK_LENGTH;
            sz_size_t vector_length, byte_count;
            vuint32m4_t counters_u32m4, keystream_u32m4, text_u32m4;
            if (blocks > maximum_blocks_per_pass) blocks = maximum_blocks_per_pass;
            vector_length = blocks * 4;
            byte_count = blocks * SZ_AES_BLOCK_LENGTH;

            counters_u32m4 = sz_aes256_counters_build_rvvcrypto_(counter_block_vec.u8s, block_index,
                                                                 broadcast_offsets_u32m4, vector_length);
            keystream_u32m4 = sz_aes256_blocks_encrypt_rvvcrypto_(key, counters_u32m4, broadcast_offsets_u32m4,
                                                                  vector_length);
            text_u32m4 = __riscv_vreinterpret_v_u8m4_u32m4(__riscv_vle8_v_u8m4(input_bytes + produced, byte_count));
            __riscv_vse8_v_u8m4(
                output_bytes + produced,
                __riscv_vreinterpret_v_u32m4_u8m4(__riscv_vxor_vv_u32m4(text_u32m4, keystream_u32m4, vector_length)),
                byte_count);

            block_index += (sz_u32_t)blocks;
            produced += byte_count;
        }
        else {
            sz_aes256_counter_index_store_rvvcrypto_(counter_block_vec.u8s, block_index);
            sz_aes256_block_store_rvvcrypto_(
                keystream_block_vec.u8s,
                sz_aes256_block_encrypt_rvvcrypto_(key, sz_aes256_block_load_rvvcrypto_(counter_block_vec.u8s)));
            for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
                output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream_block_vec.u8s[within_block]);
            within_block = 0;
            ++block_index;
        }
    }
}

#pragma endregion // Counter Mode

#pragma region Streaming Interface

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  The size is known at compile time but the register width is not, so the length-agnostic `vsetvl` idiom
 *  drives the store loop: each iteration asks for as many bytes as the machine will take and the last one
 *  narrows itself, which reaches the end of a 472-byte state without a scalar epilogue and without ever
 *  writing past it. Ordinary stores followed by one barrier, since a `volatile` view would forbid the
 *  vectorization outright.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_rvvcrypto_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *bytes = (sz_u8_t *)state;
    sz_size_t remaining = sizeof(*state);
    while (remaining != 0) {
        sz_size_t const vector_length = __riscv_vsetvl_e8m8(remaining);
        __riscv_vse8_v_u8m8(bytes, __riscv_vmv_v_x_u8m8(0, vector_length), vector_length);
        bytes += vector_length;
        remaining -= vector_length;
    }
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_rvvcrypto_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                     sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_size_t byte_index;

    state->key = *key;
    for (byte_index = 0; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
        state->accumulator[byte_index] = 0, state->partial[byte_index] = 0, state->keystream[byte_index] = 0;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    for (byte_index = 0; byte_index != 12; ++byte_index) state->counter[byte_index] = nonce[byte_index];
    sz_aes256_counter_index_store_rvvcrypto_(state->counter, 1u);
    sz_aes256_block_store_rvvcrypto_(
        state->tag_mask,
        sz_aes256_block_encrypt_rvvcrypto_(&state->key.block, sz_aes256_block_load_rvvcrypto_(state->counter)));

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_AUTO void sz_aes256_gcm_associate_rvvcrypto_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    vuint32m1_t const subkey_u32m1 = sz_aes256_block_load_rvvcrypto_(state->key.powers);
    vuint32m1_t accumulator_u32m1 = sz_aes256_block_load_rvvcrypto_(state->accumulator);
    sz_size_t consumed = 0;

    // Associated data is hashed but never encrypted, so a whole block hashes straight out of the
    // caller's buffer and only a partial one is completed in `partial`.
    while (consumed != length) {
        if (state->buffered == 0 && length - consumed >= SZ_AES_BLOCK_LENGTH) {
            accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(
                accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(input_bytes + consumed), subkey_u32m1);
            consumed += SZ_AES_BLOCK_LENGTH;
            state->associated_length += SZ_AES_BLOCK_LENGTH;
        }
        else {
            sz_size_t const wanted_bytes = SZ_AES_BLOCK_LENGTH - state->buffered;
            sz_size_t const taken_bytes = length - consumed < wanted_bytes ? length - consumed : wanted_bytes;
            for (sz_size_t byte_index = 0; byte_index != taken_bytes; ++byte_index)
                state->partial[state->buffered + byte_index] = input_bytes[consumed + byte_index];
            state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
            consumed += taken_bytes;
            state->associated_length += taken_bytes;
            if (state->buffered == SZ_AES_BLOCK_LENGTH) {
                accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(
                    accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(state->partial), subkey_u32m1);
                state->buffered = 0;
            }
        }
    }

    sz_aes256_block_store_rvvcrypto_(state->accumulator, accumulator_u32m1);
}

/**
 *  @brief Absorbs whatever `partial` holds, zero padded to a full block, and empties it.
 *  @param state The state, whose `buffered` count is cleared.
 *  @param accumulator_u32m1 The running hash.
 *  @param subkey_u32m1 The hash subkey `H`.
 *  @return The updated hash, unchanged when nothing was pending.
 */
SZ_HELPER_AUTO vuint32m1_t sz_aes256_gcm_flush_partial_rvvcrypto_(sz_aes256_gcm_state_t *state,
                                                                  vuint32m1_t accumulator_u32m1,
                                                                  vuint32m1_t subkey_u32m1) {
    sz_u128_vec_t padded_block_vec;
    sz_size_t byte_index = 0;
    if (state->buffered == 0) return accumulator_u32m1;
    for (; byte_index != (sz_size_t)state->buffered; ++byte_index)
        padded_block_vec.u8s[byte_index] = state->partial[byte_index];
    for (; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index) padded_block_vec.u8s[byte_index] = 0;
    state->buffered = 0;
    return sz_ghash_absorb_rvvcrypto_(accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(padded_block_vec.u8s),
                                      subkey_u32m1);
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *  @param direction Which buffer the hash absorbs.
 *
 *  Two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes, and neither may restart at
 *  a chunk boundary, so the wide path opens only where both are already on a block edge and closes the
 *  moment fewer than sixteen bytes remain. Everything else walks byte by byte through the carried
 *  keystream and hash blocks, which is what makes a five-byte chunk followed by an eleven-byte one spend
 *  exactly one keystream block.
 *
 *  The hash always eats ciphertext, which is the exclusive-or result when encrypting and the loaded
 *  input when decrypting. Both are held in registers across the store, because a caller may pass one
 *  pointer for both buffers and reading the ciphertext back out of the output would then hash bytes that
 *  the store had already replaced.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_rvvcrypto_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_size_t const maximum_vector_length = __riscv_vsetvlmax_e32m4();
    sz_size_t const maximum_blocks_per_pass = maximum_vector_length / 4;
    vuint32m4_t const broadcast_offsets_u32m4 = sz_aes256_broadcast_offsets_rvvcrypto_(maximum_vector_length);
    vuint32m1_t const subkey_u32m1 = sz_aes256_block_load_rvvcrypto_(state->key.powers);
    vuint32m1_t accumulator_u32m1 = sz_aes256_block_load_rvvcrypto_(state->accumulator);
    sz_size_t produced = 0;

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && length != 0)
        accumulator_u32m1 = sz_aes256_gcm_flush_partial_rvvcrypto_(state, accumulator_u32m1, subkey_u32m1);

    while (produced != length) {
        sz_size_t const remaining = length - produced;
        if (state->keystream_used == SZ_AES_BLOCK_LENGTH && state->buffered == 0 && remaining >= SZ_AES_BLOCK_LENGTH) {
            sz_u32_t const counter_index = sz_aes256_counter_index_load_rvvcrypto_(state->counter);
            sz_size_t blocks = remaining / SZ_AES_BLOCK_LENGTH;
            sz_size_t vector_length, byte_count, block_ordinal;
            vuint32m4_t counters_u32m4, keystream_u32m4, text_u32m4, result_u32m4, ciphertext_u32m4;
            if (blocks > maximum_blocks_per_pass) blocks = maximum_blocks_per_pass;
            vector_length = blocks * 4;
            byte_count = blocks * SZ_AES_BLOCK_LENGTH;

            counters_u32m4 = sz_aes256_counters_build_rvvcrypto_(state->counter, counter_index + 1u,
                                                                 broadcast_offsets_u32m4, vector_length);
            keystream_u32m4 = sz_aes256_blocks_encrypt_rvvcrypto_(&state->key.block, counters_u32m4,
                                                                  broadcast_offsets_u32m4, vector_length);
            text_u32m4 = __riscv_vreinterpret_v_u8m4_u32m4(__riscv_vle8_v_u8m4(input_bytes + produced, byte_count));
            result_u32m4 = __riscv_vxor_vv_u32m4(text_u32m4, keystream_u32m4, vector_length);
            __riscv_vse8_v_u8m4(output_bytes + produced, __riscv_vreinterpret_v_u32m4_u8m4(result_u32m4), byte_count);

            ciphertext_u32m4 = direction == sz_aes256_gcm_decrypting_k ? text_u32m4 : result_u32m4;
            for (block_ordinal = 0; block_ordinal != blocks; ++block_ordinal)
                accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(
                    accumulator_u32m1,
                    sz_aes256_group_extract_rvvcrypto_(ciphertext_u32m4, block_ordinal, vector_length), subkey_u32m1);

            // The carried keystream is spent, and a byte-wise pass would have left the last block it
            // derived sitting in the state, so leave the same one there.
            sz_aes256_block_store_rvvcrypto_(
                state->keystream, sz_aes256_group_extract_rvvcrypto_(keystream_u32m4, blocks - 1, vector_length));
            sz_aes256_counter_index_store_rvvcrypto_(state->counter, counter_index + (sz_u32_t)blocks);
            state->text_length += byte_count;
            produced += byte_count;
        }
        else {
            if (state->keystream_used == SZ_AES_BLOCK_LENGTH) {
                sz_u32_t const counter_index = sz_aes256_counter_index_load_rvvcrypto_(state->counter);
                sz_aes256_counter_index_store_rvvcrypto_(state->counter, counter_index + 1u);
                sz_aes256_block_store_rvvcrypto_(
                    state->keystream, sz_aes256_block_encrypt_rvvcrypto_(
                                          &state->key.block, sz_aes256_block_load_rvvcrypto_(state->counter)));
                state->keystream_used = 0;
            }
            {
                sz_u8_t const plaintext_byte = input_bytes[produced];
                sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
                sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
                output_bytes[produced] = transformed;
                state->partial[state->buffered++] = ciphertext;
                if (state->buffered == SZ_AES_BLOCK_LENGTH) {
                    accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(
                        accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(state->partial), subkey_u32m1);
                    state->buffered = 0;
                }
                ++state->keystream_used;
                ++state->text_length;
                ++produced;
            }
        }
    }

    sz_aes256_block_store_rvvcrypto_(state->accumulator, accumulator_u32m1);
}

/**
 *  @brief Folds the pending block and the length block into the hash, then masks it into a tag.
 *  @param state The finished state, left untouched.
 *  @param tag Receives the 16 authentication bytes.
 */
SZ_HELPER_AUTO void sz_aes256_gcm_digest_rvvcrypto_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    vuint32m1_t const subkey_u32m1 = sz_aes256_block_load_rvvcrypto_(state->key.powers);
    vuint32m1_t accumulator_u32m1 = sz_aes256_block_load_rvvcrypto_(state->accumulator);
    sz_u128_vec_t staged_block_vec;
    sz_size_t byte_index;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise. The state is not ours to modify, so the
    // pending bytes are padded in a local block rather than in `partial`.
    if (state->buffered != 0) {
        for (byte_index = 0; byte_index != (sz_size_t)state->buffered; ++byte_index)
            staged_block_vec.u8s[byte_index] = state->partial[byte_index];
        for (; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index) staged_block_vec.u8s[byte_index] = 0;
        accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(
            accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(staged_block_vec.u8s), subkey_u32m1);
    }

    sz_aes256_gcm_lengths_serial_(&staged_block_vec, state->associated_length, state->text_length);
    accumulator_u32m1 = sz_ghash_absorb_rvvcrypto_(accumulator_u32m1,
                                                   sz_aes256_block_load_rvvcrypto_(staged_block_vec.u8s), subkey_u32m1);
    accumulator_u32m1 = __riscv_vxor_vv_u32m1(accumulator_u32m1, sz_aes256_block_load_rvvcrypto_(state->tag_mask), 4);
    sz_aes256_block_store_rvvcrypto_(tag, accumulator_u32m1);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                            sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_rvvcrypto_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                 sz_size_t length) {
    sz_aes256_gcm_associate_rvvcrypto_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                              sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_rvvcrypto_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_rvvcrypto(sz_aes256_gcm_encryptor_t const *encryptor,
                                                              sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_rvvcrypto_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                            sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_rvvcrypto_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                 sz_size_t length) {
    sz_aes256_gcm_associate_rvvcrypto_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                         sz_cptr_t text, sz_size_t length,
                                                                         sz_ptr_t output) {
    sz_aes256_gcm_transform_rvvcrypto_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_rvvcrypto(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                     sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u8_t expected[SZ_AES_BLOCK_LENGTH];
    sz_aes256_gcm_digest_rvvcrypto_(&decryptor->state, expected);
    return sz_aes256_tag_equal_serial_(expected, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                     sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                     sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                     sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_rvvcrypto(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_rvvcrypto(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_rvvcrypto(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_rvvcrypto(&encryptor, tag);
    sz_aes256_gcm_state_scrub_rvvcrypto_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output,
                                                            sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_rvvcrypto(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_rvvcrypto(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_rvvcrypto(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_rvvcrypto(&decryptor, tag);
    sz_aes256_gcm_state_scrub_rvvcrypto_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVVCRYPTO

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_RVVCRYPTO_H_
