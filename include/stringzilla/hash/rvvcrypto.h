/**
 *  @file include/stringzilla/hash/rvvcrypto.h
 *  @author Ash Vardanian
 *  @date June 14, 2026
 *  @brief RISC-V Vector Crypto (Zvk) backend for hash: hardware AES & SHA-256.
 *
 *  This backend replaces the two emulated primitives of the base RVV 1.0 path in `hash/rvv.h`: the
 *  tower-field vector-permute AES round, @c sz_emulate_aesenc_rvv_, is swapped for a single
 *  @c Zvkned `vaesem.vv` instruction, and the serial SHA-256 block for the @c Zvknhb @c vsha2cl,
 *  @c vsha2ch and @c vsha2ms instructions. Both produce byte- and bit-identical results to the
 *  serial reference, so digests are interchangeable.
 *
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_RVVCRYPTO_H_
#define STRINGZILLA_HASH_RVVCRYPTO_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_RISCV64_
#if STRINGZILLA_TARGET_RVVCRYPTO

#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvkned,+zvknhb"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvkned,+zvknhb")
#endif

#pragma region RVV Crypto AES Round via Zvkned

/**
 *  @brief Bit-exact @c Zvkned implementation of a single @c _mm_aesenc_si128 round.
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, identical to
 *      @c sz_emulate_aesenc_si128_serial_ and to @c sz_emulate_aesenc_rvv_.
 *
 *  RISC-V's `vaesem.vv vd, vs2` computes `vd = MixColumns(SubBytes(ShiftRows(vd))) ^ vs2`, which is
 *  exactly the AES-NI @c AESENC operation: the state in @c vd, the round key in @c vs2. A single
 *  128-bit AES block is one element group, EGW = 128, so the operation runs with a vector length of
 *  four 32-bit lanes.
 */
STRINGZILLA_INLINE sz_u128_vec_t sz_emulate_aesenc_rvvcrypto_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    sz_size_t vector_length = __riscv_vsetvl_e32m1(4); // one 128-bit AES block = 4x u32 lanes
    vuint32m1_t state_u32m1 = __riscv_vle32_v_u32m1((sz_u32_t const *)state_vec.u32s, vector_length);
    vuint32m1_t key_u32m1 = __riscv_vle32_v_u32m1((sz_u32_t const *)round_key_vec.u32s, vector_length);
    state_u32m1 = __riscv_vaesem_vv_u32m1(state_u32m1, key_u32m1, vector_length);
    sz_u128_vec_t result_vec;
    __riscv_vse32_v_u32m1(result_vec.u32s, state_u32m1, vector_length);
    return result_vec;
}

#pragma endregion RVV Crypto AES Round via Zvkned

/*  These drivers mirror the serial and base RVV ones exactly, substituting
 *  @c sz_emulate_aesenc_rvvcrypto_ for the AES round. Every non-AES step reuses the shared serial
 *  helpers, so the digests are guaranteed value-identical to @c sz_hash_serial. */
#pragma region RVV Crypto Hash Drivers

STRINGZILLA_INLINE void sz_hash_state_short_update_rvvcrypto_(sz_hash_state_aligned_for_short_t *state,
                                                              sz_u128_vec_t block_vec) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_emulate_aesenc_rvvcrypto_(state->aes, block_vec);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block_vec.u64s[0], state->sum.u64s[1] += block_vec.u64s[1];
}

STRINGZILLA_INLINE sz_u64_t sz_hash_state_short_finalize_rvvcrypto_(sz_hash_state_aligned_for_short_t const *state,
                                                                    sz_size_t length) {
    sz_u128_vec_t key_with_length_vec = state->key;
    key_with_length_vec.u64s[0] += length;
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_rvvcrypto_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_rvvcrypto_(
        sz_emulate_aesenc_rvvcrypto_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}

STRINGZILLA_INLINE void sz_hash_state_absorb_rvvcrypto_(sz_hash_state_aligned_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        state->aes.u128s[lane_index] = sz_emulate_aesenc_rvvcrypto_(state->aes.u128s[lane_index],
                                                                    state->ins.u128s[lane_index]);
        state->sum.u128s[lane_index] = sz_emulate_shuffle_epi8_serial_(state->sum.u128s[lane_index], shuffle);
        state->sum.u128s[lane_index].u64s[0] += state->ins.u128s[lane_index].u64s[0];
        state->sum.u128s[lane_index].u64s[1] += state->ins.u128s[lane_index].u64s[1];
    }
}

STRINGZILLA_INLINE sz_u64_t sz_hash_state_finalize_rvvcrypto_(sz_hash_state_aligned_t state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t key_with_length_vec;
    key_with_length_vec.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length_vec.u64s[1] = state.key.u64s[1];

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization.
    sz_u128_vec_t aes0_vec = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[0], state.ins.u128s[0]);
    sz_u128_vec_t aes1_vec = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[1], state.ins.u128s[1]);
    sz_u128_vec_t aes2_vec = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[2], state.ins.u128s[2]);
    sz_u128_vec_t aes3_vec = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[3], state.ins.u128s[3]);
    sz_u128_vec_t sum0_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[0], shuffle);
    sz_u128_vec_t sum1_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[1], shuffle);
    sz_u128_vec_t sum2_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[2], shuffle);
    sz_u128_vec_t sum3_vec = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[3], shuffle);
    sum0_vec.u64s[0] += state.ins.u128s[0].u64s[0], sum0_vec.u64s[1] += state.ins.u128s[0].u64s[1];
    sum1_vec.u64s[0] += state.ins.u128s[1].u64s[0], sum1_vec.u64s[1] += state.ins.u128s[1].u64s[1];
    sum2_vec.u64s[0] += state.ins.u128s[2].u64s[0], sum2_vec.u64s[1] += state.ins.u128s[2].u64s[1];
    sum3_vec.u64s[0] += state.ins.u128s[3].u64s[0], sum3_vec.u64s[1] += state.ins.u128s[3].u64s[1];

    sz_u128_vec_t mixed0_vec = sz_emulate_aesenc_rvvcrypto_(sum0_vec, aes0_vec);
    sz_u128_vec_t mixed1_vec = sz_emulate_aesenc_rvvcrypto_(sum1_vec, aes1_vec);
    sz_u128_vec_t mixed2_vec = sz_emulate_aesenc_rvvcrypto_(sum2_vec, aes2_vec);
    sz_u128_vec_t mixed3_vec = sz_emulate_aesenc_rvvcrypto_(sum3_vec, aes3_vec);

    sz_u128_vec_t mixed01_vec = sz_emulate_aesenc_rvvcrypto_(mixed0_vec, mixed1_vec);
    sz_u128_vec_t mixed23_vec = sz_emulate_aesenc_rvvcrypto_(mixed2_vec, mixed3_vec);
    sz_u128_vec_t mixed_vec = sz_emulate_aesenc_rvvcrypto_(mixed01_vec, mixed23_vec);

    sz_u128_vec_t mixed_in_register_vec = sz_emulate_aesenc_rvvcrypto_(
        sz_emulate_aesenc_rvvcrypto_(mixed_vec, key_with_length_vec), mixed_vec);
    return mixed_in_register_vec.u64s[0];
}

/** Vector-copy a single AES block (`sizeof(sz_u128_vec_t)` bytes) from @p source into
 *  `target_vec->u8s`, replacing a scalar byte loop. @p source must have a full block
 *  of readable bytes. */
STRINGZILLA_INLINE void sz_hash_load_block_rvvcrypto_(sz_u128_vec_t *target_vec, sz_cptr_t source) {
    sz_size_t vector_length = __riscv_vsetvl_e8m1(sizeof(target_vec->u8s));
    __riscv_vse8_v_u8m1(target_vec->u8s, __riscv_vle8_v_u8m1((sz_u8_t const *)source, vector_length), vector_length);
}

/** Vector-copy a single AES block (`sizeof(sz_u128_vec_t)` bytes) from @p source_vec to
 *  @p target, the store counterpart of @c sz_hash_load_block_rvvcrypto_. @p target must have a
 *  full block of writable bytes. */
STRINGZILLA_INLINE void sz_hash_store_block_rvvcrypto_(sz_ptr_t target, sz_u128_vec_t source_vec) {
    sz_size_t vector_length = __riscv_vsetvl_e8m1(sizeof(source_vec.u8s));
    __riscv_vse8_v_u8m1((sz_u8_t *)target, __riscv_vle8_v_u8m1(source_vec.u8s, vector_length), vector_length);
}

/** Loads the packed public state into the aligned internal twin (one @c vle8 block
 *  per 16-byte lane). */
STRINGZILLA_INLINE sz_hash_state_aligned_t sz_hash_state_load_rvvcrypto_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state;
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        sz_hash_load_block_rvvcrypto_(&state.aes.u128s[lane_index], (sz_cptr_t)(packed->aes + offset));
        sz_hash_load_block_rvvcrypto_(&state.sum.u128s[lane_index], (sz_cptr_t)(packed->sum + offset));
        sz_hash_load_block_rvvcrypto_(&state.ins.u128s[lane_index], (sz_cptr_t)(packed->ins + offset));
    }
    sz_hash_load_block_rvvcrypto_(&state.key, (sz_cptr_t)packed->key);
    state.ins_length = packed->ins_length;
    return state;
}

/** Stores the aligned internal twin back into the packed public state (one @c vse8 block
 *  per 16-byte lane). */
STRINGZILLA_INLINE void sz_hash_state_store_rvvcrypto_(sz_hash_state_t *packed, sz_hash_state_aligned_t const *state) {
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->aes + offset), state->aes.u128s[lane_index]);
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->sum + offset), state->sum.u128s[lane_index]);
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->ins + offset), state->ins.u128s[lane_index]);
    }
    sz_hash_store_block_rvvcrypto_((sz_ptr_t)packed->key, state->key);
    packed->ins_length = state->ins_length;
}

STRINGZILLA_INLINE STRINGZILLA_NO_STACK_PROTECTOR_ sz_u64_t sz_hash_rvvcrypto_(sz_cptr_t start, sz_size_t length,
                                                                               sz_u64_t seed) {
    sz_size_t const block = sizeof(sz_u128_vec_t); // one AES block
    if (length <= block) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        // A length-agnostic load drops the zero pad and the scalar byte loop: VL covers the partial bytes.
        sz_size_t vector_length = __riscv_vsetvl_e8m1(length);
        __riscv_vse8_v_u8m1(data_vec.u8s, __riscv_vle8_v_u8m1((sz_u8_t const *)start, vector_length), vector_length);
        sz_hash_state_short_update_rvvcrypto_(&state, data_vec);
        return sz_hash_state_short_finalize_rvvcrypto_(&state, length);
    }
    else if (length <= 2 * block) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        sz_hash_load_block_rvvcrypto_(&data0_vec, start);
        sz_hash_load_block_rvvcrypto_(&data1_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(2 * block - length));
        sz_hash_state_short_update_rvvcrypto_(&state, data0_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data1_vec);
        return sz_hash_state_short_finalize_rvvcrypto_(&state, length);
    }
    else if (length <= 3 * block) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        sz_hash_load_block_rvvcrypto_(&data0_vec, start);
        sz_hash_load_block_rvvcrypto_(&data1_vec, start + block);
        sz_hash_load_block_rvvcrypto_(&data2_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(3 * block - length));
        sz_hash_state_short_update_rvvcrypto_(&state, data0_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data1_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data2_vec);
        return sz_hash_state_short_finalize_rvvcrypto_(&state, length);
    }
    else if (length <= 4 * block) {
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        sz_hash_load_block_rvvcrypto_(&data0_vec, start);
        sz_hash_load_block_rvvcrypto_(&data1_vec, start + block);
        sz_hash_load_block_rvvcrypto_(&data2_vec, start + 2 * block);
        sz_hash_load_block_rvvcrypto_(&data3_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(4 * block - length));
        sz_hash_state_short_update_rvvcrypto_(&state, data0_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data1_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data2_vec);
        sz_hash_state_short_update_rvvcrypto_(&state, data3_vec);
        return sz_hash_state_short_finalize_rvvcrypto_(&state, length);
    }
    else {
        // The aligned twin lets the kernels use clean aligned lane access; one-shot never touches the packed type
        // except to reuse `init` (layout-locked by the `static_assert`s on `sz_hash_state_aligned_t`).
        sz_align_(64) sz_hash_state_aligned_t state;
        sz_size_t const window = sizeof(state.ins.u8s); // the 64-byte hashing window
        sz_hash_state_init_serial_((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte window except the last; the final block (a full 64 or a partial
        // tail) stays buffered in `ins` for `sz_hash_state_finalize_rvvcrypto_` to fold - the same
        // deferral the streaming path uses.
        for (; state.ins_length + window < length; state.ins_length += window) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(window); // VLEN >= 128 -> one whole-window transfer
            __riscv_vse8_v_u8m8(state.ins.u8s,
                                __riscv_vle8_v_u8m8((sz_u8_t const *)start + state.ins_length, vector_length),
                                vector_length);
            sz_hash_state_absorb_rvvcrypto_(&state);
        }

        // Stage the final [ins_length, length) bytes (1..64) into a zeroed window; finalize folds them.
        sz_size_t zero_vl = __riscv_vsetvl_e8m8(window);
        __riscv_vse8_v_u8m8(state.ins.u8s, __riscv_vmv_v_x_u8m8(0, zero_vl), zero_vl);
        sz_size_t tail_vl = __riscv_vsetvl_e8m8(length - state.ins_length); // VL covers the final block natively
        __riscv_vse8_v_u8m8(state.ins.u8s, __riscv_vle8_v_u8m8((sz_u8_t const *)start + state.ins_length, tail_vl),
                            tail_vl);
        state.ins_length = length;
        return sz_hash_state_finalize_rvvcrypto_(state);
    }
}

STRINGZILLA_API STRINGZILLA_NO_STACK_PROTECTOR_ sz_status_t sz_hash_rvvcrypto(sz_cptr_t start, sz_size_t length,
                                                                              sz_u64_t seed, sz_u64_t *hash,
                                                                              sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *hash = sz_hash_rvvcrypto_(start, length, seed);
    return sz_success_k;
}

/**
 *  @brief Replays prepared text-lanes through the minimal AES state of every seed at once.
 *  @param[in] text_lanes_vec Text-lanes from @c sz_hash_multiseed_prepare_serial_.
 *  @param[in] text_lanes_count Number of populated text-lanes.
 *  @param[in] length Original byte length, folded into each digest.
 *  @param[in] seeds The seeds, one output per seed.
 *  @param[in] seeds_count Number of seeds.
 *  @param[out] hashes Receives one 64-bit hash per seed.
 *
 *  Each seed owns one 128-bit element group, holding its @c aes, @c sum and @c key exactly as
 *  @c sz_hash_state_aligned_for_short_t holds them for one seed, so every instruction advances
 *  a whole register of seeds. The text-lane is the same for every seed and reaches all groups
 *  through one indexed load, which feeds both the @c vaesem.vv round and the 64-bit additions,
 *  avoiding the @c vaesem.vs form that GCC 14 lowers with the vector length of the key operand.
 */
STRINGZILLA_INLINE void sz_hash_multiseed_replay_rvvcrypto_(sz_u512_vec_t const *text_lanes_vec,
                                                            sz_size_t text_lanes_count, sz_size_t length,
                                                            sz_u64_t const *seeds, sz_size_t seeds_count,
                                                            sz_u64_t *hashes) {
    sz_u64_t const *pi = sz_hash_pi_constants_();
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_size_t const seeds_per_register = __riscv_vsetvlmax_e64m2() / 2;

    // Byte `i` of the `sum` takes byte `shuffle[i % 16]` of its own group, via 16-bit indices.
    sz_size_t const bytes_capacity = __riscv_vsetvlmax_e8m2();
    vuint16m4_t const byte_index_u16m4 = __riscv_vid_v_u16m4(bytes_capacity);
    vuint8m2_t const shuffle_u8m2 = __riscv_vluxei16_v_u8m2(
        shuffle, __riscv_vand_vx_u16m4(byte_index_u16m4, 15, bytes_capacity), bytes_capacity);
    vuint16m4_t const sum_gather_u16m4 = __riscv_vadd_vv_u16m4(
        __riscv_vzext_vf2_u16m4(shuffle_u8m2, bytes_capacity),
        __riscv_vand_vx_u16m4(byte_index_u16m4, (sz_u16_t)~15u, bytes_capacity), bytes_capacity);

    // Even words are the low halves of the groups, odd words the high halves.
    sz_size_t const words_capacity = __riscv_vsetvlmax_e64m2();
    vuint64m2_t const word_index_u64m2 = __riscv_vid_v_u64m2(words_capacity);
    vuint64m2_t const parity_u64m2 = __riscv_vand_vx_u64m2(word_index_u64m2, 1, words_capacity);
    vbool32_t const high_b32 = __riscv_vmsne_vx_u64m2_b32(parity_u64m2, 0, words_capacity);
    vbool32_t const low_b32 = __riscv_vmseq_vx_u64m2_b32(parity_u64m2, 0, words_capacity);
    vuint64m2_t const block_offsets_u64m2 = __riscv_vsll_vx_u64m2(parity_u64m2, 3, words_capacity);
    vuint64m2_t const seed_offsets_u64m2 = __riscv_vsll_vx_u64m2(
        __riscv_vsrl_vx_u64m2(word_index_u64m2, 1, words_capacity), 3, words_capacity);
    vuint64m2_t const aes_pi_u64m2 = __riscv_vmerge_vxm_u64m2(__riscv_vmv_v_x_u64m2(pi[0], words_capacity), pi[1],
                                                              high_b32, words_capacity);
    vuint64m2_t const sum_pi_u64m2 = __riscv_vmerge_vxm_u64m2(__riscv_vmv_v_x_u64m2(pi[8], words_capacity), pi[9],
                                                              high_b32, words_capacity);
    vuint64m2_t const length_u64m2 = __riscv_vmerge_vxm_u64m2(__riscv_vmv_v_x_u64m2(length, words_capacity), 0,
                                                              high_b32, words_capacity);

    for (sz_size_t first_seed_index = 0; first_seed_index < seeds_count; first_seed_index += seeds_per_register) {
        sz_size_t const group_seeds_count = sz_min_of_two(seeds_count - first_seed_index, seeds_per_register);
        sz_size_t const words_length = group_seeds_count * 2;
        sz_size_t const state_length = group_seeds_count * 4;
        sz_size_t const bytes_length = group_seeds_count * 16;

        // The key is the seed in both halves, as `sz_hash_state_short_init_serial_` builds it.
        vuint64m2_t const key_u64m2 = __riscv_vluxei64_v_u64m2(seeds + first_seed_index, seed_offsets_u64m2,
                                                               words_length);
        vuint32m2_t aes_u32m2 = __riscv_vreinterpret_v_u64m2_u32m2(
            __riscv_vxor_vv_u64m2(key_u64m2, aes_pi_u64m2, words_length));
        vuint64m2_t sum_u64m2 = __riscv_vxor_vv_u64m2(key_u64m2, sum_pi_u64m2, words_length);

        for (sz_size_t lane_index = 0; lane_index < text_lanes_count; ++lane_index) {
            vuint64m2_t const block_u64m2 = __riscv_vluxei64_v_u64m2(text_lanes_vec->u128s[lane_index].u64s,
                                                                     block_offsets_u64m2, words_length);
            aes_u32m2 = __riscv_vaesem_vv_u32m2(aes_u32m2, __riscv_vreinterpret_v_u64m2_u32m2(block_u64m2),
                                                state_length);
            vuint8m2_t const shuffled_u8m2 = __riscv_vrgatherei16_vv_u8m2(__riscv_vreinterpret_v_u64m2_u8m2(sum_u64m2),
                                                                          sum_gather_u16m4, bytes_length);
            sum_u64m2 = __riscv_vadd_vv_u64m2(__riscv_vreinterpret_v_u8m2_u64m2(shuffled_u8m2), block_u64m2,
                                              words_length);
        }

        vuint32m2_t const key_with_length_u32m2 = __riscv_vreinterpret_v_u64m2_u32m2(
            __riscv_vadd_vv_u64m2(key_u64m2, length_u64m2, words_length));
        vuint32m2_t const mixed_u32m2 = __riscv_vaesem_vv_u32m2(__riscv_vreinterpret_v_u64m2_u32m2(sum_u64m2),
                                                                aes_u32m2, state_length);
        vuint32m2_t const mixed_in_register_u32m2 = __riscv_vaesem_vv_u32m2(
            __riscv_vaesem_vv_u32m2(mixed_u32m2, key_with_length_u32m2, state_length), mixed_u32m2, state_length);
        vuint64m2_t const low_halves_u64m2 = __riscv_vcompress_vm_u64m2(
            __riscv_vreinterpret_v_u32m2_u64m2(mixed_in_register_u32m2), low_b32, words_length);
        __riscv_vse64_v_u64m2(hashes + first_seed_index, low_halves_u64m2, group_seeds_count);
    }
}

STRINGZILLA_API sz_status_t sz_hash_multiseed_rvvcrypto(sz_cptr_t text, sz_size_t length,             //
                                                        sz_u64_t const *seeds, sz_size_t seeds_count, //
                                                        sz_u64_t *hashes, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    if (length <= 64 && seeds_count > 1) {
        sz_u512_vec_t text_lanes_vec;
        sz_size_t const text_lanes_count = sz_hash_multiseed_prepare_serial_(text, length, &text_lanes_vec);
        sz_hash_multiseed_replay_rvvcrypto_(&text_lanes_vec, text_lanes_count, length, seeds, seeds_count, hashes);
    }
    else {
        for (sz_size_t seed_index = 0; seed_index < seeds_count; ++seed_index)
            hashes[seed_index] = sz_hash_rvvcrypto_(text, length, seeds[seed_index]);
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_init_rvvcrypto(sz_hash_state_t *state, sz_u64_t seed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_hash_state_init_serial_(state, seed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_update_rvvcrypto(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length,
                                                           sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Load the packed public state (any alignment) into an aligned twin once, buffer/absorb on it, then store back.
    // `ins` is exactly one 64-byte window. Track how many bytes it holds and absorb it only once it becomes
    // interior (more bytes arrive - the deferral `digest` needs to choose minimal/full by total length). The
    // deferred trailing block reads back as `ins_length % 64 == 0 && ins_length != 0`; treat that as `buffered ==
    // 64`. The append uses `vl` as the mask, touching only `[buffered, buffered+take)`, and we re-zero `ins` after
    // each absorb so the high lanes stay zero-padded for `finalize` to fold a clean trailing block.
    sz_hash_state_aligned_t state = sz_hash_state_load_rvvcrypto_(packed);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it and re-zero the buffer
            sz_hash_state_absorb_rvvcrypto_(&state);
            sz_size_t const clear_vl = __riscv_vsetvl_e8m8(sizeof(state.ins.u8s));
            __riscv_vse8_v_u8m8(state.ins.u8s, __riscv_vmv_v_x_u8m8(0, clear_vl), clear_vl);
            buffered = 0;
        }
        sz_size_t const take = sz_min_of_two(length, (sz_size_t)64 - buffered);
        sz_size_t const vl = __riscv_vsetvl_e8m8(take); // VL is the mask: `e8m8` spans the whole 64-byte window
        __riscv_vse8_v_u8m8(state.ins.u8s + buffered, __riscv_vle8_v_u8m8((sz_u8_t const *)text, vl), vl);
        buffered += take, text += take, length -= take, state.ins_length += take;
    }
    sz_hash_state_store_rvvcrypto_(packed, &state);
    return sz_success_k;
}

STRINGZILLA_INLINE sz_u64_t sz_hash_state_digest_rvvcrypto_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t state = sz_hash_state_load_rvvcrypto_(packed);
    sz_size_t length = state.ins_length;
    // Inputs longer than one block fold through the full four-lane state, where the deferred final block buffered
    // in `ins` is folded by `sz_hash_state_finalize_rvvcrypto_`. A length of exactly 64 uses the minimal (<=64)
    // path below - matching one-shot `sz_hash`, whose `length <= 64` ladder also stays minimal.
    if (length > 64) return sz_hash_state_finalize_rvvcrypto_(state);

    // Switch back to a smaller "short" state for small inputs; the aligned twin lanes are read directly.
    sz_hash_state_aligned_for_short_t minimal_state;
    minimal_state.key = state.key;
    minimal_state.aes = state.aes.u128s[0];
    minimal_state.sum = state.sum.u128s[0];
    if (length <= 16) {
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[0]);
        return sz_hash_state_short_finalize_rvvcrypto_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[1]);
        return sz_hash_state_short_finalize_rvvcrypto_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[2]);
        return sz_hash_state_short_finalize_rvvcrypto_(&minimal_state, length);
    }
    else {
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[0]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[1]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[2]);
        sz_hash_state_short_update_rvvcrypto_(&minimal_state, state.ins.u128s[3]);
        return sz_hash_state_short_finalize_rvvcrypto_(&minimal_state, length);
    }
}

STRINGZILLA_API sz_status_t sz_hash_state_digest_rvvcrypto(sz_hash_state_t const *packed, sz_u64_t *hash,
                                                           sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *hash = sz_hash_state_digest_rvvcrypto_(packed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_fill_random_rvvcrypto(sz_ptr_t target, sz_size_t length, sz_u64_t nonce,
                                                     sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        generated_vec = sz_emulate_aesenc_rvvcrypto_(input_vec, key_vec);
        // VL deletes both the per-byte loop and the `&& length` tail guard: it clamps to the bytes left,
        // capped at the block we just generated.
        sz_size_t out_vl = __riscv_vsetvl_e8m1(sz_min_of_two(length, sizeof(generated_vec.u8s)));
        __riscv_vse8_v_u8m1((sz_u8_t *)target, __riscv_vle8_v_u8m1(generated_vec.u8s, out_vl), out_vl);
        target += out_vl, length -= out_vl;
    }
    return sz_success_k;
}

#pragma endregion RVV Crypto Hash Drivers

#pragma region RVV Crypto SHA 256 via Zvknhb

/**
 *  @brief Process a single 512-bit, 64-byte block of data using SHA-256 via @c Zvknhb.
 *  @param[inout] hash Pointer to 8x 32-bit hash values {a,b,c,d,e,f,g,h}, modified in place.
 *  @param[in] block Pointer to a 64-byte message block.
 *
 *  The @c Zvknh state is held in two 128-bit element groups, each four 32-bit lanes. Per the RISC-V
 *  Vector Crypto spec, `vsha2c[hl].vv` reads `vs2 = {a,b,e,f}` and `vd = {c,d,g,h}`, where
 *  `{x@y@z@w}` packs @c w into lane 0 and @c x into lane 3, little-endian, and writes the next
 *  `{a,b,e,f}` back to @c vd. Each instruction performs two compression rounds: @c cl consumes the
 *  low two W+K words of the group, @c ch the high two. `vsha2ms.vv` expands four message-schedule
 *  words per call.
 *
 *  The intrinsic argument order matches @c sha256_block_data_order_zvkb_zvknha_or_zvknhb of
 *  OpenSSL: `cdgh = vsha2cl(cdgh, abef, k_plus_w)` then `abef = vsha2ch(abef, cdgh, k_plus_w)` per
 *  quad-round, and `wN = vsha2ms(wN, vmerge(w_older, w_newer, mask_lane0), w_newest)` to roll the
 *  schedule forward. SHA-256 words are big-endian; we load them with a scalar byte-swap so this
 *  path needs only the @c Zvknhb extension, not the @c vrev8 of @c Zvbb or @c Zvkb.
 */
STRINGZILLA_INLINE void sz_sha256_process_block_rvvcrypto_(
    sz_u32_t hash[sz_at_least_(8)], sz_u8_t const block[sz_at_least_(STRINGZILLA_SHA256_BLOCK_LENGTH)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_size_t const vector_length = __riscv_vsetvl_e32m1(4);

    // Lane-0 selection mask for the message-schedule `vmerge` (replace the oldest word of the group).
    sz_align_(16) sz_u32_t const mask_seed[4] = {1, 0, 0, 0};
    vbool32_t const lane0_mask_b32 = __riscv_vmsne_vx_u32m1_b32(__riscv_vle32_v_u32m1(mask_seed, vector_length), 0,
                                                                vector_length);

    // Build the two state vectors: abef = {f,e,b,a} (lane0..3), cdgh = {h,g,d,c}.
    sz_align_(16) sz_u32_t const abef_seed[4] = {hash[5], hash[4], hash[1], hash[0]};
    sz_align_(16) sz_u32_t const cdgh_seed[4] = {hash[7], hash[6], hash[3], hash[2]};
    vuint32m1_t abef_u32m1 = __riscv_vle32_v_u32m1(abef_seed, vector_length);
    vuint32m1_t cdgh_u32m1 = __riscv_vle32_v_u32m1(cdgh_seed, vector_length);
    vuint32m1_t const abef_saved_u32m1 = abef_u32m1;
    vuint32m1_t const cdgh_saved_u32m1 = cdgh_u32m1;

    // Big-endian load of the 16 message words (scalar swap keeps us within `Zvknhb` only).
    sz_align_(64) sz_u32_t message_words[16];
    for (sz_size_t word_index = 0; word_index < 16; ++word_index)
        message_words[word_index] = ((sz_u32_t)block[word_index * 4 + 0] << 24) |
                                    ((sz_u32_t)block[word_index * 4 + 1] << 16) |
                                    ((sz_u32_t)block[word_index * 4 + 2] << 8) |
                                    ((sz_u32_t)block[word_index * 4 + 3] << 0);

    // Four schedule vectors hold {W3,W2,W1,W0}, {W7..W4}, {W11..W8}, {W15..W12}. RVV sizeless
    // types cannot live in an array, so each quad-round consumes `w0` and rotates the four locals.
    vuint32m1_t w0_u32m1 = __riscv_vle32_v_u32m1(&message_words[0], vector_length);
    vuint32m1_t w1_u32m1 = __riscv_vle32_v_u32m1(&message_words[4], vector_length);
    vuint32m1_t w2_u32m1 = __riscv_vle32_v_u32m1(&message_words[8], vector_length);
    vuint32m1_t w3_u32m1 = __riscv_vle32_v_u32m1(&message_words[12], vector_length);

    // Quad-rounds 0..11 compress (`cl` then `ch`) and extend the schedule via `ms`: the lane-0
    // merge of `w1` into `w2` builds {W11,W10,W9,W4}, and `w3` supplies {W15,W14,-,W12}.
    for (sz_size_t round_index = 0; round_index != 48; round_index += 4) {
        vuint32m1_t const round_key_u32m1 = __riscv_vadd_vv_u32m1(
            __riscv_vle32_v_u32m1(&round_constants[round_index], vector_length), w0_u32m1, vector_length);
        cdgh_u32m1 = __riscv_vsha2cl_vv_u32m1(cdgh_u32m1, abef_u32m1, round_key_u32m1, vector_length);
        abef_u32m1 = __riscv_vsha2ch_vv_u32m1(abef_u32m1, cdgh_u32m1, round_key_u32m1, vector_length);
        vuint32m1_t const merged_u32m1 = __riscv_vmerge_vvm_u32m1(w2_u32m1, w1_u32m1, lane0_mask_b32, vector_length);
        vuint32m1_t const extended_u32m1 = __riscv_vsha2ms_vv_u32m1(w0_u32m1, merged_u32m1, w3_u32m1, vector_length);
        w0_u32m1 = w1_u32m1, w1_u32m1 = w2_u32m1, w2_u32m1 = w3_u32m1, w3_u32m1 = extended_u32m1;
    }

    // Quad-rounds 12..15 only compress; every schedule word they consume already exists.
    for (sz_size_t round_index = 48; round_index != 64; round_index += 4) {
        vuint32m1_t const round_key_u32m1 = __riscv_vadd_vv_u32m1(
            __riscv_vle32_v_u32m1(&round_constants[round_index], vector_length), w0_u32m1, vector_length);
        cdgh_u32m1 = __riscv_vsha2cl_vv_u32m1(cdgh_u32m1, abef_u32m1, round_key_u32m1, vector_length);
        abef_u32m1 = __riscv_vsha2ch_vv_u32m1(abef_u32m1, cdgh_u32m1, round_key_u32m1, vector_length);
        w0_u32m1 = w1_u32m1, w1_u32m1 = w2_u32m1, w2_u32m1 = w3_u32m1;
    }

    // Add the compressed working state back into the running hash.
    abef_u32m1 = __riscv_vadd_vv_u32m1(abef_saved_u32m1, abef_u32m1, vector_length);
    cdgh_u32m1 = __riscv_vadd_vv_u32m1(cdgh_saved_u32m1, cdgh_u32m1, vector_length);

    // Unpack abef = {f,e,b,a}, cdgh = {h,g,d,c} back into {a,b,c,d,e,f,g,h}.
    sz_align_(16) sz_u32_t abef_out[4];
    sz_align_(16) sz_u32_t cdgh_out[4];
    __riscv_vse32_v_u32m1(abef_out, abef_u32m1, vector_length);
    __riscv_vse32_v_u32m1(cdgh_out, cdgh_u32m1, vector_length);
    hash[5] = abef_out[0], hash[4] = abef_out[1], hash[1] = abef_out[2], hash[0] = abef_out[3];
    hash[7] = cdgh_out[0], hash[6] = cdgh_out[1], hash[3] = cdgh_out[2], hash[2] = cdgh_out[3];
}

STRINGZILLA_API sz_status_t sz_sha256_state_init_rvvcrypto(sz_sha256_state_t *state_ptr, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    // Copy all 8 initial words in halves of 4: `vsetvl_e32m1(8)` would clamp to 4 lanes at VLEN=128,
    // silently leaving the upper four words uninitialized, so the half-width copy is VLEN-independent.
    sz_size_t const vector_length = __riscv_vsetvl_e32m1(4);
    __riscv_vse32_v_u32m1(state_ptr->hash + 0, __riscv_vle32_v_u32m1(initial_hash + 0, vector_length), vector_length);
    __riscv_vse32_v_u32m1(state_ptr->hash + 4, __riscv_vle32_v_u32m1(initial_hash + 4, vector_length), vector_length);
    state_ptr->block_length = 0, state_ptr->total_length = 0;
    return sz_success_k;
}

STRINGZILLA_INLINE void sz_sha256_state_update_rvvcrypto_(sz_sha256_state_t *state_ptr, sz_cptr_t text,
                                                          sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)text;
    sz_size_t const current_block_index = state_ptr->block_length / STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / STRINGZILLA_SHA256_BLOCK_LENGTH;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % STRINGZILLA_SHA256_BLOCK_LENGTH == 0;

    state_ptr->total_length += length;

    // Fast path: stays in same block and doesn't fill it
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    // Calculate head, body, and tail lengths
    sz_size_t const head_length = (STRINGZILLA_SHA256_BLOCK_LENGTH - state_ptr->block_length) %
                                  STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const tail_length = (state_ptr->block_length + length) % STRINGZILLA_SHA256_BLOCK_LENGTH;
    sz_size_t const body_length = length - head_length - tail_length;

    // Copy hash to aligned local buffer
    sz_align_(32) sz_u32_t hash[8];
    hash[0] = state_ptr->hash[0], hash[1] = state_ptr->hash[1], hash[2] = state_ptr->hash[2],
    hash[3] = state_ptr->hash[3], hash[4] = state_ptr->hash[4], hash[5] = state_ptr->hash[5],
    hash[6] = state_ptr->hash[6], hash[7] = state_ptr->hash[7];

    // Process head to complete the current block
    if (head_length) {
        for (sz_size_t byte_index = 0; byte_index < head_length; ++byte_index)
            state_ptr->block[state_ptr->block_length++] = input[byte_index];
        sz_sha256_process_block_rvvcrypto_(hash, state_ptr->block);
        state_ptr->block_length = 0;
        input += head_length;
    }

    // Process body (complete aligned blocks)
    for (sz_size_t processed = 0; processed < body_length;
         processed += STRINGZILLA_SHA256_BLOCK_LENGTH, input += STRINGZILLA_SHA256_BLOCK_LENGTH)
        sz_sha256_process_block_rvvcrypto_(hash, input);

    // Process tail (remaining bytes into block buffer)
    for (sz_size_t byte_index = 0; byte_index < tail_length; ++byte_index)
        state_ptr->block[byte_index] = input[byte_index];
    state_ptr->block_length = tail_length;

    // Copy hash back
    state_ptr->hash[0] = hash[0], state_ptr->hash[1] = hash[1], state_ptr->hash[2] = hash[2],
    state_ptr->hash[3] = hash[3];
    state_ptr->hash[4] = hash[4], state_ptr->hash[5] = hash[5], state_ptr->hash[6] = hash[6],
    state_ptr->hash[7] = hash[7];
}

STRINGZILLA_API sz_status_t sz_sha256_state_update_rvvcrypto(sz_sha256_state_t *state_ptr, sz_cptr_t text,
                                                             sz_size_t length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_sha256_state_update_rvvcrypto_(state_ptr, text, length);
    return sz_success_k;
}

STRINGZILLA_INLINE void sz_sha256_state_digest_rvvcrypto_(
    sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80 byte) after the message
    state.block[state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (state.block_length > 56) {
        sz_size_t remaining = STRINGZILLA_SHA256_BLOCK_LENGTH - state.block_length;
        for (sz_size_t byte_index = 0; byte_index < remaining; ++byte_index)
            state.block[state.block_length + byte_index] = 0;
        sz_sha256_process_block_rvvcrypto_(state.hash, state.block);
        state.block_length = 0;
    }

    // Pad with zeros until we have 56 bytes
    sz_size_t remaining = 56 - state.block_length;
    for (sz_size_t byte_index = 0; byte_index < remaining; ++byte_index)
        state.block[state.block_length + byte_index] = 0;
    state.block_length = 56;

    // Append the message length in bits as a 64-bit big-endian integer
    sz_u64_t bit_length = state.total_length * 8;
    state.block[56] = (sz_u8_t)(bit_length >> 56);
    state.block[57] = (sz_u8_t)(bit_length >> 48);
    state.block[58] = (sz_u8_t)(bit_length >> 40);
    state.block[59] = (sz_u8_t)(bit_length >> 32);
    state.block[60] = (sz_u8_t)(bit_length >> 24);
    state.block[61] = (sz_u8_t)(bit_length >> 16);
    state.block[62] = (sz_u8_t)(bit_length >> 8);
    state.block[63] = (sz_u8_t)(bit_length >> 0);

    // Process the final block
    sz_sha256_process_block_rvvcrypto_(state.hash, state.block);

    // Produce the final hash digest in big-endian format
    for (sz_size_t lane_index = 0; lane_index < 8; ++lane_index) {
        digest[lane_index * 4 + 0] = (sz_u8_t)(state.hash[lane_index] >> 24);
        digest[lane_index * 4 + 1] = (sz_u8_t)(state.hash[lane_index] >> 16);
        digest[lane_index * 4 + 2] = (sz_u8_t)(state.hash[lane_index] >> 8);
        digest[lane_index * 4 + 3] = (sz_u8_t)(state.hash[lane_index] >> 0);
    }
}

STRINGZILLA_API sz_status_t sz_sha256_state_digest_rvvcrypto(
    sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
    sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_sha256_state_digest_rvvcrypto_(state_ptr, digest);
    return sz_success_k;
}

/**
 *  @brief Replicates the 64 round constants across 16 lanes, one quad-round per 64 words, so the
 *      batched rounds read them with unit-stride loads.
 *  @param[out] broadcast_round_constants Receives 1024 words: constant 4q + j lands at index
 *      64q + 4 * lane + j.
 *  @param[in] vector_length Four 32-bit elements per lane, at most 64.
 */
STRINGZILLA_INLINE void sz_sha256_broadcast_round_constants_rvvcrypto_(sz_u32_t *broadcast_round_constants,
                                                                       sz_size_t vector_length) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    vuint32m2_t const constant_offsets_u32m2 = __riscv_vsll_vx_u32m2(
        __riscv_vand_vx_u32m2(__riscv_vid_v_u32m2(vector_length), 3, vector_length), 2, vector_length);
    for (sz_size_t round_index = 0; round_index != 64; round_index += 4)
        __riscv_vse32_v_u32m2(
            broadcast_round_constants + round_index * 16,
            __riscv_vluxei32_v_u32m2(round_constants + round_index, constant_offsets_u32m2, vector_length),
            vector_length);
}

/** Offsets that scatter the eight 64-bit words of a block into the quarter-major stage:
 *  word i lands at byte 256 * ⌊i / 2⌋ + 8 * (i mod 2). */
STRINGZILLA_INLINE vuint16m1_t sz_sha256_multistate_quarter_offsets_rvvcrypto_(void) {
    sz_size_t const words_length = STRINGZILLA_SHA256_BLOCK_LENGTH / 8;
    vuint16m1_t const word_u16m1 = __riscv_vid_v_u16m1(words_length);
    return __riscv_vor_vv_u16m1(
        __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(word_u16m1, 1, words_length), 8, words_length),
        __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(word_u16m1, 1, words_length), 3, words_length), words_length);
}

/**
 *  @brief Stages one lane's message block, so quarter k of every lane sits side by side and the
 *      compression reads each message vector with one aligned unit-stride load.
 *  @param[out] stage_lane The lane's slot in the first quarter, 16 * lane bytes into the stage.
 *  @param[in] block The 64-byte block, at any alignment.
 *  @param[in] length Either 64, or 0 to stage nothing.
 *  @param[in] quarter_offsets_u16m1 Offsets from the quarter-offsets helper above.
 */
STRINGZILLA_INLINE void sz_sha256_multistate_stage_rvvcrypto_(sz_u8_t *stage_lane, sz_u8_t const *block,
                                                              sz_size_t length, vuint16m1_t quarter_offsets_u16m1) {
    sz_size_t const bytes_length = __riscv_vsetvl_e8m4(length);
    vuint64m4_t const words_u64m4 = __riscv_vreinterpret_v_u8m4_u64m4(__riscv_vle8_v_u8m4(block, bytes_length));
    __riscv_vsuxei16_v_u64m4((sz_u64_t *)stage_lane, quarter_offsets_u16m1, words_u64m4, length / 8);
}

/**
 *  @brief Compresses one staged block per lane into the batched state, one lane per element group.
 *  @param[inout] abef_u32m2 Every lane's {f,e,b,a} group, as in the single-lane block process.
 *  @param[inout] cdgh_u32m2 Every lane's {h,g,d,c} group.
 *  @param[in] stage Blocks staged by @c sz_sha256_multistate_stage_rvvcrypto_, 64 words a quarter.
 *  @param[in] broadcast_round_constants Table from the broadcast-round-constants helper.
 *  @param[in] active_b16 Lanes whose hash takes the result; the rest keep their state bit-for-bit.
 *  @param[in] vector_length Four 32-bit elements per lane.
 *
 *  @c vsha2cl, @c vsha2ch and @c vsha2ms work on each element group alone, so the single-state
 *  round sequence runs unchanged over every lane. Kept out of line: inlined into the block loop,
 *  the round-constant loads are hoisted out of it and spilled.
 */
STRINGZILLA_OUTLINED_ void sz_sha256_multistate_compress_rvvcrypto_(vuint32m2_t *abef_u32m2, vuint32m2_t *cdgh_u32m2,
                                                                    sz_u32_t const *stage,
                                                                    sz_u32_t const *broadcast_round_constants,
                                                                    vbool16_t active_b16, sz_size_t vector_length) {
    vuint32m2_t const word_u32m2 = __riscv_vand_vx_u32m2(__riscv_vid_v_u32m2(vector_length), 3, vector_length);
    vbool16_t const lane0_mask_b16 = __riscv_vmseq_vx_u32m2_b16(word_u32m2, 0, vector_length);

    // Message words are big-endian; at most 16 lanes keep every byte index below 256.
    sz_size_t const bytes_length = vector_length * 4;
    vuint8m2_t const byte_swap_u8m2 = __riscv_vxor_vx_u8m2(__riscv_vid_v_u8m2(bytes_length), 3, bytes_length);
    vuint32m2_t w0_u32m2 = __riscv_vreinterpret_v_u8m2_u32m2(
        __riscv_vrgather_vv_u8m2(__riscv_vreinterpret_v_u32m2_u8m2(__riscv_vle32_v_u32m2(stage + 0, vector_length)),
                                 byte_swap_u8m2, bytes_length));
    vuint32m2_t w1_u32m2 = __riscv_vreinterpret_v_u8m2_u32m2(
        __riscv_vrgather_vv_u8m2(__riscv_vreinterpret_v_u32m2_u8m2(__riscv_vle32_v_u32m2(stage + 64, vector_length)),
                                 byte_swap_u8m2, bytes_length));
    vuint32m2_t w2_u32m2 = __riscv_vreinterpret_v_u8m2_u32m2(
        __riscv_vrgather_vv_u8m2(__riscv_vreinterpret_v_u32m2_u8m2(__riscv_vle32_v_u32m2(stage + 128, vector_length)),
                                 byte_swap_u8m2, bytes_length));
    vuint32m2_t w3_u32m2 = __riscv_vreinterpret_v_u8m2_u32m2(
        __riscv_vrgather_vv_u8m2(__riscv_vreinterpret_v_u32m2_u8m2(__riscv_vle32_v_u32m2(stage + 192, vector_length)),
                                 byte_swap_u8m2, bytes_length));

    vuint32m2_t abef_compressed_u32m2 = *abef_u32m2;
    vuint32m2_t cdgh_compressed_u32m2 = *cdgh_u32m2;
    for (sz_size_t round_index = 0; round_index != 48; round_index += 4) {
        vuint32m2_t const round_key_u32m2 = __riscv_vadd_vv_u32m2(
            __riscv_vle32_v_u32m2(broadcast_round_constants + round_index * 16, vector_length), w0_u32m2,
            vector_length);
        cdgh_compressed_u32m2 = __riscv_vsha2cl_vv_u32m2(cdgh_compressed_u32m2, abef_compressed_u32m2, round_key_u32m2,
                                                         vector_length);
        abef_compressed_u32m2 = __riscv_vsha2ch_vv_u32m2(abef_compressed_u32m2, cdgh_compressed_u32m2, round_key_u32m2,
                                                         vector_length);
        vuint32m2_t const merged_u32m2 = __riscv_vmerge_vvm_u32m2(w2_u32m2, w1_u32m2, lane0_mask_b16, vector_length);
        vuint32m2_t const extended_u32m2 = __riscv_vsha2ms_vv_u32m2(w0_u32m2, merged_u32m2, w3_u32m2, vector_length);
        w0_u32m2 = w1_u32m2, w1_u32m2 = w2_u32m2, w2_u32m2 = w3_u32m2, w3_u32m2 = extended_u32m2;
    }
    for (sz_size_t round_index = 48; round_index != 64; round_index += 4) {
        vuint32m2_t const round_key_u32m2 = __riscv_vadd_vv_u32m2(
            __riscv_vle32_v_u32m2(broadcast_round_constants + round_index * 16, vector_length), w0_u32m2,
            vector_length);
        cdgh_compressed_u32m2 = __riscv_vsha2cl_vv_u32m2(cdgh_compressed_u32m2, abef_compressed_u32m2, round_key_u32m2,
                                                         vector_length);
        abef_compressed_u32m2 = __riscv_vsha2ch_vv_u32m2(abef_compressed_u32m2, cdgh_compressed_u32m2, round_key_u32m2,
                                                         vector_length);
        w0_u32m2 = w1_u32m2, w1_u32m2 = w2_u32m2, w2_u32m2 = w3_u32m2;
    }

    *abef_u32m2 = __riscv_vadd_vv_u32m2_mu(active_b16, *abef_u32m2, *abef_u32m2, abef_compressed_u32m2, vector_length);
    *cdgh_u32m2 = __riscv_vadd_vv_u32m2_mu(active_b16, *cdgh_u32m2, *cdgh_u32m2, cdgh_compressed_u32m2, vector_length);
}

/**
 *  @brief Offsets of every lane's `{f,e,b,a}` words, relative to the first lane's @c hash, one
 *      lane per element group; the `{h,g,d,c}` words sit two words further.
 *  @param[in] vector_length Four 32-bit elements per lane.
 */
STRINGZILLA_INLINE vuint32m2_t sz_sha256_multistate_abef_offsets_rvvcrypto_(sz_size_t vector_length) {
    vuint32m2_t const element_u32m2 = __riscv_vid_v_u32m2(vector_length);
    vuint32m2_t const word_u32m2 = __riscv_vand_vx_u32m2(element_u32m2, 3, vector_length);
    // Words 0, 1, 2, 3 of a group hold hash words 5, 4, 1, 0.
    vuint32m2_t const hash_word_u32m2 = __riscv_vsub_vv_u32m2(__riscv_vrsub_vx_u32m2(word_u32m2, 5, vector_length),
                                                              __riscv_vand_vx_u32m2(word_u32m2, 2, vector_length),
                                                              vector_length);
    return __riscv_vmacc_vx_u32m2(__riscv_vsll_vx_u32m2(hash_word_u32m2, 2, vector_length),
                                  (sz_u32_t)sizeof(sz_sha256_state_t),
                                  __riscv_vsrl_vx_u32m2(element_u32m2, 2, vector_length), vector_length);
}

/** Lane mask, four elements per lane, from a 32-bit flag or count per lane against @p threshold. */
STRINGZILLA_INLINE vbool16_t sz_sha256_multistate_lanes_above_rvvcrypto_(sz_u32_t const *per_lane, sz_u32_t threshold,
                                                                         sz_size_t vector_length) {
    vuint32m2_t const lane_offsets_u32m2 = __riscv_vsll_vx_u32m2(
        __riscv_vsrl_vx_u32m2(__riscv_vid_v_u32m2(vector_length), 2, vector_length), 2, vector_length);
    return __riscv_vmsgtu_vx_u32m2_b16(__riscv_vluxei32_v_u32m2(per_lane, lane_offsets_u32m2, vector_length), threshold,
                                       vector_length);
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_update_rvvcrypto(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_size_t const lanes_count = texts->count;
    // One lane per 128-bit element group of an `m2` register, capped at the 16 stack-table lanes.
    sz_size_t const lanes_per_register = sz_min_of_two(__riscv_vsetvlmax_e32m2() / 4, (sz_size_t)16);
    sz_align_(64) sz_u32_t broadcast_round_constants[1024];
    sz_align_(64) sz_u32_t stage[256];
    sz_u32_t buffered_flags[16], blocks_per_lane[16];
    sz_u8_t const *cursors[16];
    sz_size_t remaining[16];
    sz_bool_t constants_broadcasted = sz_false_k;
    vuint16m1_t const quarter_offsets_u16m1 = sz_sha256_multistate_quarter_offsets_rvvcrypto_();

    for (sz_size_t first_lane_index = 0; first_lane_index < lanes_count; first_lane_index += lanes_per_register) {
        sz_size_t const active_lanes_count = sz_min_of_two(lanes_count - first_lane_index, lanes_per_register);
        sz_sha256_state_t *const batch = &states[first_lane_index];
        sz_u32_t buffered_lanes_count = 0, largest_blocks_count = 0;

        // Top up any buffered partial block into the state's own buffer, which then becomes the
        // lane's first block. Every lane's whole chunk is charged to `total_length` here, once.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &batch[lane_index];
            cursors[lane_index] = (sz_u8_t const *)texts->get_start(texts->handle, first_lane_index + lane_index);
            remaining[lane_index] = texts->get_length(texts->handle, first_lane_index + lane_index);
            buffered_flags[lane_index] = 0, blocks_per_lane[lane_index] = 0;

            // The block counts ride in 32-bit lanes, so a longer chunk takes the single kernel.
            if (remaining[lane_index] / STRINGZILLA_SHA256_BLOCK_LENGTH > 0xFFFFFFFFull) {
                sz_sha256_state_update_rvvcrypto_(state, (sz_cptr_t)cursors[lane_index], remaining[lane_index]);
                remaining[lane_index] = 0;
                continue;
            }

            state->total_length += remaining[lane_index];
            sz_size_t const missing = STRINGZILLA_SHA256_BLOCK_LENGTH - state->block_length;
            if (state->block_length != 0 && remaining[lane_index] >= missing) {
                sz_size_t const head_length = __riscv_vsetvl_e8m8(missing);
                __riscv_vse8_v_u8m8(state->block + state->block_length,
                                    __riscv_vle8_v_u8m8(cursors[lane_index], head_length), head_length);
                buffered_flags[lane_index] = 1, ++buffered_lanes_count;
                state->block_length = 0;
                cursors[lane_index] += missing, remaining[lane_index] -= missing;
            }
            blocks_per_lane[lane_index] = (sz_u32_t)(remaining[lane_index] / STRINGZILLA_SHA256_BLOCK_LENGTH);
            if (blocks_per_lane[lane_index] > largest_blocks_count) largest_blocks_count = blocks_per_lane[lane_index];
        }

        // Token-sized chunks only buffer, so they skip the state round trip and constants.
        if (buffered_lanes_count != 0 || largest_blocks_count != 0) {
            sz_size_t const vector_length = active_lanes_count * 4;
            if (constants_broadcasted == sz_false_k) {
                sz_sha256_broadcast_round_constants_rvvcrypto_(broadcast_round_constants, lanes_per_register * 4);
                constants_broadcasted = sz_true_k;
            }
            vuint32m2_t const abef_offsets_u32m2 = sz_sha256_multistate_abef_offsets_rvvcrypto_(vector_length);
            vuint32m2_t abef_u32m2 = __riscv_vluxei32_v_u32m2(batch->hash, abef_offsets_u32m2, vector_length);
            vuint32m2_t cdgh_u32m2 = __riscv_vluxei32_v_u32m2(batch->hash + 2, abef_offsets_u32m2, vector_length);

            // A lane stages a block only while it owns one; the rest keep stale bytes, masked off.
            if (buffered_lanes_count != 0) {
                for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
                    sz_sha256_multistate_stage_rvvcrypto_((sz_u8_t *)stage + lane_index * 16, batch[lane_index].block,
                                                          buffered_flags[lane_index] * STRINGZILLA_SHA256_BLOCK_LENGTH,
                                                          quarter_offsets_u16m1);
                vbool16_t const buffered_b16 = sz_sha256_multistate_lanes_above_rvvcrypto_(buffered_flags, 0,
                                                                                           vector_length);
                sz_sha256_multistate_compress_rvvcrypto_(&abef_u32m2, &cdgh_u32m2, stage, broadcast_round_constants,
                                                         buffered_b16, vector_length);
            }

            for (sz_u32_t block_index = 0; block_index != largest_blocks_count; ++block_index) {
                for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
                    sz_size_t const block_length = (blocks_per_lane[lane_index] > block_index) *
                                                   STRINGZILLA_SHA256_BLOCK_LENGTH;
                    sz_sha256_multistate_stage_rvvcrypto_((sz_u8_t *)stage + lane_index * 16, cursors[lane_index],
                                                          block_length, quarter_offsets_u16m1);
                    cursors[lane_index] += block_length;
                }
                vbool16_t const active_b16 = sz_sha256_multistate_lanes_above_rvvcrypto_(blocks_per_lane, block_index,
                                                                                         vector_length);
                sz_sha256_multistate_compress_rvvcrypto_(&abef_u32m2, &cdgh_u32m2, stage, broadcast_round_constants,
                                                         active_b16, vector_length);
            }

            __riscv_vsuxei32_v_u32m2(batch->hash, abef_offsets_u32m2, abef_u32m2, vector_length);
            __riscv_vsuxei32_v_u32m2(batch->hash + 2, abef_offsets_u32m2, cdgh_u32m2, vector_length);
        }

        // Whatever is left cannot fill a block, so it only ever buffers.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &batch[lane_index];
            sz_size_t const tail_length = __riscv_vsetvl_e8m8(remaining[lane_index] % STRINGZILLA_SHA256_BLOCK_LENGTH);
            __riscv_vse8_v_u8m8(state->block + state->block_length,
                                __riscv_vle8_v_u8m8(cursors[lane_index], tail_length), tail_length);
            state->block_length += (sz_u8_t)tail_length;
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_rvvcrypto(sz_sha256_state_t const *states,
                                                                  sz_size_t states_count, sz_u8_t *digests,
                                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_size_t const lanes_per_register = sz_min_of_two(__riscv_vsetvlmax_e32m2() / 4, (sz_size_t)16);
    sz_align_(64) sz_u32_t broadcast_round_constants[1024];
    sz_align_(64) sz_u32_t stage[256];
    sz_u32_t overflow_flags[16];
    vuint16m1_t const quarter_offsets_u16m1 = sz_sha256_multistate_quarter_offsets_rvvcrypto_();
    if (states_count != 0)
        sz_sha256_broadcast_round_constants_rvvcrypto_(broadcast_round_constants, lanes_per_register * 4);

    for (sz_size_t first_lane_index = 0; first_lane_index < states_count; first_lane_index += lanes_per_register) {
        sz_size_t const active_lanes_count = sz_min_of_two(states_count - first_lane_index, lanes_per_register);
        sz_sha256_state_t const *const batch = &states[first_lane_index];
        sz_size_t const vector_length = active_lanes_count * 4;
        sz_size_t const padded_length = __riscv_vsetvl_e8m4(STRINGZILLA_SHA256_BLOCK_LENGTH);
        vuint8m4_t const zeros_u8m4 = __riscv_vmv_v_x_u8m4(0, padded_length);
        sz_align_(64) sz_u512_vec_t padded_vec;
        sz_u32_t overflow_lanes_count = 0;

        vuint32m2_t const abef_offsets_u32m2 = sz_sha256_multistate_abef_offsets_rvvcrypto_(vector_length);
        vuint32m2_t abef_u32m2 = __riscv_vluxei32_v_u32m2(batch->hash, abef_offsets_u32m2, vector_length);
        vuint32m2_t cdgh_u32m2 = __riscv_vluxei32_v_u32m2(batch->hash + 2, abef_offsets_u32m2, vector_length);

        // The terminator lands in a carrier block whenever it would crowd out the bit length.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_size_t const buffered = batch[lane_index].block_length;
            overflow_flags[lane_index] = buffered + 1 > STRINGZILLA_SHA256_BLOCK_LENGTH - 8;
            overflow_lanes_count += overflow_flags[lane_index];
            if (!overflow_flags[lane_index]) continue;
            __riscv_vse8_v_u8m4(padded_vec.u8s, zeros_u8m4, padded_length);
            sz_size_t const copy_length = __riscv_vsetvl_e8m4(buffered);
            __riscv_vse8_v_u8m4(padded_vec.u8s, __riscv_vle8_v_u8m4(batch[lane_index].block, copy_length), copy_length);
            padded_vec.u8s[buffered] = 0x80;
            sz_sha256_multistate_stage_rvvcrypto_((sz_u8_t *)stage + lane_index * 16, padded_vec.u8s,
                                                  STRINGZILLA_SHA256_BLOCK_LENGTH, quarter_offsets_u16m1);
        }
        if (overflow_lanes_count != 0)
            sz_sha256_multistate_compress_rvvcrypto_(
                &abef_u32m2, &cdgh_u32m2, stage, broadcast_round_constants,
                sz_sha256_multistate_lanes_above_rvvcrypto_(overflow_flags, 0, vector_length), vector_length);

        // A lane that overflowed spent its buffer on the carrier, so its last block is the length.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_size_t const buffered = overflow_flags[lane_index] ? 0 : batch[lane_index].block_length;
            __riscv_vse8_v_u8m4(padded_vec.u8s, zeros_u8m4, padded_length);
            sz_size_t const copy_length = __riscv_vsetvl_e8m4(buffered);
            __riscv_vse8_v_u8m4(padded_vec.u8s, __riscv_vle8_v_u8m4(batch[lane_index].block, copy_length), copy_length);
            if (!overflow_flags[lane_index]) padded_vec.u8s[buffered] = 0x80;
            padded_vec.u64s[7] = sz_u64_bytes_reverse(batch[lane_index].total_length * 8);
            sz_sha256_multistate_stage_rvvcrypto_((sz_u8_t *)stage + lane_index * 16, padded_vec.u8s,
                                                  STRINGZILLA_SHA256_BLOCK_LENGTH, quarter_offsets_u16m1);
        }
        sz_sha256_multistate_compress_rvvcrypto_(&abef_u32m2, &cdgh_u32m2, stage, broadcast_round_constants,
                                                 __riscv_vmset_m_b16(vector_length), vector_length);

        // Digests are big-endian words, so the byte scatter reverses each word on its way out.
        sz_size_t const bytes_length = vector_length * 4;
        vuint16m4_t const byte_index_u16m4 = __riscv_vid_v_u16m4(bytes_length);
        vuint16m4_t const word_u16m4 = __riscv_vand_vx_u16m4(__riscv_vsrl_vx_u16m4(byte_index_u16m4, 2, bytes_length),
                                                             3, bytes_length);
        vuint16m4_t const hash_word_u16m4 = __riscv_vsub_vv_u16m4(__riscv_vrsub_vx_u16m4(word_u16m4, 5, bytes_length),
                                                                  __riscv_vand_vx_u16m4(word_u16m4, 2, bytes_length),
                                                                  bytes_length);
        vuint16m4_t const digest_offsets_u16m4 = __riscv_vadd_vv_u16m4(
            __riscv_vadd_vv_u16m4(
                __riscv_vsll_vx_u16m4(__riscv_vsrl_vx_u16m4(byte_index_u16m4, 4, bytes_length), 5, bytes_length),
                __riscv_vsll_vx_u16m4(hash_word_u16m4, 2, bytes_length), bytes_length),
            __riscv_vxor_vx_u16m4(__riscv_vand_vx_u16m4(byte_index_u16m4, 3, bytes_length), 3, bytes_length),
            bytes_length);
        sz_u8_t *const batch_digests = digests + first_lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH;
        __riscv_vsuxei16_v_u8m2(batch_digests, digest_offsets_u16m4, __riscv_vreinterpret_v_u32m2_u8m2(abef_u32m2),
                                bytes_length);
        __riscv_vsuxei16_v_u8m2(batch_digests + 8, digest_offsets_u16m4, __riscv_vreinterpret_v_u32m2_u8m2(cdgh_u32m2),
                                bytes_length);
    }
    return sz_success_k;
}

#pragma endregion RVV Crypto SHA 256 via Zvknhb

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_RVVCRYPTO
#endif // STRINGZILLA_ARCH_RISCV64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_RVVCRYPTO_H_
