/**
 *  @brief RISC-V Vector Crypto (Zvk) backend for hash: hardware AES & SHA-256.
 *  @file include/stringzilla/hash/rvvcrypto.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 *
 *  This backend replaces the two emulated primitives of the base RVV 1.0 path (`hash/rvv.h`):
 *    - the tower-field vector-permute AES round (`sz_emulate_aesenc_rvv_`) is swapped for a single
 *      `Zvkned` `vaesem.vv` instruction, and
 *    - the serial SHA-256 block is swapped for the `Zvknhb` `vsha2cl/vsha2ch/vsha2ms` instructions.
 *  Both produce byte/bit-identical results to the serial reference, so digests are interchangeable.
 */
#ifndef STRINGZILLA_HASH_RVVCRYPTO_H_
#define STRINGZILLA_HASH_RVVCRYPTO_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVVCRYPTO

#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvkned,+zvknhb"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvkned,+zvknhb")
#endif

#pragma region RVV Crypto AES Round (Zvkned)

/**
 *  @brief Bit-exact `Zvkned` implementation of a single `_mm_aesenc_si128` round.
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, identical to
 *          `sz_emulate_aesenc_si128_serial_` and to `sz_emulate_aesenc_rvv_`.
 *
 *  RISC-V's `vaesem.vv vd, vs2` computes `vd = MixColumns(SubBytes(ShiftRows(vd))) ^ vs2`, which is
 *  exactly the AES-NI `AESENC` operation: the state in `vd`, the round key in `vs2`. A single 128-bit
 *  AES block is one element group (`EGW = 128`), so the operation runs with `vector_length = 4` 32-bit lanes.
 */
SZ_INTERNAL sz_u128_vec_t sz_emulate_aesenc_rvvcrypto_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    sz_size_t vector_length = __riscv_vsetvl_e32m1(4); // one 128-bit AES block = 4x u32 lanes
    vuint32m1_t state_u32 = __riscv_vle32_v_u32m1((sz_u32_t const *)state_vec.u32s, vector_length);
    vuint32m1_t key_u32 = __riscv_vle32_v_u32m1((sz_u32_t const *)round_key_vec.u32s, vector_length);
    state_u32 = __riscv_vaesem_vv_u32m1(state_u32, key_u32, vector_length);
    sz_u128_vec_t result;
    __riscv_vse32_v_u32m1(result.u32s, state_u32, vector_length);
    return result;
}

#pragma endregion // RVV Crypto AES Round

#pragma region RVV Crypto Hash Drivers

/*  These drivers mirror the serial / base-RVV ones exactly, substituting `sz_emulate_aesenc_rvvcrypto_`
 *  for the AES round. Every non-AES step reuses the shared serial helpers, so the digests are
 *  guaranteed value-identical to `sz_hash_serial`. */

SZ_INTERNAL void sz_hash_state_short_update_rvvcrypto_(sz_hash_state_aligned_for_short_t_ *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_emulate_aesenc_rvvcrypto_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

SZ_INTERNAL sz_u64_t sz_hash_state_short_finalize_rvvcrypto_(sz_hash_state_aligned_for_short_t_ const *state,
                                                             sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    sz_u128_vec_t mixed = sz_emulate_aesenc_rvvcrypto_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_rvvcrypto_(sz_emulate_aesenc_rvvcrypto_(mixed, key_with_length),
                                                                   mixed);
    return mixed_in_register.u64s[0];
}

SZ_INTERNAL void sz_hash_state_update_rvvcrypto_(sz_hash_state_aligned_t_ *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        state->aes.u128s[lane_index] = sz_emulate_aesenc_rvvcrypto_(state->aes.u128s[lane_index],
                                                                    state->ins.u128s[lane_index]);
        state->sum.u128s[lane_index] = sz_emulate_shuffle_epi8_serial_(state->sum.u128s[lane_index], shuffle);
        state->sum.u128s[lane_index].u64s[0] += state->ins.u128s[lane_index].u64s[0];
        state->sum.u128s[lane_index].u64s[1] += state->ins.u128s[lane_index].u64s[1];
    }
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_rvvcrypto_(sz_hash_state_aligned_t_ state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = state.key.u64s[0] + state.ins_length;
    key_with_length.u64s[1] = state.key.u64s[1];

    // Fold the deferred final block (still buffered in `ins` - a full 64 bytes or a zero-padded tail) into each
    // lane. Folding the last block here, rather than in `update`, lets both one-shot `sz_hash` and the streaming
    // digest defer it and share this single finalization.
    sz_u128_vec_t aes0 = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[0], state.ins.u128s[0]);
    sz_u128_vec_t aes1 = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[1], state.ins.u128s[1]);
    sz_u128_vec_t aes2 = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[2], state.ins.u128s[2]);
    sz_u128_vec_t aes3 = sz_emulate_aesenc_rvvcrypto_(state.aes.u128s[3], state.ins.u128s[3]);
    sz_u128_vec_t sum0 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[0], shuffle);
    sz_u128_vec_t sum1 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[1], shuffle);
    sz_u128_vec_t sum2 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[2], shuffle);
    sz_u128_vec_t sum3 = sz_emulate_shuffle_epi8_serial_(state.sum.u128s[3], shuffle);
    sum0.u64s[0] += state.ins.u128s[0].u64s[0], sum0.u64s[1] += state.ins.u128s[0].u64s[1];
    sum1.u64s[0] += state.ins.u128s[1].u64s[0], sum1.u64s[1] += state.ins.u128s[1].u64s[1];
    sum2.u64s[0] += state.ins.u128s[2].u64s[0], sum2.u64s[1] += state.ins.u128s[2].u64s[1];
    sum3.u64s[0] += state.ins.u128s[3].u64s[0], sum3.u64s[1] += state.ins.u128s[3].u64s[1];

    sz_u128_vec_t mixed0 = sz_emulate_aesenc_rvvcrypto_(sum0, aes0);
    sz_u128_vec_t mixed1 = sz_emulate_aesenc_rvvcrypto_(sum1, aes1);
    sz_u128_vec_t mixed2 = sz_emulate_aesenc_rvvcrypto_(sum2, aes2);
    sz_u128_vec_t mixed3 = sz_emulate_aesenc_rvvcrypto_(sum3, aes3);

    sz_u128_vec_t mixed01 = sz_emulate_aesenc_rvvcrypto_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_emulate_aesenc_rvvcrypto_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_emulate_aesenc_rvvcrypto_(mixed01, mixed23);

    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_rvvcrypto_(sz_emulate_aesenc_rvvcrypto_(mixed, key_with_length),
                                                                   mixed);
    return mixed_in_register.u64s[0];
}

/** @brief  Vector-copy a single AES block (`sizeof(sz_u128_vec_t)` bytes) from `source` into `target->u8s`,
 *          replacing a scalar byte loop. `source` must have a full block of readable bytes. */
SZ_INTERNAL void sz_hash_load_block_rvvcrypto_(sz_u128_vec_t *target, sz_cptr_t source) {
    sz_size_t vector_length = __riscv_vsetvl_e8m1(sizeof(target->u8s));
    __riscv_vse8_v_u8m1(target->u8s, __riscv_vle8_v_u8m1((sz_u8_t const *)source, vector_length), vector_length);
}

/** @brief  Vector-copy a single AES block (`sizeof(sz_u128_vec_t)` bytes) from `source` to `target`, the store
 *          counterpart of `sz_hash_load_block_rvvcrypto_`. `target` must have a full block of writable bytes. */
SZ_INTERNAL void sz_hash_store_block_rvvcrypto_(sz_ptr_t target, sz_u128_vec_t source) {
    sz_size_t vector_length = __riscv_vsetvl_e8m1(sizeof(source.u8s));
    __riscv_vse8_v_u8m1((sz_u8_t *)target, __riscv_vle8_v_u8m1(source.u8s, vector_length), vector_length);
}

/**
 *  @brief Loads the packed public state into the aligned internal twin (one `vle8` block per 16-byte lane).
 */
SZ_INTERNAL sz_hash_state_aligned_t_ sz_hash_state_load_rvvcrypto_(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t_ state;
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

/** @brief Stores the aligned internal twin back into the packed public state (one `vse8` block per 16-byte lane). */
SZ_INTERNAL void sz_hash_state_store_rvvcrypto_(sz_hash_state_t *packed, sz_hash_state_aligned_t_ const *state) {
    for (sz_size_t lane_index = 0; lane_index < 4; ++lane_index) {
        sz_size_t const offset = lane_index * 16;
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->aes + offset), state->aes.u128s[lane_index]);
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->sum + offset), state->sum.u128s[lane_index]);
        sz_hash_store_block_rvvcrypto_((sz_ptr_t)(packed->ins + offset), state->ins.u128s[lane_index]);
    }
    sz_hash_store_block_rvvcrypto_((sz_ptr_t)packed->key, state->key);
    packed->ins_length = state->ins_length;
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_rvvcrypto(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    sz_size_t const block = sizeof(sz_u128_vec_t); // one AES block
    if (length <= block) {
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
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
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
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
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
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
        sz_align_(16) sz_hash_state_aligned_for_short_t_ state;
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
        // except to reuse `init` (layout-locked by the `static_assert`s on `sz_hash_state_aligned_t_`).
        sz_align_(64) sz_hash_state_aligned_t_ state;
        sz_size_t const window = sizeof(state.ins.u8s); // the 64-byte hashing window
        sz_hash_state_init_serial((sz_hash_state_t *)&state, seed);

        // Absorb every full 64-byte window EXCEPT the last; the final block (a full 64 or a partial tail) stays
        // buffered in `ins` for `sz_hash_state_finalize_rvvcrypto_` to fold - the same deferral the streaming path uses.
        for (; state.ins_length + window < length; state.ins_length += window) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(window); // VLEN >= 128 -> one whole-window transfer
            __riscv_vse8_v_u8m8(state.ins.u8s,
                                __riscv_vle8_v_u8m8((sz_u8_t const *)start + state.ins_length, vector_length),
                                vector_length);
            sz_hash_state_update_rvvcrypto_(&state);
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

SZ_PUBLIC void sz_hash_state_init_rvvcrypto(sz_hash_state_t *state, sz_u64_t seed) {
    sz_hash_state_init_serial(state, seed);
}

SZ_PUBLIC void sz_hash_state_update_rvvcrypto(sz_hash_state_t *packed, sz_cptr_t text, sz_size_t length) {
    // Load the packed public state (any alignment) into an aligned twin once, buffer/absorb on it, then store back.
    // `ins` is exactly one 64-byte window. Track how many bytes it holds and absorb it only once it becomes
    // interior (more bytes arrive - the deferral `digest` needs to choose minimal/full by total length). The
    // deferred trailing block reads back as `ins_length % 64 == 0 && ins_length != 0`; treat that as `buffered ==
    // 64`. The append uses `vl` as the mask, touching only `[buffered, buffered+take)`, and we re-zero `ins` after
    // each absorb so the high lanes stay zero-padded for `finalize` to fold a clean trailing block.
    sz_hash_state_aligned_t_ state = sz_hash_state_load_rvvcrypto_(packed);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it and re-zero the buffer
            sz_hash_state_update_rvvcrypto_(&state);
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
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_rvvcrypto(sz_hash_state_t const *packed) {
    sz_hash_state_aligned_t_ state = sz_hash_state_load_rvvcrypto_(packed);
    sz_size_t length = state.ins_length;
    // Inputs longer than one block fold through the full four-lane state, where the deferred final block buffered
    // in `ins` is folded by `sz_hash_state_finalize_rvvcrypto_`. A length of exactly 64 uses the minimal (<=64)
    // path below - matching one-shot `sz_hash`, whose `length <= 64` ladder also stays minimal.
    if (length > 64) return sz_hash_state_finalize_rvvcrypto_(state);

    // Switch back to a smaller "short" state for small inputs; the aligned twin lanes are read directly.
    sz_hash_state_aligned_for_short_t_ minimal_state;
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

SZ_PUBLIC void sz_fill_random_rvvcrypto(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
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
        __riscv_vse8_v_u8m1((sz_u8_t *)text, __riscv_vle8_v_u8m1(generated_vec.u8s, out_vl), out_vl);
        text += out_vl, length -= out_vl;
    }
}

#pragma endregion // RVV Crypto Hash Drivers

#pragma region RVV Crypto SHA 256 (Zvknhb)

/**
 *  @brief Process a single 512-bit (64-byte) block of data using SHA-256 via `Zvknhb`.
 *  @param hash Pointer to 8x 32-bit hash values {a,b,c,d,e,f,g,h}, modified in place.
 *  @param block Pointer to a 64-byte message block.
 *
 *  The `Zvknh` state is held in two 128-bit element groups, each four 32-bit lanes. Per the RISC-V
 *  Vector Crypto spec, `vsha2c[hl].vv` reads `vs2 = {a,b,e,f}` and `vd = {c,d,g,h}` (where `{x@y@z@w}`
 *  packs `w` into lane 0 and `x` into lane 3, little-endian), and writes the next `{a,b,e,f}` back to
 *  `vd`. Each instruction performs two compression rounds (`cl` consumes the low two W+K words of the
 *  group, `ch` the high two). `vsha2ms.vv` expands four message-schedule words per call.
 *
 *  The intrinsic argument order (matching OpenSSL's `sha256_block_data_order_zvkb_zvknha_or_zvknhb`):
 *    - `cdgh = vsha2cl(cdgh, abef, k_plus_w)` then `abef = vsha2ch(abef, cdgh, k_plus_w)` per quad-round,
 *    - `wN = vsha2ms(wN, vmerge(w_older, w_newer, mask_lane0), w_newest)` to roll the schedule forward.
 *  SHA-256 words are big-endian; we load them with a scalar byte-swap so this path needs only the
 *  `Zvknhb` extension (no `Zvbb`/`Zvkb` `vrev8`).
 */
SZ_INTERNAL void sz_sha256_process_block_rvvcrypto_(sz_u32_t hash[sz_at_least_(8)],
                                                    sz_u8_t const block[sz_at_least_(64)]) {
    sz_u32_t const *round_constants = sz_sha256_round_constants_();
    sz_size_t const vector_length = __riscv_vsetvl_e32m1(4);

    // Lane-0 selection mask for the message-schedule `vmerge` (replace the oldest word of the group).
    sz_align_(16) sz_u32_t const mask_seed[4] = {1, 0, 0, 0};
    vbool32_t const lane0_mask = __riscv_vmsne_vx_u32m1_b32(__riscv_vle32_v_u32m1(mask_seed, vector_length), 0,
                                                            vector_length);

    // Build the two state vectors: abef = {f,e,b,a} (lane0..3), cdgh = {h,g,d,c}.
    sz_align_(16) sz_u32_t const abef_seed[4] = {hash[5], hash[4], hash[1], hash[0]};
    sz_align_(16) sz_u32_t const cdgh_seed[4] = {hash[7], hash[6], hash[3], hash[2]};
    vuint32m1_t abef = __riscv_vle32_v_u32m1(abef_seed, vector_length);
    vuint32m1_t cdgh = __riscv_vle32_v_u32m1(cdgh_seed, vector_length);
    vuint32m1_t const abef_saved = abef;
    vuint32m1_t const cdgh_saved = cdgh;

    // Big-endian load of the 16 message words (scalar swap keeps us within `Zvknhb` only).
    sz_align_(64) sz_u32_t message_words[16];
    for (sz_size_t word_index = 0; word_index < 16; ++word_index)
        message_words[word_index] = ((sz_u32_t)block[word_index * 4 + 0] << 24) |
                                    ((sz_u32_t)block[word_index * 4 + 1] << 16) |
                                    ((sz_u32_t)block[word_index * 4 + 2] << 8) |
                                    ((sz_u32_t)block[word_index * 4 + 3] << 0);

    // Four schedule vectors hold {W3,W2,W1,W0}, {W7..W4}, {W11..W8}, {W15..W12}; they roll forward
    // by `vsha2ms`. RVV sizeless types cannot live in an array, so the four lanes are named locals and
    // the 16 quad-rounds are emitted by macros that cycle (w0 -> w1 -> w2 -> w3 -> w0 ...).
    vuint32m1_t w0 = __riscv_vle32_v_u32m1(&message_words[0], vector_length);
    vuint32m1_t w1 = __riscv_vle32_v_u32m1(&message_words[4], vector_length);
    vuint32m1_t w2 = __riscv_vle32_v_u32m1(&message_words[8], vector_length);
    vuint32m1_t w3 = __riscv_vle32_v_u32m1(&message_words[12], vector_length);

    // One quad-round: two compression rounds (`cl` then `ch`), then roll `current` forward via `ms`.
    // `current` is the group being consumed/produced; `newer`/`older` feed the lane-0 merge that builds
    // {W11,W10,W9,W4}; `newest` supplies {W15,W14,-,W12}.
#define SZ_RVVCRYPTO_SHA_QUAD_(current, newer, older, newest, k_offset)                                    \
    do {                                                                                                   \
        vuint32m1_t const round_key = __riscv_vadd_vv_u32m1(                                               \
            __riscv_vle32_v_u32m1(&round_constants[(k_offset)], vector_length), (current), vector_length); \
        cdgh = __riscv_vsha2cl_vv_u32m1(cdgh, abef, round_key, vector_length);                             \
        abef = __riscv_vsha2ch_vv_u32m1(abef, cdgh, round_key, vector_length);                             \
        vuint32m1_t const merged = __riscv_vmerge_vvm_u32m1((older), (newer), lane0_mask, vector_length);  \
        (current) = __riscv_vsha2ms_vv_u32m1((current), merged, (newest), vector_length);                  \
    } while (0)

    // Quad-rounds 0..11 compress and extend the message schedule.
    SZ_RVVCRYPTO_SHA_QUAD_(w0, w1, w2, w3, 0);
    SZ_RVVCRYPTO_SHA_QUAD_(w1, w2, w3, w0, 4);
    SZ_RVVCRYPTO_SHA_QUAD_(w2, w3, w0, w1, 8);
    SZ_RVVCRYPTO_SHA_QUAD_(w3, w0, w1, w2, 12);
    SZ_RVVCRYPTO_SHA_QUAD_(w0, w1, w2, w3, 16);
    SZ_RVVCRYPTO_SHA_QUAD_(w1, w2, w3, w0, 20);
    SZ_RVVCRYPTO_SHA_QUAD_(w2, w3, w0, w1, 24);
    SZ_RVVCRYPTO_SHA_QUAD_(w3, w0, w1, w2, 28);
    SZ_RVVCRYPTO_SHA_QUAD_(w0, w1, w2, w3, 32);
    SZ_RVVCRYPTO_SHA_QUAD_(w1, w2, w3, w0, 36);
    SZ_RVVCRYPTO_SHA_QUAD_(w2, w3, w0, w1, 40);
    SZ_RVVCRYPTO_SHA_QUAD_(w3, w0, w1, w2, 44);
#undef SZ_RVVCRYPTO_SHA_QUAD_

    // Quad-rounds 12..15 only compress; every schedule word we still consume already exists.
#define SZ_RVVCRYPTO_SHA_TAIL_(current, k_offset)                                                          \
    do {                                                                                                   \
        vuint32m1_t const round_key = __riscv_vadd_vv_u32m1(                                               \
            __riscv_vle32_v_u32m1(&round_constants[(k_offset)], vector_length), (current), vector_length); \
        cdgh = __riscv_vsha2cl_vv_u32m1(cdgh, abef, round_key, vector_length);                             \
        abef = __riscv_vsha2ch_vv_u32m1(abef, cdgh, round_key, vector_length);                             \
    } while (0)
    SZ_RVVCRYPTO_SHA_TAIL_(w0, 48);
    SZ_RVVCRYPTO_SHA_TAIL_(w1, 52);
    SZ_RVVCRYPTO_SHA_TAIL_(w2, 56);
    SZ_RVVCRYPTO_SHA_TAIL_(w3, 60);
#undef SZ_RVVCRYPTO_SHA_TAIL_

    // Add the compressed working state back into the running hash.
    abef = __riscv_vadd_vv_u32m1(abef_saved, abef, vector_length);
    cdgh = __riscv_vadd_vv_u32m1(cdgh_saved, cdgh, vector_length);

    // Unpack abef = {f,e,b,a}, cdgh = {h,g,d,c} back into {a,b,c,d,e,f,g,h}.
    sz_align_(16) sz_u32_t abef_out[4];
    sz_align_(16) sz_u32_t cdgh_out[4];
    __riscv_vse32_v_u32m1(abef_out, abef, vector_length);
    __riscv_vse32_v_u32m1(cdgh_out, cdgh, vector_length);
    hash[5] = abef_out[0], hash[4] = abef_out[1], hash[1] = abef_out[2], hash[0] = abef_out[3];
    hash[7] = cdgh_out[0], hash[6] = cdgh_out[1], hash[3] = cdgh_out[2], hash[2] = cdgh_out[3];
}

SZ_PUBLIC void sz_sha256_state_init_rvvcrypto(sz_sha256_state_t *state_ptr) {
    sz_u32_t const *initial_hash = sz_sha256_initial_hash_();
    // Copy all 8 initial words in halves of 4: `vsetvl_e32m1(8)` would clamp to 4 lanes at VLEN=128,
    // silently leaving the upper four words uninitialized, so the half-width copy is VLEN-independent.
    sz_size_t const vector_length = __riscv_vsetvl_e32m1(4);
    __riscv_vse32_v_u32m1(state_ptr->hash + 0, __riscv_vle32_v_u32m1(initial_hash + 0, vector_length), vector_length);
    __riscv_vse32_v_u32m1(state_ptr->hash + 4, __riscv_vle32_v_u32m1(initial_hash + 4, vector_length), vector_length);
    state_ptr->block_length = 0, state_ptr->total_length = 0;
}

SZ_PUBLIC void sz_sha256_state_update_rvvcrypto(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_u8_t const *input = (sz_u8_t const *)data;
    sz_size_t const current_block_index = state_ptr->block_length / 64;
    sz_size_t const final_block_index = (state_ptr->block_length + length) / 64;
    int const stays_in_the_block = current_block_index == final_block_index;
    int const fills_the_block = (state_ptr->block_length + length) % 64 == 0;

    state_ptr->total_length += length;

    // Fast path: stays in same block and doesn't fill it
    if (stays_in_the_block && !fills_the_block) {
        for (; length; --length, ++state_ptr->block_length, ++input) state_ptr->block[state_ptr->block_length] = *input;
        return;
    }

    // Calculate head, body, and tail lengths
    sz_size_t const head_length = (64 - state_ptr->block_length) % 64;
    sz_size_t const tail_length = (state_ptr->block_length + length) % 64;
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
    for (sz_size_t processed = 0; processed < body_length; processed += 64, input += 64)
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

SZ_PUBLIC void sz_sha256_state_digest_rvvcrypto(sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(32)]) {
    // Create a copy of the state for padding
    sz_sha256_state_t state = *state_ptr;

    // Append the '1' bit (0x80 byte) after the message
    state.block[state.block_length++] = 0x80;

    // If there's not enough room for the 64-bit length, pad this block and process it
    if (state.block_length > 56) {
        sz_size_t remaining = 64 - state.block_length;
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

#pragma endregion // RVV Crypto SHA 256

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVVCRYPTO

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_RVVCRYPTO_H_
