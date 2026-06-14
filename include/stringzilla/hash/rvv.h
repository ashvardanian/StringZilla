/**
 *  @brief RISC-V Vector (RVV 1.0) backend for hash.
 *  @file include/stringzilla/hash/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_RVV_H_
#define STRINGZILLA_HASH_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/hash/serial.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV

#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/*  Sum of all bytes, identical to `sz_bytesum_serial`. We process the buffer in strips capped at
 *  256 elements so a single `vwredsumu` (widening unsigned reduction, u8 -> u16) cannot overflow:
 *  the maximum per-strip sum is 256 * 255 = 65280, which still fits a u16 accumulator. The per-strip
 *  partial is then promoted into a 64-bit running total.
 *
 *  Hoisting the reduction out of the loop (a per-lane `u16` accumulator fed by `vwaddu.wv`, reduced
 *  once at the end) was tried and measured ~2x slower: the widening add forces the byte load down to
 *  `u8m4` (half the per-strip throughput), and that loss dwarfs the saving from a single cheap
 *  per-strip `vwredsumu`. So the cheap-reduction-per-strip form below is kept deliberately. */
SZ_PUBLIC sz_u64_t sz_bytesum_rvv(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u64_t bytesum = 0;
    while (length) {
        sz_size_t want = length < 256 ? length : 256;
        sz_size_t vl = __riscv_vsetvl_e8m8(want);
        vuint8m8_t text_u8m8 = __riscv_vle8_v_u8m8(text_u8, vl);
        // Seed the reduction accumulator with zero; widen u8 lanes and reduce into one u16 lane.
        vuint16m1_t zero = __riscv_vmv_v_x_u16m1(0, __riscv_vsetvl_e16m1(1));
        vuint16m1_t partial = __riscv_vwredsumu_vs_u8m8_u16m1(text_u8m8, zero, vl);
        bytesum += (sz_u64_t)__riscv_vmv_x_s_u16m1_u16(partial);
        text_u8 += vl, length -= vl;
    }
    return bytesum;
}

#pragma region RVV AES Round (vector-permute tower field)

/*  AES-based hashing on RVV without the optional `Zvkned` AES instructions.
 *
 *  The whole `sz_hash` family is built on a single AES encryption round
 *  (`sz_emulate_aesenc_si128_serial_`) so every backend must reproduce the *exact* same digest.
 *  RVV's base 1.0 profile (`rv64gcv`) has no AES opcodes, so we synthesize one round entirely from
 *  byte permutes and finite-field arithmetic, using Mike Hamburg's "vector permute AES" technique:
 *  the GF(2^8) S-box is evaluated through the composite (tower) field GF((2^4)^2), turning every
 *  256-entry table into a handful of 16-entry nibble lookups served by `vrgather.vv` over a single
 *  16-lane `u8m1` register (`vsetvl e8m1, vl = 16`).
 *
 *  Pipeline of one round, all on the 16 bytes of state:
 *    1. SubBytes  - map each byte into the tower field via two nibble gathers (a GF(2)-linear basis
 *                   change split over low/high input nibbles), invert in GF((2^4)^2) using nibble
 *                   square / multiply-by-constant / general-multiply (log+antilog) / GF(2^4)-inverse
 *                   gathers, then fold the inverse-isomorphism and the AES affine map (incl. the
 *                   `0x63` constant) back into two output nibble gathers.
 *    2. ShiftRows - a single `vrgather` permutation that also folds in the serial code's combined
 *                   ShiftRows byte ordering.
 *    3. MixColumns- per 4-byte group: `u = a0^a1^a2^a3`, then `out[j] = a[j] ^ u ^ xtime(a[j]^a[j+1])`
 *                   with `xtime` as `vsll` plus a conditional `^0x1b` selected by `vmsne`/`vmerge`.
 *    4. AddRoundKey - a `vxor` with the round key.
 *
 *  The tower-field isomorphism (subfield root `r = 0x5c`, minimal polynomial `Y^2 = A*Y + B` with
 *  `A = 2`, `B = 6` over GF(2^4) using reduction polynomial `x^4 + x + 1`) and all derived tables are
 *  fixed compile-time constants. The construction is validated to be bit-exact against
 *  `sz_emulate_aesenc_si128_serial_` over millions of random inputs at `vlen` 128 and 256. */

/*  Tower-field S-box tables (16 entries each, indexed by a nibble):
 *  - `M*Tbl` : GF(2)-linear basis change from the AES field into GF((2^4)^2), split by input nibble.
 *  - `G*Tbl` : composition of the inverse basis change with the AES affine *linear* part, split by
 *              the packed tower nibble (the constant `0x63` is added once, separately).
 *  - `SQ`    : `x^2` in GF(2^4).
 *  - `INV4`  : multiplicative inverse in GF(2^4).
 *  - `LOG`/`ALOG` : discrete log / antilog (base the GF(2^4) generator `2`) for general multiply.
 *  - `MUL_A`/`MUL_B` : multiply-by-`A` (=2) and by-`B` (=6) in GF(2^4). */
SZ_INTERNAL sz_u8_t const *sz_aes_tables_rvv_(void) {
    static sz_align_(16) sz_u8_t const tables[16 * 10] = {
        /* MlowTbl  */ 0, 1,   16,  17,  38,  39,  54,  55,  44,  45,  60,  61,  10,  11,  26,  27,
        /* MhighTbl */ 0, 140, 245, 121, 132, 8,   113, 253, 117, 249, 128, 12,  241, 125, 4,   136,
        /* GlowTbl  */ 0, 31,  178, 173, 171, 180, 25,  6,   54,  41,  132, 155, 157, 130, 47,  48,
        /* GhighTbl */ 0, 62,  101, 91,  96,  94,  5,   59,  108, 82,  9,   55,  12,  50,  105, 87,
        /* SQ       */ 0, 1,   4,   5,   3,   2,   7,   6,   12,  13,  8,   9,   15,  14,  11,  10,
        /* INV4     */ 0, 1,   9,   14,  13,  11,  7,   6,   15,  2,   12,  5,   10,  4,   3,   8,
        /* LOG      */ 0, 0,   1,   4,   2,   8,   5,   10,  3,   14,  9,   7,   6,   13,  11,  12,
        /* ALOG     */ 1, 2,   4,   8,   3,   6,   12,  11,  5,   10,  7,   14,  15,  13,  9,   1,
        /* MUL_A=*2 */ 0, 2,   4,   6,   8,   10,  12,  14,  3,   1,   7,   5,   11,  9,   15,  13,
        /* MUL_B=*6 */ 0, 6,   12,  10,  11,  13,  7,   1,   5,   3,   9,   15,  14,  8,   2,   4};
    return &tables[0];
}

/*  Combined ShiftRows permutation matching the serial code: `premix[j] = sbox[state[shiftrows[j]]]`. */
SZ_INTERNAL sz_u8_t const *sz_aes_shiftrows_rvv_(void) {
    static sz_align_(16) sz_u8_t const order[16] = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};
    return &order[0];
}

/*  Within-group rotate by +1 for MixColumns: lane `j -> 4*(j/4) + (j+1)%4`. */
SZ_INTERNAL sz_u8_t const *sz_aes_rot1_rvv_(void) {
    static sz_align_(16) sz_u8_t const rot1[16] = {1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12};
    return &rot1[0];
}

/*  General GF(2^4) multiply of two vector operands via log/antilog, with the discrete-log sum
 *  reduced modulo 15 so the antilog gather index stays inside the 16 active lanes, and a zero-select
 *  for the `0 * x` and `x * 0` cases. */
SZ_INTERNAL vuint8m1_t sz_gf16_mul_rvv_(vuint8m1_t a_u8m1, vuint8m1_t b_u8m1, vuint8m1_t log_table_u8m1,
                                        vuint8m1_t antilog_table_u8m1, sz_size_t vl) {
    vuint8m1_t log_a_vec = __riscv_vrgather_vv_u8m1(log_table_u8m1, a_u8m1, vl);
    vuint8m1_t log_b_vec = __riscv_vrgather_vv_u8m1(log_table_u8m1, b_u8m1, vl);
    vuint8m1_t log_sum_vec = __riscv_vadd_vv_u8m1(log_a_vec, log_b_vec, vl);
    vbool8_t overflow_mask = __riscv_vmsgeu_vx_u8m1_b8(log_sum_vec, 15, vl);
    log_sum_vec = __riscv_vmerge_vvm_u8m1(log_sum_vec, __riscv_vsub_vx_u8m1(log_sum_vec, 15, vl), overflow_mask, vl);
    vuint8m1_t product_vec = __riscv_vrgather_vv_u8m1(antilog_table_u8m1, log_sum_vec, vl);
    vbool8_t zero_mask = __riscv_vmor_mm_b8(__riscv_vmseq_vx_u8m1_b8(a_u8m1, 0, vl),
                                            __riscv_vmseq_vx_u8m1_b8(b_u8m1, 0, vl), vl);
    return __riscv_vmerge_vvm_u8m1(product_vec, __riscv_vmv_v_x_u8m1(0, vl), zero_mask, vl);
}

/**
 *  @brief Bit-exact RVV emulation of a single `_mm_aesenc_si128` round.
 *  @return Result of `MixColumns(SubBytes(ShiftRows(state))) ^ round_key`, identical to
 *          `sz_emulate_aesenc_si128_serial_`.
 *  @see Mike Hamburg, "Accelerating AES with Vector Permute Instructions" (CHES 2009).
 */
SZ_INTERNAL sz_u128_vec_t sz_emulate_aesenc_rvv_(sz_u128_vec_t state_vec, sz_u128_vec_t round_key_vec) {
    sz_size_t vl = __riscv_vsetvl_e8m1(sizeof(sz_u128_vec_t)); // the AES state is exactly one 128-bit block
    sz_u8_t const *tables = sz_aes_tables_rvv_();

    vuint8m1_t state_bytes_vec = __riscv_vle8_v_u8m1(state_vec.u8s, vl);
    vuint8m1_t key_vec = __riscv_vle8_v_u8m1(round_key_vec.u8s, vl);
    vuint8m1_t mlow_vec = __riscv_vle8_v_u8m1(tables + 16 * 0, vl);
    vuint8m1_t mhigh_vec = __riscv_vle8_v_u8m1(tables + 16 * 1, vl);
    vuint8m1_t glow_vec = __riscv_vle8_v_u8m1(tables + 16 * 2, vl);
    vuint8m1_t ghigh_vec = __riscv_vle8_v_u8m1(tables + 16 * 3, vl);
    vuint8m1_t square_vec = __riscv_vle8_v_u8m1(tables + 16 * 4, vl);
    vuint8m1_t inverse4_vec = __riscv_vle8_v_u8m1(tables + 16 * 5, vl);
    vuint8m1_t log_vec = __riscv_vle8_v_u8m1(tables + 16 * 6, vl);
    vuint8m1_t antilog_vec = __riscv_vle8_v_u8m1(tables + 16 * 7, vl);
    vuint8m1_t mul_a_vec = __riscv_vle8_v_u8m1(tables + 16 * 8, vl);
    vuint8m1_t mul_b_vec = __riscv_vle8_v_u8m1(tables + 16 * 9, vl);

    // SubBytes via tower-field inversion + affine.
    // Map each byte into GF((2^4)^2) as a packed nibble pair (hi << 4 | lo).
    vuint8m1_t low_nibble_vec = __riscv_vand_vx_u8m1(state_bytes_vec, 0x0f, vl);
    vuint8m1_t high_nibble_vec = __riscv_vsrl_vx_u8m1(state_bytes_vec, 4, vl);
    vuint8m1_t tower_vec = __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(mlow_vec, low_nibble_vec, vl),
                                                __riscv_vrgather_vv_u8m1(mhigh_vec, high_nibble_vec, vl), vl);
    vuint8m1_t tower_high_vec = __riscv_vsrl_vx_u8m1(tower_vec, 4, vl);
    vuint8m1_t tower_low_vec = __riscv_vand_vx_u8m1(tower_vec, 0x0f, vl);

    // Norm: norm = lo^2 + lo*hi*A + hi^2*B  (all in GF(2^4)).
    vuint8m1_t low_squared_vec = __riscv_vrgather_vv_u8m1(square_vec, tower_low_vec, vl);
    vuint8m1_t high_squared_vec = __riscv_vrgather_vv_u8m1(square_vec, tower_high_vec, vl);
    vuint8m1_t low_high_vec = sz_gf16_mul_rvv_(tower_low_vec, tower_high_vec, log_vec, antilog_vec, vl);
    vuint8m1_t low_high_a_vec = __riscv_vrgather_vv_u8m1(mul_a_vec, low_high_vec, vl);
    vuint8m1_t high_squared_b_vec = __riscv_vrgather_vv_u8m1(mul_b_vec, high_squared_vec, vl);
    vuint8m1_t norm_vec = __riscv_vxor_vv_u8m1(__riscv_vxor_vv_u8m1(low_squared_vec, low_high_a_vec, vl),
                                               high_squared_b_vec, vl);
    vuint8m1_t norm_inverse_vec = __riscv_vrgather_vv_u8m1(inverse4_vec, norm_vec, vl);

    // Inverse in the tower: new_high = hi*norm_inverse, new_low = (lo + hi*A) * norm_inverse.
    vuint8m1_t new_high_vec = sz_gf16_mul_rvv_(tower_high_vec, norm_inverse_vec, log_vec, antilog_vec, vl);
    vuint8m1_t high_a_vec = __riscv_vrgather_vv_u8m1(mul_a_vec, tower_high_vec, vl);
    vuint8m1_t low_xor_high_a_vec = __riscv_vxor_vv_u8m1(tower_low_vec, high_a_vec, vl);
    vuint8m1_t new_low_vec = sz_gf16_mul_rvv_(low_xor_high_a_vec, norm_inverse_vec, log_vec, antilog_vec, vl);

    // Map back out of the tower and apply the AES affine transform (linear part + 0x63).
    vuint8m1_t packed_inverse_vec = __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(new_high_vec, 4, vl), new_low_vec, vl);
    vuint8m1_t packed_low_vec = __riscv_vand_vx_u8m1(packed_inverse_vec, 0x0f, vl);
    vuint8m1_t packed_high_vec = __riscv_vsrl_vx_u8m1(packed_inverse_vec, 4, vl);
    vuint8m1_t sbox_vec = __riscv_vxor_vv_u8m1(__riscv_vrgather_vv_u8m1(glow_vec, packed_low_vec, vl),
                                               __riscv_vrgather_vv_u8m1(ghigh_vec, packed_high_vec, vl), vl);
    sbox_vec = __riscv_vxor_vx_u8m1(sbox_vec, 0x63, vl);

    // ShiftRows (combined ordering): premix[j] = sbox[shiftrows[j]].
    vuint8m1_t shiftrows_index_vec = __riscv_vle8_v_u8m1(sz_aes_shiftrows_rvv_(), vl);
    vuint8m1_t premix_vec = __riscv_vrgather_vv_u8m1(sbox_vec, shiftrows_index_vec, vl);

    // MixColumns over groups of 4: out[j] = premix[j] ^ group_xor ^ xtime(premix[j] ^ premix[(j+1)%4]).
    vuint8m1_t rot1_index_vec = __riscv_vle8_v_u8m1(sz_aes_rot1_rvv_(), vl);
    vuint8m1_t premix_next_vec = __riscv_vrgather_vv_u8m1(premix_vec, rot1_index_vec, vl);
    vuint8m1_t diff_vec = __riscv_vxor_vv_u8m1(premix_vec, premix_next_vec, vl);
    vuint8m1_t shifted_vec = __riscv_vsll_vx_u8m1(diff_vec, 1, vl);
    vbool8_t high_bit_mask = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(diff_vec, 0x80, vl), 0, vl);
    vuint8m1_t shifted_reduced_vec = __riscv_vxor_vx_u8m1(shifted_vec, 0x1b, vl);
    vuint8m1_t xtime_vec = __riscv_vmerge_vvm_u8m1(shifted_vec, shifted_reduced_vec, high_bit_mask, vl);

    // group_xor = xor of the 4 lanes in each group, broadcast across the group via successive rotations.
    vuint8m1_t group_xor_vec = premix_vec;
    group_xor_vec = __riscv_vxor_vv_u8m1(group_xor_vec, premix_next_vec, vl);
    vuint8m1_t rot2_index_vec = __riscv_vrgather_vv_u8m1(rot1_index_vec, rot1_index_vec, vl);
    group_xor_vec = __riscv_vxor_vv_u8m1(group_xor_vec, __riscv_vrgather_vv_u8m1(premix_vec, rot2_index_vec, vl), vl);
    vuint8m1_t rot3_index_vec = __riscv_vrgather_vv_u8m1(rot2_index_vec, rot1_index_vec, vl);
    group_xor_vec = __riscv_vxor_vv_u8m1(group_xor_vec, __riscv_vrgather_vv_u8m1(premix_vec, rot3_index_vec, vl), vl);

    vuint8m1_t out_vec = __riscv_vxor_vv_u8m1(__riscv_vxor_vv_u8m1(premix_vec, group_xor_vec, vl), xtime_vec, vl);

    // AddRoundKey.
    out_vec = __riscv_vxor_vv_u8m1(out_vec, key_vec, vl);

    sz_u128_vec_t result;
    __riscv_vse8_v_u8m1(result.u8s, out_vec, vl);
    return result;
}

#pragma endregion // RVV AES Round

#pragma region RVV Hash Drivers

/*  These drivers mirror the serial ones exactly, substituting `sz_emulate_aesenc_rvv_` for the
 *  serial AES round. Every non-AES step (the additive `sum` shuffle, length folding, block layout)
 *  reuses the shared serial helpers, so the digests are guaranteed value-identical. */

SZ_INTERNAL void sz_hash_minimal_update_rvv_(sz_hash_minimal_t_ *state, sz_u128_vec_t block) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    state->aes = sz_emulate_aesenc_rvv_(state->aes, block);
    state->sum = sz_emulate_shuffle_epi8_serial_(state->sum, shuffle);
    state->sum.u64s[0] += block.u64s[0], state->sum.u64s[1] += block.u64s[1];
}

SZ_INTERNAL sz_u64_t sz_hash_minimal_finalize_rvv_(sz_hash_minimal_t_ const *state, sz_size_t length) {
    sz_u128_vec_t key_with_length = state->key;
    key_with_length.u64s[0] += length;
    sz_u128_vec_t mixed = sz_emulate_aesenc_rvv_(state->sum, state->aes);
    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_rvv_(sz_emulate_aesenc_rvv_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

SZ_INTERNAL void sz_hash_state_update_rvv_(sz_hash_state_t *state) {
    sz_u8_t const *shuffle = sz_hash_u8x16x4_shuffle_();
    sz_u128_vec_t *aes_vecs = (sz_u128_vec_t *)state->aes;
    sz_u128_vec_t *sum_vecs = (sz_u128_vec_t *)state->sum;
    sz_u128_vec_t *ins_vecs = (sz_u128_vec_t *)state->ins;

    aes_vecs[0] = sz_emulate_aesenc_rvv_(aes_vecs[0], ins_vecs[0]);
    sum_vecs[0] = sz_emulate_shuffle_epi8_serial_(sum_vecs[0], shuffle);
    sum_vecs[0].u64s[0] += ins_vecs[0].u64s[0], sum_vecs[0].u64s[1] += ins_vecs[0].u64s[1];

    aes_vecs[1] = sz_emulate_aesenc_rvv_(aes_vecs[1], ins_vecs[1]);
    sum_vecs[1] = sz_emulate_shuffle_epi8_serial_(sum_vecs[1], shuffle);
    sum_vecs[1].u64s[0] += ins_vecs[1].u64s[0], sum_vecs[1].u64s[1] += ins_vecs[1].u64s[1];

    aes_vecs[2] = sz_emulate_aesenc_rvv_(aes_vecs[2], ins_vecs[2]);
    sum_vecs[2] = sz_emulate_shuffle_epi8_serial_(sum_vecs[2], shuffle);
    sum_vecs[2].u64s[0] += ins_vecs[2].u64s[0], sum_vecs[2].u64s[1] += ins_vecs[2].u64s[1];

    aes_vecs[3] = sz_emulate_aesenc_rvv_(aes_vecs[3], ins_vecs[3]);
    sum_vecs[3] = sz_emulate_shuffle_epi8_serial_(sum_vecs[3], shuffle);
    sum_vecs[3].u64s[0] += ins_vecs[3].u64s[0], sum_vecs[3].u64s[1] += ins_vecs[3].u64s[1];
}

SZ_INTERNAL sz_u64_t sz_hash_state_finalize_rvv_(sz_hash_state_t const *state) {
    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_u128_vec_t key_with_length;
    key_with_length.u64s[0] = key_u64s[0] + state->ins_length;
    key_with_length.u64s[1] = key_u64s[1];

    sz_u128_vec_t const *aes_vecs = (sz_u128_vec_t const *)state->aes;
    sz_u128_vec_t const *sum_vecs = (sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t mixed0 = sz_emulate_aesenc_rvv_(sum_vecs[0], aes_vecs[0]);
    sz_u128_vec_t mixed1 = sz_emulate_aesenc_rvv_(sum_vecs[1], aes_vecs[1]);
    sz_u128_vec_t mixed2 = sz_emulate_aesenc_rvv_(sum_vecs[2], aes_vecs[2]);
    sz_u128_vec_t mixed3 = sz_emulate_aesenc_rvv_(sum_vecs[3], aes_vecs[3]);

    sz_u128_vec_t mixed01 = sz_emulate_aesenc_rvv_(mixed0, mixed1);
    sz_u128_vec_t mixed23 = sz_emulate_aesenc_rvv_(mixed2, mixed3);
    sz_u128_vec_t mixed = sz_emulate_aesenc_rvv_(mixed01, mixed23);

    sz_u128_vec_t mixed_in_register = sz_emulate_aesenc_rvv_(sz_emulate_aesenc_rvv_(mixed, key_with_length), mixed);
    return mixed_in_register.u64s[0];
}

/** @brief  Vector-copy a single AES block (`sizeof(sz_u128_vec_t)` bytes) from `source` into `target->u8s`,
 *          replacing a scalar byte loop. `source` must have a full block of readable bytes. */
SZ_INTERNAL void sz_hash_load_block_rvv_(sz_u128_vec_t *target, sz_cptr_t source) {
    sz_size_t vl = __riscv_vsetvl_e8m1(sizeof(target->u8s));
    __riscv_vse8_v_u8m1(target->u8s, __riscv_vle8_v_u8m1((sz_u8_t const *)source, vl), vl);
}

SZ_PUBLIC SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_rvv(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {
    sz_size_t const block = sizeof(sz_u128_vec_t); // one AES block
    if (length <= block) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data_vec;
        data_vec.u64s[0] = data_vec.u64s[1] = 0;
        // A length-agnostic load drops the zero pad and the scalar byte loop: VL covers the partial bytes.
        sz_size_t vl = __riscv_vsetvl_e8m1(length);
        __riscv_vse8_v_u8m1(data_vec.u8s, __riscv_vle8_v_u8m1((sz_u8_t const *)start, vl), vl);
        sz_hash_minimal_update_rvv_(&state, data_vec);
        return sz_hash_minimal_finalize_rvv_(&state, length);
    }
    else if (length <= 2 * block) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec;
        sz_hash_load_block_rvv_(&data0_vec, start);
        sz_hash_load_block_rvv_(&data1_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data1_vec, (int)(2 * block - length));
        sz_hash_minimal_update_rvv_(&state, data0_vec);
        sz_hash_minimal_update_rvv_(&state, data1_vec);
        return sz_hash_minimal_finalize_rvv_(&state, length);
    }
    else if (length <= 3 * block) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        sz_hash_load_block_rvv_(&data0_vec, start);
        sz_hash_load_block_rvv_(&data1_vec, start + block);
        sz_hash_load_block_rvv_(&data2_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data2_vec, (int)(3 * block - length));
        sz_hash_minimal_update_rvv_(&state, data0_vec);
        sz_hash_minimal_update_rvv_(&state, data1_vec);
        sz_hash_minimal_update_rvv_(&state, data2_vec);
        return sz_hash_minimal_finalize_rvv_(&state, length);
    }
    else if (length <= 4 * block) {
        sz_align_(16) sz_hash_minimal_t_ state;
        sz_hash_minimal_init_serial_(&state, seed);
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        sz_hash_load_block_rvv_(&data0_vec, start);
        sz_hash_load_block_rvv_(&data1_vec, start + block);
        sz_hash_load_block_rvv_(&data2_vec, start + 2 * block);
        sz_hash_load_block_rvv_(&data3_vec, start + length - block);
        sz_hash_shift_in_register_serial_(&data3_vec, (int)(4 * block - length));
        sz_hash_minimal_update_rvv_(&state, data0_vec);
        sz_hash_minimal_update_rvv_(&state, data1_vec);
        sz_hash_minimal_update_rvv_(&state, data2_vec);
        sz_hash_minimal_update_rvv_(&state, data3_vec);
        return sz_hash_minimal_finalize_rvv_(&state, length);
    }
    else {
        sz_align_(64) sz_hash_state_t state;
        sz_size_t const window = sizeof(state.ins); // the 64-byte hashing window
        sz_hash_state_init_serial(&state, seed);
        for (; state.ins_length + window <= length; state.ins_length += window) {
            sz_size_t vl = __riscv_vsetvl_e8m8(window); // VLEN >= 128 -> one whole-window transfer
            __riscv_vse8_v_u8m8(state.ins, __riscv_vle8_v_u8m8((sz_u8_t const *)start + state.ins_length, vl), vl);
            sz_hash_state_update_rvv_(&state);
        }
        if (state.ins_length < length) {
            sz_size_t zero_vl = __riscv_vsetvl_e8m8(window);
            __riscv_vse8_v_u8m8(state.ins, __riscv_vmv_v_x_u8m8(0, zero_vl), zero_vl);
            sz_size_t tail_vl = __riscv_vsetvl_e8m8(length - state.ins_length); // VL covers the partial tail natively
            __riscv_vse8_v_u8m8(state.ins, __riscv_vle8_v_u8m8((sz_u8_t const *)start + state.ins_length, tail_vl),
                                tail_vl);
            sz_hash_state_update_rvv_(&state);
            state.ins_length = length;
        }
        return sz_hash_state_finalize_rvv_(&state);
    }
}

SZ_PUBLIC void sz_hash_state_init_rvv(sz_hash_state_t *state, sz_u64_t seed) { sz_hash_state_init_serial(state, seed); }

SZ_PUBLIC void sz_hash_state_update_rvv(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    while (length) {
        sz_size_t progress_in_block = state->ins_length % 64;
        sz_size_t to_copy = sz_min_of_two(length, 64 - progress_in_block);
        int const will_fill_block = progress_in_block + to_copy == 64;
        state->ins_length += to_copy;
        length -= to_copy;
        while (to_copy--) state->ins[progress_in_block++] = *text++;
        if (will_fill_block) {
            sz_hash_state_update_rvv_(state);
            sz_size_t clear_vl = __riscv_vsetvl_e8m8(sizeof(state->ins));
            __riscv_vse8_v_u8m8(state->ins, __riscv_vmv_v_x_u8m8(0, clear_vl), clear_vl);
        }
    }
}

SZ_PUBLIC sz_u64_t sz_hash_state_digest_rvv(sz_hash_state_t const *state) {
    sz_size_t length = state->ins_length;
    if (length >= 64) return sz_hash_state_finalize_rvv_(state);

    sz_u64_t const *key_u64s = (sz_u64_t const *)state->key;
    sz_hash_minimal_t_ minimal_state;
    minimal_state.key.u64s[0] = key_u64s[0];
    minimal_state.key.u64s[1] = key_u64s[1];
    minimal_state.aes = *(sz_u128_vec_t const *)state->aes;
    minimal_state.sum = *(sz_u128_vec_t const *)state->sum;

    sz_u128_vec_t const *ins_vecs = (sz_u128_vec_t const *)state->ins;
    if (length <= 16) {
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[0]);
        return sz_hash_minimal_finalize_rvv_(&minimal_state, length);
    }
    else if (length <= 32) {
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[1]);
        return sz_hash_minimal_finalize_rvv_(&minimal_state, length);
    }
    else if (length <= 48) {
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[2]);
        return sz_hash_minimal_finalize_rvv_(&minimal_state, length);
    }
    else {
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[0]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[1]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[2]);
        sz_hash_minimal_update_rvv_(&minimal_state, ins_vecs[3]);
        return sz_hash_minimal_finalize_rvv_(&minimal_state, length);
    }
}

SZ_PUBLIC void sz_fill_random_rvv(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_u64_t const *pi_constants = sz_hash_pi_constants_();
    sz_u128_vec_t input_vec, pi_vec, key_vec, generated_vec;
    for (sz_size_t lane_index = 0; length; ++lane_index) {
        input_vec.u64s[0] = input_vec.u64s[1] = nonce + lane_index;
        pi_vec = ((sz_u128_vec_t const *)pi_constants)[lane_index % 4];
        key_vec.u64s[0] = nonce ^ pi_vec.u64s[0];
        key_vec.u64s[1] = nonce ^ pi_vec.u64s[1];
        generated_vec = sz_emulate_aesenc_rvv_(input_vec, key_vec);
        // VL deletes both the per-byte loop and the `&& length` tail guard: it clamps to the bytes left,
        // capped at the block we just generated.
        sz_size_t out_vl = __riscv_vsetvl_e8m1(sz_min_of_two(length, sizeof(generated_vec.u8s)));
        __riscv_vse8_v_u8m1((sz_u8_t *)text, __riscv_vle8_v_u8m1(generated_vec.u8s, out_vl), out_vl);
        text += out_vl, length -= out_vl;
    }
}

#pragma endregion // RVV Hash Drivers

/*  SHA-256 has no AES structure to vectorize within RVV's base profile, so it stays serial. */
SZ_PUBLIC void sz_sha256_state_init_rvv(sz_sha256_state_t *state_ptr) { sz_sha256_state_init_serial(state_ptr); }

SZ_PUBLIC void sz_sha256_state_update_rvv(sz_sha256_state_t *state_ptr, sz_cptr_t data, sz_size_t length) {
    sz_sha256_state_update_serial(state_ptr, data, length);
}

SZ_PUBLIC void sz_sha256_state_digest_rvv(sz_sha256_state_t const *state_ptr, sz_u8_t digest[sz_at_least_(32)]) {
    sz_sha256_state_digest_serial(state_ptr, digest);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_RVV_H_
