/**
 *  @file include/stringzilla/cipher/serial.h
 *  @author Ash Vardanian
 *  @date August 4, 2026
 *  @brief Serial (scalar) backend for AES-256 encryption in counter and Galois/counter modes.
 *
 *  @sa include/stringzilla/cipher.h
 */
#ifndef STRINGZILLA_CIPHER_SERIAL_H_
#define STRINGZILLA_CIPHER_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h" // `sz_fill_serial_`
#include "stringzilla/cipher/tables.h" // `sz_aes_sbox_`

#ifdef __cplusplus
extern "C" {
#endif

/** Bytes in an AES block, fixed by FIPS 197. */
#define STRINGZILLA_AES_BLOCK_LENGTH (16)

/** Bytes in an AES-256 secret key. */
#define STRINGZILLA_AES256_KEY_LENGTH (32)

/** Bytes in the nonce both modes accept, as NIST SP 800-38D recommends for Galois/counter mode. */
#define STRINGZILLA_AES256_NONCE_LENGTH (12)

/** Bytes in a Galois/counter mode authentication tag. */
#define STRINGZILLA_AES256_TAG_LENGTH (16)

/** Round keys in an AES-256 schedule: 15 of them, four 32-bit words each. */
#define STRINGZILLA_AES256_ROUND_KEYS (60)

/**
 *  @brief An expanded AES-256 round-key schedule, all that counter mode, or CTR, needs.
 *  @sa sz_aes256_key_init_best, sz_aes256_ctr_xor_best
 *
 *  @note Deliberately separate from @c sz_aes256_gcm_key_t. Folding the Galois hash powers in here
 *      would make every counter-mode caller carry 128 bytes it never reads, and pay eight field
 *      multiplications at key setup for a mode that has no authentication tag.
 */
typedef struct sz_aes256_key_t {

    /** 15 round keys of 4 words each, in encryption order. */
    sz_u32_t round_keys[STRINGZILLA_AES256_ROUND_KEYS];
} sz_aes256_key_t;

/**
 *  @brief An AES-256 schedule together with the Galois hash subkey powers, the key Galois/counter
 *      mode needs. Galois/counter mode is abbreviated GCM, and its hash is GHASH.
 *  @sa sz_aes256_gcm_key_init_best, sz_aes256_gcm_encrypt_best, sz_aes256_gcm_encryptor_init_best
 *
 *  The powers H¹ through H⁸ are derived once per key so the wide backends can aggregate eight
 *  blocks between field reductions. Counter mode never touches them, which is why they live here
 *  rather than in @c sz_aes256_key_t.
 */
typedef struct sz_aes256_gcm_key_t {

    /** The round-key schedule. */
    sz_aes256_key_t block;

    /** H¹ through H⁸, ascending. */
    sz_u8_t powers[8 * STRINGZILLA_AES_BLOCK_LENGTH];
} sz_aes256_gcm_key_t;

/**
 *  @brief What both directions of a chunked transformation carry.
 *  @sa sz_aes256_gcm_encryptor_t, sz_aes256_gcm_decryptor_t
 *
 *  Callers construct an encryptor or a decryptor, never this. It is public because the per-backend
 *  kernels operate on it and the differential tests compare it field by field.
 *
 *  @note The key is embedded rather than referenced. A pointer would save the copy, but nothing in
 *      a transparent C struct can stop a state from outliving the key it points at, and that
 *      failure is silent. The copy costs about two percent of a 16 KB record.
 */
typedef struct sz_aes256_gcm_state_t {

    /** Embedded, so the state cannot outlive its key. */
    sz_aes256_gcm_key_t key;

    /** Running Galois hash value. */
    sz_u8_t accumulator[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Current counter block, big-endian in the low four bytes. */
    sz_u8_t counter[STRINGZILLA_AES_BLOCK_LENGTH];

    /** The encrypted initial counter, folded in at digest time. */
    sz_u8_t tag_mask[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Hash bytes awaiting a full block, associated then ciphertext. */
    sz_u8_t partial[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Current keystream block, carried across chunk boundaries. */
    sz_u8_t keystream[STRINGZILLA_AES_BLOCK_LENGTH];

    /** Total associated bytes absorbed. */
    sz_u64_t associated_length;

    /** Total message bytes transformed. */
    sz_u64_t text_length;

    /** Valid bytes in @c partial, 0 to 15. */
    sz_u8_t buffered;

    /** Bytes of @c keystream already spent, 0 to 16. */
    sz_u8_t keystream_used;
} sz_aes256_gcm_state_t;

/**
 *  @brief Seals a chunked message under Galois/counter mode, or GCM, emitting a tag at the end.
 *  @sa sz_aes256_gcm_encryptor_init_best, sz_aes256_gcm_encryptor_update_best,
 *      sz_aes256_gcm_encryptor_digest_best
 *
 *  Distinct from @c sz_aes256_gcm_decryptor_t so that the direction is carried by the type rather
 *  than by a field. The Galois hash absorbs ciphertext either way - the output buffer when sealing,
 *  the input buffer when opening - and a single state that could be pointed either way makes that a
 *  runtime question the compiler cannot help with. Passing one of these where the other belongs
 *  will not compile.
 *
 *  A chunk boundary must be invisible to the result. The keystream block and the hash block are
 *  both sixteen bytes wide and a caller's chunks are not, so both have to survive between calls -
 *  which is why @c keystream_used exists alongside @c buffered instead of one shared counter.
 */
typedef struct sz_aes256_gcm_encryptor_t {

    /** The shared payload. */
    sz_aes256_gcm_state_t state;
} sz_aes256_gcm_encryptor_t;

/**
 *  @brief Opens a chunked message under Galois/counter mode, or GCM, checking its tag at the end.
 *  @sa sz_aes256_gcm_decryptor_init_best, sz_aes256_gcm_decryptor_update_unverified_best,
 *      sz_aes256_gcm_decryptor_verify_best
 */
typedef struct sz_aes256_gcm_decryptor_t {

    /** The shared payload. */
    sz_aes256_gcm_state_t state;
} sz_aes256_gcm_decryptor_t;

/*  Optimize this tier for size. The round function is a table-driven scalar loop that no amount of
 *  unrolling makes competitive with AES-NI, so what it expands into is pure footprint - 41 KB of
 *  `.text` under GCC, 6 KB under Clang, against 3 KB apiece once the compilers stop widening it.
 *  The scope is exact: the annotation reaches functions alone, leaving the types and tables
 *  untouched, and neither compiler moves a byte of the vector backends that include them. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((minsize)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("Os")
#endif

/*  This backend exists so every target has a correct implementation, and so the vector backends
 *  have a reference to be differentiated against. It is not the fast path anywhere.
 *
 *  The substitution box is a lookup table, which means the round function's timing depends on the
 *  key through the cache. That is acceptable for @c sz_hash_best, whose inputs are public, and it
 *  is not acceptable in principle for a cipher. Platforms with AES instructions never reach this
 *  code; those without it have no constant-time alternative short of bit-slicing, which costs an
 *  order of magnitude. The Galois hash below deliberately does @b not follow suit: its subkey is
 *  derived from the key, so a subkey-indexed table would leak the key directly, and it uses a
 *  branch-free shift-and-add instead. */

/**
 *  @brief Which buffer the Galois hash absorbs, for the one transform both directions share.
 *
 *  It is internal vocabulary, absent from every public signature, so it lives here, not in the hub.
 *
 *  @sa sz_aes256_gcm_encryptor_t, sz_aes256_gcm_decryptor_t.
 */
typedef enum sz_aes256_gcm_direction_t {

    /** The hash absorbs the transformed bytes, which are the output. */
    sz_aes256_gcm_encrypting_k = 0,

    /** The hash absorbs the untransformed bytes, which are the input. */
    sz_aes256_gcm_decrypting_k = 1,
} sz_aes256_gcm_direction_t;

#pragma region Key Schedule

/**
 *  @brief Packs four schedule bytes into one word, byte zero in the least significant position.
 *  @return The packed word.
 *
 *  FIPS 197 writes a key-schedule word as the byte quadruple @b (a0,a1,a2,a3).
 */
STRINGZILLA_INLINE sz_u32_t sz_aes256_word_pack_serial_(sz_u8_t const *bytes) {
    return (sz_u32_t)bytes[0] | ((sz_u32_t)bytes[1] << 8) | ((sz_u32_t)bytes[2] << 16) | ((sz_u32_t)bytes[3] << 24);
}

/** Substitutes every byte of a schedule word through the substitution box. */
STRINGZILLA_INLINE sz_u32_t sz_aes256_word_substitute_serial_(sz_u32_t word) {
    sz_u8_t const *sbox = sz_aes_sbox_();
    return (sz_u32_t)sbox[word & 0xFFu] |                 //
           ((sz_u32_t)sbox[(word >> 8) & 0xFFu] << 8) |   //
           ((sz_u32_t)sbox[(word >> 16) & 0xFFu] << 16) | //
           ((sz_u32_t)sbox[(word >> 24) & 0xFFu] << 24);
}

/** Rotates a schedule word so @b (a0,a1,a2,a3) becomes @b (a1,a2,a3,a0). */
STRINGZILLA_INLINE sz_u32_t sz_aes256_word_rotate_serial_(sz_u32_t word) { return (word >> 8) | (word << 24); }

STRINGZILLA_INLINE void sz_aes256_key_init_serial_(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    static sz_u8_t const round_constants[7] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40};
    sz_size_t word_index = 0;
    for (; word_index != 8; ++word_index)
        key->round_keys[word_index] = sz_aes256_word_pack_serial_(secret + word_index * 4);
    for (; word_index != STRINGZILLA_AES256_ROUND_KEYS; ++word_index) {
        sz_u32_t carried_word = key->round_keys[word_index - 1];
        if ((word_index & 7) == 0) {
            carried_word = sz_aes256_word_substitute_serial_(sz_aes256_word_rotate_serial_(carried_word));
            carried_word ^= (sz_u32_t)round_constants[(word_index >> 3) - 1];
        }
        else if ((word_index & 7) == 4) { carried_word = sz_aes256_word_substitute_serial_(carried_word); }
        key->round_keys[word_index] = key->round_keys[word_index - 8] ^ carried_word;
    }
}

#pragma endregion Key Schedule

#pragma region Block Encryption

/**
 *  @brief Applies @c SubBytes and @c ShiftRows to a block, writing the column-major result.
 *  @param[in] block The 16 input bytes.
 *  @param[out] shifted Receives the substituted and row-shifted bytes.
 */
STRINGZILLA_INLINE void sz_aes256_substitute_and_shift_serial_(sz_u8_t const *block, sz_u8_t *shifted) {
    sz_u8_t const *sbox = sz_aes_sbox_();
    static sz_u8_t const source_of[16] = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};
    for (sz_size_t byte_index = 0; byte_index != 16; ++byte_index)
        shifted[byte_index] = sbox[block[source_of[byte_index]]];
}

/**
 *  @brief Smears the most significant bit of @p value across the whole byte.
 *  @return All ones when the bit was set, all zeros otherwise.
 *
 *  Negating a zero-or-one replicates it, which is how a byte selects a value without a branch.
 */
STRINGZILLA_INLINE sz_u8_t sz_u8_top_bit_smear_(sz_u8_t value) { return (sz_u8_t)(0u - (sz_u8_t)(value >> 7)); }

/** Doubles a byte in GF(2⁸) under the AES reduction polynomial. */
STRINGZILLA_INLINE sz_u8_t sz_aes256_gf_double_serial_(sz_u8_t value) {
    return (sz_u8_t)((sz_u8_t)(value << 1) ^ (sz_u8_t)(0x1Bu & sz_u8_top_bit_smear_(value)));
}

/**
 *  @brief Encrypts one 16-byte block with the expanded schedule.
 *  @param[in] key The expanded schedule.
 *  @param[in] block The 16 plaintext bytes.
 *  @param[out] output Receives the 16 ciphertext bytes; may alias @p block.
 */
STRINGZILLA_INLINE void sz_aes256_block_encrypt_serial_(sz_aes256_key_t const *key, sz_u8_t const *block,
                                                        sz_u8_t *output) {
    sz_u8_t state[16], shifted[16];
    sz_size_t round_index, byte_index, column_index;

    for (byte_index = 0; byte_index != 16; ++byte_index) {
        sz_u32_t const round_key = key->round_keys[byte_index >> 2];
        state[byte_index] = (sz_u8_t)(block[byte_index] ^ (sz_u8_t)(round_key >> ((byte_index & 3) * 8)));
    }

    for (round_index = 1; round_index != 14; ++round_index) {
        sz_aes256_substitute_and_shift_serial_(state, shifted);
        for (column_index = 0; column_index != 4; ++column_index) {
            sz_u8_t const *column = shifted + column_index * 4;
            sz_u8_t const parity = (sz_u8_t)(column[0] ^ column[1] ^ column[2] ^ column[3]);
            sz_u8_t const first_byte = column[0];
            sz_u8_t mixed[4];
            mixed[0] = (sz_u8_t)(column[0] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[0] ^ column[1])));
            mixed[1] = (sz_u8_t)(column[1] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[1] ^ column[2])));
            mixed[2] = (sz_u8_t)(column[2] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[2] ^ column[3])));
            mixed[3] = (sz_u8_t)(column[3] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[3] ^ first_byte)));
            for (byte_index = 0; byte_index != 4; ++byte_index) {
                sz_size_t const target = column_index * 4 + byte_index;
                sz_u32_t const round_key = key->round_keys[round_index * 4 + column_index];
                state[target] = (sz_u8_t)(mixed[byte_index] ^ (sz_u8_t)(round_key >> (byte_index * 8)));
            }
        }
    }

    sz_aes256_substitute_and_shift_serial_(state, shifted);
    for (byte_index = 0; byte_index != 16; ++byte_index) {
        sz_u32_t const round_key = key->round_keys[14 * 4 + (byte_index >> 2)];
        output[byte_index] = (sz_u8_t)(shifted[byte_index] ^ (sz_u8_t)(round_key >> ((byte_index & 3) * 8)));
    }
}

#pragma endregion Block Encryption

#pragma region Counter Mode

/** Builds the counter block for a given block index: the nonce then a big-endian 32-bit counter. */
STRINGZILLA_INLINE void sz_aes256_counter_block_serial_(sz_u8_t const *nonce, sz_u32_t block_index, sz_u8_t *block) {
    for (sz_size_t byte_index = 0; byte_index != 12; ++byte_index) block[byte_index] = nonce[byte_index];
    block[12] = (sz_u8_t)(block_index >> 24);
    block[13] = (sz_u8_t)(block_index >> 16);
    block[14] = (sz_u8_t)(block_index >> 8);
    block[15] = (sz_u8_t)(block_index >> 0);
}

#pragma endregion Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Multiplies @p accumulator by @p subkey in the Galois field the tag is built over.
 *  @param[inout] accumulator The running hash, replaced by the product.
 *  @param[in] subkey One of the precomputed powers of the hash subkey.
 *
 *  Deliberately branch free and table free.
 */
STRINGZILLA_CONSTEXPR void sz_ghash_multiply_serial_(sz_u8_t *accumulator, sz_u8_t const *subkey) {
    sz_u8_t product[16], operand[16];
    sz_size_t byte_index, bit_index;

    for (byte_index = 0; byte_index != 16; ++byte_index)
        product[byte_index] = 0, operand[byte_index] = subkey[byte_index];

    for (bit_index = 0; bit_index != 128; ++bit_index) {
        sz_u8_t const selected_bit = (sz_u8_t)((accumulator[bit_index >> 3] >> (7 - (bit_index & 7))) & 1u);
        sz_u8_t const mask = (sz_u8_t)(0u - selected_bit);
        sz_u8_t const carried_bit = (sz_u8_t)(operand[15] & 1u);
        for (byte_index = 0; byte_index != 16; ++byte_index)
            product[byte_index] ^= (sz_u8_t)(operand[byte_index] & mask);
        for (byte_index = 15; byte_index != 0; --byte_index)
            operand[byte_index] = (sz_u8_t)((operand[byte_index] >> 1) | (sz_u8_t)(operand[byte_index - 1] << 7));
        operand[0] = (sz_u8_t)(operand[0] >> 1);
        operand[0] ^= (sz_u8_t)(0xE1u & (sz_u8_t)(0u - carried_bit));
    }

    for (byte_index = 0; byte_index != 16; ++byte_index) accumulator[byte_index] = product[byte_index];
}

/** Absorbs one whole block into the running hash. */
STRINGZILLA_CONSTEXPR void sz_ghash_absorb_serial_(sz_u8_t *accumulator, sz_u8_t const *block, sz_u8_t const *subkey) {
    for (sz_size_t byte_index = 0; byte_index != 16; ++byte_index) accumulator[byte_index] ^= block[byte_index];
    sz_ghash_multiply_serial_(accumulator, subkey);
}

#pragma endregion Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Compares two authentication tags in time that does not depend on their contents.
 *  @return `sz_true_k` when all sixteen bytes match.
 *
 *  Accumulates every difference instead of returning at the first one, so an attacker cannot
 *  recover a forged tag byte by byte from how long the comparison ran.
 */
STRINGZILLA_INLINE sz_bool_t sz_aes256_tag_equal_serial_(sz_u8_t const *first, sz_u8_t const *second) {
    sz_u8_t difference = 0;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != STRINGZILLA_AES_BLOCK_LENGTH; ++byte_index)
        difference |= (sz_u8_t)(first[byte_index] ^ second[byte_index]);
    return difference == 0 ? sz_true_k : sz_false_k;
}

/**
 *  @brief Fills the closing hash block with the associated-data and message
 *      bit lengths, big-endian.
 *  @param[out] lengths_vec Receives the sixteen bytes.
 *  @param[in] associated_length Bytes of associated data absorbed.
 *  @param[in] text_length Bytes of message absorbed.
 *
 *  Every backend closes its hash with this block, so it lives here rather than eight times over.
 */
STRINGZILLA_INLINE void sz_aes256_gcm_lengths_serial_(sz_u128_vec_t *lengths_vec, sz_u64_t associated_length,
                                                      sz_u64_t text_length) {
    sz_u64_t const associated_bits = associated_length * 8, text_bits = text_length * 8;
#if STRINGZILLA_ARCH_BIG_ENDIAN_
    lengths_vec->u64s[0] = associated_bits;
    lengths_vec->u64s[1] = text_bits;
#else
    lengths_vec->u64s[0] = sz_u64_bytes_reverse(associated_bits);
    lengths_vec->u64s[1] = sz_u64_bytes_reverse(text_bits);
#endif
}

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  Ordinary stores followed by one barrier, rather than writes through a @c volatile view:
 *  @c volatile forbids vectorization, so that form would cost 472 single-byte stores on
 *  every one-shot call.
 */
STRINGZILLA_INLINE void sz_aes256_gcm_state_scrub_serial_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != sizeof(*state); ++byte_index) bytes[byte_index] = 0;
    sz_keep_alive_(state);
}

/** Prepares the payload both directions share: counter block, tag mask and empty carries. */
STRINGZILLA_INLINE void sz_aes256_gcm_begin_serial_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
                                                    sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_size_t byte_index;
    sz_u8_t initial[16];

    state->key = *key;
    for (byte_index = 0; byte_index != 16; ++byte_index)
        state->accumulator[byte_index] = 0, state->partial[byte_index] = 0, state->keystream[byte_index] = 0;

    // With a twelve-byte nonce the initial counter block is the nonce followed by a one, and the value
    // encrypted under it masks the finished hash. Data blocks start one past it.
    sz_aes256_counter_block_serial_(nonce, 1u, initial);
    sz_aes256_block_encrypt_serial_(&state->key.block, initial, state->tag_mask);
    for (byte_index = 0; byte_index != 16; ++byte_index) state->counter[byte_index] = initial[byte_index];

    state->associated_length = 0;
    state->text_length = 0;
    state->buffered = 0;
    state->keystream_used = STRINGZILLA_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** Advances the big-endian counter occupying the last four bytes of the counter block. */
STRINGZILLA_INLINE void sz_aes256_counter_advance_serial_(sz_u8_t *counter) {
    for (sz_size_t byte_index = 16; byte_index-- != 12;)
        if (++counter[byte_index] != 0) break;
}

/** Absorbs associated data into the payload both directions share. */
STRINGZILLA_INLINE void sz_aes256_gcm_associate_serial_(sz_aes256_gcm_state_t *state, sz_cptr_t text,
                                                        sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_size_t consumed = 0;
    sz_assert_(state->text_length == 0 && "Associated data must precede the message");

    // Associated data is hashed but never encrypted, so a partial block is completed in place.
    while (consumed != length) {
        sz_size_t const wanted_bytes = STRINGZILLA_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const taken_bytes = length - consumed < wanted_bytes ? length - consumed : wanted_bytes;
        for (sz_size_t byte_index = 0; byte_index != taken_bytes; ++byte_index)
            state->partial[state->buffered + byte_index] = input_bytes[consumed + byte_index];
        state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
        consumed += taken_bytes;
        state->associated_length += taken_bytes;
        if (state->buffered == STRINGZILLA_AES_BLOCK_LENGTH) {
            sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
            state->buffered = 0;
        }
    }
}

/** Absorbs whatever @c partial holds, zero padded to a full block, and empties it. */
STRINGZILLA_CONSTEXPR void sz_aes256_gcm_flush_partial_serial_(sz_aes256_gcm_state_t *state) {
    if (state->buffered == 0) return;
    for (sz_size_t byte_index = state->buffered; byte_index != STRINGZILLA_AES_BLOCK_LENGTH; ++byte_index)
        state->partial[byte_index] = 0;
    sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
    state->buffered = 0;
}

/** Appends one ciphertext byte to the hash block, absorbing whenever it fills. */
STRINGZILLA_INLINE void sz_aes256_gcm_hash_byte_serial_(sz_aes256_gcm_state_t *state, sz_u8_t byte) {
    state->partial[state->buffered++] = byte;
    if (state->buffered == STRINGZILLA_AES_BLOCK_LENGTH) {
        sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
        state->buffered = 0;
    }
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param[inout] state The state.
 *  @param[in] text The chunk to transform.
 *  @param[in] length Bytes in the chunk.
 *  @param[out] output Receives the transformed bytes.
 *
 *  Two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes, and neither may
 *  restart at a chunk boundary.
 */
STRINGZILLA_INLINE void sz_aes256_gcm_transform_serial_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_size_t produced = 0;
    sz_assert_no_overlap_(output, length, text, length);

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && length != 0) sz_aes256_gcm_flush_partial_serial_(state);

    while (produced != length) {
        if (state->keystream_used == STRINGZILLA_AES_BLOCK_LENGTH) {
            sz_aes256_counter_advance_serial_(state->counter);
            sz_aes256_block_encrypt_serial_(&state->key.block, state->counter, state->keystream);
            state->keystream_used = 0;
        }
        {
            sz_u8_t const plaintext_byte = input_bytes[produced];
            sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
            sz_u8_t const ciphertext_byte = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
            output_bytes[produced] = transformed;
            sz_aes256_gcm_hash_byte_serial_(state, ciphertext_byte);
            ++state->keystream_used;
            ++state->text_length;
            ++produced;
        }
    }
}

/** Pads whatever is still pending, folds in the length block, and masks out the tag. */
STRINGZILLA_INLINE void sz_aes256_gcm_digest_serial_(sz_aes256_gcm_state_t const *state,
                                                     sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_state_t finishing = *state;
    sz_u128_vec_t lengths_vec;
    sz_size_t byte_index;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    sz_aes256_gcm_flush_partial_serial_(&finishing);

    sz_aes256_gcm_lengths_serial_(&lengths_vec, finishing.associated_length, finishing.text_length);
    sz_ghash_absorb_serial_(finishing.accumulator, lengths_vec.u8s, finishing.key.powers);

    for (byte_index = 0; byte_index != 16; ++byte_index)
        tag[byte_index] = (sz_u8_t)(finishing.accumulator[byte_index] ^ finishing.tag_mask[byte_index]);
}

STRINGZILLA_INLINE sz_status_t sz_aes256_gcm_decryptor_verify_serial_(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                      sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u8_t expected[STRINGZILLA_AES_BLOCK_LENGTH];
    sz_aes256_gcm_digest_serial_(&decryptor->state, expected);
    return sz_aes256_tag_equal_serial_(expected, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion Streaming Interface

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_aes256_key_init_serial(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                      void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_key_init_serial_(key, secret);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_serial(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                     sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                     sz_ptr_t target, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)target;
    sz_u8_t counter[16], keystream[16];
    sz_size_t produced = 0;

    // The first block may start part way in, so its leading bytes are generated and discarded.
    sz_u32_t block_index = (sz_u32_t)(byte_offset / STRINGZILLA_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % STRINGZILLA_AES_BLOCK_LENGTH);
    sz_assert_no_overlap_(target, length, text, length);

    while (produced != length) {
        sz_aes256_counter_block_serial_(nonce, block_index, counter);
        sz_aes256_block_encrypt_serial_(key, counter, keystream);
        for (; within_block != STRINGZILLA_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream[within_block]);
        within_block = 0;
        ++block_index;
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_serial(sz_aes256_gcm_key_t *key,
                                                          sz_u8_t const secret[sz_at_least_(32)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_u8_t zeros[16], subkey[16];
    sz_size_t byte_index, power_index;

    sz_aes256_key_init_serial_(&key->block, secret);
    for (byte_index = 0; byte_index != 16; ++byte_index) zeros[byte_index] = 0;
    sz_aes256_block_encrypt_serial_(&key->block, zeros, subkey);

    for (byte_index = 0; byte_index != 16; ++byte_index) key->powers[byte_index] = subkey[byte_index];
    for (power_index = 1; power_index != 8; ++power_index) {
        sz_u8_t *target = key->powers + power_index * 16;
        for (byte_index = 0; byte_index != 16; ++byte_index)
            target[byte_index] = key->powers[(power_index - 1) * 16 + byte_index];
        sz_ghash_multiply_serial_(target, subkey);
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_begin_serial_(&encryptor->state, key, nonce);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_associate_serial_(&encryptor->state, text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_serial(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                  sz_size_t length, sz_ptr_t target, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_transform_serial_(&encryptor->state, text, length, target, sz_aes256_gcm_encrypting_k);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_serial(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                  sz_u8_t tag[sz_at_least_(16)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_digest_serial_(&encryptor->state, tag);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_begin_serial_(&decryptor->state, key, nonce);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                     sz_cptr_t text, sz_size_t length, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_associate_serial_(&decryptor->state, text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                             sz_cptr_t text, sz_size_t length,
                                                                             sz_ptr_t target, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_transform_serial_(&decryptor->state, text, length, target, sz_aes256_gcm_decrypting_k);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_serial(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                  sz_u8_t const tag[sz_at_least_(16)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    return sz_aes256_gcm_decryptor_verify_serial_(decryptor, tag);
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_serial(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_begin_serial_(&encryptor.state, key, nonce);
    if (associated_length) sz_aes256_gcm_associate_serial_(&encryptor.state, associated, associated_length);
    sz_aes256_gcm_transform_serial_(&encryptor.state, text, length, target, sz_aes256_gcm_encrypting_k);
    sz_aes256_gcm_digest_serial_(&encryptor.state, tag);
    sz_aes256_gcm_state_scrub_serial_(&encryptor.state);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_serial(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                         void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_begin_serial_(&decryptor.state, key, nonce);
    if (associated_length) sz_aes256_gcm_associate_serial_(&decryptor.state, associated, associated_length);
    sz_aes256_gcm_transform_serial_(&decryptor.state, text, length, target, sz_aes256_gcm_decrypting_k);
    verdict = sz_aes256_gcm_decryptor_verify_serial_(&decryptor, tag);
    sz_aes256_gcm_state_scrub_serial_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) {
        sz_fill_serial_(target, length, 0);
        sz_keep_alive_(target);
    }
    return verdict;
}

#endif // STRINGZILLA_TARGET_SERIAL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_SERIAL_H_
