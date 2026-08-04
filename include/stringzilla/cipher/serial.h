/**
 *  @brief Serial (scalar) backend for AES-256 encryption in counter and Galois/counter modes.
 *  @file include/stringzilla/cipher/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/cipher.h
 */
#ifndef STRINGZILLA_CIPHER_SERIAL_H_
#define STRINGZILLA_CIPHER_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/memory.h"        // `sz_copy`, `sz_fill`
#include "stringzilla/cipher/tables.h" // `sz_aes_sbox_`

#ifdef __cplusplus
extern "C" {
#endif

/*  This backend exists so every target has a correct implementation, and so the vector backends have a
 *  reference to be differentiated against. It is not the fast path anywhere.
 *
 *  The substitution box is a lookup table, which means the round function's timing depends on the key
 *  through the cache. That is acceptable for `sz_hash`, whose inputs are public, and it is not
 *  acceptable in principle for a cipher. Platforms with AES instructions never reach this code; those
 *  without it have no constant-time alternative short of bit-slicing, which costs an order of magnitude.
 *  The Galois hash below deliberately does @b not follow suit: its subkey is derived from the key, so a
 *  subkey-indexed table would leak the key directly, and it uses a branch-free shift-and-add instead.
 */

/**
 *  @brief Which buffer the Galois hash absorbs, for the one transform both directions share.
 *  @see sz_aes256_gcm_encryptor_t, sz_aes256_gcm_decryptor_t.
 *
 *  Internal vocabulary, which is why it lives here rather than in the hub: it appears in no public
 *  signature. The public tier spells the direction as a type, so a caller cannot point a state the wrong
 *  way. The backends share one transform body, because only the choice of which register feeds the hash
 *  differs - a single line out of dozens - and duplicating the rest would trade a branch for a
 *  maintenance hazard. That tier therefore spells the direction as this parameter. Every call site
 *  passes a constant into an always-inlined helper, so the selection folds away and no branch survives
 *  into the hot loop. Every backend includes this header, so every backend sees it.
 */
typedef enum sz_aes256_gcm_direction_t {
    sz_aes256_gcm_encrypting_k = 0, ///< The hash absorbs the transformed bytes, which are the output
    sz_aes256_gcm_decrypting_k = 1, ///< The hash absorbs the untransformed bytes, which are the input
} sz_aes256_gcm_direction_t;

#pragma region Key Schedule

/**
 *  @brief Packs four schedule bytes into one word, byte zero in the least significant position.
 *  @return The packed word.
 *
 *  FIPS 197 writes a key-schedule word as the byte quadruple `(a0, a1, a2, a3)`. Keeping `a0` low means
 *  the rotation the schedule needs is an ordinary shift rather than an endian-dependent permutation.
 */
SZ_HELPER_INLINE sz_u32_t sz_aes256_word_pack_serial_(sz_u8_t const *bytes) {
    return (sz_u32_t)bytes[0] | ((sz_u32_t)bytes[1] << 8) | ((sz_u32_t)bytes[2] << 16) | ((sz_u32_t)bytes[3] << 24);
}

/** @brief Substitutes every byte of a schedule word through the substitution box. */
SZ_HELPER_INLINE sz_u32_t sz_aes256_word_substitute_serial_(sz_u32_t word) {
    sz_u8_t const *sbox = sz_aes_sbox_();
    return (sz_u32_t)sbox[word & 0xFFu] |                 //
           ((sz_u32_t)sbox[(word >> 8) & 0xFFu] << 8) |   //
           ((sz_u32_t)sbox[(word >> 16) & 0xFFu] << 16) | //
           ((sz_u32_t)sbox[(word >> 24) & 0xFFu] << 24);
}

/** @brief Rotates a schedule word so `(a0, a1, a2, a3)` becomes `(a1, a2, a3, a0)`. */
SZ_HELPER_INLINE sz_u32_t sz_aes256_word_rotate_serial_(sz_u32_t word) { return (word >> 8) | (word << 24); }

SZ_API_COMPTIME void sz_aes256_key_init_serial(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    static sz_u8_t const round_constants[7] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40};
    sz_size_t word_index = 0;
    for (; word_index != 8; ++word_index)
        key->round_keys[word_index] = sz_aes256_word_pack_serial_(secret + word_index * 4);
    for (; word_index != SZ_AES256_ROUND_KEYS; ++word_index) {
        sz_u32_t carried_word = key->round_keys[word_index - 1];
        if ((word_index & 7) == 0) {
            carried_word = sz_aes256_word_substitute_serial_(sz_aes256_word_rotate_serial_(carried_word));
            carried_word ^= (sz_u32_t)round_constants[(word_index >> 3) - 1];
        }
        else if ((word_index & 7) == 4) { carried_word = sz_aes256_word_substitute_serial_(carried_word); }
        key->round_keys[word_index] = key->round_keys[word_index - 8] ^ carried_word;
    }
}

#pragma endregion // Key Schedule

#pragma region Block Encryption

/**
 *  @brief Applies `SubBytes` and `ShiftRows` to a block, writing the column-major result.
 *  @param block The 16 input bytes.
 *  @param shifted Receives the substituted and row-shifted bytes.
 */
SZ_HELPER_INLINE void sz_aes256_substitute_and_shift_serial_(sz_u8_t const *block, sz_u8_t *shifted) {
    sz_u8_t const *sbox = sz_aes_sbox_();
    static sz_u8_t const source_of[16] = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};
    for (sz_size_t byte_index = 0; byte_index != 16; ++byte_index)
        shifted[byte_index] = sbox[block[source_of[byte_index]]];
}

/** @brief Doubles a byte in `GF(2^8)` under the AES reduction polynomial. */
SZ_HELPER_INLINE sz_u8_t sz_aes256_gf_double_serial_(sz_u8_t value) {
    return (sz_u8_t)((sz_u8_t)(value << 1) ^ (sz_u8_t)(0x1Bu & (sz_u8_t)(0u - (sz_u8_t)(value >> 7))));
}

/**
 *  @brief Encrypts one 16-byte block with the expanded schedule.
 *  @param key The expanded schedule.
 *  @param block The 16 plaintext bytes.
 *  @param output Receives the 16 ciphertext bytes; may alias @p block.
 */
SZ_HELPER_AUTO void sz_aes256_block_encrypt_serial_(sz_aes256_key_t const *key, sz_u8_t const *block, sz_u8_t *output) {
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
            sz_u8_t const first = column[0];
            sz_u8_t mixed[4];
            mixed[0] = (sz_u8_t)(column[0] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[0] ^ column[1])));
            mixed[1] = (sz_u8_t)(column[1] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[1] ^ column[2])));
            mixed[2] = (sz_u8_t)(column[2] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[2] ^ column[3])));
            mixed[3] = (sz_u8_t)(column[3] ^ parity ^ sz_aes256_gf_double_serial_((sz_u8_t)(column[3] ^ first)));
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

#pragma endregion // Block Encryption

#pragma region Counter Mode

/** @brief Builds the counter block for a given block index: the nonce then a big-endian 32-bit counter. */
SZ_HELPER_INLINE void sz_aes256_counter_block_serial_(sz_u8_t const *nonce, sz_u32_t block_index, sz_u8_t *block) {
    for (sz_size_t byte_index = 0; byte_index != 12; ++byte_index) block[byte_index] = nonce[byte_index];
    block[12] = (sz_u8_t)(block_index >> 24);
    block[13] = (sz_u8_t)(block_index >> 16);
    block[14] = (sz_u8_t)(block_index >> 8);
    block[15] = (sz_u8_t)(block_index >> 0);
}

SZ_API_COMPTIME void sz_aes256_ctr_xor_serial(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                              sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length, sz_ptr_t output) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_u8_t counter[16], keystream[16];
    sz_size_t produced = 0;

    // The first block may start part way in, so its leading bytes are generated and discarded.
    sz_u32_t block_index = (sz_u32_t)(byte_offset / SZ_AES_BLOCK_LENGTH);
    sz_size_t within_block = (sz_size_t)(byte_offset % SZ_AES_BLOCK_LENGTH);

    while (produced != length) {
        sz_aes256_counter_block_serial_(nonce, block_index, counter);
        sz_aes256_block_encrypt_serial_(key, counter, keystream);
        for (; within_block != SZ_AES_BLOCK_LENGTH && produced != length; ++within_block, ++produced)
            output_bytes[produced] = (sz_u8_t)(input_bytes[produced] ^ keystream[within_block]);
        within_block = 0;
        ++block_index;
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Hashing

/**
 *  @brief Multiplies @p accumulator by @p subkey in the Galois field the tag is built over.
 *  @param accumulator The running hash, replaced by the product.
 *  @param subkey One of the precomputed powers of the hash subkey.
 *
 *  Deliberately branch free and table free. The obvious four-bit windowed implementation indexes a table
 *  derived from the subkey, which is derived from the key, so its cache footprint would leak the key on
 *  exactly the platforms that reach this code. Selecting each partial product through an all-ones or
 *  all-zeros mask costs 128 iterations and reveals nothing.
 */
SZ_HELPER_AUTO void sz_ghash_multiply_serial_(sz_u8_t *accumulator, sz_u8_t const *subkey) {
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

/** @brief Absorbs one whole block into the running hash. */
SZ_HELPER_AUTO void sz_ghash_absorb_serial_(sz_u8_t *accumulator, sz_u8_t const *block, sz_u8_t const *subkey) {
    for (sz_size_t byte_index = 0; byte_index != 16; ++byte_index) accumulator[byte_index] ^= block[byte_index];
    sz_ghash_multiply_serial_(accumulator, subkey);
}

SZ_API_COMPTIME void sz_aes256_gcm_key_init_serial(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_u8_t zeros[16], subkey[16];
    sz_size_t byte_index, power_index;

    sz_aes256_key_init_serial(&key->block, secret);
    for (byte_index = 0; byte_index != 16; ++byte_index) zeros[byte_index] = 0;
    sz_aes256_block_encrypt_serial_(&key->block, zeros, subkey);

    for (byte_index = 0; byte_index != 16; ++byte_index) key->powers[byte_index] = subkey[byte_index];
    for (power_index = 1; power_index != 8; ++power_index) {
        sz_u8_t *target = key->powers + power_index * 16;
        for (byte_index = 0; byte_index != 16; ++byte_index)
            target[byte_index] = key->powers[(power_index - 1) * 16 + byte_index];
        sz_ghash_multiply_serial_(target, subkey);
    }
}

#pragma endregion // Galois Hashing

#pragma region Streaming Interface

/**
 *  @brief Compares two authentication tags in time that does not depend on their contents.
 *  @return `sz_true_k` when all sixteen bytes match.
 *
 *  Accumulates every difference instead of returning at the first one, so an attacker cannot recover a
 *  forged tag byte by byte from how long the comparison ran. `sz_equal` short-circuits by design and is
 *  the wrong tool here. The width is fixed because a tag is always one block.
 */
SZ_HELPER_INLINE sz_bool_t sz_aes256_tag_equal_serial_(sz_u8_t const *first, sz_u8_t const *second) {
    sz_u8_t difference = 0;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
        difference |= (sz_u8_t)(first[byte_index] ^ second[byte_index]);
    return difference == 0 ? sz_true_k : sz_false_k;
}

/**
 *  @brief Overwrites a finished state so the key schedule it embeds does not outlive the call.
 *
 *  Ordinary stores followed by one barrier, rather than writes through a `volatile` view: `volatile`
 *  forbids vectorization, so that form would cost 472 single-byte stores on every one-shot call. The
 *  bound is a compile-time constant, so a compiler is free to widen this however it likes - on x86 it
 *  becomes a single `rep stos`. The barrier is what stops it being dropped as a dead store.
 *
 *  @note The vector backends replace this with stores of their own native width. This is the tier with
 *        no vector registers by definition, so a byte loop already is the native width here.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_state_scrub_serial_(sz_aes256_gcm_state_t *state) {
    sz_u8_t *const bytes = (sz_u8_t *)state;
    sz_size_t byte_index;
    for (byte_index = 0; byte_index != sizeof(*state); ++byte_index) bytes[byte_index] = 0;
    sz_keep_alive_(state);
}

/** @brief Prepares the payload both directions share: counter block, tag mask and empty carries. */
SZ_HELPER_INLINE void sz_aes256_gcm_begin_serial_(sz_aes256_gcm_state_t *state, sz_aes256_gcm_key_t const *key,
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
    state->keystream_used = SZ_AES_BLOCK_LENGTH; // ? Forces the first message byte to derive a fresh block
}

/** @brief Advances the big-endian counter occupying the last four bytes of the counter block. */
SZ_HELPER_INLINE void sz_aes256_counter_advance_serial_(sz_u8_t *counter) {
    for (sz_size_t byte_index = 16; byte_index-- != 12;)
        if (++counter[byte_index] != 0) break;
}

/** @brief Absorbs associated data into the payload both directions share. */
SZ_HELPER_INLINE void sz_aes256_gcm_associate_serial_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_size_t consumed = 0;

    // Associated data is hashed but never encrypted, so a partial block is completed in place.
    while (consumed != length) {
        sz_size_t const wanted_bytes = SZ_AES_BLOCK_LENGTH - state->buffered;
        sz_size_t const taken_bytes = length - consumed < wanted_bytes ? length - consumed : wanted_bytes;
        for (sz_size_t byte_index = 0; byte_index != taken_bytes; ++byte_index)
            state->partial[state->buffered + byte_index] = input_bytes[consumed + byte_index];
        state->buffered = (sz_u8_t)(state->buffered + taken_bytes);
        consumed += taken_bytes;
        state->associated_length += taken_bytes;
        if (state->buffered == SZ_AES_BLOCK_LENGTH) {
            sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
            state->buffered = 0;
        }
    }
}

/** @brief Absorbs whatever `partial` holds, zero padded to a full block, and empties it. */
SZ_HELPER_AUTO void sz_aes256_gcm_flush_partial_serial_(sz_aes256_gcm_state_t *state) {
    if (state->buffered == 0) return;
    for (sz_size_t byte_index = state->buffered; byte_index != SZ_AES_BLOCK_LENGTH; ++byte_index)
        state->partial[byte_index] = 0;
    sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
    state->buffered = 0;
}

/** @brief Appends one ciphertext byte to the hash block, absorbing whenever it fills. */
SZ_HELPER_INLINE void sz_aes256_gcm_hash_byte_serial_(sz_aes256_gcm_state_t *state, sz_u8_t byte) {
    state->partial[state->buffered++] = byte;
    if (state->buffered == SZ_AES_BLOCK_LENGTH) {
        sz_ghash_absorb_serial_(state->accumulator, state->partial, state->key.powers);
        state->buffered = 0;
    }
}

/**
 *  @brief Transforms a chunk and absorbs its ciphertext, whichever side of the call that is.
 *  @param state The state.
 *  @param text The chunk to transform.
 *  @param length Bytes in the chunk.
 *  @param output Receives the transformed bytes.
 *
 *  Two sixteen-byte rhythms run underneath a caller's arbitrary chunk sizes, and neither may restart at
 *  a chunk boundary. The keystream block is carried in the state and consumed byte by byte, so a
 *  five-byte chunk followed by an eleven-byte one spends exactly one block rather than two. The hash
 *  block is carried the same way, so only the final block of the whole message is ever zero padded.
 *
 *  The hash always eats ciphertext, which is the output when encrypting and the input when decrypting.
 *  The ciphertext byte is captured before the store because a caller may pass one pointer for both.
 */
SZ_HELPER_INLINE void sz_aes256_gcm_transform_serial_(sz_aes256_gcm_state_t *state, sz_cptr_t text, sz_size_t length,
                                                      sz_ptr_t output, sz_aes256_gcm_direction_t direction) {
    sz_u8_t const *input_bytes = (sz_u8_t const *)text;
    sz_u8_t *output_bytes = (sz_u8_t *)output;
    sz_size_t produced = 0;

    // Associated data ends the moment the first message byte arrives, and its tail needs padding.
    if (state->text_length == 0 && length != 0) sz_aes256_gcm_flush_partial_serial_(state);

    while (produced != length) {
        if (state->keystream_used == SZ_AES_BLOCK_LENGTH) {
            sz_aes256_counter_advance_serial_(state->counter);
            sz_aes256_block_encrypt_serial_(&state->key.block, state->counter, state->keystream);
            state->keystream_used = 0;
        }
        {
            sz_u8_t const plaintext_byte = input_bytes[produced];
            sz_u8_t const transformed = (sz_u8_t)(plaintext_byte ^ state->keystream[state->keystream_used]);
            sz_u8_t const ciphertext = direction == sz_aes256_gcm_decrypting_k ? plaintext_byte : transformed;
            output_bytes[produced] = transformed;
            sz_aes256_gcm_hash_byte_serial_(state, ciphertext);
            ++state->keystream_used;
            ++state->text_length;
            ++produced;
        }
    }
}

/** @brief Pads whatever is still pending, folds in the length block, and masks out the tag. */
SZ_HELPER_INLINE void sz_aes256_gcm_digest_serial_(sz_aes256_gcm_state_t const *state, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_state_t finishing = *state;
    sz_u8_t lengths[16];
    sz_u64_t const associated_bits = finishing.associated_length * 8;
    sz_u64_t const text_bits = finishing.text_length * 8;
    sz_size_t byte_index;

    // Whatever is still pending gets padded here: an associated-data tail when the message was empty,
    // or the message's own trailing ciphertext bytes otherwise.
    sz_aes256_gcm_flush_partial_serial_(&finishing);

    for (byte_index = 0; byte_index != 8; ++byte_index) {
        lengths[byte_index] = (sz_u8_t)(associated_bits >> (56 - byte_index * 8));
        lengths[8 + byte_index] = (sz_u8_t)(text_bits >> (56 - byte_index * 8));
    }
    sz_ghash_absorb_serial_(finishing.accumulator, lengths, finishing.key.powers);

    for (byte_index = 0; byte_index != 16; ++byte_index)
        tag[byte_index] = (sz_u8_t)(finishing.accumulator[byte_index] ^ finishing.tag_mask[byte_index]);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_init_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                         sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_serial_(&encryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_associate_serial(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                              sz_size_t length) {
    sz_aes256_gcm_associate_serial_(&encryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_update_serial(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t output) {
    sz_aes256_gcm_transform_serial_(&encryptor->state, text, length, output, sz_aes256_gcm_encrypting_k);
}

SZ_API_COMPTIME void sz_aes256_gcm_encryptor_digest_serial(sz_aes256_gcm_encryptor_t const *encryptor,
                                                           sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_digest_serial_(&encryptor->state, tag);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_init_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                         sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_aes256_gcm_begin_serial_(&decryptor->state, key, nonce);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_associate_serial(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                              sz_size_t length) {
    sz_aes256_gcm_associate_serial_(&decryptor->state, text, length);
}

SZ_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                      sz_cptr_t text, sz_size_t length,
                                                                      sz_ptr_t output) {
    sz_aes256_gcm_transform_serial_(&decryptor->state, text, length, output, sz_aes256_gcm_decrypting_k);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_serial(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                  sz_u8_t const tag[sz_at_least_(16)]) {
    sz_u8_t expected[SZ_AES_BLOCK_LENGTH];
    sz_aes256_gcm_digest_serial_(&decryptor->state, expected);
    return sz_aes256_tag_equal_serial_(expected, tag) == sz_true_k ? sz_success_k : sz_authentication_failed_k;
}

#pragma endregion // Streaming Interface

#pragma region One Shot Interface

SZ_API_COMPTIME void sz_aes256_gcm_encrypt_serial(sz_aes256_gcm_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                  sz_cptr_t associated, sz_size_t associated_length, sz_cptr_t text,
                                                  sz_size_t length, sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_aes256_gcm_encryptor_t encryptor;
    sz_aes256_gcm_encryptor_init_serial(&encryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_encryptor_associate_serial(&encryptor, associated, associated_length);
    sz_aes256_gcm_encryptor_update_serial(&encryptor, text, length, output);
    sz_aes256_gcm_encryptor_digest_serial(&encryptor, tag);
    sz_aes256_gcm_state_scrub_serial_(&encryptor.state);
}

SZ_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_serial(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
    sz_aes256_gcm_decryptor_t decryptor;
    sz_status_t verdict;
    sz_aes256_gcm_decryptor_init_serial(&decryptor, key, nonce);
    if (associated_length) sz_aes256_gcm_decryptor_associate_serial(&decryptor, associated, associated_length);
    sz_aes256_gcm_decryptor_update_unverified_serial(&decryptor, text, length, output);
    verdict = sz_aes256_gcm_decryptor_verify_serial(&decryptor, tag);
    sz_aes256_gcm_state_scrub_serial_(&decryptor.state);

    // A caller who drops the status must still be unable to act on forged plaintext.
    if (verdict != sz_success_k) sz_fill(output, length, 0);
    return verdict;
}

#pragma endregion // One Shot Interface

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_CIPHER_SERIAL_H_
