/**
 *  @file include/stringzilla/cipher.h
 *  @author Ash Vardanian
 *  @date August 4, 2026
 *  @brief Hardware-accelerated AES-256 encryption in counter and Galois/counter modes.
 *
 *  Two mode names appear throughout, each in both of its usual spellings, because the standards and
 *  the symbol names here disagree about which to use. @b Counter @b mode is abbreviated @b CTR, and
 *  appears in symbols as `sz_aes256_ctr_*`. @b Galois/counter @b mode is abbreviated @b GCM, and
 *  appears as `sz_aes256_gcm_*`. The Galois hash inside the latter is also called @b GHASH.
 *
 *  Includes core APIs with hardware-specific backends:
 *
 *  - @c sz_aes256_key_init - round-key schedule for counter mode.
 *  - @c sz_aes256_ctr_xor - seekable, unauthenticated counter mode over bulk data.
 *  - @c sz_aes256_gcm_key_init - schedule plus the Galois hash subkey powers.
 *  - @c sz_aes256_gcm_encrypt, @c sz_aes256_gcm_decrypt - one-shot authenticated encryption.
 *  - @c sz_aes256_gcm_encryptor_init, @c _associate, @c _update, @c _digest - sealing in chunks.
 *  - @c sz_aes256_gcm_decryptor_init, @c _associate, @c _update_unverified, @c _verify - opening.
 *
 *  Two modes rather than one, because they have genuinely different contracts and hiding that
 *  behind a shared interface would be a disservice. Counter mode is @b unauthenticated and
 *  @b seekable: block @c n of the keystream depends on nothing but the key, the nonce and @c n, so
 *  a caller may start anywhere in a stream and decryption is the same call as encryption. That is
 *  what makes it the mode the columnar and block-storage formats reach for. Galois/counter mode
 *  adds a message authentication tag over the ciphertext and any associated data, which costs the
 *  ability to seek and introduces a failure mode the caller must handle.
 *
 *  @b Nonce @b discipline is the caller's, and it is the one thing that cannot be gotten wrong
 *  twice. Reusing a nonce under one key in counter mode exposes the exclusive-or of two plaintexts.
 *  Reusing one in Galois/counter mode is worse: it exposes the hash subkey, which is fixed for the
 *  whole key, and from there every message under that key can be forged. Derive nonces from a
 *  counter, never at random, unless a fresh key accompanies each message.
 *
 *  @b Data @b volume has a ceiling that counter-mode constructions inherit from the 128-bit block.
 *  TLS caps a connection at roughly 2^24.5 records of 16 KiB for exactly this reason, and callers
 *  moving more than a few hundred gigabytes under one key should rekey instead. One Galois/counter
 *  mode nonce seals at most 2^36 - 32 bytes, past which its 32-bit block counter would wrap.
 *
 *  The AES round instructions are those @c sz_hash and @c sz_fill_random already use, so the tier
 *  structure matches `hash.h`. Carry-less multiplication for the Galois hash is new; platforms
 *  lacking it - WebAssembly, base RISC-V vector, LoongArch - use a constant-time shift-and-add
 *  reduction rather than a subkey-indexed table, which would leak the key through the cache.
 *
 *  @b Constant @b time is a property of the backend, not of this API. Every tier with cipher
 *  instructions is constant time, and so are the WebAssembly tiers, which substitute through
 *  swizzles over a fixed address stream. The @c serial tier is @b not: its round indexes a 256-byte
 *  table, so its timing depends on the key through the cache. It runs only where no cipher
 *  instruction exists, the alternative being bit-slicing at an order of magnitude. The Galois hash
 *  is constant time everywhere, including @c serial, since a table indexed by its subkey would leak
 *  the key outright.
 *
 *  Galois/counter mode comes from The Galois/Counter Mode of Operation by David McGrew and John
 *  Viega. NIST specifies both in its Recommendation for Block Cipher Modes of Operation series.
 *
 *  @see NIST SP 800-38D, Galois/Counter Mode: https://doi.org/10.6028/NIST.SP.800-38D
 *  @see NIST SP 800-38A, Methods and Techniques: https://doi.org/10.6028/NIST.SP.800-38A
 */
#ifndef STRINGZILLA_CIPHER_H_
#define STRINGZILLA_CIPHER_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

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
 *  @sa sz_aes256_key_init, sz_aes256_ctr_xor
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
 *  @brief Expands a 32-byte secret into the AES-256 round-key schedule.
 *
 *  @param[out] key Receives the expanded schedule.
 *  @param[in] secret The 32 secret bytes.
 *
 *  Expanding an all-zero secret:
 *
 *  @code{.c}
 *      #include <stringzilla/cipher.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0};
 *          sz_aes256_key_t key;
 *          sz_aes256_key_init(&key, secret);
 *          return 0;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_aes256_key_init_serial, sz_aes256_key_init_westmere, sz_aes256_key_init_icelake,
 *      sz_aes256_key_init_neonaes, sz_aes256_key_init_sve2aes, sz_aes256_key_init_rvvcrypto,
 *      sz_aes256_key_init_powervsx, sz_aes256_key_init_v128
 */
STRINGZILLA_API_RUNTIME void sz_aes256_key_init(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/**
 *  @brief Exclusive-ors @p length bytes against the AES-256 counter-mode keystream.
 *
 *  @param[in] key The expanded schedule.
 *  @param[in] nonce The 12 nonce bytes, which must never repeat under one key.
 *  @param[in] byte_offset Position of @p text within the stream, so a caller may start anywhere.
 *  @param[in] text The input bytes.
 *  @param[in] length Number of bytes to transform.
 *  @param[out] output Receives @p length bytes; may equal @p text, or must not overlap it at all.
 *
 *  Counter mode is its own inverse, so one call serves both directions. The counter block is the
 *  nonce followed by a big-endian 32-bit block index that starts at zero, so block n covers bytes
 *  [16n, 16n + 16) and a @p byte_offset that is not a multiple of 16 simply discards the leading
 *  bytes of its first keystream block.
 *
 *  Encrypting a message, then decrypting only its second half:
 *
 *  @code{.c}
 *      #include <stringzilla/cipher.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0};
 *          char plain[64] = "the second half is decrypted on its own", cipher[64], back[64];
 *          sz_aes256_key_t key;
 *          sz_aes256_key_init(&key, secret);
 *          sz_aes256_ctr_xor(&key, nonce, 0, plain, 64, cipher);
 *          sz_aes256_ctr_xor(&key, nonce, 32, cipher + 32, 32, back + 32); // ? Seeks straight to byte 32
 *          return back[32] == plain[32] ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_aes256_ctr_xor_serial, sz_aes256_ctr_xor_westmere, sz_aes256_ctr_xor_icelake,
 *      sz_aes256_ctr_xor_neonaes, sz_aes256_ctr_xor_sve2aes, sz_aes256_ctr_xor_rvvcrypto,
 *      sz_aes256_ctr_xor_powervsx, sz_aes256_ctr_xor_v128
 */
STRINGZILLA_API_RUNTIME void sz_aes256_ctr_xor( //
    sz_aes256_key_t const *key,                 //
    sz_u8_t const nonce[sz_at_least_(12)],      //
    sz_u64_t byte_offset,                       //
    sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/**
 *  @brief An AES-256 schedule together with the Galois hash subkey powers, the key Galois/counter
 *      mode needs. Galois/counter mode is abbreviated GCM, and its hash is GHASH.
 *  @sa sz_aes256_gcm_key_init, sz_aes256_gcm_encrypt, sz_aes256_gcm_encryptor_init
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
 *  @brief Expands a 32-byte secret into a schedule and the Galois hash subkey powers.
 *
 *  @param[out] key Receives the schedule and the powers.
 *  @param[in] secret The 32 secret bytes.
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *
 *  The backends are sz_aes256_gcm_key_init_serial, sz_aes256_gcm_key_init_westmere,
 *  sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_key_init_sve2aes,
 *  sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_key_init_powervsx and
 *  sz_aes256_gcm_key_init_v128, one per target.
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_key_init(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/**
 *  @brief Encrypts and authenticates a whole message in one call.
 *
 *  @param[in] key The expanded key.
 *  @param[in] nonce The 12 nonce bytes, which must never repeat under one key.
 *  @param[in] associated Authenticated, unencrypted bytes like a routing header,
 *      or @c STRINGZILLA_NULL.
 *  @param[in] associated_length Number of associated bytes, possibly zero.
 *  @param[in] text The plaintext.
 *  @param[in] length Number of plaintext bytes.
 *  @param[out] output Receives @p length ciphertext bytes, at @p text or not overlapping it at all.
 *  @param[out] tag Receives the 16-byte authentication tag.
 *
 *  Sealing a short message and opening it again:
 *
 *  @code{.c}
 *      #include <stringzilla/cipher.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0}, tag[16];
 *          char plain[5] = "hello", cipher[5], back[5];
 *          sz_aes256_gcm_key_t key;
 *          sz_aes256_gcm_key_init(&key, secret);
 *          sz_aes256_gcm_encrypt(&key, nonce, STRINGZILLA_NULL, 0, plain, 5, cipher, tag);
 *          return sz_aes256_gcm_decrypt(&key, nonce, STRINGZILLA_NULL, 0, cipher, 5, back, tag) == sz_success_k ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_aes256_gcm_encrypt_serial, sz_aes256_gcm_encrypt_westmere, sz_aes256_gcm_encrypt_icelake,
 *      sz_aes256_gcm_encrypt_neonaes, sz_aes256_gcm_encrypt_sve2aes,
 *      sz_aes256_gcm_encrypt_rvvcrypto, sz_aes256_gcm_encrypt_powervsx, sz_aes256_gcm_encrypt_v128
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encrypt(    //
    sz_aes256_gcm_key_t const *key,                    //
    sz_u8_t const nonce[sz_at_least_(12)],             //
    sz_cptr_t associated, sz_size_t associated_length, //
    sz_cptr_t text, sz_size_t length, sz_ptr_t output, //
    sz_u8_t tag[sz_at_least_(16)]);

/**
 *  @brief Verifies a tag and decrypts a whole message in one call.
 *
 *  @param[in] key The expanded key.
 *  @param[in] nonce The 12 nonce bytes used to encrypt.
 *  @param[in] associated Bytes authenticated but not encrypted, or @c STRINGZILLA_NULL.
 *  @param[in] associated_length Number of associated bytes, possibly zero.
 *  @param[in] text The ciphertext.
 *  @param[in] length Number of ciphertext bytes.
 *  @param[out] output Receives @p length plaintext bytes on success, and zeros on failure; may
 *      equal @p text, or must not overlap it at all.
 *  @param[in] tag The 16-byte tag to check.
 *  @return @c sz_success_k when the tag matched and @p output holds the plaintext, or
 *      @c sz_authentication_failed_k when it did not and @p output is zeroed.
 *
 *  The comparison runs in time independent of where the tag first differs, and the output is
 *  cleared rather than left holding the unauthenticated plaintext, so a caller who ignores the
 *  status still cannot act on forged data. Decrypting in place therefore destroys the ciphertext
 *  when the tag is rejected, which suits a caller that drops the message anyway and rules out one
 *  that must retry.
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_aes256_gcm_decrypt_serial, sz_aes256_gcm_decrypt_westmere, sz_aes256_gcm_decrypt_icelake,
 *      sz_aes256_gcm_decrypt_neonaes, sz_aes256_gcm_decrypt_sve2aes,
 *      sz_aes256_gcm_decrypt_rvvcrypto, sz_aes256_gcm_decrypt_powervsx, sz_aes256_gcm_decrypt_v128
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_aes256_gcm_decrypt( //
    sz_aes256_gcm_key_t const *key,                        //
    sz_u8_t const nonce[sz_at_least_(12)],                 //
    sz_cptr_t associated, sz_size_t associated_length,     //
    sz_cptr_t text, sz_size_t length, sz_ptr_t output,     //
    sz_u8_t const tag[sz_at_least_(16)]);

/**
 *  @brief What both directions of a chunked transformation carry.
 *  @sa sz_aes256_gcm_encryptor_t, sz_aes256_gcm_decryptor_t
 *
 *  Callers construct an encryptor or a decryptor, never this. It is public because the per-backend
 *  kernels operate on it and the differential tests compare it field by field.
 *
 *  @note The key is embedded rather than referenced. A pointer would save the copy, but nothing in
 *      a transparent C struct can stop a state from outliving the key it points at, and that
 *      failure is silent. The copy costs about two percent of a 16 KiB record.
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
 *  @sa sz_aes256_gcm_encryptor_init, sz_aes256_gcm_encryptor_update, sz_aes256_gcm_encryptor_digest
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
 *  @sa sz_aes256_gcm_decryptor_init, sz_aes256_gcm_decryptor_update_unverified,
 *      sz_aes256_gcm_decryptor_verify
 */
typedef struct sz_aes256_gcm_decryptor_t {

    /** The shared payload. */
    sz_aes256_gcm_state_t state;
} sz_aes256_gcm_decryptor_t;

/**
 *  @brief Begins sealing a message delivered in chunks.
 *
 *  @param[out] encryptor The encryptor to initialize.
 *  @param[in] key The expanded key, copied into @p encryptor.
 *  @param[in] nonce The 12 nonce bytes, which must never repeat under one key.
 *
 *  Sealing an eight-byte buffer in place:
 *
 *  @code{.c}
 *      #include <stringzilla/cipher.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0}, tag[16], buffer[8] = {0};
 *          sz_aes256_gcm_key_t key;
 *          sz_aes256_gcm_encryptor_t encryptor;
 *          sz_aes256_gcm_key_init(&key, secret);
 *          sz_aes256_gcm_encryptor_init(&encryptor, &key, nonce);
 *          sz_aes256_gcm_encryptor_update(&encryptor, (sz_cptr_t)buffer, 8, (sz_ptr_t)buffer);
 *          sz_aes256_gcm_encryptor_digest(&encryptor, tag);
 *          return 0;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *
 *  The backends are sz_aes256_gcm_encryptor_init_serial, sz_aes256_gcm_encryptor_init_westmere,
 *  sz_aes256_gcm_encryptor_init_icelake, sz_aes256_gcm_encryptor_init_neonaes,
 *  sz_aes256_gcm_encryptor_init_sve2aes, sz_aes256_gcm_encryptor_init_rvvcrypto,
 *  sz_aes256_gcm_encryptor_init_powervsx and sz_aes256_gcm_encryptor_init_v128.
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_init(sz_aes256_gcm_encryptor_t *encryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]);

/**
 *  @brief Absorbs associated data into a seal, authenticated but not encrypted.
 *
 *  @param[inout] encryptor The encryptor.
 *  @param[in] text The associated bytes.
 *  @param[in] length Number of associated bytes.
 *  @note All associated data must be absorbed before the first call transforming message bytes.
 *  @sa sz_aes256_gcm_encryptor_associate_serial
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_associate(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                               sz_size_t length);

/**
 *  @brief Encrypts one chunk and absorbs its ciphertext into the running tag.
 *
 *  @param[inout] encryptor The encryptor.
 *  @param[in] text The plaintext chunk.
 *  @param[in] length Number of bytes in the chunk.
 *  @param[out] output Receives @p length ciphertext bytes, at @p text or not overlapping it at all.
 *  @sa sz_aes256_gcm_encryptor_update_serial
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_update(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output);

/**
 *  @brief Finalizes a seal and emits its authentication tag.
 *
 *  @param[in] encryptor The encryptor, left unmodified so a caller may keep appending.
 *  @param[out] tag Receives the 16 tag bytes.
 *  @sa sz_aes256_gcm_encryptor_digest_serial
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_digest(sz_aes256_gcm_encryptor_t const *encryptor,
                                                            sz_u8_t tag[sz_at_least_(16)]);

/**
 *  @brief Begins opening a message delivered in chunks.
 *
 *  @param[out] decryptor The decryptor to initialize.
 *  @param[in] key The expanded key, copied into @p decryptor.
 *  @param[in] nonce The 12 nonce bytes the message was sealed under.
 *  @note Selects the fastest backend at compile- or run-time based on
 *      @c STRINGZILLA_RUNTIME_DISPATCH.
 *  @sa sz_aes256_gcm_decryptor_init_serial
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_init(sz_aes256_gcm_decryptor_t *decryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]);

/**
 *  @brief Absorbs associated data into an open, authenticated but not encrypted.
 *
 *  @param[inout] decryptor The decryptor.
 *  @param[in] text The associated bytes.
 *  @param[in] length Number of associated bytes.
 *  @note All associated data must be absorbed before the first call transforming message bytes.
 *  @sa sz_aes256_gcm_decryptor_associate_serial
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_associate(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                               sz_size_t length);

/**
 *  @brief Decrypts one chunk and absorbs its ciphertext into the running tag.
 *
 *  @param[inout] decryptor The decryptor.
 *  @param[in] text The ciphertext chunk.
 *  @param[in] length Number of bytes in the chunk.
 *  @param[out] output Receives @p length plaintext bytes, at @p text or not overlapping it at all.
 *  @warning The emitted bytes are @b not yet authenticated.
 *  @sa sz_aes256_gcm_decryptor_update_unverified_serial
 *
 *  Nothing has verified that this ciphertext is genuine, and a caller that acts on the output
 *  before @c sz_aes256_gcm_decryptor_verify succeeds is trusting an attacker's data. Buffer it, or
 *  accept that risk knowingly. The name says so because a streaming decryption cannot avoid it.
 */
STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_update_unverified(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output);

/**
 *  @brief Finalizes an open and checks its authentication tag.
 *
 *  @param[in] decryptor The decryptor, left unmodified.
 *  @param[in] tag The 16 tag bytes to check.
 *  @return @c sz_success_k when the tag matched every byte the decryptor absorbed, or
 *      @c sz_authentication_failed_k when it did not.
 *  @sa sz_aes256_gcm_decryptor_verify_serial
 *
 *  The comparison runs in time independent of where the tag first differs, so a caller cannot leak
 *  the expected tag by timing repeated attempts. Emitting the expected tag and letting the caller
 *  compare it would hand that guarantee to code this library does not own.
 */
STRINGZILLA_API_RUNTIME sz_status_t sz_aes256_gcm_decryptor_verify(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_serial(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_serial(sz_aes256_gcm_key_t *key,
                                                            sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_serial(sz_aes256_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                       sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_serial(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t output,
                                                           sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_serial(sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)],
                                                                  sz_cptr_t associated, sz_size_t associated_length,
                                                                  sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                  sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_serial(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                    sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                               sz_cptr_t text, sz_size_t length,
                                                                               sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_serial(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                           sz_u8_t const tag[sz_at_least_(16)]);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_westmere(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_westmere(sz_aes256_gcm_key_t *key,
                                                              sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_westmere(sz_aes256_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                         sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                             sz_u8_t const nonce[sz_at_least_(12)],
                                                             sz_cptr_t associated, sz_size_t associated_length,
                                                             sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                             sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)],
                                                                    sz_cptr_t associated, sz_size_t associated_length,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                    sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                    sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                         sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                      sz_cptr_t text, sz_size_t length,
                                                                      sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_westmere(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                      sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                    sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                         sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                 sz_cptr_t text, sz_size_t length,
                                                                                 sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_westmere(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                             sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_icelake(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_icelake(sz_aes256_gcm_key_t *key,
                                                             sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_icelake(sz_aes256_key_t const *key,
                                                        sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                        sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output,
                                                            sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)],
                                                                   sz_cptr_t associated, sz_size_t associated_length,
                                                                   sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                   sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_icelake(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                     sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                sz_cptr_t text, sz_size_t length,
                                                                                sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_icelake(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                            sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_NEONAES

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_neonaes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_neonaes(sz_aes256_gcm_key_t *key,
                                                             sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_neonaes(sz_aes256_key_t const *key,
                                                        sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                        sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output,
                                                            sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)],
                                                                   sz_cptr_t associated, sz_size_t associated_length,
                                                                   sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                   sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_neonaes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                     sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                sz_cptr_t text, sz_size_t length,
                                                                                sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_neonaes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                            sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_SVE2AES

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_sve2aes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_sve2aes(sz_aes256_gcm_key_t *key,
                                                             sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_sve2aes(sz_aes256_key_t const *key,
                                                        sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                        sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output,
                                                            sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)],
                                                                   sz_cptr_t associated, sz_size_t associated_length,
                                                                   sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                   sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_sve2aes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                     sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                sz_cptr_t text, sz_size_t length,
                                                                                sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_sve2aes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                            sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_v128relaxed(sz_aes256_key_t *key,
                                                             sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_v128relaxed(sz_aes256_gcm_key_t *key,
                                                                 sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_v128relaxed(sz_aes256_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                            sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)],
                                                                sz_cptr_t associated, sz_size_t associated_length,
                                                                sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                                       sz_u8_t const nonce[sz_at_least_(12)],
                                                                       sz_cptr_t associated,
                                                                       sz_size_t associated_length, sz_cptr_t text,
                                                                       sz_size_t length, sz_ptr_t output,
                                                                       sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_aes256_gcm_key_t const *key,
                                                                       sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                            sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                         sz_cptr_t text, sz_size_t length,
                                                                         sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_v128relaxed(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                         sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_aes256_gcm_key_t const *key,
                                                                       sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                            sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_v128relaxed(
    sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_v128relaxed(
    sz_aes256_gcm_decryptor_t const *decryptor, sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_v128(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_v128(sz_aes256_gcm_key_t *key,
                                                          sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_v128(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                     sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                     sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_v128(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_v128(sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)],
                                                                sz_cptr_t associated, sz_size_t associated_length,
                                                                sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_v128(sz_aes256_gcm_encryptor_t *encryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_v128(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_v128(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                  sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_v128(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                  sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                                     sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                                             sz_cptr_t text, sz_size_t length,
                                                                             sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_v128(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                         sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_RVVCRYPTO

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_rvvcrypto(sz_aes256_key_t *key,
                                                           sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_rvvcrypto(sz_aes256_gcm_key_t *key,
                                                               sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_rvvcrypto(sz_aes256_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                          sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_cptr_t associated, sz_size_t associated_length,
                                                              sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                              sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                                     sz_u8_t const nonce[sz_at_least_(12)],
                                                                     sz_cptr_t associated, sz_size_t associated_length,
                                                                     sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                     sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_aes256_gcm_key_t const *key,
                                                                     sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                          sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_rvvcrypto(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                       sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                     sz_aes256_gcm_key_t const *key,
                                                                     sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                          sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                  sz_cptr_t text, sz_size_t length,
                                                                                  sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_rvvcrypto(
    sz_aes256_gcm_decryptor_t const *decryptor, sz_u8_t const tag[sz_at_least_(16)]);

#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_aes256_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_key_init_powervsx(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_gcm_key_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_key_init_powervsx(sz_aes256_gcm_key_t *key,
                                                              sz_u8_t const secret[sz_at_least_(32)]);

/** @copydoc sz_aes256_ctr_xor */
STRINGZILLA_API_COMPTIME void sz_aes256_ctr_xor_powervsx(sz_aes256_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                         sz_cptr_t text, sz_size_t length, sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encrypt */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                             sz_u8_t const nonce[sz_at_least_(12)],
                                                             sz_cptr_t associated, sz_size_t associated_length,
                                                             sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                             sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decrypt */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)],
                                                                    sz_cptr_t associated, sz_size_t associated_length,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t output,
                                                                    sz_u8_t const tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_encryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_init_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                    sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_encryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_associate_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                         sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_encryptor_update */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_update_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                      sz_cptr_t text, sz_size_t length,
                                                                      sz_ptr_t output);

/** @copydoc sz_aes256_gcm_encryptor_digest */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_encryptor_digest_powervsx(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                      sz_u8_t tag[sz_at_least_(16)]);

/** @copydoc sz_aes256_gcm_decryptor_init */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_init_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                    sz_aes256_gcm_key_t const *key,
                                                                    sz_u8_t const nonce[sz_at_least_(12)]);

/** @copydoc sz_aes256_gcm_decryptor_associate */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_associate_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                         sz_cptr_t text, sz_size_t length);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified */
STRINGZILLA_API_COMPTIME void sz_aes256_gcm_decryptor_update_unverified_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                 sz_cptr_t text, sz_size_t length,
                                                                                 sz_ptr_t output);

/** @copydoc sz_aes256_gcm_decryptor_verify */
STRINGZILLA_API_COMPTIME sz_status_t sz_aes256_gcm_decryptor_verify_powervsx(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                             sz_u8_t const tag[sz_at_least_(16)]);

#endif

#pragma endregion Core API

#include "stringzilla/cipher/serial.h"
#include "stringzilla/cipher/westmere.h"
#include "stringzilla/cipher/icelake.h"
#include "stringzilla/cipher/neonaes.h"
#include "stringzilla/cipher/sve2aes.h"
#include "stringzilla/cipher/v128relaxed.h"
#include "stringzilla/cipher/v128.h"
#include "stringzilla/cipher/rvvcrypto.h"
#include "stringzilla/cipher/powervsx.h"

/*  Pick the right implementation for the cipher. To override this behavior and precompile all
 *  backends - set @c STRINGZILLA_RUNTIME_DISPATCH to 1. */
#pragma region Compile Time Dispatching
#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME void sz_aes256_key_init(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_key_init_v128relaxed(key, secret);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_key_init_v128(key, secret);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_key_init_rvvcrypto(key, secret);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_key_init_powervsx(key, secret);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_key_init_icelake(key, secret);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_key_init_westmere(key, secret);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_key_init_sve2aes(key, secret);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_key_init_neonaes(key, secret);
#else
    sz_aes256_key_init_serial(key, secret);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_key_init(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_key_init_v128relaxed(key, secret);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_key_init_v128(key, secret);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_key_init_rvvcrypto(key, secret);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_key_init_powervsx(key, secret);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_key_init_icelake(key, secret);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_key_init_westmere(key, secret);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_key_init_sve2aes(key, secret);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_key_init_neonaes(key, secret);
#else
    sz_aes256_gcm_key_init_serial(key, secret);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_ctr_xor(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                               sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                               sz_ptr_t output) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_ctr_xor_v128relaxed(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_ctr_xor_v128(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_ctr_xor_rvvcrypto(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_ctr_xor_powervsx(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_ctr_xor_icelake(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_ctr_xor_westmere(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_ctr_xor_sve2aes(key, nonce, byte_offset, text, length, output);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_ctr_xor_neonaes(key, nonce, byte_offset, text, length, output);
#else
    sz_aes256_ctr_xor_serial(key, nonce, byte_offset, text, length, output);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encrypt(sz_aes256_gcm_key_t const *key,
                                                   sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                   sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_encrypt_v128relaxed(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_encrypt_v128(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_encrypt_rvvcrypto(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_encrypt_powervsx(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_encrypt_icelake(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_encrypt_westmere(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_encrypt_sve2aes(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_encrypt_neonaes(key, nonce, associated, associated_length, text, length, output, tag);
#else
    sz_aes256_gcm_encrypt_serial(key, nonce, associated, associated_length, text, length, output, tag);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_aes256_gcm_decrypt(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t output, sz_u8_t const tag[sz_at_least_(16)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_aes256_gcm_decrypt_v128relaxed(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_V128
    return sz_aes256_gcm_decrypt_v128(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    return sz_aes256_gcm_decrypt_rvvcrypto(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_aes256_gcm_decrypt_powervsx(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_aes256_gcm_decrypt_icelake(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_aes256_gcm_decrypt_westmere(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_SVE2AES
    return sz_aes256_gcm_decrypt_sve2aes(key, nonce, associated, associated_length, text, length, output, tag);
#elif STRINGZILLA_TARGET_NEONAES
    return sz_aes256_gcm_decrypt_neonaes(key, nonce, associated, associated_length, text, length, output, tag);
#else
    return sz_aes256_gcm_decrypt_serial(key, nonce, associated, associated_length, text, length, output, tag);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_init(sz_aes256_gcm_encryptor_t *encryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_encryptor_init_v128relaxed(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_encryptor_init_v128(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_encryptor_init_rvvcrypto(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_encryptor_init_powervsx(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_encryptor_init_icelake(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_encryptor_init_westmere(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_encryptor_init_sve2aes(encryptor, key, nonce);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_encryptor_init_neonaes(encryptor, key, nonce);
#else
    sz_aes256_gcm_encryptor_init_serial(encryptor, key, nonce);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_associate(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                               sz_size_t length) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_encryptor_associate_v128relaxed(encryptor, text, length);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_encryptor_associate_v128(encryptor, text, length);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_encryptor_associate_rvvcrypto(encryptor, text, length);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_encryptor_associate_powervsx(encryptor, text, length);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_encryptor_associate_icelake(encryptor, text, length);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_encryptor_associate_westmere(encryptor, text, length);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_encryptor_associate_sve2aes(encryptor, text, length);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_encryptor_associate_neonaes(encryptor, text, length);
#else
    sz_aes256_gcm_encryptor_associate_serial(encryptor, text, length);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_update(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t output) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_encryptor_update_v128relaxed(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_encryptor_update_v128(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_encryptor_update_rvvcrypto(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_encryptor_update_powervsx(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_encryptor_update_icelake(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_encryptor_update_westmere(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_encryptor_update_sve2aes(encryptor, text, length, output);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_encryptor_update_neonaes(encryptor, text, length, output);
#else
    sz_aes256_gcm_encryptor_update_serial(encryptor, text, length, output);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_encryptor_digest(sz_aes256_gcm_encryptor_t const *encryptor,
                                                            sz_u8_t tag[sz_at_least_(16)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_encryptor_digest_v128relaxed(encryptor, tag);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_encryptor_digest_v128(encryptor, tag);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_encryptor_digest_rvvcrypto(encryptor, tag);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_encryptor_digest_powervsx(encryptor, tag);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_encryptor_digest_icelake(encryptor, tag);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_encryptor_digest_westmere(encryptor, tag);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_encryptor_digest_sve2aes(encryptor, tag);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_encryptor_digest_neonaes(encryptor, tag);
#else
    sz_aes256_gcm_encryptor_digest_serial(encryptor, tag);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_init(sz_aes256_gcm_decryptor_t *decryptor,
                                                          sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_decryptor_init_v128relaxed(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_decryptor_init_v128(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_decryptor_init_rvvcrypto(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_decryptor_init_powervsx(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_decryptor_init_icelake(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_decryptor_init_westmere(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_decryptor_init_sve2aes(decryptor, key, nonce);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_decryptor_init_neonaes(decryptor, key, nonce);
#else
    sz_aes256_gcm_decryptor_init_serial(decryptor, key, nonce);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_associate(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                               sz_size_t length) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_decryptor_associate_v128relaxed(decryptor, text, length);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_decryptor_associate_v128(decryptor, text, length);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_decryptor_associate_rvvcrypto(decryptor, text, length);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_decryptor_associate_powervsx(decryptor, text, length);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_decryptor_associate_icelake(decryptor, text, length);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_decryptor_associate_westmere(decryptor, text, length);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_decryptor_associate_sve2aes(decryptor, text, length);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_decryptor_associate_neonaes(decryptor, text, length);
#else
    sz_aes256_gcm_decryptor_associate_serial(decryptor, text, length);
#endif
}

STRINGZILLA_API_RUNTIME void sz_aes256_gcm_decryptor_update_unverified(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t output) {
#if STRINGZILLA_TARGET_V128RELAXED
    sz_aes256_gcm_decryptor_update_unverified_v128relaxed(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_V128
    sz_aes256_gcm_decryptor_update_unverified_v128(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    sz_aes256_gcm_decryptor_update_unverified_rvvcrypto(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_POWERVSX
    sz_aes256_gcm_decryptor_update_unverified_powervsx(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_ICELAKE
    sz_aes256_gcm_decryptor_update_unverified_icelake(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_WESTMERE
    sz_aes256_gcm_decryptor_update_unverified_westmere(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_SVE2AES
    sz_aes256_gcm_decryptor_update_unverified_sve2aes(decryptor, text, length, output);
#elif STRINGZILLA_TARGET_NEONAES
    sz_aes256_gcm_decryptor_update_unverified_neonaes(decryptor, text, length, output);
#else
    sz_aes256_gcm_decryptor_update_unverified_serial(decryptor, text, length, output);
#endif
}

STRINGZILLA_API_RUNTIME sz_status_t sz_aes256_gcm_decryptor_verify(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)]) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_aes256_gcm_decryptor_verify_v128relaxed(decryptor, tag);
#elif STRINGZILLA_TARGET_V128
    return sz_aes256_gcm_decryptor_verify_v128(decryptor, tag);
#elif STRINGZILLA_TARGET_RVVCRYPTO
    return sz_aes256_gcm_decryptor_verify_rvvcrypto(decryptor, tag);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_aes256_gcm_decryptor_verify_powervsx(decryptor, tag);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_aes256_gcm_decryptor_verify_icelake(decryptor, tag);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_aes256_gcm_decryptor_verify_westmere(decryptor, tag);
#elif STRINGZILLA_TARGET_SVE2AES
    return sz_aes256_gcm_decryptor_verify_sve2aes(decryptor, tag);
#elif STRINGZILLA_TARGET_NEONAES
    return sz_aes256_gcm_decryptor_verify_neonaes(decryptor, tag);
#else
    return sz_aes256_gcm_decryptor_verify_serial(decryptor, tag);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLA_CIPHER_H_
