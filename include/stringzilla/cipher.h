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
 *  - @c sz_aes256_key_init_best - round-key schedule for counter mode.
 *  - @c sz_aes256_ctr_xor_best - seekable, unauthenticated counter mode over bulk data.
 *  - @c sz_aes256_gcm_key_init_best - schedule plus the Galois hash subkey powers.
 *  - @c sz_aes256_gcm_encrypt_best, @c sz_aes256_gcm_decrypt_best - one-shot authenticated
 *    encryption.
 *  - @c sz_aes256_gcm_encryptor_init_best, @c _associate_best, @c _update_best, @c _digest_best -
 *    sealing in chunks.
 *  - @c sz_aes256_gcm_decryptor_init_best, @c _associate_best, @c _update_unverified_best,
 *    @c _verify_best - opening.
 *
 *  Every verb takes the capabilities to dispatch on, and the streaming ones take them per call: the
 *  encryptor and decryptor states are laid out the same for every backend.
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
 *  TLS caps a connection at roughly 2^24.5 records of 16 KB for exactly this reason, and callers
 *  moving more than a few hundred gigabytes under one key should rekey instead. One Galois/counter
 *  mode nonce seals at most 2^36 - 32 bytes, past which its 32-bit block counter would wrap.
 *
 *  The AES round instructions are those @c sz_hash_best and @c sz_fill_random_best already use, so
 *  the tier structure matches `hash.h`. Carry-less multiplication for the Galois hash is new;
 *  platforms lacking it - WebAssembly, base RISC-V vector, LoongArch - use a constant-time
 *  shift-and-add reduction rather than a subkey-indexed table, which would leak the key through
 *  the data cache.
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
#include "stringzilla/capabilities.h"  // `sz_capability_t`
#include "stringzilla/cipher/serial.h" // `sz_aes256_key_t`, `sz_aes256_gcm_encryptor_t` and their twins

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Expands a 32-byte secret into the AES-256 round-key schedule.
 *
 *  @param[out] key Receives the expanded schedule.
 *  @param[in] secret The 32 secret bytes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Expanding an all-zero secret:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0};
 *          sz_aes256_key_t key;
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          return sz_aes256_key_init_best(&key, secret, capabilities, NULL) == sz_success_k ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_aes256_key_init_serial, sz_aes256_key_init_westmere, sz_aes256_key_init_icelake,
 *      sz_aes256_key_init_neonaes, sz_aes256_key_init_sve2aes, sz_aes256_key_init_rvvcrypto,
 *      sz_aes256_key_init_v128, sz_aes256_key_init_v128relaxed, sz_aes256_key_init_powervsx
 */
STRINGZILLA_API sz_status_t sz_aes256_key_init_best(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                    sz_capability_t capabilities, void *stream);

/**
 *  @brief Exclusive-ors @p length bytes against the AES-256 counter-mode keystream.
 *
 *  @param[in] key The expanded schedule.
 *  @param[in] nonce The 12 nonce bytes, which must never repeat under one key.
 *  @param[in] byte_offset Position of @p text within the stream, so a caller may start anywhere.
 *  @param[in] text The input bytes.
 *  @param[in] length Number of bytes to transform.
 *  @param[out] target Receives @p length bytes; may equal @p text, or must not overlap it at all.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Counter mode is its own inverse, so one call serves both directions. The counter block is the
 *  nonce followed by a big-endian 32-bit block index that starts at zero, so block n covers bytes
 *  [16n, 16n + 16) and a @p byte_offset that is not a multiple of 16 simply discards the leading
 *  bytes of its first keystream block.
 *
 *  Encrypting a message, then decrypting only its second half:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0};
 *          char plain[64] = "the second half is decrypted on its own", cipher[64], back[64];
 *          sz_aes256_key_t key;
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_aes256_key_init_best(&key, secret, capabilities, NULL);
 *          sz_aes256_ctr_xor_best(&key, nonce, 0, plain, 64, cipher, capabilities, NULL);
 *          sz_aes256_ctr_xor_best(&key, nonce, 32, cipher + 32, 32, back + 32, capabilities, NULL); // ? Seeks
 *          return back[32] == plain[32] ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_aes256_ctr_xor_serial, sz_aes256_ctr_xor_westmere, sz_aes256_ctr_xor_icelake,
 *      sz_aes256_ctr_xor_neonaes, sz_aes256_ctr_xor_sve2aes, sz_aes256_ctr_xor_rvvcrypto,
 *      sz_aes256_ctr_xor_v128, sz_aes256_ctr_xor_v128relaxed, sz_aes256_ctr_xor_powervsx
 */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_best(    //
    sz_aes256_key_t const *key,                        //
    sz_u8_t const nonce[sz_at_least_(12)],             //
    sz_u64_t byte_offset,                              //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target, //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Expands a 32-byte secret into a schedule and the Galois hash subkey powers.
 *
 *  @param[out] key Receives the schedule and the powers.
 *  @param[in] secret The 32 secret bytes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  The backends are sz_aes256_gcm_key_init_serial, sz_aes256_gcm_key_init_westmere,
 *  sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_key_init_sve2aes,
 *  sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_key_init_v128,
 *  sz_aes256_gcm_key_init_v128relaxed and sz_aes256_gcm_key_init_powervsx, one per target.
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_best(sz_aes256_gcm_key_t *key,
                                                        sz_u8_t const secret[sz_at_least_(32)],
                                                        sz_capability_t capabilities, void *stream);

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
 *  @param[out] target Receives @p length ciphertext bytes, at @p text or not overlapping it at all.
 *  @param[out] tag Receives the 16-byte authentication tag.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Sealing a short message and opening it again:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0}, tag[16];
 *          char plain[5] = "hello", cipher[5], back[5];
 *          sz_aes256_gcm_key_t key;
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_aes256_gcm_key_init_best(&key, secret, capabilities, NULL);
 *          sz_aes256_gcm_encrypt_best(&key, nonce, NULL, 0, plain, 5, cipher, tag, capabilities, NULL);
 *          sz_status_t status = sz_aes256_gcm_decrypt_best(&key, nonce, NULL, 0, cipher, 5, back, tag, capabilities,
 *              NULL);
 *          return status == sz_success_k ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @sa sz_aes256_gcm_encrypt_serial, sz_aes256_gcm_encrypt_westmere, sz_aes256_gcm_encrypt_icelake,
 *      sz_aes256_gcm_encrypt_neonaes, sz_aes256_gcm_encrypt_sve2aes,
 *      sz_aes256_gcm_encrypt_rvvcrypto, sz_aes256_gcm_encrypt_v128,
 *      sz_aes256_gcm_encrypt_v128relaxed, sz_aes256_gcm_encrypt_powervsx
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t tag[sz_at_least_(16)],                      //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Verifies a tag and decrypts a whole message in one call.
 *
 *  @param[in] key The expanded key.
 *  @param[in] nonce The 12 nonce bytes used to encrypt.
 *  @param[in] associated Bytes authenticated but not encrypted, or @c STRINGZILLA_NULL.
 *  @param[in] associated_length Number of associated bytes, possibly zero.
 *  @param[in] text The ciphertext.
 *  @param[in] length Number of ciphertext bytes.
 *  @param[out] target Receives @p length plaintext bytes on success, and zeros on failure; may
 *      equal @p text, or must not overlap it at all.
 *  @param[in] tag The 16-byte tag to check.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k when the tag matched and @p target holds the plaintext,
 *      @c sz_authentication_failed_k when it did not and @p target is zeroed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it and nothing was written.
 *
 *  The comparison runs in time independent of where the tag first differs, and the output is
 *  cleared rather than left holding the unauthenticated plaintext, so a caller who ignores the
 *  status still cannot act on forged data. Decrypting in place therefore destroys the ciphertext
 *  when the tag is rejected, which suits a caller that drops the message anyway and rules out one
 *  that must retry.
 *
 *  @sa sz_aes256_gcm_decrypt_serial, sz_aes256_gcm_decrypt_westmere, sz_aes256_gcm_decrypt_icelake,
 *      sz_aes256_gcm_decrypt_neonaes, sz_aes256_gcm_decrypt_sve2aes,
 *      sz_aes256_gcm_decrypt_rvvcrypto, sz_aes256_gcm_decrypt_v128,
 *      sz_aes256_gcm_decrypt_v128relaxed, sz_aes256_gcm_decrypt_powervsx
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t const tag[sz_at_least_(16)],                //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Begins sealing a message delivered in chunks.
 *
 *  @param[out] encryptor The encryptor to initialize.
 *  @param[in] key The expanded key, copied into @p encryptor.
 *  @param[in] nonce The 12 nonce bytes, which must never repeat under one key.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Sealing an eight-byte buffer in place:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          sz_u8_t secret[32] = {0}, nonce[12] = {0}, tag[16], buffer[8] = {0};
 *          sz_aes256_gcm_key_t key;
 *          sz_aes256_gcm_encryptor_t encryptor;
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_aes256_gcm_key_init_best(&key, secret, capabilities, NULL);
 *          sz_aes256_gcm_encryptor_init_best(&encryptor, &key, nonce, capabilities, NULL);
 *          sz_aes256_gcm_encryptor_update_best(&encryptor, (sz_cptr_t)buffer, 8, (sz_ptr_t)buffer, capabilities, NULL);
 *          sz_aes256_gcm_encryptor_digest_best(&encryptor, tag, capabilities, NULL);
 *          return 0;
 *      }
 *  @endcode
 *
 *  The backends are sz_aes256_gcm_encryptor_init_serial, sz_aes256_gcm_encryptor_init_westmere,
 *  sz_aes256_gcm_encryptor_init_icelake, sz_aes256_gcm_encryptor_init_neonaes,
 *  sz_aes256_gcm_encryptor_init_sve2aes, sz_aes256_gcm_encryptor_init_rvvcrypto,
 *  sz_aes256_gcm_encryptor_init_v128, sz_aes256_gcm_encryptor_init_v128relaxed and
 *  sz_aes256_gcm_encryptor_init_powervsx.
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_best(sz_aes256_gcm_encryptor_t *encryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream);

/**
 *  @brief Absorbs associated data into a seal, authenticated but not encrypted.
 *
 *  @param[inout] encryptor The encryptor.
 *  @param[in] text The associated bytes.
 *  @param[in] length Number of associated bytes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @note All associated data must be absorbed before the first call transforming message bytes.
 *  @sa sz_aes256_gcm_encryptor_associate_serial
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream);

/**
 *  @brief Encrypts one chunk and absorbs its ciphertext into the running tag.
 *
 *  @param[inout] encryptor The encryptor.
 *  @param[in] text The plaintext chunk.
 *  @param[in] length Number of bytes in the chunk.
 *  @param[out] target Receives @p length ciphertext bytes, at @p text or not overlapping it at all.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @sa sz_aes256_gcm_encryptor_update_serial
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length, sz_ptr_t target,
                                                                sz_capability_t capabilities, void *stream);

/**
 *  @brief Finalizes a seal and emits its authentication tag.
 *
 *  @param[in] encryptor The encryptor, left unmodified so a caller may keep appending.
 *  @param[out] tag Receives the 16 tag bytes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @sa sz_aes256_gcm_encryptor_digest_serial
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_best(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                sz_u8_t tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream);

/**
 *  @brief Begins opening a message delivered in chunks.
 *
 *  @param[out] decryptor The decryptor to initialize.
 *  @param[in] key The expanded key, copied into @p decryptor.
 *  @param[in] nonce The 12 nonce bytes the message was sealed under.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @sa sz_aes256_gcm_decryptor_init_serial
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream);

/**
 *  @brief Absorbs associated data into an open, authenticated but not encrypted.
 *
 *  @param[inout] decryptor The decryptor.
 *  @param[in] text The associated bytes.
 *  @param[in] length Number of associated bytes.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @note All associated data must be absorbed before the first call transforming message bytes.
 *  @sa sz_aes256_gcm_decryptor_associate_serial
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_best(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream);

/**
 *  @brief Decrypts one chunk and absorbs its ciphertext into the running tag.
 *
 *  @param[inout] decryptor The decryptor.
 *  @param[in] text The ciphertext chunk.
 *  @param[in] length Number of bytes in the chunk.
 *  @param[out] target Receives @p length plaintext bytes, at @p text or not overlapping it at all.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @warning The emitted bytes are @b not yet authenticated.
 *  @sa sz_aes256_gcm_decryptor_update_unverified_serial
 *
 *  Nothing has verified that this ciphertext is genuine, and a caller that acts on the output
 *  before @c sz_aes256_gcm_decryptor_verify_best succeeds is trusting an attacker's data. Buffer
 *  it, or accept that risk knowingly: a streaming decryption cannot avoid it, so the name says so.
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                                           sz_cptr_t text, sz_size_t length,
                                                                           sz_ptr_t target,
                                                                           sz_capability_t capabilities, void *stream);

/**
 *  @brief Finalizes an open and checks its authentication tag.
 *
 *  @param[in] decryptor The decryptor, left unmodified.
 *  @param[in] tag The 16 tag bytes to check.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k when the tag matched every byte the decryptor absorbed,
 *      @c sz_authentication_failed_k when it did not, or @c sz_missing_kernel_k when no
 *      capability in @p capabilities has it.
 *  @sa sz_aes256_gcm_decryptor_verify_serial
 *
 *  The comparison runs in time independent of where the tag first differs, so a caller cannot leak
 *  the expected tag by timing repeated attempts. Emitting the expected tag and letting the caller
 *  compare it would hand that guarantee to code this library does not own.
 */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_best(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                sz_u8_t const tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream);

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_serial(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                      void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_serial(sz_aes256_gcm_key_t *key,
                                                          sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_serial(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                     sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                     sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_serial(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_serial(sz_aes256_gcm_key_t const *key,
                                                         sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                         sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                         sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                         void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_serial(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_serial(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                  sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_serial(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                  sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                sz_aes256_gcm_key_t const *key,
                                                                sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                     sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_serial(sz_aes256_gcm_decryptor_t *decryptor,
                                                                             sz_cptr_t text, sz_size_t length,
                                                                             sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_serial(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                  sz_u8_t const tag[sz_at_least_(16)], void *stream);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_westmere(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                        void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_westmere(sz_aes256_gcm_key_t *key,
                                                            sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_westmere(sz_aes256_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                       sz_cptr_t text, sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t target,
                                                           sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_westmere(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t target,
                                                           sz_u8_t const tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_westmere(sz_aes256_gcm_encryptor_t *encryptor,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                                    void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_westmere(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                    sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_westmere(sz_aes256_gcm_decryptor_t *decryptor,
                                                                               sz_cptr_t text, sz_size_t length,
                                                                               sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_westmere(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                    sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_icelake(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                       void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_icelake(sz_aes256_gcm_key_t *key,
                                                           sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_icelake(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                      sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                      sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_icelake(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                          void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_icelake(sz_aes256_gcm_encryptor_t *encryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_icelake(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_icelake(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                   sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_icelake(sz_aes256_gcm_decryptor_t *decryptor,
                                                                              sz_cptr_t text, sz_size_t length,
                                                                              sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_icelake(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_NEONAES

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_neonaes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                       void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_neonaes(sz_aes256_gcm_key_t *key,
                                                           sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_neonaes(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                      sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                      sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_neonaes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                          void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_neonaes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_neonaes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_neonaes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                   sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_neonaes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                              sz_cptr_t text, sz_size_t length,
                                                                              sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_neonaes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_SVE2AES

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_sve2aes(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                       void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_sve2aes(sz_aes256_gcm_key_t *key,
                                                           sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_sve2aes(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                      sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                      sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_sve2aes(sz_aes256_gcm_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                          sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                          sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                          void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_sve2aes(sz_aes256_gcm_encryptor_t *encryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_sve2aes(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_sve2aes(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                   sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                 sz_aes256_gcm_key_t const *key,
                                                                 sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                      sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_sve2aes(sz_aes256_gcm_decryptor_t *decryptor,
                                                                              sz_cptr_t text, sz_size_t length,
                                                                              sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_sve2aes(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                   sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_v128relaxed(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                           void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_v128relaxed(sz_aes256_gcm_key_t *key,
                                                               sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_v128relaxed(sz_aes256_key_t const *key,
                                                          sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                          sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                          void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_cptr_t associated, sz_size_t associated_length,
                                                              sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                              sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_v128relaxed(sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_cptr_t associated, sz_size_t associated_length,
                                                              sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                              sz_u8_t const tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_aes256_gcm_key_t const *key,
                                                                     sz_u8_t const nonce[sz_at_least_(12)],
                                                                     void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                          sz_cptr_t text, sz_size_t length,
                                                                          void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_v128relaxed(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_cptr_t text, sz_size_t length,
                                                                       sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_v128relaxed(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                       sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                     sz_aes256_gcm_key_t const *key,
                                                                     sz_u8_t const nonce[sz_at_least_(12)],
                                                                     void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                          sz_cptr_t text, sz_size_t length,
                                                                          void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_v128relaxed(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                  sz_cptr_t text, sz_size_t length,
                                                                                  sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_v128relaxed(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                       sz_u8_t const tag[sz_at_least_(16)],
                                                                       void *stream);

#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_v128(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                    void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_v128(sz_aes256_gcm_key_t *key,
                                                        sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_v128(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                   sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length,
                                                   sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_v128(sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                       sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t target, sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_v128(sz_aes256_gcm_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                       sz_size_t associated_length, sz_cptr_t text, sz_size_t length,
                                                       sz_ptr_t target, sz_u8_t const tag[sz_at_least_(16)],
                                                       void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_v128(sz_aes256_gcm_encryptor_t *encryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_v128(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_v128(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_v128(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_v128(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                   sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_v128(sz_aes256_gcm_decryptor_t *decryptor,
                                                                           sz_cptr_t text, sz_size_t length,
                                                                           sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_v128(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_RVVCRYPTO

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_rvvcrypto(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                         void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_rvvcrypto(sz_aes256_gcm_key_t *key,
                                                             sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_rvvcrypto(sz_aes256_key_t const *key,
                                                        sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                        sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                        void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t target,
                                                            sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_rvvcrypto(sz_aes256_gcm_key_t const *key,
                                                            sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                            sz_size_t associated_length, sz_cptr_t text,
                                                            sz_size_t length, sz_ptr_t target,
                                                            sz_u8_t const tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                        sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_rvvcrypto(sz_aes256_gcm_encryptor_t *encryptor,
                                                                     sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                                     void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_rvvcrypto(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                     sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                   sz_aes256_gcm_key_t const *key,
                                                                   sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                        sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_rvvcrypto(sz_aes256_gcm_decryptor_t *decryptor,
                                                                                sz_cptr_t text, sz_size_t length,
                                                                                sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_rvvcrypto(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                     sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_aes256_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_key_init_powervsx(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                        void *stream);

/** @copydoc sz_aes256_gcm_key_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_powervsx(sz_aes256_gcm_key_t *key,
                                                            sz_u8_t const secret[sz_at_least_(32)], void *stream);

/** @copydoc sz_aes256_ctr_xor_best */
STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_powervsx(sz_aes256_key_t const *key,
                                                       sz_u8_t const nonce[sz_at_least_(12)], sz_u64_t byte_offset,
                                                       sz_cptr_t text, sz_size_t length, sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_encrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t target,
                                                           sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decrypt_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_powervsx(sz_aes256_gcm_key_t const *key,
                                                           sz_u8_t const nonce[sz_at_least_(12)], sz_cptr_t associated,
                                                           sz_size_t associated_length, sz_cptr_t text,
                                                           sz_size_t length, sz_ptr_t target,
                                                           sz_u8_t const tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_encryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                       sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_encryptor_update_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_powervsx(sz_aes256_gcm_encryptor_t *encryptor,
                                                                    sz_cptr_t text, sz_size_t length, sz_ptr_t target,
                                                                    void *stream);

/** @copydoc sz_aes256_gcm_encryptor_digest_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_powervsx(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                    sz_u8_t tag[sz_at_least_(16)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_init_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                  sz_aes256_gcm_key_t const *key,
                                                                  sz_u8_t const nonce[sz_at_least_(12)], void *stream);

/** @copydoc sz_aes256_gcm_decryptor_associate_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                       sz_cptr_t text, sz_size_t length, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_update_unverified_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_powervsx(sz_aes256_gcm_decryptor_t *decryptor,
                                                                               sz_cptr_t text, sz_size_t length,
                                                                               sz_ptr_t target, void *stream);

/** @copydoc sz_aes256_gcm_decryptor_verify_best */
STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_powervsx(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                    sz_u8_t const tag[sz_at_least_(16)], void *stream);

#endif

/**
 *  @brief Finds the cipher kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_cipher_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/cipher/westmere.h"
#include "stringzilla/cipher/icelake.h"
#include "stringzilla/cipher/neonaes.h"
#include "stringzilla/cipher/sve2aes.h"
#include "stringzilla/cipher/v128relaxed.h"
#include "stringzilla/cipher/v128.h"
#include "stringzilla/cipher/rvvcrypto.h"
#include "stringzilla/cipher/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma region Dispatch

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_aes256_key_init_best(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                    sz_capability_t capabilities, void *stream) {
    sz_unused_(key), sz_unused_(secret), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_best(sz_aes256_gcm_key_t *key,
                                                        sz_u8_t const secret[sz_at_least_(32)],
                                                        sz_capability_t capabilities, void *stream) {
    sz_unused_(key), sz_unused_(secret), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_best(    //
    sz_aes256_key_t const *key,                        //
    sz_u8_t const nonce[sz_at_least_(12)],             //
    sz_u64_t byte_offset,                              //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target, //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(key), sz_unused_(nonce), sz_unused_(byte_offset), sz_unused_(text), sz_unused_(length),
        sz_unused_(target), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t tag[sz_at_least_(16)],                      //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(key), sz_unused_(nonce), sz_unused_(associated), sz_unused_(associated_length), sz_unused_(text),
        sz_unused_(length), sz_unused_(target), sz_unused_(tag), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t const tag[sz_at_least_(16)],                //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(key), sz_unused_(nonce), sz_unused_(associated), sz_unused_(associated_length), sz_unused_(text),
        sz_unused_(length), sz_unused_(target), sz_unused_(tag), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_best(sz_aes256_gcm_encryptor_t *encryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream) {
    sz_unused_(encryptor), sz_unused_(key), sz_unused_(nonce), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream) {
    sz_unused_(encryptor), sz_unused_(text), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length, sz_ptr_t target,
                                                                sz_capability_t capabilities, void *stream) {
    sz_unused_(encryptor), sz_unused_(text), sz_unused_(length), sz_unused_(target), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_best(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                sz_u8_t tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream) {
    sz_unused_(encryptor), sz_unused_(tag), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream) {
    sz_unused_(decryptor), sz_unused_(key), sz_unused_(nonce), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_best(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream) {
    sz_unused_(decryptor), sz_unused_(text), sz_unused_(length), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                                           sz_cptr_t text, sz_size_t length,
                                                                           sz_ptr_t target,
                                                                           sz_capability_t capabilities, void *stream) {
    sz_unused_(decryptor), sz_unused_(text), sz_unused_(length), sz_unused_(target), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_best(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                sz_u8_t const tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream) {
    sz_unused_(decryptor), sz_unused_(tag), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_cipher_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY
#pragma endregion Dispatch

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLA_CIPHER_H_
