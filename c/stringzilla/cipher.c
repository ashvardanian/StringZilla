/**
 *  @file c/stringzilla/cipher.c
 *  @brief Per-domain dispatch shim for AES-256 counter and Galois/counter mode encryption.
 *  @author Ash Vardanian
 *  @date August 3, 2026
 */
#if !defined(SZ_OVERRIDE_LIBC)
#define SZ_OVERRIDE_LIBC SZ_AVOID_LIBC
#endif
#include "dispatch.h"
#include <stringzilla/cipher.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_cipher_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->aes256_key_init = sz_aes256_key_init_serial;
    impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_serial;
    impl->aes256_ctr_xor = sz_aes256_ctr_xor_serial;
    impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_serial;
    impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_serial;
    impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_serial;
    impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_serial;
    impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_serial;
    impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_serial;
    impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_serial;
    impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_serial;
    impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_serial;
    impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_serial;

    // Blocks run weakest to strongest, because a later assignment simply overwrites an earlier one.

#if SZ_USE_WESTMERE
    if (caps & sz_cap_westmere_k) {
        impl->aes256_key_init = sz_aes256_key_init_westmere;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_westmere;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_westmere;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_westmere;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_westmere;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_westmere;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_westmere;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_westmere;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_westmere;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_westmere;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_westmere;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_westmere;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_westmere;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->aes256_key_init = sz_aes256_key_init_icelake;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_icelake;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_icelake;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_icelake;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_icelake;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_icelake;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_icelake;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_icelake;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_icelake;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_icelake;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_icelake;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_icelake;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_icelake;
    }
#endif

#if SZ_USE_NEONAES
    if (caps & sz_cap_neonaes_k) {
        impl->aes256_key_init = sz_aes256_key_init_neonaes;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_neonaes;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_neonaes;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_neonaes;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_neonaes;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_neonaes;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_neonaes;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_neonaes;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_neonaes;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_neonaes;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_neonaes;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_neonaes;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_neonaes;
    }
#endif

#if SZ_USE_SVE2AES
    if (caps & sz_cap_sve2aes_k) {
        impl->aes256_key_init = sz_aes256_key_init_sve2aes;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_sve2aes;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_sve2aes;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_sve2aes;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_sve2aes;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_sve2aes;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_sve2aes;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_sve2aes;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_sve2aes;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_sve2aes;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_sve2aes;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_sve2aes;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_sve2aes;
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->aes256_key_init = sz_aes256_key_init_v128;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_v128;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_v128;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_v128;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_v128;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_v128;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_v128;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_v128;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_v128;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_v128;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_v128;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_v128;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_v128;
    }
#endif

#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->aes256_key_init = sz_aes256_key_init_v128relaxed;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_v128relaxed;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_v128relaxed;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_v128relaxed;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_v128relaxed;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_v128relaxed;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_v128relaxed;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_v128relaxed;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_v128relaxed;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_v128relaxed;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_v128relaxed;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_v128relaxed;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_v128relaxed;
    }
#endif

#if SZ_USE_RVVCRYPTO
    if (caps & sz_cap_rvvcrypto_k) {
        impl->aes256_key_init = sz_aes256_key_init_rvvcrypto;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_rvvcrypto;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_rvvcrypto;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_rvvcrypto;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_rvvcrypto;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_rvvcrypto;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_rvvcrypto;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_rvvcrypto;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_rvvcrypto;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_rvvcrypto;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_rvvcrypto;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_rvvcrypto;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_rvvcrypto;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->aes256_key_init = sz_aes256_key_init_powervsx;
        impl->aes256_gcm_key_init = sz_aes256_gcm_key_init_powervsx;
        impl->aes256_ctr_xor = sz_aes256_ctr_xor_powervsx;
        impl->aes256_gcm_encrypt = sz_aes256_gcm_encrypt_powervsx;
        impl->aes256_gcm_decrypt = sz_aes256_gcm_decrypt_powervsx;
        impl->aes256_gcm_encryptor_init = sz_aes256_gcm_encryptor_init_powervsx;
        impl->aes256_gcm_encryptor_associate = sz_aes256_gcm_encryptor_associate_powervsx;
        impl->aes256_gcm_encryptor_update = sz_aes256_gcm_encryptor_update_powervsx;
        impl->aes256_gcm_encryptor_digest = sz_aes256_gcm_encryptor_digest_powervsx;
        impl->aes256_gcm_decryptor_init = sz_aes256_gcm_decryptor_init_powervsx;
        impl->aes256_gcm_decryptor_associate = sz_aes256_gcm_decryptor_associate_powervsx;
        impl->aes256_gcm_decryptor_update_unverified = sz_aes256_gcm_decryptor_update_unverified_powervsx;
        impl->aes256_gcm_decryptor_verify = sz_aes256_gcm_decryptor_verify_powervsx;
    }
#endif
}

SZ_API_RUNTIME void sz_aes256_key_init(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_dispatch_table.aes256_key_init(key, secret);
}

SZ_API_RUNTIME void sz_aes256_gcm_key_init(sz_aes256_gcm_key_t *key, sz_u8_t const secret[sz_at_least_(32)]) {
    sz_dispatch_table.aes256_gcm_key_init(key, secret);
}

SZ_API_RUNTIME void sz_aes256_ctr_xor(sz_aes256_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                      sz_u64_t byte_offset, sz_cptr_t text, sz_size_t length, sz_ptr_t output) {
    sz_dispatch_table.aes256_ctr_xor(key, nonce, byte_offset, text, length, output);
}

SZ_API_RUNTIME void sz_aes256_gcm_encrypt(sz_aes256_gcm_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                          sz_cptr_t associated, sz_size_t associated_length, sz_cptr_t text,
                                          sz_size_t length, sz_ptr_t output, sz_u8_t tag[sz_at_least_(16)]) {
    sz_dispatch_table.aes256_gcm_encrypt(key, nonce, associated, associated_length, text, length, output, tag);
}

SZ_API_RUNTIME sz_status_t sz_aes256_gcm_decrypt(sz_aes256_gcm_key_t const *key, sz_u8_t const nonce[sz_at_least_(12)],
                                                 sz_cptr_t associated, sz_size_t associated_length, sz_cptr_t text,
                                                 sz_size_t length, sz_ptr_t output,
                                                 sz_u8_t const tag[sz_at_least_(16)]) {
    return sz_dispatch_table.aes256_gcm_decrypt(key, nonce, associated, associated_length, text, length, output, tag);
}

SZ_API_RUNTIME void sz_aes256_gcm_encryptor_init(sz_aes256_gcm_encryptor_t *encryptor, sz_aes256_gcm_key_t const *key,
                                                 sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_dispatch_table.aes256_gcm_encryptor_init(encryptor, key, nonce);
}

SZ_API_RUNTIME void sz_aes256_gcm_encryptor_associate(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                      sz_size_t length) {
    sz_dispatch_table.aes256_gcm_encryptor_associate(encryptor, text, length);
}

SZ_API_RUNTIME void sz_aes256_gcm_encryptor_update(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                   sz_size_t length, sz_ptr_t output) {
    sz_dispatch_table.aes256_gcm_encryptor_update(encryptor, text, length, output);
}

SZ_API_RUNTIME void sz_aes256_gcm_encryptor_digest(sz_aes256_gcm_encryptor_t const *encryptor,
                                                   sz_u8_t tag[sz_at_least_(16)]) {
    sz_dispatch_table.aes256_gcm_encryptor_digest(encryptor, tag);
}

SZ_API_RUNTIME void sz_aes256_gcm_decryptor_init(sz_aes256_gcm_decryptor_t *decryptor, sz_aes256_gcm_key_t const *key,
                                                 sz_u8_t const nonce[sz_at_least_(12)]) {
    sz_dispatch_table.aes256_gcm_decryptor_init(decryptor, key, nonce);
}

SZ_API_RUNTIME void sz_aes256_gcm_decryptor_associate(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                      sz_size_t length) {
    sz_dispatch_table.aes256_gcm_decryptor_associate(decryptor, text, length);
}

SZ_API_RUNTIME void sz_aes256_gcm_decryptor_update_unverified(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                              sz_size_t length, sz_ptr_t output) {
    sz_dispatch_table.aes256_gcm_decryptor_update_unverified(decryptor, text, length, output);
}

SZ_API_RUNTIME sz_status_t sz_aes256_gcm_decryptor_verify(sz_aes256_gcm_decryptor_t const *decryptor,
                                                          sz_u8_t const tag[sz_at_least_(16)]) {
    return sz_dispatch_table.aes256_gcm_decryptor_verify(decryptor, tag);
}
