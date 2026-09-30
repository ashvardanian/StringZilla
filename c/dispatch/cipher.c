/**
 *  @file c/dispatch/cipher.c
 *  @author Ash Vardanian
 *  @date August 3, 2026
 *  @brief The AES-256 counter and Galois/counter mode capability lists, dispatch points and finder.
 */
#include <stringzilla/cipher.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_aes256_key_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_key_init_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_key_init_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_key_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_key_init_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_key_init_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_key_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_key_init_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_key_init_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_key_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_key_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_key_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_ctr_xor_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_ctr_xor_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_encrypt_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_encrypt_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_decrypt_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_decrypt_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_encryptor_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_encryptor_associate_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_associate_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_encryptor_update_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_update_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_encryptor_digest_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_encryptor_digest_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_decryptor_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_decryptor_associate_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_associate_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_decryptor_update_unverified_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_update_unverified_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_aes256_gcm_decryptor_verify_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_aes256_gcm_decryptor_verify_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_aes256_key_init_best(sz_aes256_key_t *key, sz_u8_t const secret[sz_at_least_(32)],
                                                    sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_key_init_t const kernel = (sz_kernel_aes256_key_init_t)sz_kernel_pick_(
        capabilities, sz_aes256_key_init_capabilities());
    return kernel ? kernel(key, secret, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_key_init_best(sz_aes256_gcm_key_t *key,
                                                        sz_u8_t const secret[sz_at_least_(32)],
                                                        sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_key_init_t const kernel = (sz_kernel_aes256_gcm_key_init_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_key_init_capabilities());
    return kernel ? kernel(key, secret, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_ctr_xor_best(    //
    sz_aes256_key_t const *key,                        //
    sz_u8_t const nonce[sz_at_least_(12)],             //
    sz_u64_t byte_offset,                              //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target, //
    sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_ctr_xor_t const kernel = (sz_kernel_aes256_ctr_xor_t)sz_kernel_pick_(
        capabilities, sz_aes256_ctr_xor_capabilities());
    return kernel ? kernel(key, nonce, byte_offset, text, length, target, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t tag[sz_at_least_(16)],                      //
    sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_encrypt_t const kernel = (sz_kernel_aes256_gcm_encrypt_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_encrypt_capabilities());
    return kernel ? kernel(key, nonce, associated, associated_length, text, length, target, tag, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decrypt_best( //
    sz_aes256_gcm_key_t const *key,                     //
    sz_u8_t const nonce[sz_at_least_(12)],              //
    sz_cptr_t associated, sz_size_t associated_length,  //
    sz_cptr_t text, sz_size_t length, sz_ptr_t target,  //
    sz_u8_t const tag[sz_at_least_(16)],                //
    sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_decrypt_t const kernel = (sz_kernel_aes256_gcm_decrypt_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_decrypt_capabilities());
    return kernel ? kernel(key, nonce, associated, associated_length, text, length, target, tag, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_init_best(sz_aes256_gcm_encryptor_t *encryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_encryptor_init_t const kernel = (sz_kernel_aes256_gcm_encryptor_init_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_encryptor_init_capabilities());
    return kernel ? kernel(encryptor, key, nonce, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_associate_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream) {
    sz_kernel_aes256_gcm_encryptor_associate_t const kernel = (sz_kernel_aes256_gcm_encryptor_associate_t)
        sz_kernel_pick_(capabilities, sz_aes256_gcm_encryptor_associate_capabilities());
    return kernel ? kernel(encryptor, text, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_update_best(sz_aes256_gcm_encryptor_t *encryptor, sz_cptr_t text,
                                                                sz_size_t length, sz_ptr_t target,
                                                                sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_encryptor_update_t const kernel = (sz_kernel_aes256_gcm_encryptor_update_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_encryptor_update_capabilities());
    return kernel ? kernel(encryptor, text, length, target, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_encryptor_digest_best(sz_aes256_gcm_encryptor_t const *encryptor,
                                                                sz_u8_t tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_encryptor_digest_t const kernel = (sz_kernel_aes256_gcm_encryptor_digest_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_encryptor_digest_capabilities());
    return kernel ? kernel(encryptor, tag, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_init_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                              sz_aes256_gcm_key_t const *key,
                                                              sz_u8_t const nonce[sz_at_least_(12)],
                                                              sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_decryptor_init_t const kernel = (sz_kernel_aes256_gcm_decryptor_init_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_decryptor_init_capabilities());
    return kernel ? kernel(decryptor, key, nonce, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_associate_best(sz_aes256_gcm_decryptor_t *decryptor, sz_cptr_t text,
                                                                   sz_size_t length, sz_capability_t capabilities,
                                                                   void *stream) {
    sz_kernel_aes256_gcm_decryptor_associate_t const kernel = (sz_kernel_aes256_gcm_decryptor_associate_t)
        sz_kernel_pick_(capabilities, sz_aes256_gcm_decryptor_associate_capabilities());
    return kernel ? kernel(decryptor, text, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_update_unverified_best(sz_aes256_gcm_decryptor_t *decryptor,
                                                                           sz_cptr_t text, sz_size_t length,
                                                                           sz_ptr_t target,
                                                                           sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_decryptor_update_unverified_t const kernel =
        (sz_kernel_aes256_gcm_decryptor_update_unverified_t)sz_kernel_pick_(
            capabilities, sz_aes256_gcm_decryptor_update_unverified_capabilities());
    return kernel ? kernel(decryptor, text, length, target, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_aes256_gcm_decryptor_verify_best(sz_aes256_gcm_decryptor_t const *decryptor,
                                                                sz_u8_t const tag[sz_at_least_(16)],
                                                                sz_capability_t capabilities, void *stream) {
    sz_kernel_aes256_gcm_decryptor_verify_t const kernel = (sz_kernel_aes256_gcm_decryptor_verify_t)sz_kernel_pick_(
        capabilities, sz_aes256_gcm_decryptor_verify_capabilities());
    return kernel ? kernel(decryptor, tag, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_cipher_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_aes256_key_init_k: lists = sz_aes256_key_init_capabilities(); break;
    case sz_kernel_aes256_gcm_key_init_k: lists = sz_aes256_gcm_key_init_capabilities(); break;
    case sz_kernel_aes256_ctr_xor_k: lists = sz_aes256_ctr_xor_capabilities(); break;
    case sz_kernel_aes256_gcm_encrypt_k: lists = sz_aes256_gcm_encrypt_capabilities(); break;
    case sz_kernel_aes256_gcm_decrypt_k: lists = sz_aes256_gcm_decrypt_capabilities(); break;
    case sz_kernel_aes256_gcm_encryptor_init_k: lists = sz_aes256_gcm_encryptor_init_capabilities(); break;
    case sz_kernel_aes256_gcm_encryptor_associate_k: lists = sz_aes256_gcm_encryptor_associate_capabilities(); break;
    case sz_kernel_aes256_gcm_encryptor_update_k: lists = sz_aes256_gcm_encryptor_update_capabilities(); break;
    case sz_kernel_aes256_gcm_encryptor_digest_k: lists = sz_aes256_gcm_encryptor_digest_capabilities(); break;
    case sz_kernel_aes256_gcm_decryptor_init_k: lists = sz_aes256_gcm_decryptor_init_capabilities(); break;
    case sz_kernel_aes256_gcm_decryptor_associate_k: lists = sz_aes256_gcm_decryptor_associate_capabilities(); break;
    case sz_kernel_aes256_gcm_decryptor_update_unverified_k:
        lists = sz_aes256_gcm_decryptor_update_unverified_capabilities();
        break;
    case sz_kernel_aes256_gcm_decryptor_verify_k: lists = sz_aes256_gcm_decryptor_verify_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
