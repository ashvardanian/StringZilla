/**
 *  @file c/dispatch/hash.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The hashing, checksum, SHA-256 and random-fill lists, dispatch points and finder.
 */
#include <stringzilla/hash.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_bytesum_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_bytesum_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_bytesum_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_bytesum_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_bytesum_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_bytesum_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_bytesum_sve,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_bytesum_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_bytesum_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_bytesum_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_bytesum_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_bytesum_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_bytesum_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_sve_k * STRINGZILLA_TARGET_SVE |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_hash_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_hash_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_hash_westmere,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_hash_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_hash_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_hash_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_hash_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_hash_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_hash_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_hash_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_hash_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_hash_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_hash_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES | sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_hash_multiseed_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_hash_multiseed_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_hash_multiseed_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_hash_multiseed_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_hash_multiseed_neonaes,
#endif
    // No SVE2 AES kernel: a scalable batch needs a keyed substrate the Z/Q bridge makes slow.
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_hash_multiseed_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_hash_multiseed_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_hash_multiseed_v128relaxed,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_hash_multiseed_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES |
             sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_fill_random_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_fill_random_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_fill_random_westmere,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_fill_random_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_fill_random_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_fill_random_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_fill_random_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_fill_random_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_fill_random_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_fill_random_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_fill_random_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_fill_random_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_fill_random_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES | sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_hash_state_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_hash_state_init_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_hash_state_init_westmere,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_hash_state_init_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_hash_state_init_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_hash_state_init_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_hash_state_init_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_hash_state_init_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_hash_state_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_hash_state_init_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_hash_state_init_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_hash_state_init_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_hash_state_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES | sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_hash_state_update_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_hash_state_update_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_hash_state_update_westmere,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_hash_state_update_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_hash_state_update_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_hash_state_update_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_hash_state_update_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_hash_state_update_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_hash_state_update_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_hash_state_update_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_hash_state_update_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_hash_state_update_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_hash_state_update_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES | sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_hash_state_digest_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_hash_state_digest_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_hash_state_digest_westmere,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_hash_state_digest_skylake,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_hash_state_digest_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_hash_state_digest_neonaes,
#endif
#if STRINGZILLA_TARGET_SVE2AES
        (sz_kernel_punned_t)&sz_hash_state_digest_sve2aes,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_hash_state_digest_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_hash_state_digest_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_hash_state_digest_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_hash_state_digest_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_hash_state_digest_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_hash_state_digest_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
             sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES | sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

/*  No SHA-256 list holds a V128 Relaxed kernel: its byteswap is a fixed shuffle, so V128 serves. */

static sz_capability_kernels_t const *sz_sha256_state_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sha256_state_init_serial,
#if STRINGZILLA_TARGET_GOLDMONT
        (sz_kernel_punned_t)&sz_sha256_state_init_goldmont,
#endif
#if STRINGZILLA_TARGET_NEONSHA
        (sz_kernel_punned_t)&sz_sha256_state_init_neonsha,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_sha256_state_init_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_sha256_state_init_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_sha256_state_init_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_sha256_state_init_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_sha256_state_init_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT |
             sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_sha256_state_update_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sha256_state_update_serial,
#if STRINGZILLA_TARGET_GOLDMONT
        (sz_kernel_punned_t)&sz_sha256_state_update_goldmont,
#endif
#if STRINGZILLA_TARGET_NEONSHA
        (sz_kernel_punned_t)&sz_sha256_state_update_neonsha,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_sha256_state_update_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_sha256_state_update_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_sha256_state_update_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_sha256_state_update_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_sha256_state_update_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT |
             sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_sha256_state_digest_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sha256_state_digest_serial,
#if STRINGZILLA_TARGET_GOLDMONT
        (sz_kernel_punned_t)&sz_sha256_state_digest_goldmont,
#endif
#if STRINGZILLA_TARGET_NEONSHA
        (sz_kernel_punned_t)&sz_sha256_state_digest_neonsha,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_sha256_state_digest_rvv,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_sha256_state_digest_rvvcrypto,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_sha256_state_digest_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_sha256_state_digest_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_sha256_state_digest_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT |
             sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_sha256_multistate_update_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sha256_multistate_update_serial,
#if STRINGZILLA_TARGET_GOLDMONT
        (sz_kernel_punned_t)&sz_sha256_multistate_update_goldmont,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_sha256_multistate_update_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_sha256_multistate_update_skylake,
#endif
#if STRINGZILLA_TARGET_NEONSHA
        (sz_kernel_punned_t)&sz_sha256_multistate_update_neonsha,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_sha256_multistate_update_rvvcrypto,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_sha256_multistate_update_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_sha256_multistate_update_rocm,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT |
             sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
             sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_sha256_multistate_digest_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_serial,
#if STRINGZILLA_TARGET_GOLDMONT
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_goldmont,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_skylake,
#endif
#if STRINGZILLA_TARGET_NEONSHA
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_neonsha,
#endif
#if STRINGZILLA_TARGET_RVVCRYPTO
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_rvvcrypto,
#endif
    };
    static sz_kernel_punned_t const cuda[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_CUDA
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_cuda,
#endif
    };
    static sz_kernel_punned_t const rocm[] = {
        STRINGZILLA_NULL,
#if STRINGZILLA_TARGET_ROCM
        (sz_kernel_punned_t)&sz_sha256_multistate_digest_rocm,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT |
             sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
             sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA | sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO,
         cpu},
        {sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA, cuda},
        {sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM, rocm},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_bytesum_best(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum,
                                            sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_bytesum_t const kernel = (sz_kernel_bytesum_t)sz_kernel_pick_(capabilities, sz_bytesum_capabilities());
    return kernel ? kernel(text, length, checksum, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_best(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash,
                                         sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_hash_t const kernel = (sz_kernel_hash_t)sz_kernel_pick_(capabilities, sz_hash_capabilities());
    return kernel ? kernel(text, length, seed, hash, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_multiseed_best( //
    sz_cptr_t text, sz_size_t length,               //
    sz_u64_t const *seeds, sz_size_t seeds_count,   //
    sz_u64_t *hashes, sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_hash_multiseed_t const kernel = (sz_kernel_hash_multiseed_t)sz_kernel_pick_(
        capabilities, sz_hash_multiseed_capabilities());
    return kernel ? kernel(text, length, seeds, seeds_count, hashes, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_fill_random_best(sz_ptr_t target, sz_size_t length, sz_u64_t nonce,
                                                sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_fill_random_t const kernel = (sz_kernel_fill_random_t)sz_kernel_pick_(capabilities,
                                                                                    sz_fill_random_capabilities());
    return kernel ? kernel(target, length, nonce, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_init_best(sz_hash_state_t *state, sz_u64_t seed, sz_capability_t capabilities,
                                                    sz_stream_t stream) {
    sz_kernel_hash_state_init_t const kernel = (sz_kernel_hash_state_init_t)sz_kernel_pick_(
        capabilities, sz_hash_state_init_capabilities());
    return kernel ? kernel(state, seed, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_update_best(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length,
                                                      sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_hash_state_update_t const kernel = (sz_kernel_hash_state_update_t)sz_kernel_pick_(
        capabilities, sz_hash_state_update_capabilities());
    return kernel ? kernel(state, text, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_state_digest_best(sz_hash_state_t const *state, sz_u64_t *hash,
                                                      sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_hash_state_digest_t const kernel = (sz_kernel_hash_state_digest_t)sz_kernel_pick_(
        capabilities, sz_hash_state_digest_capabilities());
    return kernel ? kernel(state, hash, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_init_best(sz_sha256_state_t *state, sz_capability_t capabilities,
                                                      sz_stream_t stream) {
    sz_kernel_sha256_state_init_t const kernel = (sz_kernel_sha256_state_init_t)sz_kernel_pick_(
        capabilities, sz_sha256_state_init_capabilities());
    return kernel ? kernel(state, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_update_best(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length,
                                                        sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sha256_state_update_t const kernel = (sz_kernel_sha256_state_update_t)sz_kernel_pick_(
        capabilities, sz_sha256_state_update_capabilities());
    return kernel ? kernel(state, text, length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sha256_state_digest_best(sz_sha256_state_t const *state,
                                                        sz_u8_t digest[sz_at_least_(STRINGZILLA_SHA256_DIGEST_LENGTH)],
                                                        sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sha256_state_digest_t const kernel = (sz_kernel_sha256_state_digest_t)sz_kernel_pick_(
        capabilities, sz_sha256_state_digest_capabilities());
    return kernel ? kernel(state, digest, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_update_best(sz_sha256_state_t *states, sz_sequence_t const *texts,
                                                             sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sha256_multistate_update_t const kernel = (sz_kernel_sha256_multistate_update_t)sz_kernel_pick_(
        capabilities, sz_sha256_multistate_update_capabilities());
    return kernel ? kernel(states, texts, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sha256_multistate_digest_best(sz_sha256_state_t const *states, sz_size_t states_count,
                                                             sz_u8_t *digests, sz_capability_t capabilities,
                                                             sz_stream_t stream) {
    sz_kernel_sha256_multistate_digest_t const kernel = (sz_kernel_sha256_multistate_digest_t)sz_kernel_pick_(
        capabilities, sz_sha256_multistate_digest_capabilities());
    return kernel ? kernel(states, states_count, digests, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_hash_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_bytesum_k: lists = sz_bytesum_capabilities(); break;
    case sz_kernel_hash_k: lists = sz_hash_capabilities(); break;
    case sz_kernel_hash_multiseed_k: lists = sz_hash_multiseed_capabilities(); break;
    case sz_kernel_fill_random_k: lists = sz_fill_random_capabilities(); break;
    case sz_kernel_hash_state_init_k: lists = sz_hash_state_init_capabilities(); break;
    case sz_kernel_hash_state_update_k: lists = sz_hash_state_update_capabilities(); break;
    case sz_kernel_hash_state_digest_k: lists = sz_hash_state_digest_capabilities(); break;
    case sz_kernel_sha256_state_init_k: lists = sz_sha256_state_init_capabilities(); break;
    case sz_kernel_sha256_state_update_k: lists = sz_sha256_state_update_capabilities(); break;
    case sz_kernel_sha256_state_digest_k: lists = sz_sha256_state_digest_capabilities(); break;
    case sz_kernel_sha256_multistate_update_k: lists = sz_sha256_multistate_update_capabilities(); break;
    case sz_kernel_sha256_multistate_digest_k: lists = sz_sha256_multistate_digest_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
