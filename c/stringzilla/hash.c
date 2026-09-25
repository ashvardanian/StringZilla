/**
 *  @file c/stringzilla/hash.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Per-domain dispatch shim for hashing, checksums, SHA-256, and random fills.
 */
#if !defined(STRINGZILLA_OVERRIDE_LIBC)
#define STRINGZILLA_OVERRIDE_LIBC (!STRINGZILLA_WITH_LIBC)
#endif
#include <stringzilla/hash.h>

#include "dispatch.h"

#if !STRINGZILLA_WITH_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_hash_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->bytesum = sz_bytesum_serial;
    impl->hash = sz_hash_serial;
    impl->hash_multiseed = sz_hash_multiseed_serial;
    impl->hash_state_init = sz_hash_state_init_serial;
    impl->hash_state_update = sz_hash_state_update_serial;
    impl->hash_state_digest = sz_hash_state_digest_serial;
    impl->fill_random = sz_fill_random_serial;

    impl->sha256_state_init = sz_sha256_state_init_serial;
    impl->sha256_state_update = sz_sha256_state_update_serial;
    impl->sha256_state_digest = sz_sha256_state_digest_serial;
    impl->sha256_multistate_update = sz_sha256_multistate_update_serial;
    impl->sha256_multistate_digest = sz_sha256_multistate_digest_serial;

#if STRINGZILLA_TARGET_WESTMERE
    if (caps & sz_cap_westmere_k) {
        impl->hash = sz_hash_westmere;
        impl->hash_multiseed = sz_hash_multiseed_westmere;
        impl->hash_state_init = sz_hash_state_init_westmere;
        impl->hash_state_update = sz_hash_state_update_westmere;
        impl->hash_state_digest = sz_hash_state_digest_westmere;
        impl->fill_random = sz_fill_random_westmere;
    }
#endif

#if STRINGZILLA_TARGET_GOLDMONT
    if (caps & sz_cap_goldmont_k) {
        impl->sha256_state_init = sz_sha256_state_init_goldmont;
        impl->sha256_state_update = sz_sha256_state_update_goldmont;
        impl->sha256_state_digest = sz_sha256_state_digest_goldmont;
        impl->sha256_multistate_update = sz_sha256_multistate_update_goldmont;
        impl->sha256_multistate_digest = sz_sha256_multistate_digest_goldmont;
    }
#endif

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->bytesum = sz_bytesum_haswell;

        impl->sha256_multistate_update = sz_sha256_multistate_update_haswell;
        impl->sha256_multistate_digest = sz_sha256_multistate_digest_haswell;
    }
#endif

#if STRINGZILLA_TARGET_SKYLAKE
    if (caps & sz_cap_skylake_k) {
        impl->bytesum = sz_bytesum_skylake;
        impl->hash = sz_hash_skylake;
        impl->hash_state_init = sz_hash_state_init_skylake;
        impl->hash_state_update = sz_hash_state_update_skylake;
        impl->hash_state_digest = sz_hash_state_digest_skylake;
        impl->fill_random = sz_fill_random_skylake;

        impl->sha256_multistate_update = sz_sha256_multistate_update_skylake;
        impl->sha256_multistate_digest = sz_sha256_multistate_digest_skylake;
    }
#endif

#if STRINGZILLA_TARGET_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->bytesum = sz_bytesum_icelake;
        impl->hash = sz_hash_icelake;
        impl->hash_multiseed = sz_hash_multiseed_icelake;
        impl->hash_state_init = sz_hash_state_init_icelake;
        impl->hash_state_update = sz_hash_state_update_icelake;
        impl->hash_state_digest = sz_hash_state_digest_icelake;
        impl->fill_random = sz_fill_random_icelake;
    }
#endif

#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) { impl->bytesum = sz_bytesum_neon; }
#endif

#if STRINGZILLA_TARGET_NEONAES
    if (caps & sz_cap_neonaes_k) {
        impl->hash = sz_hash_neonaes;
        impl->hash_multiseed = sz_hash_multiseed_neonaes;
        impl->hash_state_init = sz_hash_state_init_neonaes;
        impl->hash_state_update = sz_hash_state_update_neonaes;
        impl->hash_state_digest = sz_hash_state_digest_neonaes;
        impl->fill_random = sz_fill_random_neonaes;
    }
#endif

#if STRINGZILLA_TARGET_NEONSHA
    if (caps & sz_cap_neonsha_k) {
        impl->sha256_state_init = sz_sha256_state_init_neonsha;
        impl->sha256_state_update = sz_sha256_state_update_neonsha;
        impl->sha256_state_digest = sz_sha256_state_digest_neonsha;
    }
#endif

#if STRINGZILLA_TARGET_SVE
    if (caps & sz_cap_sve_k) { impl->bytesum = sz_bytesum_sve; }
#endif

#if STRINGZILLA_TARGET_SVE2
    if (caps & sz_cap_sve2_k) { impl->bytesum = sz_bytesum_sve2; }
#endif

#if STRINGZILLA_TARGET_SVE2AES
    if (caps & sz_cap_sve2aes_k) {
        impl->hash = sz_hash_sve2aes;
        impl->hash_state_init = sz_hash_state_init_sve2aes;
        impl->hash_state_update = sz_hash_state_update_sve2aes;
        impl->hash_state_digest = sz_hash_state_digest_sve2aes;
        impl->fill_random = sz_fill_random_sve2aes;
        // `hash_multiseed` intentionally keeps NEONAES: a scalable batch would need the up-to-16-byte
        // x-wide keyed substrate the Z/Q bridge makes unprofitable at 128 bits (see hash/sve2aes.h).
    }
#endif

#if STRINGZILLA_TARGET_V128
    if (caps & sz_cap_v128_k) {
        impl->bytesum = sz_bytesum_v128;
        impl->hash = sz_hash_v128;
        impl->hash_state_init = sz_hash_state_init_v128;
        impl->hash_state_update = sz_hash_state_update_v128;
        impl->hash_state_digest = sz_hash_state_digest_v128;
        impl->fill_random = sz_fill_random_v128;

        impl->sha256_state_init = sz_sha256_state_init_v128;
        impl->sha256_state_update = sz_sha256_state_update_v128;
        impl->sha256_state_digest = sz_sha256_state_digest_v128;
    }
#endif

#if STRINGZILLA_TARGET_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->bytesum = sz_bytesum_v128relaxed;
        impl->hash = sz_hash_v128relaxed;
        impl->hash_state_init = sz_hash_state_init_v128relaxed;
        impl->hash_state_update = sz_hash_state_update_v128relaxed;
        impl->hash_state_digest = sz_hash_state_digest_v128relaxed;
        impl->fill_random = sz_fill_random_v128relaxed;
        // SHA-256's only permutation is the big-endian byteswap, a compile-time-constant `i8x16.shuffle`
        // (not a data-dependent table swizzle), so relaxed-SIMD offers nothing here; reuse the SIMD128 path.
        impl->sha256_state_init = sz_sha256_state_init_v128;
        impl->sha256_state_update = sz_sha256_state_update_v128;
        impl->sha256_state_digest = sz_sha256_state_digest_v128;
    }
#endif

#if STRINGZILLA_TARGET_RVV
    if (caps & sz_cap_rvv_k) {
        impl->bytesum = sz_bytesum_rvv;
        impl->hash = sz_hash_rvv;
        impl->hash_state_init = sz_hash_state_init_rvv;
        impl->hash_state_update = sz_hash_state_update_rvv;
        impl->hash_state_digest = sz_hash_state_digest_rvv;
        impl->fill_random = sz_fill_random_rvv;

        impl->sha256_state_init = sz_sha256_state_init_rvv;
        impl->sha256_state_update = sz_sha256_state_update_rvv;
        impl->sha256_state_digest = sz_sha256_state_digest_rvv;
    }
#endif

#if STRINGZILLA_TARGET_RVVCRYPTO
    if (caps & sz_cap_rvvcrypto_k) {
        impl->hash = sz_hash_rvvcrypto;
        impl->hash_state_init = sz_hash_state_init_rvvcrypto;
        impl->hash_state_update = sz_hash_state_update_rvvcrypto;
        impl->hash_state_digest = sz_hash_state_digest_rvvcrypto;
        impl->fill_random = sz_fill_random_rvvcrypto;

        impl->sha256_state_init = sz_sha256_state_init_rvvcrypto;
        impl->sha256_state_update = sz_sha256_state_update_rvvcrypto;
        impl->sha256_state_digest = sz_sha256_state_digest_rvvcrypto;
    }
#endif

#if STRINGZILLA_TARGET_LASX
    if (caps & sz_cap_lasx_k) {
        impl->bytesum = sz_bytesum_lasx;
        impl->hash = sz_hash_lasx;
        impl->hash_state_init = sz_hash_state_init_lasx;
        impl->hash_state_update = sz_hash_state_update_lasx;
        impl->hash_state_digest = sz_hash_state_digest_lasx;
        impl->fill_random = sz_fill_random_lasx;

        impl->sha256_state_init = sz_sha256_state_init_lasx;
        impl->sha256_state_update = sz_sha256_state_update_lasx;
        impl->sha256_state_digest = sz_sha256_state_digest_lasx;
    }
#endif

#if STRINGZILLA_TARGET_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->bytesum = sz_bytesum_powervsx;
        impl->hash = sz_hash_powervsx;
        impl->hash_state_init = sz_hash_state_init_powervsx;
        impl->hash_state_update = sz_hash_state_update_powervsx;
        impl->hash_state_digest = sz_hash_state_digest_powervsx;
        impl->fill_random = sz_fill_random_powervsx;

        impl->sha256_state_init = sz_sha256_state_init_powervsx;
        impl->sha256_state_update = sz_sha256_state_update_powervsx;
        impl->sha256_state_digest = sz_sha256_state_digest_powervsx;
    }
#endif
}

STRINGZILLA_API_RUNTIME sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length) {
    return sz_dispatch_cpu_table.bytesum(text, length);
}

STRINGZILLA_API_RUNTIME sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed) {
    return sz_dispatch_cpu_table.hash(text, length, seed);
}

STRINGZILLA_API_RUNTIME void sz_hash_multiseed(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds,
                                               sz_size_t seeds_count, sz_u64_t *hashes) {
    sz_dispatch_cpu_table.hash_multiseed(text, length, seeds, seeds_count, hashes);
}

STRINGZILLA_API_RUNTIME void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed) {
    sz_dispatch_cpu_table.hash_state_init(state, seed);
}

STRINGZILLA_API_RUNTIME void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length) {
    sz_dispatch_cpu_table.hash_state_update(state, text, length);
}

STRINGZILLA_API_RUNTIME sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state) {
    return sz_dispatch_cpu_table.hash_state_digest(state);
}

STRINGZILLA_API_RUNTIME void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_dispatch_cpu_table.fill_random(text, length, nonce);
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_init(sz_sha256_state_t *state) {
    sz_dispatch_cpu_table.sha256_state_init(state);
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length) {
    sz_dispatch_cpu_table.sha256_state_update(state, data, length);
}

STRINGZILLA_API_RUNTIME void sz_sha256_state_digest(sz_sha256_state_t const *state, sz_u8_t digest[sz_at_least_(32)]) {
    sz_dispatch_cpu_table.sha256_state_digest(state, digest);
}

STRINGZILLA_API_RUNTIME void sz_sha256_multistate_update(sz_sha256_state_t *states, sz_sequence_t const *texts) {
    sz_dispatch_cpu_table.sha256_multistate_update(states, texts);
}

STRINGZILLA_API_RUNTIME void sz_sha256_multistate_digest(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests) {
    sz_dispatch_cpu_table.sha256_multistate_digest(states, states_count, digests);
}

/* Provide overrides for the LibC `mem*` functions. */
#if STRINGZILLA_OVERRIDE_LIBC && !defined(__CYGWIN__)
#if !defined(_MSC_VER)
STRINGZILLA_API_RUNTIME void memfrob(void *target, size_t length) {
    static sz_u64_t nonce = 42;
    sz_fill_random(target, length, nonce++);
}
#endif
#endif // STRINGZILLA_OVERRIDE_LIBC
