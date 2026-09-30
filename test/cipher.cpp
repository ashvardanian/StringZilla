/**
 *  @file test/cipher.cpp
 *  @author Ash Vardanian
 *  @date August 3, 2026
 *  @brief AES-256 counter and Galois/counter mode known-answer, safety, and equivalence tests.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cstring> // `std::memcpy`

#include "cross.hpp"   // `check_cipher_unit_`, `check_cipher_equivalence_`
#include "harness.hpp" // `verify`, `randomize_string`, `test_context_t`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

#pragma region Helpers

/** The dispatched counter-mode entry points, in the shape of their capability kernels. */
static ctr_backend_t const ctr_dispatched {"dispatched", cpu_best<sz_aes256_key_init_best>,
                                           cpu_best<sz_aes256_ctr_xor_best>};

/** The dispatched authenticated entry points, one-shot and streaming alike. */
static gcm_backend_t const gcm_dispatched {
    "dispatched",
    cpu_best<sz_aes256_gcm_key_init_best>,
    cpu_best<sz_aes256_gcm_encrypt_best>,
    cpu_best<sz_aes256_gcm_decrypt_best>,
    cpu_best<sz_aes256_gcm_encryptor_init_best>,
    cpu_best<sz_aes256_gcm_encryptor_associate_best>,
    cpu_best<sz_aes256_gcm_encryptor_update_best>,
    cpu_best<sz_aes256_gcm_encryptor_digest_best>,
    cpu_best<sz_aes256_gcm_decryptor_init_best>,
    cpu_best<sz_aes256_gcm_decryptor_associate_best>,
    cpu_best<sz_aes256_gcm_decryptor_update_unverified_best>,
    cpu_best<sz_aes256_gcm_decryptor_verify_best>,
};

#pragma endregion Helpers

#pragma region Unit

/** Holds the dispatched entry points to the published vectors, since a vector has to reach whatever
 *  the dispatcher picks as well as each kernel `cross_<arch>.cpp` names outright. */
void test_cipher_unit() { check_cipher_unit_(ctr_dispatched, gcm_dispatched); }

#pragma endregion Unit

#pragma region Drivers

/**
 *  @brief Confirms the kernels stay inside their buffers when transforming in place.
 *
 *  Counter mode and Galois/counter mode both permit the output pointer to equal the input pointer,
 *  and both must leave the bytes on either side of the buffer untouched. Whether the bytes they
 *  write are the right ones is the unit tier's question, not this one's.
 */
void test_cipher_safety(test_context_t &context) {
    sz_u8_t secret[32], nonce[12], tag[16];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 11 + 3);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index * 2 + 1);

    sz_aes256_key_t counter_key;
    sz_aes256_gcm_key_t authenticated_key;
    sz_capability_t const capabilities = sz::default_capabilities();
    verify(sz_aes256_key_init_best(&counter_key, secret, capabilities, nullptr) == sz_success_k);
    verify(sz_aes256_gcm_key_init_best(&authenticated_key, secret, capabilities, nullptr) == sz_success_k);

    sz_size_t const longest = (sz_size_t)context.iterations(200);
    for (sz_size_t length = 0; length <= longest; ++length) {
        with_guarded_buffer_(length, [&](sz_ptr_t pointer, std::size_t usable) {
            randomize_string(context.generator, {pointer, usable});
            verify(sz_aes256_ctr_xor_best(&counter_key, nonce, 0, pointer, (sz_size_t)usable, pointer, capabilities,
                                          nullptr) == sz_success_k);
        });
        with_guarded_buffer_(length, [&](sz_ptr_t pointer, std::size_t usable) {
            randomize_string(context.generator, {pointer, usable});
            verify(sz_aes256_gcm_encrypt_best(&authenticated_key, nonce, STRINGZILLA_NULL, 0, pointer,
                                              (sz_size_t)usable, pointer, tag, capabilities, nullptr) == sz_success_k);
        });
        // Opening in place has to stay inside the buffer on both outcomes, and the rejected
        // path writes the most: it clears every byte it was given.
        with_guarded_buffer_(length, [&](sz_ptr_t pointer, std::size_t usable) {
            randomize_string(context.generator, {pointer, usable});
            verify(sz_aes256_gcm_encrypt_best(&authenticated_key, nonce, STRINGZILLA_NULL, 0, pointer,
                                              (sz_size_t)usable, pointer, tag, capabilities, nullptr) == sz_success_k);
            verify(sz_aes256_gcm_decrypt_best(&authenticated_key, nonce, STRINGZILLA_NULL, 0, pointer,
                                              (sz_size_t)usable, pointer, tag, capabilities, nullptr) == sz_success_k);
            sz_u8_t forged[16];
            std::memcpy(forged, tag, 16);
            forged[0] ^= 0x01;
            verify(sz_aes256_gcm_decrypt_best(&authenticated_key, nonce, STRINGZILLA_NULL, 0, pointer,
                                              (sz_size_t)usable, pointer, forged, capabilities,
                                              nullptr) == sz_authentication_failed_k);
        });
    }
}

/** Drives the serial-versus-dispatched differential over both cipher modes. */
void test_cipher_all(test_context_t &context) { check_cipher_equivalence_(context, ctr_dispatched, gcm_dispatched); }

#pragma endregion Drivers
