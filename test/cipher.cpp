/**
 *  @brief  AES-256 counter and Galois/counter mode known-answer, safety, and equivalence tests.
 *  @file   test/cipher.cpp
 *  @author Ash Vardanian
 *  @date August 3, 2026
 */
#undef NDEBUG // ! Enable all assertions for testing

#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/**
 *  Make sure to include the StringZilla headers before anything else,
 *  to intercept missing `#include` directives and other issues.
 */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <cstring> // `std::memcmp`
#include <string>  // `std::string`
#include <vector>  // `std::vector`

#include "stringzilla.hpp" // `verify`, `randomize_string`, `scale_iterations`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;

#pragma region Helpers

/** @brief Decodes a hexadecimal literal into bytes, returning how many were written. */
static std::size_t bytes_from_hex_(char const *hex, sz_u8_t *output) noexcept {
    auto nibble = [](char character) -> sz_u8_t {
        return (sz_u8_t)(character <= '9' ? character - '0' : (character | 32) - 'a' + 10);
    };
    std::size_t written = 0;
    for (; hex[written * 2] != '\0' && hex[written * 2 + 1] != '\0'; ++written)
        output[written] = (sz_u8_t)((nibble(hex[written * 2]) << 4) | nibble(hex[written * 2 + 1]));
    return written;
}

/** @brief One published Galois/counter mode vector, spelled out rather than derived from any backend. */
struct known_gcm_t {
    char const *key_hex;
    char const *nonce_hex;
    char const *associated_hex;
    char const *plaintext_hex;
    char const *ciphertext_hex;
    char const *tag_hex;
};

/**
 *  @brief Cases 13 through 16 of McGrew and Viega's Galois/counter mode note, the ones NIST's own
 *         validation suite is built from.
 *
 *  Values longer than a line are split into adjacent literals at 32-byte boundaries, so the formatter
 *  has no reason to re-break them somewhere less readable.
 */
static known_gcm_t const known_gcm_vectors_[] = {
    // Empty message, empty associated data.
    {"0000000000000000000000000000000000000000000000000000000000000000", //
     "000000000000000000000000",                                         //
     "",                                                                 //
     "",                                                                 //
     "",                                                                 //
     "530f8afbc74536b9a963b4f1c4cb738b"},
    // One zero block, empty associated data.
    {"0000000000000000000000000000000000000000000000000000000000000000", //
     "000000000000000000000000",                                         //
     "",                                                                 //
     "00000000000000000000000000000000",                                 //
     "cea7403d4d606b6e074ec5d3baf39d18",                                 //
     "d0d1c8a799996bf0265b98b5d48ab919"},
    // Four whole blocks, empty associated data.
    {"feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308", //
     "cafebabefacedbaddecaf888",                                         //
     "",                                                                 //
     "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"  //
     "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255", //
     "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"  //
     "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad", //
     "b094dac5d93471bdec1a502270e3cc6c"},
    // A partial trailing block plus associated data, which exercises both zero pads.
    {"feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308", //
     "cafebabefacedbaddecaf888",                                         //
     "feedfacedeadbeeffeedfacedeadbeefabaddad2",                         //
     "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"  //
     "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39",         //
     "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"  //
     "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",         //
     "76fc6ece0f4e1768cddf8853bb2d551b"},
};

/**
 *  @brief NIST SP 800-38A F.5.5, the AES-256 counter mode encryption vector.
 *
 *  That vector names a full 128-bit initial counter block incremented across its whole width, while
 *  `sz_aes256_ctr_xor` builds its counter from a twelve-byte nonce and a 32-bit block index starting at
 *  zero. The two coincide exactly when the stream is entered at the block index the published counter
 *  spells out, which is what the byte offset reaches, and the low 32 bits carry nowhere across four blocks.
 */
struct known_ctr_t {
    char const *key_hex;
    char const *nonce_hex;
    sz_u64_t byte_offset;
    char const *plaintext_hex;
    char const *ciphertext_hex;
};

static known_ctr_t const known_ctr_vectors_[] = {
    {"603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", //
     "f0f1f2f3f4f5f6f7f8f9fafb",                                         //
     (sz_u64_t)0xFCFDFEFFull * SZ_AES_BLOCK_LENGTH,                      //
     "6bc1bee22e409f96e93d7e117393172a"                                  //
     "ae2d8a571e03ac9c9eb76fac45af8e51"                                  //
     "30c81c46a35ce411e5fbc1191a0a52ef"                                  //
     "f69f2445df4f9b17ad2b417be66c3710",                                 //
     "601ec313775789a5b7a7f504bbf3d228"                                  //
     "f443e3ca4d62b59aca84e990cacaf5c5"                                  //
     "2b0930daa23de94ce87017ba2d84988d"                                  //
     "dfc9c58db67aada613c2dd08457941a6"},
};

/** @brief One counter-mode backend, named so a failing check can say which one disagreed. */
struct ctr_backend_t {
    char const *name;
    sz_aes256_key_init_t key_init;
    sz_aes256_ctr_xor_t xor_bytes;
};

/** @brief One authenticated backend, one-shot and streaming kernels alike. */
struct gcm_backend_t {
    char const *name;
    sz_aes256_gcm_key_init_t key_init;
    sz_aes256_gcm_encrypt_t encrypt;
    sz_aes256_gcm_decrypt_t decrypt;
    sz_aes256_gcm_encryptor_init_t sealer_init;
    sz_aes256_gcm_encryptor_associate_t sealer_associate;
    sz_aes256_gcm_encryptor_update_t sealer_update;
    sz_aes256_gcm_encryptor_digest_t sealer_digest;
    sz_aes256_gcm_decryptor_init_t opener_init;
    sz_aes256_gcm_decryptor_associate_t opener_associate;
    sz_aes256_gcm_decryptor_update_unverified_t opener_update;
    sz_aes256_gcm_decryptor_verify_t opener_verify;
};

/**
 *  @brief Every counter-mode backend compiled into this translation unit, serial first.
 *
 *  The dispatched entry points lead, because a published vector has to reach whatever the dispatcher
 *  picks as well as each kernel named outright.
 */
static ctr_backend_t const ctr_backends_[] = {
    {"dispatched", sz_aes256_key_init, sz_aes256_ctr_xor},
    {"serial", sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial},
#if SZ_USE_WESTMERE
    {"westmere", sz_aes256_key_init_westmere, sz_aes256_ctr_xor_westmere},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_aes256_key_init_icelake, sz_aes256_ctr_xor_icelake},
#endif
#if SZ_USE_NEONAES
    {"neonaes", sz_aes256_key_init_neonaes, sz_aes256_ctr_xor_neonaes},
#endif
#if SZ_USE_SVE2AES
    {"sve2aes", sz_aes256_key_init_sve2aes, sz_aes256_ctr_xor_sve2aes},
#endif
#if SZ_USE_RVVCRYPTO
    {"rvvcrypto", sz_aes256_key_init_rvvcrypto, sz_aes256_ctr_xor_rvvcrypto},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_aes256_key_init_powervsx, sz_aes256_ctr_xor_powervsx},
#endif
#if SZ_USE_V128
    {"v128", sz_aes256_key_init_v128, sz_aes256_ctr_xor_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_aes256_key_init_v128relaxed, sz_aes256_ctr_xor_v128relaxed},
#endif
};

/** @brief Every authenticated backend compiled into this translation unit, serial first. */
static gcm_backend_t const gcm_backends_[] = {
    {"dispatched", sz_aes256_gcm_key_init, sz_aes256_gcm_encrypt, sz_aes256_gcm_decrypt, sz_aes256_gcm_encryptor_init,
     sz_aes256_gcm_encryptor_associate, sz_aes256_gcm_encryptor_update, sz_aes256_gcm_encryptor_digest,
     sz_aes256_gcm_decryptor_init, sz_aes256_gcm_decryptor_associate, sz_aes256_gcm_decryptor_update_unverified,
     sz_aes256_gcm_decryptor_verify},
    {"serial", sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encrypt_serial, sz_aes256_gcm_decrypt_serial,
     sz_aes256_gcm_encryptor_init_serial, sz_aes256_gcm_encryptor_associate_serial,
     sz_aes256_gcm_encryptor_update_serial, sz_aes256_gcm_encryptor_digest_serial, sz_aes256_gcm_decryptor_init_serial,
     sz_aes256_gcm_decryptor_associate_serial, sz_aes256_gcm_decryptor_update_unverified_serial,
     sz_aes256_gcm_decryptor_verify_serial},
#if SZ_USE_WESTMERE
    {"westmere", sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encrypt_westmere, sz_aes256_gcm_decrypt_westmere,
     sz_aes256_gcm_encryptor_init_westmere, sz_aes256_gcm_encryptor_associate_westmere,
     sz_aes256_gcm_encryptor_update_westmere, sz_aes256_gcm_encryptor_digest_westmere,
     sz_aes256_gcm_decryptor_init_westmere, sz_aes256_gcm_decryptor_associate_westmere,
     sz_aes256_gcm_decryptor_update_unverified_westmere, sz_aes256_gcm_decryptor_verify_westmere},
#endif
#if SZ_USE_ICELAKE
    {"icelake", sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encrypt_icelake, sz_aes256_gcm_decrypt_icelake,
     sz_aes256_gcm_encryptor_init_icelake, sz_aes256_gcm_encryptor_associate_icelake,
     sz_aes256_gcm_encryptor_update_icelake, sz_aes256_gcm_encryptor_digest_icelake,
     sz_aes256_gcm_decryptor_init_icelake, sz_aes256_gcm_decryptor_associate_icelake,
     sz_aes256_gcm_decryptor_update_unverified_icelake, sz_aes256_gcm_decryptor_verify_icelake},
#endif
#if SZ_USE_NEONAES
    {"neonaes", sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encrypt_neonaes, sz_aes256_gcm_decrypt_neonaes,
     sz_aes256_gcm_encryptor_init_neonaes, sz_aes256_gcm_encryptor_associate_neonaes,
     sz_aes256_gcm_encryptor_update_neonaes, sz_aes256_gcm_encryptor_digest_neonaes,
     sz_aes256_gcm_decryptor_init_neonaes, sz_aes256_gcm_decryptor_associate_neonaes,
     sz_aes256_gcm_decryptor_update_unverified_neonaes, sz_aes256_gcm_decryptor_verify_neonaes},
#endif
#if SZ_USE_SVE2AES
    {"sve2aes", sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encrypt_sve2aes, sz_aes256_gcm_decrypt_sve2aes,
     sz_aes256_gcm_encryptor_init_sve2aes, sz_aes256_gcm_encryptor_associate_sve2aes,
     sz_aes256_gcm_encryptor_update_sve2aes, sz_aes256_gcm_encryptor_digest_sve2aes,
     sz_aes256_gcm_decryptor_init_sve2aes, sz_aes256_gcm_decryptor_associate_sve2aes,
     sz_aes256_gcm_decryptor_update_unverified_sve2aes, sz_aes256_gcm_decryptor_verify_sve2aes},
#endif
#if SZ_USE_RVVCRYPTO
    {"rvvcrypto", sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_encrypt_rvvcrypto, sz_aes256_gcm_decrypt_rvvcrypto,
     sz_aes256_gcm_encryptor_init_rvvcrypto, sz_aes256_gcm_encryptor_associate_rvvcrypto,
     sz_aes256_gcm_encryptor_update_rvvcrypto, sz_aes256_gcm_encryptor_digest_rvvcrypto,
     sz_aes256_gcm_decryptor_init_rvvcrypto, sz_aes256_gcm_decryptor_associate_rvvcrypto,
     sz_aes256_gcm_decryptor_update_unverified_rvvcrypto, sz_aes256_gcm_decryptor_verify_rvvcrypto},
#endif
#if SZ_USE_POWERVSX
    {"powervsx", sz_aes256_gcm_key_init_powervsx, sz_aes256_gcm_encrypt_powervsx, sz_aes256_gcm_decrypt_powervsx,
     sz_aes256_gcm_encryptor_init_powervsx, sz_aes256_gcm_encryptor_associate_powervsx,
     sz_aes256_gcm_encryptor_update_powervsx, sz_aes256_gcm_encryptor_digest_powervsx,
     sz_aes256_gcm_decryptor_init_powervsx, sz_aes256_gcm_decryptor_associate_powervsx,
     sz_aes256_gcm_decryptor_update_unverified_powervsx, sz_aes256_gcm_decryptor_verify_powervsx},
#endif
#if SZ_USE_V128
    {"v128", sz_aes256_gcm_key_init_v128, sz_aes256_gcm_encrypt_v128, sz_aes256_gcm_decrypt_v128,
     sz_aes256_gcm_encryptor_init_v128, sz_aes256_gcm_encryptor_associate_v128, sz_aes256_gcm_encryptor_update_v128,
     sz_aes256_gcm_encryptor_digest_v128, sz_aes256_gcm_decryptor_init_v128, sz_aes256_gcm_decryptor_associate_v128,
     sz_aes256_gcm_decryptor_update_unverified_v128, sz_aes256_gcm_decryptor_verify_v128},
#endif
#if SZ_USE_V128RELAXED
    {"v128relaxed", sz_aes256_gcm_key_init_v128relaxed, sz_aes256_gcm_encrypt_v128relaxed,
     sz_aes256_gcm_decrypt_v128relaxed, sz_aes256_gcm_encryptor_init_v128relaxed,
     sz_aes256_gcm_encryptor_associate_v128relaxed, sz_aes256_gcm_encryptor_update_v128relaxed,
     sz_aes256_gcm_encryptor_digest_v128relaxed, sz_aes256_gcm_decryptor_init_v128relaxed,
     sz_aes256_gcm_decryptor_associate_v128relaxed, sz_aes256_gcm_decryptor_update_unverified_v128relaxed,
     sz_aes256_gcm_decryptor_verify_v128relaxed},
#endif
};

/** @brief Reports which backend broke a check, then aborts through the suite's oracle. */
static void fail_backend_(char const *name, char const *what) noexcept {
    std::fprintf(stderr, "Backend %s failed: %s\n", name, what);
    verify(false && "A cipher backend disagreed with its published vector or with serial");
}

#pragma endregion // Helpers

#pragma region Unit

/** @brief Checks one counter-mode backend against a published vector, seeking to its block index. */
static void check_ctr_unit_(ctr_backend_t const &backend, known_ctr_t const &vector) {
    sz_u8_t secret[32], nonce[12];
    std::vector<sz_u8_t> plaintext(64), expected(64), produced(64), recovered(64);
    bytes_from_hex_(vector.key_hex, secret);
    bytes_from_hex_(vector.nonce_hex, nonce);
    std::size_t const length = bytes_from_hex_(vector.plaintext_hex, plaintext.data());
    bytes_from_hex_(vector.ciphertext_hex, expected.data());

    sz_aes256_key_t key;
    backend.key_init(&key, secret);
    backend.xor_bytes(&key, nonce, vector.byte_offset, (sz_cptr_t)plaintext.data(), (sz_size_t)length,
                      (sz_ptr_t)produced.data());
    if (std::memcmp(produced.data(), expected.data(), length) != 0)
        fail_backend_(backend.name, "counter mode ciphertext differs from NIST SP 800-38A F.5.5");

    // Counter mode is its own inverse, so the same call must walk the vector back.
    backend.xor_bytes(&key, nonce, vector.byte_offset, (sz_cptr_t)produced.data(), (sz_size_t)length,
                      (sz_ptr_t)recovered.data());
    if (std::memcmp(recovered.data(), plaintext.data(), length) != 0)
        fail_backend_(backend.name, "counter mode did not recover its own plaintext");
}

/** @brief Checks one authenticated backend against a published vector, both directions and a forged tag. */
static void check_gcm_unit_(gcm_backend_t const &backend, known_gcm_t const &vector) {
    sz_u8_t secret[32], nonce[12], associated[64], expected_tag[16], produced_tag[16];
    std::vector<sz_u8_t> plaintext(256), expected(256), produced(256), recovered(256);
    bytes_from_hex_(vector.key_hex, secret);
    bytes_from_hex_(vector.nonce_hex, nonce);
    std::size_t const associated_length = bytes_from_hex_(vector.associated_hex, associated);
    std::size_t const length = bytes_from_hex_(vector.plaintext_hex, plaintext.data());
    bytes_from_hex_(vector.ciphertext_hex, expected.data());
    bytes_from_hex_(vector.tag_hex, expected_tag);

    sz_aes256_gcm_key_t key;
    backend.key_init(&key, secret);
    backend.encrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length, (sz_cptr_t)plaintext.data(),
                    (sz_size_t)length, (sz_ptr_t)produced.data(), produced_tag);
    if (std::memcmp(produced.data(), expected.data(), length) != 0)
        fail_backend_(backend.name, "authenticated ciphertext differs from the published vector");
    if (std::memcmp(produced_tag, expected_tag, 16) != 0)
        fail_backend_(backend.name, "authentication tag differs from the published vector");

    // The same vector must decrypt back, and a single flipped tag bit must be refused.
    if (backend.decrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length, (sz_cptr_t)produced.data(),
                        (sz_size_t)length, (sz_ptr_t)recovered.data(), produced_tag) != sz_success_k)
        fail_backend_(backend.name, "a genuine tag was refused");
    if (std::memcmp(recovered.data(), plaintext.data(), length) != 0)
        fail_backend_(backend.name, "decryption did not recover the published plaintext");

    produced_tag[0] ^= 0x01;
    for (std::size_t index = 0; index != recovered.size(); ++index) recovered[index] = 0xA5;
    if (backend.decrypt(&key, nonce, (sz_cptr_t)associated, (sz_size_t)associated_length, (sz_cptr_t)produced.data(),
                        (sz_size_t)length, (sz_ptr_t)recovered.data(), produced_tag) != sz_authentication_failed_k)
        fail_backend_(backend.name, "a forged tag was accepted");
    // A caller who drops the status must not find forged plaintext waiting in the buffer.
    for (std::size_t index = 0; index != length; ++index)
        if (recovered[index] != 0) fail_backend_(backend.name, "forged plaintext survived a failed authentication");
}

/**
 *  @brief Checks every compiled kernel against literal expectations only.
 *
 *  The counter-mode vector is NIST SP 800-38A F.5.5 and the four authenticated vectors are cases 13
 *  through 16 of McGrew and Viega's Galois/counter mode note, the ones NIST's own validation suite is
 *  built from. Nothing here is derived by calling another backend, so a shared mistake cannot hide.
 *  Every vector reaches the dispatched entry point, the serial kernel, and each per-ISA kernel in turn,
 *  because agreement between backends is blind to a mistake all of them share.
 */
void test_cipher_unit() {

    for (ctr_backend_t const &backend : ctr_backends_)
        for (known_ctr_t const &vector : known_ctr_vectors_) check_ctr_unit_(backend, vector);

    for (gcm_backend_t const &backend : gcm_backends_)
        for (known_gcm_t const &vector : known_gcm_vectors_) check_gcm_unit_(backend, vector);

    // Transforming in place must agree with transforming into a separate buffer, in both modes.
    for (ctr_backend_t const &backend : ctr_backends_) {
        sz_u8_t secret[32], nonce[12];
        for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 7 + 1);
        for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index * 5 + 2);

        sz_aes256_key_t key;
        backend.key_init(&key, secret);
        std::string original(300, '\0'), separate(300, '\0'), in_place;
        randomize_string(&original[0], original.size());
        backend.xor_bytes(&key, nonce, 0, original.data(), 300, &separate[0]);
        in_place = original;
        backend.xor_bytes(&key, nonce, 0, in_place.data(), 300, &in_place[0]);
        if (in_place != separate) fail_backend_(backend.name, "in-place counter mode differs from out-of-place");
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief Cross-checks a counter-mode backend against a reference across lengths and seek offsets.
 *
 *  Seeking is the whole reason counter mode is exposed separately, so every offset is compared against
 *  the same bytes taken from a from-zero encryption rather than only against the reference backend.
 */
void test_ctr_equivalence(ctr_backend_t const &reference, ctr_backend_t const &candidate, sz_size_t inputs) {
    sz_u8_t secret[32], nonce[12];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 7 + 1);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index * 5 + 2);

    sz_aes256_key_t reference_key, candidate_key;
    reference.key_init(&reference_key, secret);
    candidate.key_init(&candidate_key, secret);
    if (std::memcmp(&reference_key, &candidate_key, sizeof(reference_key)) != 0)
        fail_backend_(candidate.name, "expanded a different round-key schedule than serial");

    std::string text, from_reference, from_candidate;
    for (sz_size_t length = 0; length <= inputs; ++length) {
        text.resize(length), from_reference.resize(length), from_candidate.resize(length);
        if (length) randomize_string(&text[0], length);
        reference.xor_bytes(&reference_key, nonce, 0, text.data(), length, &from_reference[0]);
        candidate.xor_bytes(&candidate_key, nonce, 0, text.data(), length, &from_candidate[0]);
        if (from_reference != from_candidate) fail_backend_(candidate.name, "counter mode disagreed with serial");
    }

    // Every offset must land on the same keystream the whole-stream encryption used.
    std::size_t const span = 1024;
    std::string whole(span, '\0'), sliced;
    randomize_string(&whole[0], span);
    std::string whole_out(span, '\0');
    reference.xor_bytes(&reference_key, nonce, 0, whole.data(), span, &whole_out[0]);
    for (std::size_t offset = 0; offset <= 200; ++offset) {
        sliced.assign(span - offset, '\0');
        candidate.xor_bytes(&candidate_key, nonce, offset, whole.data() + offset, span - offset, &sliced[0]);
        if (std::memcmp(sliced.data(), whole_out.data() + offset, span - offset) != 0)
            fail_backend_(candidate.name, "seeking into the keystream landed on different bytes");
    }
}

/**
 *  @brief Cross-checks an authenticated backend against a reference, one-shot and in chunks.
 *  @param inputs The longest message fuzzed, inclusive.
 */
void test_gcm_equivalence(gcm_backend_t const &reference, gcm_backend_t const &candidate, sz_size_t inputs) {
    sz_u8_t secret[32], nonce[12], reference_tag[16], candidate_tag[16];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 3 + 5);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index + 9);

    sz_aes256_gcm_key_t reference_key, candidate_key;
    reference.key_init(&reference_key, secret);
    candidate.key_init(&candidate_key, secret);
    if (std::memcmp(&reference_key, &candidate_key, sizeof(reference_key)) != 0)
        fail_backend_(candidate.name, "expanded a different schedule or different subkey powers than serial");

    std::vector<std::size_t> const associated_lengths = {0, 1, 15, 16, 17, 40};
    std::string text, from_reference, from_candidate, associated;

    for (sz_size_t length = 0; length <= inputs; ++length) {
        text.resize(length), from_reference.resize(length), from_candidate.resize(length);
        if (length) randomize_string(&text[0], length);
        std::size_t const associated_length = associated_lengths[length % associated_lengths.size()];
        associated.resize(associated_length);
        if (associated_length) randomize_string(&associated[0], associated_length);

        reference.encrypt(&reference_key, nonce, associated.data(), (sz_size_t)associated_length, text.data(), length,
                          &from_reference[0], reference_tag);
        candidate.encrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length, text.data(), length,
                          &from_candidate[0], candidate_tag);
        if (from_reference != from_candidate) fail_backend_(candidate.name, "ciphertext disagreed with serial");
        if (std::memcmp(reference_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "authentication tag disagreed with serial");

        // Decryption must recover the plaintext and report success on the genuine tag.
        std::string recovered(length, '\0');
        if (candidate.decrypt(&candidate_key, nonce, associated.data(), (sz_size_t)associated_length,
                              from_candidate.data(), length, &recovered[0], candidate_tag) != sz_success_k)
            fail_backend_(candidate.name, "refused a tag it had just produced");
        if (recovered != text) fail_backend_(candidate.name, "decryption did not recover its own plaintext");
    }

    // A chunk boundary must be invisible: the keystream block, the hash block, and the associated-data
    // tail all straddle it, and the associated data is chunked on its own rhythm rather than the text's.
    std::size_t const streamed_length = 512, streamed_associated_length = 37;
    text.resize(streamed_length);
    randomize_string(&text[0], streamed_length);
    associated.resize(streamed_associated_length);
    randomize_string(&associated[0], streamed_associated_length);
    from_reference.assign(streamed_length, '\0');
    reference.encrypt(&reference_key, nonce, associated.data(), (sz_size_t)streamed_associated_length, text.data(),
                      streamed_length, &from_reference[0], reference_tag);
    for (std::size_t chunk = 1; chunk <= 40; ++chunk) {
        std::string streamed(streamed_length, '\0');
        sz_aes256_gcm_encryptor_t encryptor;
        candidate.sealer_init(&encryptor, &candidate_key, nonce);
        for (std::size_t offset = 0; offset < streamed_associated_length; offset += chunk) {
            std::size_t const taken = streamed_associated_length - offset < chunk ? streamed_associated_length - offset
                                                                                  : chunk;
            candidate.sealer_associate(&encryptor, associated.data() + offset, (sz_size_t)taken);
        }
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            candidate.sealer_update(&encryptor, text.data() + offset, (sz_size_t)taken, &streamed[offset]);
        }
        candidate.sealer_digest(&encryptor, candidate_tag);
        if (streamed != from_reference) fail_backend_(candidate.name, "chunked sealing differs from one-shot");
        if (std::memcmp(reference_tag, candidate_tag, 16) != 0)
            fail_backend_(candidate.name, "chunked sealing produced a different tag than one-shot");

        // The opposite direction is a separate type, so it needs its own pass over the same chunking.
        std::string reopened(streamed_length, '\0');
        sz_aes256_gcm_decryptor_t decryptor;
        candidate.opener_init(&decryptor, &candidate_key, nonce);
        for (std::size_t offset = 0; offset < streamed_associated_length; offset += chunk) {
            std::size_t const taken = streamed_associated_length - offset < chunk ? streamed_associated_length - offset
                                                                                  : chunk;
            candidate.opener_associate(&decryptor, associated.data() + offset, (sz_size_t)taken);
        }
        for (std::size_t offset = 0; offset < streamed_length; offset += chunk) {
            std::size_t const taken = streamed_length - offset < chunk ? streamed_length - offset : chunk;
            candidate.opener_update(&decryptor, from_reference.data() + offset, (sz_size_t)taken, &reopened[offset]);
        }
        if (reopened != text) fail_backend_(candidate.name, "chunked opening did not recover the plaintext");
        if (candidate.opener_verify(&decryptor, reference_tag) != sz_success_k)
            fail_backend_(candidate.name, "chunked opening refused a genuine tag");
    }
}

#pragma endregion // Equivalence

#pragma region Drivers

/**
 *  @brief Confirms the kernels stay inside their buffers when transforming in place.
 *
 *  Counter mode and Galois/counter mode both permit the output pointer to equal the input pointer, and
 *  both must leave the bytes on either side of the buffer untouched. Whether the bytes they write are
 *  the right ones is the unit tier's question, not this one's.
 */
void test_cipher_safety() {
    sz_u8_t secret[32], nonce[12], tag[16];
    for (std::size_t index = 0; index != 32; ++index) secret[index] = (sz_u8_t)(index * 11 + 3);
    for (std::size_t index = 0; index != 12; ++index) nonce[index] = (sz_u8_t)(index * 2 + 1);

    sz_aes256_key_t counter_key;
    sz_aes256_gcm_key_t authenticated_key;
    sz_aes256_key_init(&counter_key, secret);
    sz_aes256_gcm_key_init(&authenticated_key, secret);

    sz_size_t const longest = (sz_size_t)scale_iterations(200);
    for (sz_size_t length = 0; length <= longest; ++length) {
        with_guarded_buffer_(length, [&](sz_ptr_t pointer, std::size_t usable) {
            randomize_string(pointer, usable);
            sz_aes256_ctr_xor(&counter_key, nonce, 0, pointer, (sz_size_t)usable, pointer);
        });
        with_guarded_buffer_(length, [&](sz_ptr_t pointer, std::size_t usable) {
            randomize_string(pointer, usable);
            sz_aes256_gcm_encrypt(&authenticated_key, nonce, SZ_NULL, 0, pointer, (sz_size_t)usable, pointer, tag);
        });
    }
}

/** @brief Drives the serial-versus-SIMD differential across every cipher backend compiled here. */
void test_cipher_all() {

    // Each length sweeps a fresh buffer, so the work grows with the square of the count.
    sz_size_t const cipher_inputs = (sz_size_t)scale_iterations_quadratic(160);

    // The serial backend leads the table and is the reference for everything after it. Running it
    // against itself first catches a streaming path that disagrees with its own one-shot kernel.
    ctr_backend_t const &ctr_reference = ctr_backends_[1];
    gcm_backend_t const &gcm_reference = gcm_backends_[1];
    for (ctr_backend_t const &candidate : ctr_backends_) test_ctr_equivalence(ctr_reference, candidate, cipher_inputs);
    for (gcm_backend_t const &candidate : gcm_backends_) test_gcm_equivalence(gcm_reference, candidate, cipher_inputs);
}

#pragma endregion // Drivers
