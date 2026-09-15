/**
 *  @file bench/cipher.cpp
 *  @brief Benchmarks AES-256 counter and Galois/counter mode against the serial reference.
 *
 *  Bulk shaped rather than token shaped, so this sits alongside `bench/memory.cpp`'s random fill rather
 *  than in `bench/token.cpp`. The corpus decides how long each message is drawn from, but the content is
 *  a deterministic fill: a cipher's cost depends on message length alone, never on the bytes themselves.
 *  Sizes sweep a short record up to a page-sized buffer, because that range is where the fixed cost of a
 *  key schedule and a tag stops dominating and the bulk kernels take over.
 *
 *  Use the following environment variables to control the benchmark:
 *
 *  - `STRINGWARS_DATASET` : path to the input dataset, used to size the pool of messages.
 *  - `STRINGWARS_FILTER` : regular expression selecting which kernels to run.
 *  - `STRINGWARS_DURATION` : seconds per kernel.
 */
#include <cstring> // `std::memcmp`
#include <string>  // `std::string`
#include <vector>  // `std::vector`

#include "shared.hpp"

using namespace ashvardanian::stringzilla::scripts;

/** @brief The message sizes every kernel is measured at, from a short record to a page. */
static constexpr std::size_t cipher_message_sizes_[] = {256, 1024, 4096, 16384};

/** @brief Bytes of messages cycled per size, large enough that no single message stays in a register file. */
static constexpr std::size_t cipher_pool_bytes_ = 1024ull * 1024ull;

/** @brief Counter-mode throughput for one kernel, cycling a pool so each call touches fresh bytes. */
template <sz_aes256_key_init_t init_, sz_aes256_ctr_xor_t transform_>
struct ctr_from_sz {

    std::size_t message_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_key_t key;
    sz_u8_t nonce[SZ_AES256_NONCE_LENGTH];

    ctr_from_sz(std::size_t configured, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), pool(input), target(output) {
        sz_u8_t secret[SZ_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != SZ_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 7 + 1);
        for (std::size_t index = 0; index != SZ_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index * 5 + 2);
        init_(&key, secret);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        transform_(&key, nonce, 0, pool.data() + offset, message_bytes, target.data() + offset);
        sz_u64_t check = 0;
        std::memcpy(&check, target.data() + offset, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations = 1;
        result.inputs_processed = 1;
        return result;
    }
};

/** @brief Authenticated throughput for one kernel, one whole message per call. */
template <sz_aes256_gcm_key_init_t init_, sz_aes256_gcm_encrypt_t encrypt_>
struct gcm_from_sz {

    std::size_t message_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_gcm_key_t key;
    sz_u8_t nonce[SZ_AES256_NONCE_LENGTH];

    gcm_from_sz(std::size_t configured, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), pool(input), target(output) {
        sz_u8_t secret[SZ_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != SZ_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 3 + 5);
        for (std::size_t index = 0; index != SZ_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index + 9);
        init_(&key, secret);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        sz_u8_t tag[SZ_AES256_TAG_LENGTH];
        encrypt_(&key, nonce, SZ_NULL, 0, pool.data() + offset, message_bytes, target.data() + offset, tag);
        sz_u64_t check = 0;
        std::memcpy(&check, tag, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations = 1;
        result.inputs_processed = 1;
        return result;
    }
};

/** @brief Fills a pool with a deterministic pattern, since cipher cost never depends on the bytes. */
static void fill_cipher_pool_(std::vector<char> &pool) noexcept {
    for (std::size_t index = 0; index != pool.size(); ++index) pool[index] = static_cast<char>(index * 31 + 7);
}

#pragma region Counter Mode

void bench_cipher_ctr(environment_t const &env) {

    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        std::vector<char> target(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        fill_cipher_pool_(pool);

        std::string const suffix = ":" + std::to_string(message_bytes);
        auto validator = ctr_from_sz<sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial> {message_bytes, pool, target};
        bench_result_t base = bench_unary(env, "sz_aes256_ctr_xor_serial" + suffix, validator).log();

#if SZ_USE_WESTMERE
        bench_unary(env, "sz_aes256_ctr_xor_westmere" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_westmere, sz_aes256_ctr_xor_westmere> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_ICELAKE
        bench_unary(env, "sz_aes256_ctr_xor_icelake" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_icelake, sz_aes256_ctr_xor_icelake> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_NEONAES
        bench_unary(env, "sz_aes256_ctr_xor_neonaes" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_neonaes, sz_aes256_ctr_xor_neonaes> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_SVE2AES
        bench_unary(env, "sz_aes256_ctr_xor_sve2aes" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_sve2aes, sz_aes256_ctr_xor_sve2aes> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128
        bench_unary(env, "sz_aes256_ctr_xor_v128" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_v128, sz_aes256_ctr_xor_v128> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128RELAXED
        bench_unary(
            env, "sz_aes256_ctr_xor_v128relaxed" + suffix, validator,
            ctr_from_sz<sz_aes256_key_init_v128relaxed, sz_aes256_ctr_xor_v128relaxed> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_RVVCRYPTO
        bench_unary(
            env, "sz_aes256_ctr_xor_rvvcrypto" + suffix, validator,
            ctr_from_sz<sz_aes256_key_init_rvvcrypto, sz_aes256_ctr_xor_rvvcrypto> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_POWERVSX
        bench_unary(env, "sz_aes256_ctr_xor_powervsx" + suffix, validator,
                    ctr_from_sz<sz_aes256_key_init_powervsx, sz_aes256_ctr_xor_powervsx> {message_bytes, pool, target})
            .log(base);
#endif
    }
}

#pragma endregion // Counter Mode

#pragma region Galois Counter Mode

void bench_cipher_gcm(environment_t const &env) {

    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        std::vector<char> target(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        fill_cipher_pool_(pool);

        std::string const suffix = ":" + std::to_string(message_bytes);
        auto validator = gcm_from_sz<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encrypt_serial> {message_bytes, pool,
                                                                                                   target};
        bench_result_t base = bench_unary(env, "sz_aes256_gcm_encrypt_serial" + suffix, validator).log();

#if SZ_USE_WESTMERE
        bench_unary(
            env, "sz_aes256_gcm_encrypt_westmere" + suffix, validator,
            gcm_from_sz<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encrypt_westmere> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_ICELAKE
        bench_unary(
            env, "sz_aes256_gcm_encrypt_icelake" + suffix, validator,
            gcm_from_sz<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encrypt_icelake> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_NEONAES
        bench_unary(
            env, "sz_aes256_gcm_encrypt_neonaes" + suffix, validator,
            gcm_from_sz<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encrypt_neonaes> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_SVE2AES
        bench_unary(
            env, "sz_aes256_gcm_encrypt_sve2aes" + suffix, validator,
            gcm_from_sz<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encrypt_sve2aes> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128
        bench_unary(env, "sz_aes256_gcm_encrypt_v128" + suffix, validator,
                    gcm_from_sz<sz_aes256_gcm_key_init_v128, sz_aes256_gcm_encrypt_v128> {message_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128RELAXED
        bench_unary(env, "sz_aes256_gcm_encrypt_v128relaxed" + suffix, validator,
                    gcm_from_sz<sz_aes256_gcm_key_init_v128relaxed, sz_aes256_gcm_encrypt_v128relaxed> {message_bytes,
                                                                                                        pool, target})
            .log(base);
#endif
#if SZ_USE_RVVCRYPTO
        bench_unary(env, "sz_aes256_gcm_encrypt_rvvcrypto" + suffix, validator,
                    gcm_from_sz<sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_encrypt_rvvcrypto> {message_bytes, pool,
                                                                                                    target})
            .log(base);
#endif
#if SZ_USE_POWERVSX
        bench_unary(
            env, "sz_aes256_gcm_encrypt_powervsx" + suffix, validator,
            gcm_from_sz<sz_aes256_gcm_key_init_powervsx, sz_aes256_gcm_encrypt_powervsx> {message_bytes, pool, target})
            .log(base);
#endif
    }
}

#pragma endregion // Galois Counter Mode

#pragma region Streaming

/** @brief The chunk sizes the streaming path is measured at, straddling the sixteen-byte block. */
static constexpr std::size_t cipher_chunk_sizes_[] = {1, 7, 16, 40};

/**
 *  @brief Seals one message in fixed-size chunks, which is where the partial-block path dominates.
 *
 *  A chunk below the block width leaves bytes staged in the state between calls, so the cost per byte
 *  is the staging rather than the cipher. Sweeping across sixteen shows where the two cross over.
 */
template <sz_aes256_gcm_key_init_t init_, sz_aes256_gcm_encryptor_init_t begin_,
          sz_aes256_gcm_encryptor_update_t update_, sz_aes256_gcm_encryptor_digest_t digest_>
struct gcm_stream_from_sz {

    std::size_t message_bytes;
    std::size_t chunk_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_gcm_key_t key;
    sz_u8_t nonce[SZ_AES256_NONCE_LENGTH];

    gcm_stream_from_sz(std::size_t configured, std::size_t chunk, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), chunk_bytes(chunk), pool(input), target(output) {
        sz_u8_t secret[SZ_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != SZ_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 3 + 5);
        for (std::size_t index = 0; index != SZ_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index + 9);
        init_(&key, secret);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        sz_u8_t tag[SZ_AES256_TAG_LENGTH];
        sz_aes256_gcm_encryptor_t encryptor;
        begin_(&encryptor, &key, nonce);
        for (std::size_t consumed = 0; consumed < message_bytes; consumed += chunk_bytes) {
            std::size_t const taken = chunk_bytes < message_bytes - consumed ? chunk_bytes : message_bytes - consumed;
            update_(&encryptor, pool.data() + offset + consumed, taken, target.data() + offset + consumed);
        }
        digest_(&encryptor, tag);
        sz_u64_t check = 0;
        std::memcpy(&check, tag, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations = 1;
        result.inputs_processed = 1;
        return result;
    }
};

void bench_cipher_stream(environment_t const &env) {

    std::size_t const message_bytes = 4096;
    for (std::size_t chunk_bytes : cipher_chunk_sizes_) {
        std::vector<char> pool(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        std::vector<char> target(cipher_pool_bytes_ + SZ_AES_BLOCK_LENGTH);
        fill_cipher_pool_(pool);

        std::string const suffix = ":chunk" + std::to_string(chunk_bytes);
        auto validator =
            gcm_stream_from_sz<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encryptor_init_serial,
                               sz_aes256_gcm_encryptor_update_serial, sz_aes256_gcm_encryptor_digest_serial> {
                message_bytes, chunk_bytes, pool, target};
        bench_result_t base = bench_unary(env, "sz_aes256_gcm_stream_serial" + suffix, validator).log();

#if SZ_USE_WESTMERE
        bench_unary(
            env, "sz_aes256_gcm_stream_westmere" + suffix, validator,
            gcm_stream_from_sz<sz_aes256_gcm_key_init_westmere, sz_aes256_gcm_encryptor_init_westmere,
                               sz_aes256_gcm_encryptor_update_westmere, sz_aes256_gcm_encryptor_digest_westmere> {
                message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_ICELAKE
        bench_unary(env, "sz_aes256_gcm_stream_icelake" + suffix, validator,
                    gcm_stream_from_sz<sz_aes256_gcm_key_init_icelake, sz_aes256_gcm_encryptor_init_icelake,
                                       sz_aes256_gcm_encryptor_update_icelake, sz_aes256_gcm_encryptor_digest_icelake> {
                        message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_NEONAES
        bench_unary(env, "sz_aes256_gcm_stream_neonaes" + suffix, validator,
                    gcm_stream_from_sz<sz_aes256_gcm_key_init_neonaes, sz_aes256_gcm_encryptor_init_neonaes,
                                       sz_aes256_gcm_encryptor_update_neonaes, sz_aes256_gcm_encryptor_digest_neonaes> {
                        message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_SVE2AES
        bench_unary(env, "sz_aes256_gcm_stream_sve2aes" + suffix, validator,
                    gcm_stream_from_sz<sz_aes256_gcm_key_init_sve2aes, sz_aes256_gcm_encryptor_init_sve2aes,
                                       sz_aes256_gcm_encryptor_update_sve2aes, sz_aes256_gcm_encryptor_digest_sve2aes> {
                        message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128
        bench_unary(env, "sz_aes256_gcm_stream_v128" + suffix, validator,
                    gcm_stream_from_sz<sz_aes256_gcm_key_init_v128, sz_aes256_gcm_encryptor_init_v128,
                                       sz_aes256_gcm_encryptor_update_v128, sz_aes256_gcm_encryptor_digest_v128> {
                        message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_V128RELAXED
        bench_unary(
            env, "sz_aes256_gcm_stream_v128relaxed" + suffix, validator,
            gcm_stream_from_sz<sz_aes256_gcm_key_init_v128relaxed, sz_aes256_gcm_encryptor_init_v128relaxed,
                               sz_aes256_gcm_encryptor_update_v128relaxed, sz_aes256_gcm_encryptor_digest_v128relaxed> {
                message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_RVVCRYPTO
        bench_unary(
            env, "sz_aes256_gcm_stream_rvvcrypto" + suffix, validator,
            gcm_stream_from_sz<sz_aes256_gcm_key_init_rvvcrypto, sz_aes256_gcm_encryptor_init_rvvcrypto,
                               sz_aes256_gcm_encryptor_update_rvvcrypto, sz_aes256_gcm_encryptor_digest_rvvcrypto> {
                message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
#if SZ_USE_POWERVSX
        bench_unary(
            env, "sz_aes256_gcm_stream_powervsx" + suffix, validator,
            gcm_stream_from_sz<sz_aes256_gcm_key_init_powervsx, sz_aes256_gcm_encryptor_init_powervsx,
                               sz_aes256_gcm_encryptor_update_powervsx, sz_aes256_gcm_encryptor_digest_powervsx> {
                message_bytes, chunk_bytes, pool, target})
            .log(base);
#endif
    }
}

#pragma endregion // Streaming

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla AES-256 Cipher Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::lines_k);

    std::printf("Starting AES-256 cipher benchmarks...\n");

    bench_cipher_ctr(env);
    bench_cipher_gcm(env);
    bench_cipher_stream(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
