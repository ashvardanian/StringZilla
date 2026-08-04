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

namespace {

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
void fill_cipher_pool_(std::vector<char> &pool) noexcept {
    for (std::size_t index = 0; index != pool.size(); ++index) pool[index] = static_cast<char>(index * 31 + 7);
}

} // namespace

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
    }
}

#pragma endregion // Galois Counter Mode

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

    std::printf("All benchmarks passed.\n");
    return 0;
}
