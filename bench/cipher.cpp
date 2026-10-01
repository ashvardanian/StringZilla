/**
 *  @file bench/cipher.cpp
 *  @author Ash Vardanian
 *  @date August 4, 2026
 *  @brief Benchmarks AES-256 counter and Galois/counter mode through the dispatch points.
 *
 *  Bulk shaped rather than token shaped, so this sits alongside the random fill of
 *  `bench/memory.cpp` rather than in `bench/token.cpp`. The corpus decides how long each message is
 *  drawn from, but the content is a deterministic fill: a cipher's cost depends on message length
 *  alone, never on the bytes themselves. Sizes sweep a short record up to a page-sized buffer,
 *  because that range is where the fixed cost of a key schedule and a tag stops dominating and bulk
 *  kernels take over. Every capability's kernels are timed against the serial ones by the
 *  `cross_<arch>.cpp` files, through the adapters in `cross.hpp`.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_FILTER=aes256 build_release/stringzilla_cpu_bench
 *  @endcode
 */
#include <string> // `std::string`
#include <vector> // `std::vector`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_cipher_ctr(environment_t const &env, corpus_t const &corpus) {
    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":" + std::to_string(message_bytes);
        using best_t = ctr_from_sz<cpu_best<sz_aes256_key_init_best>, cpu_best<sz_aes256_ctr_xor_best>>;
        print(bench_unary(env, corpus, "sz_aes256_ctr_xor_best" + suffix, best_t {message_bytes, pool, target}));
    }
}

void bench_cipher_gcm(environment_t const &env, corpus_t const &corpus) {
    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":" + std::to_string(message_bytes);
        using best_t = gcm_from_sz<cpu_best<sz_aes256_gcm_key_init_best>, cpu_best<sz_aes256_gcm_encrypt_best>>;
        print(bench_unary(env, corpus, "sz_aes256_gcm_encrypt_best" + suffix, best_t {message_bytes, pool, target}));
    }
}

void bench_cipher_stream(environment_t const &env, corpus_t const &corpus) {
    std::size_t const message_bytes = 4096;
    for (std::size_t chunk_bytes : cipher_chunk_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":chunk" + std::to_string(chunk_bytes);
        using best_t =
            gcm_stream_from_sz<cpu_best<sz_aes256_gcm_key_init_best>, cpu_best<sz_aes256_gcm_encryptor_init_best>,
                               cpu_best<sz_aes256_gcm_encryptor_update_best>,
                               cpu_best<sz_aes256_gcm_encryptor_digest_best>>;
        print(bench_unary(env, corpus, "sz_aes256_gcm_stream_best" + suffix,
                          best_t {message_bytes, chunk_bytes, pool, target}));
    }
}

void bench_cipher(environment_t &env) {
    corpus_t const &corpus = env.corpora.lines();
    fmt::println("Starting AES-256 cipher benchmarks...");
    bench_cipher_ctr(env, corpus);
    bench_cipher_gcm(env, corpus);
    bench_cipher_stream(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
