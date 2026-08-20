/**
 *  @file bench/token.cpp
 *  @brief Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *         The program accepts a file path to a dataset, tokenizes it, and benchmarks the search operations,
 *         validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Memory-bound: token hashing, equality, ordering, and copies are bandwidth-limited, so it reads the whole file by default.
 *
 *  Benchmarks include:
 *  - Checksum calculation and hashing for each token - @b bytesum and @b hash.
 *  - Stream hashing of a token (file, lines, or words) - @b sz_hash_state_init, @b sz_hash_state_update,
 *    @b sz_hash_state_digest.
 *  - Equality check between two tokens and their relative order - @b equal and @b ordering.
 *  - Multi-seed hashing, and SHA-256 digest computation - @b bench_hashing_multiseed, @b bench_sha256,
 *    @b bench_sha256_multistate.
 *
 *  For token operations, the number of operations per second are reported as the number of bytes processed
 *  or comparisons performed, depending on the specific operation being benchmarked.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=0` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_token_cpp20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines build_release/stringzilla_bench_token_cpp20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_token_cpp20
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `find.cpp`, `sequence.cpp`, and `memory.cpp`.
 */
#include <numeric> // `std::accumulate`
#include <array>   // `std::array`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Unary Functions

/** @brief Wraps a hardware-specific hashing backend into something similar to @b `std::accumulate`. */
template <sz_bytesum_t func_>
struct bytesum_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_u64_t bytesum = func_(buffer.data(), buffer.size());
        do_not_optimize(bytesum);
        return {buffer.size(), static_cast<check_value_t>(bytesum), 1};
    }
};

/** @brief Wraps @b `std::accumulate` into a function object compatible with our benchmarking suite. */
struct bytesum_from_std_t {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        std::size_t bytesum = std::accumulate(
            buffer.begin(), buffer.end(), (std::size_t)0,
            [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
        do_not_optimize(bytesum);
        return {buffer.size(), static_cast<check_value_t>(bytesum), 1};
    }
};

/** @brief Wraps a hardware-specific hashing backend into something similar to @b `std::hash`. */
template <sz_hash_t func_>
struct hash_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_u64_t hash = func_(buffer.data(), buffer.size(), 0);
        do_not_optimize(hash);
        return {buffer.size(), static_cast<check_value_t>(hash), 1};
    }
};

/** @brief Wraps @b `std::hash` into a function object compatible with our benchmarking suite. */
struct hash_from_std_t {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        std::size_t hash = std::hash<std::string_view> {}(buffer);
        do_not_optimize(hash); //! The used function is not documented and can't be tested against anything
        return {buffer.size(), 0, 1};
    }
};

/** @brief Fixed seed schedule shared by the multi-seed hashing baseline and kernels. */
inline std::array<sz_u64_t, 8> multiway_seeds() noexcept {
    return {0u, 1u, 42u, 314159u, 2654435761u, 11400714819323198485ull, 7u, 8u};
}

/** @brief Baseline: hashes one token under every seed via independent `sz_hash` calls. */
template <sz_hash_t func_>
struct hash_multiseed_loop_from_sz {
    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }
    inline call_result_t operator()(std::string_view buffer) const noexcept {
        auto const seeds = multiway_seeds();
        sz_u64_t mixed = 0;
        for (sz_u64_t seed : seeds) mixed ^= func_(buffer.data(), buffer.size(), seed);
        do_not_optimize(mixed);
        return {buffer.size() * seeds.size(), static_cast<check_value_t>(mixed), seeds.size()};
    }
};

/** @brief Hashes one token under every seed in a single `sz_hash_multiseed` call. */
template <sz_hash_multiseed_t func_>
struct hash_multiseed_from_sz {
    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }
    inline call_result_t operator()(std::string_view buffer) const noexcept {
        auto seeds = multiway_seeds();
        decltype(seeds) hashes; // Same std::array type - carries the compile-time seed count.
        func_(buffer.data(), buffer.size(), seeds.data(), seeds.size(), hashes.data());
        sz_u64_t mixed = 0;
        for (sz_u64_t hash : hashes) mixed ^= hash;
        do_not_optimize(mixed);
        return {buffer.size() * seeds.size(), static_cast<check_value_t>(mixed), seeds.size()};
    }
};

/** @brief Wraps hash state initialization, streaming, and folding for streaming benchmarks. */
template <sz_hash_state_init_t init_, sz_hash_state_update_t stream_, sz_hash_state_digest_t fold_>
struct hash_stream_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    call_result_t operator()(std::string_view s) const noexcept {
        sz_hash_state_t state;
        init_(&state, 42);
        stream_(&state, s.data(), s.size());
        sz_u64_t hash = fold_(&state);
        do_not_optimize(hash);
        return {s.size(), static_cast<check_value_t>(hash), 1};
    }
};

void bench_checksums(environment_t const &env) {

    auto validator = bytesum_from_std_t {env};
    bench_result_t base_stl = bench_unary(env, "bytesum<std::accumulate>", validator).log();
    bench_result_t base =
        bench_unary(env, "sz_bytesum_serial", validator, bytesum_from_sz<sz_bytesum_serial> {env}).log(base_stl);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_bytesum_haswell", validator, bytesum_from_sz<sz_bytesum_haswell> {env}).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_bytesum_skylake", validator, bytesum_from_sz<sz_bytesum_skylake> {env}).log(base, base_stl);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_bytesum_icelake", validator, bytesum_from_sz<sz_bytesum_icelake> {env}).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_bytesum_neon", validator, bytesum_from_sz<sz_bytesum_neon> {env}).log(base, base_stl);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_bytesum_sve", validator, bytesum_from_sz<sz_bytesum_sve> {env}).log(base, base_stl);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_bytesum_sve2", validator, bytesum_from_sz<sz_bytesum_sve2> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_bytesum_v128", validator, bytesum_from_sz<sz_bytesum_v128> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_bytesum_v128relaxed", validator, bytesum_from_sz<sz_bytesum_v128relaxed> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_bytesum_rvv", validator, bytesum_from_sz<sz_bytesum_rvv> {env}).log(base, base_stl);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_bytesum_lasx", validator, bytesum_from_sz<sz_bytesum_lasx> {env}).log(base, base_stl);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_bytesum_powervsx", validator, bytesum_from_sz<sz_bytesum_powervsx> {env}).log(base, base_stl);
#endif
}

void bench_hashing(environment_t const &env) {

    auto validator = hash_from_sz<sz_hash_serial> {env};
    bench_result_t base = bench_unary(env, "sz_hash_serial", validator).log();
    bench_result_t base_stl = bench_unary(env, "std::hash", hash_from_std_t {env}).log(base);
#if SZ_USE_WESTMERE
    bench_unary(env, "sz_hash_westmere", validator, hash_from_sz<sz_hash_westmere> {env}).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_hash_skylake", validator, hash_from_sz<sz_hash_skylake> {env}).log(base, base_stl);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_hash_icelake", validator, hash_from_sz<sz_hash_icelake> {env}).log(base, base_stl);
#endif
#if SZ_USE_NEONAES
    bench_unary(env, "sz_hash_neonaes", validator, hash_from_sz<sz_hash_neonaes> {env}).log(base, base_stl);
#endif
#if SZ_USE_SVE2AES
    bench_unary(env, "sz_hash_sve2aes", validator, hash_from_sz<sz_hash_sve2aes> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_hash_v128", validator, hash_from_sz<sz_hash_v128> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_hash_v128relaxed", validator, hash_from_sz<sz_hash_v128relaxed> {env}).log(base, base_stl);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_hash_rvv", validator, hash_from_sz<sz_hash_rvv> {env}).log(base, base_stl);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_hash_lasx", validator, hash_from_sz<sz_hash_lasx> {env}).log(base, base_stl);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_hash_powervsx", validator, hash_from_sz<sz_hash_powervsx> {env}).log(base, base_stl);
#endif
}

void bench_hashing_multiseed(environment_t const &env) {

    // Baseline is the realistic status quo: K dispatched single-shot `sz_hash` calls per token, so the
    // reported speedup isolates the multi-seed structural win rather than a backend-vs-serial difference.
    auto validator = hash_multiseed_loop_from_sz<sz_hash> {env};
    bench_result_t base = bench_unary(env, "sz_hash_loop", validator).log();

    bench_unary(env, "sz_hash_multiseed", validator, hash_multiseed_from_sz<sz_hash_multiseed> {env}).log(base);
    bench_unary(env, "sz_hash_multiseed_serial", validator, hash_multiseed_from_sz<sz_hash_multiseed_serial> {env})
        .log(base);
#if SZ_USE_WESTMERE
    bench_unary(env, "sz_hash_multiseed_westmere", validator, hash_multiseed_from_sz<sz_hash_multiseed_westmere> {env})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_hash_multiseed_icelake", validator, hash_multiseed_from_sz<sz_hash_multiseed_icelake> {env})
        .log(base);
#endif
#if SZ_USE_NEONAES
    bench_unary(env, "sz_hash_multiseed_neonaes", validator, hash_multiseed_from_sz<sz_hash_multiseed_neonaes> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_hash_multiseed_v128", validator, hash_multiseed_from_sz<sz_hash_multiseed_v128> {env})
        .log(base);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_hash_multiseed_v128relaxed", validator,
                hash_multiseed_from_sz<sz_hash_multiseed_v128relaxed> {env})
        .log(base);
#endif
}

void bench_stream_hashing(environment_t const &env) {

    auto validator =
        hash_stream_from_sz<sz_hash_state_init_serial, sz_hash_state_update_serial, sz_hash_state_digest_serial> {env};
    bench_result_t base = bench_unary(env, "sz_hash_stream_serial", validator).log();
    bench_result_t base_stl = bench_unary(env, "std::hash", hash_from_std_t {env}).log(base);

#if SZ_USE_WESTMERE
    bench_unary(
        env, "sz_hash_stream_westmere", validator,
        hash_stream_from_sz<sz_hash_state_init_westmere, sz_hash_state_update_westmere, sz_hash_state_digest_westmere> {
            env})
        .log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_hash_stream_skylake", validator,
        hash_stream_from_sz<sz_hash_state_init_skylake, sz_hash_state_update_skylake, sz_hash_state_digest_skylake> {
            env})
        .log(base, base_stl);
#endif
#if SZ_USE_ICELAKE
    bench_unary(
        env, "sz_hash_stream_icelake", validator,
        hash_stream_from_sz<sz_hash_state_init_icelake, sz_hash_state_update_icelake, sz_hash_state_digest_icelake> {
            env})
        .log(base, base_stl);
#endif
#if SZ_USE_NEONAES
    bench_unary(
        env, "sz_hash_stream_neonaes", validator,
        hash_stream_from_sz<sz_hash_state_init_neonaes, sz_hash_state_update_neonaes, sz_hash_state_digest_neonaes> {
            env})
        .log(base, base_stl);
#endif
#if SZ_USE_SVE2AES
    bench_unary(
        env, "sz_hash_stream_sve2aes", validator,
        hash_stream_from_sz<sz_hash_state_init_sve2aes, sz_hash_state_update_sve2aes, sz_hash_state_digest_sve2aes> {
            env})
        .log(base, base_stl);
#endif
#if SZ_USE_V128
    bench_unary(
        env, "sz_hash_stream_v128", validator,
        hash_stream_from_sz<sz_hash_state_init_v128, sz_hash_state_update_v128, sz_hash_state_digest_v128> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_hash_stream_v128relaxed", validator,
                hash_stream_from_sz<sz_hash_state_init_v128relaxed, sz_hash_state_update_v128relaxed,
                                    sz_hash_state_digest_v128relaxed> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_hash_stream_rvv", validator,
                hash_stream_from_sz<sz_hash_state_init_rvv, sz_hash_state_update_rvv, sz_hash_state_digest_rvv> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_LASX
    bench_unary(
        env, "sz_hash_stream_lasx", validator,
        hash_stream_from_sz<sz_hash_state_init_lasx, sz_hash_state_update_lasx, sz_hash_state_digest_lasx> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_POWERVSX
    bench_unary(
        env, "sz_hash_stream_powervsx", validator,
        hash_stream_from_sz<sz_hash_state_init_powervsx, sz_hash_state_update_powervsx, sz_hash_state_digest_powervsx> {
            env})
        .log(base, base_stl);
#endif
}

/** @brief Wraps SHA256 state initialization, streaming, and digesting for streaming benchmarks. */
template <sz_sha256_state_init_t init_, sz_sha256_state_update_t stream_, sz_sha256_state_digest_t fold_>
struct sha256_stream_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    call_result_t operator()(std::string_view s) const noexcept {
        sz_sha256_state_t state;
        init_(&state);
        stream_(&state, s.data(), s.size());
        sz_u8_t digest[SZ_SHA256_DIGEST_LENGTH];
        fold_(&state, digest);
        // Use first 8 bytes of digest as check value
        sz_u64_t check = 0;
        std::memcpy(&check, digest, sizeof(sz_u64_t));
        do_not_optimize(check);
        return {s.size(), static_cast<check_value_t>(check), 1};
    }
};

/** @brief Number of independent messages driven through one multi-state SHA256 call. */
enum : std::size_t { multistate_lanes_k = 16 };

/**
 *  @brief Lane-length policies, trimming the corpus tokens into a chosen shape.
 *
 *  Lanes in a group advance in lockstep, so the spread of lengths within a group is what the batched kernels
 *  are actually sensitive to - a corpus alone only ever shows whatever spread it happens to have. Every
 *  policy trims rather than extends, so lanes stay inside the tokens the environment already owns.
 */
struct sha256_lanes_uniform_t {
    static constexpr char const *name_k = "";
    static inline std::size_t length(std::size_t, std::size_t token_length) noexcept { return token_length; }
};

/** @brief One lane far shorter than the rest, which drops its whole group to the scalar kernel. */
struct sha256_lanes_one_short_t {
    static constexpr char const *name_k = "_one_short";
    static inline std::size_t length(std::size_t lane_index, std::size_t token_length) noexcept {
        return lane_index == 0 ? (token_length < 21 ? token_length : 21) : token_length;
    }
};

/** @brief One lane far longer than the rest, the shape lockstep pays the most for. */
struct sha256_lanes_one_long_t {
    static constexpr char const *name_k = "_one_long";
    static inline std::size_t length(std::size_t lane_index, std::size_t token_length) noexcept {
        return lane_index == 0 ? token_length : token_length / 8;
    }
};

/** @brief Baseline: digests a batch of tokens through independent single-state SHA256 calls. */
template <typename lanes_>
struct sha256_multistate_loop_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        sz_sha256_state_t states[multistate_lanes_k];
        sz_u8_t digests[multistate_lanes_k * SZ_SHA256_DIGEST_LENGTH];
        std::size_t bytes_passed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            std::string_view const token = env.tokens[(token_index + lane_index) % env.tokens.size()];
            std::size_t const lane_length = lanes_::length(lane_index, token.size());
            sz_sha256_state_init(&states[lane_index]);
            sz_sha256_state_update(&states[lane_index], token.data(), lane_length);
            sz_sha256_state_digest(&states[lane_index], &digests[lane_index * SZ_SHA256_DIGEST_LENGTH]);
            bytes_passed += lane_length;
        }
        // Multiplied rather than XOR-ed, so two lanes swapping digests cannot cancel out - lane ordering is
        // exactly what a batched kernel gets wrong.
        sz_u64_t mixed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            sz_u64_t lane_word;
            std::memcpy(&lane_word, &digests[lane_index * SZ_SHA256_DIGEST_LENGTH], sizeof(sz_u64_t));
            mixed = mixed * 31u + lane_word;
        }
        do_not_optimize(mixed);
        call_result_t result;
        result.bytes_passed = bytes_passed;
        result.check_value = static_cast<check_value_t>(mixed);
        result.operations = multistate_lanes_k;
        result.inputs_processed = multistate_lanes_k;
        return result;
    }
};

/**
 *  @brief Digests a batch of tokens through one `sz_sha256_multistate_update` call.
 *
 *  Update and digest come from the same tier on purpose. Goldmont is legacy-SSE SHA-NI while Skylake and
 *  above are AVX-512, and pairing a legacy-SSE update with a wide digest in one loop makes every SHA-NI
 *  instruction pay an AVX-SSE transition, which reads as a slow kernel rather than as a mixed measurement.
 */
template <sz_sha256_multistate_update_t update_, sz_sha256_multistate_digest_t digest_, typename lanes_>
struct sha256_multistate_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        sz_string_view_t lanes[multistate_lanes_k];
        sz_sha256_state_t states[multistate_lanes_k];
        sz_u8_t digests[multistate_lanes_k * SZ_SHA256_DIGEST_LENGTH];
        std::size_t bytes_passed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            std::string_view const token = env.tokens[(token_index + lane_index) % env.tokens.size()];
            std::size_t const lane_length = lanes_::length(lane_index, token.size());
            lanes[lane_index].start = token.data();
            lanes[lane_index].length = lane_length;
            sz_sha256_state_init(&states[lane_index]);
            bytes_passed += lane_length;
        }
        sz_sequence_t texts;
        sz_sequence_from_string_views(lanes, multistate_lanes_k, &texts);
        update_(states, &texts);
        digest_(states, multistate_lanes_k, digests);
        // Multiplied rather than XOR-ed, so two lanes swapping digests cannot cancel out - lane ordering is
        // exactly what a batched kernel gets wrong.
        sz_u64_t mixed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            sz_u64_t lane_word;
            std::memcpy(&lane_word, &digests[lane_index * SZ_SHA256_DIGEST_LENGTH], sizeof(sz_u64_t));
            mixed = mixed * 31u + lane_word;
        }
        do_not_optimize(mixed);
        call_result_t result;
        result.bytes_passed = bytes_passed;
        result.check_value = static_cast<check_value_t>(mixed);
        result.operations = multistate_lanes_k;
        result.inputs_processed = multistate_lanes_k;
        return result;
    }
};

/** @brief Runs every multi-state backend against one lane-length shape. */
template <typename lanes_>
static void bench_sha256_multistate_shape(environment_t const &env, std::string const &suffix) {

    // Baseline is the realistic status quo: one dispatched single-state hash per message, so the reported
    // speedup isolates the structural batch win rather than a backend-versus-serial difference. The baseline
    // is rebuilt per shape, so each row compares like with like.
    auto validator = sha256_multistate_loop_from_sz<lanes_> {env};
    bench_result_t base = bench_unary(env, "sz_sha256_multistate_loop" + suffix, validator).log();

    bench_unary(
        env, "sz_sha256_multistate_serial" + suffix, validator,
        sha256_multistate_from_sz<sz_sha256_multistate_update_serial, sz_sha256_multistate_digest_serial, lanes_> {env})
        .log(base);
#if SZ_USE_GOLDMONT
    bench_unary(
        env, "sz_sha256_multistate_goldmont" + suffix, validator,
        sha256_multistate_from_sz<sz_sha256_multistate_update_goldmont, sz_sha256_multistate_digest_goldmont, lanes_> {
            env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(
        env, "sz_sha256_multistate_haswell" + suffix, validator,
        sha256_multistate_from_sz<sz_sha256_multistate_update_haswell, sz_sha256_multistate_digest_haswell, lanes_> {
            env})
        .log(base);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_sha256_multistate_skylake" + suffix, validator,
        sha256_multistate_from_sz<sz_sha256_multistate_update_skylake, sz_sha256_multistate_digest_skylake, lanes_> {
            env})
        .log(base);
#endif
}

void bench_sha256_multistate(environment_t const &env) {
    bench_sha256_multistate_shape<sha256_lanes_uniform_t>(env, sha256_lanes_uniform_t::name_k);
    bench_sha256_multistate_shape<sha256_lanes_one_short_t>(env, sha256_lanes_one_short_t::name_k);
    bench_sha256_multistate_shape<sha256_lanes_one_long_t>(env, sha256_lanes_one_long_t::name_k);
}

void bench_sha256(environment_t const &env) {

    auto validator = sha256_stream_from_sz<sz_sha256_state_init_serial, sz_sha256_state_update_serial,
                                           sz_sha256_state_digest_serial> {env};
    bench_result_t base = bench_unary(env, "sz_sha256_serial", validator).log();

#if SZ_USE_GOLDMONT
    bench_unary(env, "sz_sha256_goldmont", validator,
                sha256_stream_from_sz<sz_sha256_state_init_goldmont, sz_sha256_state_update_goldmont,
                                      sz_sha256_state_digest_goldmont> {env})
        .log(base);
#endif
#if SZ_USE_NEONSHA
    bench_unary(env, "sz_sha256_neonsha", validator,
                sha256_stream_from_sz<sz_sha256_state_init_neonsha, sz_sha256_state_update_neonsha,
                                      sz_sha256_state_digest_neonsha> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(
        env, "sz_sha256_v128", validator,
        sha256_stream_from_sz<sz_sha256_state_init_v128, sz_sha256_state_update_v128, sz_sha256_state_digest_v128> {
            env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(
        env, "sz_sha256_rvv", validator,
        sha256_stream_from_sz<sz_sha256_state_init_rvv, sz_sha256_state_update_rvv, sz_sha256_state_digest_rvv> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(
        env, "sz_sha256_lasx", validator,
        sha256_stream_from_sz<sz_sha256_state_init_lasx, sz_sha256_state_update_lasx, sz_sha256_state_digest_lasx> {
            env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_sha256_powervsx", validator,
                sha256_stream_from_sz<sz_sha256_state_init_powervsx, sz_sha256_state_update_powervsx,
                                      sz_sha256_state_digest_powervsx> {env})
        .log(base);
#endif
}

#pragma endregion

#pragma region Binary Functions

/**
 *  @brief Wraps a hardware-specific equality-checking backend into something similar to @b `std::equal_to`.
 *         Assuming that almost any random pair of strings would differ in the very first byte, to make benchmarks
 *         more similar to mixed cases, like Hash Table lookups, where during probing we meet both differing
 *         and equivalent strings.
 */
template <sz_equal_t func_>
struct equality_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        bool ab = func_(a.data(), b.data(), std::min(a.size(), b.size())) == sz_true_k;
        bool aa = func_(a.data(), a.data(), a.size()) == sz_true_k;
        bool bb = func_(b.data(), b.data(), b.size()) == sz_true_k;
        bool ba = func_(b.data(), a.data(), std::min(a.size(), b.size())) == sz_true_k;
        std::size_t max_bytes_passed = a.size() + b.size() + std::min(a.size(), b.size());
        check_value_t check_value = ab;
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }
};

/** @brief Wraps LibC's string equality check for potentially different length inputs. */
struct equality_from_memcmp_t {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        bool ab = std::memcmp(a.data(), b.data(), std::min(a.size(), b.size())) == 0;
        bool aa = std::memcmp(a.data(), a.data(), a.size()) == 0;
        bool bb = std::memcmp(b.data(), b.data(), b.size()) == 0;
        bool ba = std::memcmp(b.data(), a.data(), std::min(a.size(), b.size())) == 0;
        std::size_t max_bytes_passed = a.size() + b.size() + std::min(a.size(), b.size());
        check_value_t check_value = ab;
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }
};

/**
 *  @brief Wraps a hardware-specific order-checking backend into something similar to @b `std::less`.
 *         Assuming that almost any random pair of strings would differ in the very first byte, to make benchmarks
 *         more similar to mixed cases, like Hash Table lookups, where during probing we meet both differing
 *         and equivalent strings.
 */
template <sz_order_t func_>
struct ordering_from_sz {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_ordering_t ab = func_(a.data(), a.size(), b.data(), b.size());
        sz_ordering_t aa = func_(a.data(), a.size(), a.data(), a.size());
        sz_ordering_t bb = func_(b.data(), b.size(), b.data(), b.size());
        sz_ordering_t ba = func_(b.data(), b.size(), a.data(), a.size());
        std::size_t max_bytes_passed = 4 * std::min(a.size(), b.size());
        check_value_t check_value = ab + aa * 3 + bb * 9 + ba * 27; // Each can have 3 unique values
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }
};

/** @brief Wraps LibC's string order-checking for potentially different length inputs. */
struct ordering_from_memcmp_t {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index], env.tokens[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        int ab = memcmp_for_ordering(a, b);
        int aa = memcmp_for_ordering(a, a);
        int bb = memcmp_for_ordering(b, b);
        int ba = memcmp_for_ordering(b, a);
        std::size_t max_bytes_passed = 4 * std::min(a.size(), b.size());
        check_value_t check_value = ab + aa * 3 + bb * 9 + ba * 27; // Each can have 3 unique values
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }

    /** @brief Wraps LibC's string comparison for potentially different length inputs. */
    static int memcmp_for_ordering(std::string_view a, std::string_view b) noexcept {
        auto order = memcmp(a.data(), b.data(), a.size() < b.size() ? a.size() : b.size());
        if (order == 0) return a.size() == b.size() ? 0 : (a.size() < b.size() ? -1 : 1);
        return order < 0 ? -1 : 1; // Normalize to sz_ordering_t's {-1,0,1}, not the raw memcmp delta.
    }
};

void bench_comparing_equality(environment_t const &env) {

    auto validator = equality_from_memcmp_t {env};
    bench_result_t base = bench_unary(env, "sz_equal_serial", validator, equality_from_sz<sz_equal_serial> {env}).log();
    bench_result_t base_stl = bench_unary(env, "equal<std::memcmp>", validator).log(base);

#if SZ_USE_WESTMERE
    bench_unary(env, "sz_equal_westmere", validator, equality_from_sz<sz_equal_westmere> {env}).log(base, base_stl);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_equal_haswell", validator, equality_from_sz<sz_equal_haswell> {env}).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_equal_skylake", validator, equality_from_sz<sz_equal_skylake> {env}).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_equal_neon", validator, equality_from_sz<sz_equal_neon> {env}).log(base, base_stl);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_equal_sve", validator, equality_from_sz<sz_equal_sve> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_equal_v128", validator, equality_from_sz<sz_equal_v128> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_equal_v128relaxed", validator, equality_from_sz<sz_equal_v128relaxed> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_equal_rvv", validator, equality_from_sz<sz_equal_rvv> {env}).log(base, base_stl);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_equal_lasx", validator, equality_from_sz<sz_equal_lasx> {env}).log(base, base_stl);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_equal_powervsx", validator, equality_from_sz<sz_equal_powervsx> {env}).log(base, base_stl);
#endif
}

void bench_comparing_order(environment_t const &env) {

    auto validator = ordering_from_memcmp_t {env};
    bench_result_t base = bench_unary(env, "sz_order_serial", validator, ordering_from_sz<sz_order_serial> {env}).log();
    bench_result_t base_stl = bench_unary(env, "order<std::memcmp>", validator).log(base);

#if SZ_USE_WESTMERE
    bench_unary(env, "sz_order_westmere", validator, ordering_from_sz<sz_order_westmere> {env}).log(base, base_stl);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_order_haswell", validator, ordering_from_sz<sz_order_haswell> {env}).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_order_skylake", validator, ordering_from_sz<sz_order_skylake> {env}).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_order_neon", validator, ordering_from_sz<sz_order_neon> {env}).log(base, base_stl);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_order_sve", validator, ordering_from_sz<sz_order_sve> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_order_v128", validator, ordering_from_sz<sz_order_v128> {env}).log(base, base_stl);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_order_v128relaxed", validator, ordering_from_sz<sz_order_v128relaxed> {env})
        .log(base, base_stl);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_order_rvv", validator, ordering_from_sz<sz_order_rvv> {env}).log(base, base_stl);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_order_lasx", validator, ordering_from_sz<sz_order_lasx> {env}).log(base, base_stl);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_order_powervsx", validator, ordering_from_sz<sz_order_powervsx> {env}).log(base, base_stl);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::lines_k);

    std::printf("Starting individual token-level benchmarks...\n");

    // Unary operations
    bench_checksums(env);
    bench_hashing(env);
    bench_hashing_multiseed(env);
    bench_stream_hashing(env);
    bench_sha256(env);
    bench_sha256_multistate(env);

    // Binary operations
    bench_comparing_equality(env);
    bench_comparing_order(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
