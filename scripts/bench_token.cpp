/**
 *  @file   bench_token.cpp
 *  @brief  Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *          The program accepts a file path to a dataset, tokenizes it, and benchmarks the search operations,
 *          validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Checksum calculation and hashing for each token - @b bytesum and @b hash.
 *  - Stream hashing of a token (file, lines, or words) - @b hash_init, @b hash_stream, @b hash_fold.
 *  - Equality check between two tokens and their relative order - @b equal and @b ordering.
 *
 *  For token operations, the number of operations per second are reported as the number of bytes processed
 *  or comparisons performed, depending on the specific operation being benchmarked.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
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
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_find.cpp`, `bench_sequence.cpp`, `bench_similarity.cpp`, and `bench_memory.cpp`.
 */
#include <numeric> // `std::accumulate`

#include "bench.hpp"

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
        return {buffer.size(), static_cast<check_value_t>(bytesum)};
    }
};

/** @brief Wraps @b `std::accumulate` into a function object compatible with our benchmarking suite. */
struct bytesum_from_std_t {

    environment_t const &env;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        std::size_t bytesum =
            std::accumulate(buffer.begin(), buffer.end(), (std::size_t)0,
                            [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
        do_not_optimize(bytesum);
        return {buffer.size(), static_cast<check_value_t>(bytesum)};
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
        return {buffer.size(), static_cast<check_value_t>(hash)};
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
        return {buffer.size() /* static_cast<check_value_t>(hash) */};
    }
};

/** @brief Wraps hash state initialization, streaming, and folding for streaming benchmarks. */
template <sz_hash_state_init_t init_, sz_hash_state_stream_t stream_, sz_hash_state_fold_t fold_>
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
        return {s.size(), static_cast<check_value_t>(hash)};
    }
};

void bench_checksums(environment_t const &env) {

    auto validator = bytesum_from_std_t(env);
    bench_result_t base_stl = bench_unary(env, "bytesum<std::accumulate>", validator).log();
    bench_result_t base =
        bench_unary(env, "sz_bytesum_serial", validator, bytesum_from_sz<sz_bytesum_serial>(env)).log(base_stl);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_bytesum_haswell", validator, bytesum_from_sz<sz_bytesum_haswell>(env)).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_bytesum_skylake", validator, bytesum_from_sz<sz_bytesum_skylake>(env)).log(base, base_stl);
#endif
#if SZ_USE_ICE
    bench_unary(env, "sz_bytesum_ice", validator, bytesum_from_sz<sz_bytesum_ice>(env)).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_bytesum_neon", validator, bytesum_from_sz<sz_bytesum_neon>(env)).log(base, base_stl);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_bytesum_sve", validator, bytesum_from_sz<sz_bytesum_sve>(env)).log(base, base_stl);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_bytesum_sve2", validator, bytesum_from_sz<sz_bytesum_sve2>(env)).log(base, base_stl);
#endif
}

void bench_hashing(environment_t const &env) {

    auto validator = hash_from_sz<sz_hash_serial>(env);
    bench_result_t base = bench_unary(env, "sz_hash_serial", validator).log();
    bench_result_t base_stl = bench_unary(env, "std::hash", hash_from_std_t(env)).log(base);
#if SZ_USE_HASWELL
    bench_unary(env, "sz_hash_haswell", validator, hash_from_sz<sz_hash_haswell>(env)).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_hash_skylake", validator, hash_from_sz<sz_hash_skylake>(env)).log(base, base_stl);
#endif
#if SZ_USE_ICE
    bench_unary(env, "sz_hash_ice", validator, hash_from_sz<sz_hash_ice>(env)).log(base, base_stl);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_hash_sve2", validator, hash_from_sz<sz_hash_sve2>(env)).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_hash_neon", validator, hash_from_sz<sz_hash_neon>(env)).log(base, base_stl);
#endif
}

void bench_stream_hashing(environment_t const &env) {

    auto validator =
        hash_stream_from_sz<sz_hash_state_init_serial, sz_hash_state_stream_serial, sz_hash_state_fold_serial>(env);
    bench_result_t base = bench_unary(env, "sz_hash_stream_serial", validator).log();
    bench_result_t base_stl = bench_unary(env, "std::hash", hash_from_std_t(env)).log(base);

#if SZ_USE_HASWELL
    bench_unary(
        env, "sz_hash_stream_haswell", validator,
        hash_stream_from_sz<sz_hash_state_init_haswell, sz_hash_state_stream_haswell, sz_hash_state_fold_haswell>(env))
        .log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(
        env, "sz_hash_stream_skylake", validator,
        hash_stream_from_sz<sz_hash_state_init_skylake, sz_hash_state_stream_skylake, sz_hash_state_fold_skylake>(env))
        .log(base, base_stl);
#endif
#if SZ_USE_ICE
    bench_unary(env, "sz_hash_stream_ice", validator,
                hash_stream_from_sz<sz_hash_state_init_ice, sz_hash_state_stream_ice, sz_hash_state_fold_ice>(env))
        .log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_hash_stream_neon", validator,
                hash_stream_from_sz<sz_hash_state_init_neon, sz_hash_state_stream_neon, sz_hash_state_fold_neon>(env))
        .log(base, base_stl);
#endif
}

#pragma endregion

#pragma region Binary Functions

/**
 *  @brief  Wraps a hardware-specific equality-checking backend into something similar to @b `std::equal_to`.
 *          Assuming that almost any random pair of strings would differ in the very first byte, to make benchmarks
 *          more similar to mixed cases, like Hash Table lookups, where during probing we meet both differing
 *          and equivalent strings.
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
 *  @brief  Wraps a hardware-specific order-checking backend into something similar to @b `std::equal_to`.
 *          Assuming that almost any random pair of strings would differ in the very first byte, to make benchmarks
 *          more similar to mixed cases, like Hash Table lookups, where during probing we meet both differing
 *          and equivalent strings.
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
        sz_ordering_t ba = func_(b.data(), a.size(), a.data(), a.size());
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
        return order;
    }
};

void bench_comparing_equality(environment_t const &env) {

    auto validator = equality_from_memcmp_t(env);
    bench_result_t base = bench_unary(env, "sz_equal_serial", validator, equality_from_sz<sz_equal_serial>(env)).log();
    bench_result_t base_stl = bench_unary(env, "equal<std::memcmp>", validator).log(base);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_equal_haswell", validator, equality_from_sz<sz_equal_haswell>(env)).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_equal_skylake", validator, equality_from_sz<sz_equal_skylake>(env)).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_equal_neon", validator, equality_from_sz<sz_equal_neon>(env)).log(base, base_stl);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_equal_sve", validator, equality_from_sz<sz_equal_sve>(env)).log(base, base_stl);
#endif
}

void bench_comparing_order(environment_t const &env) {

    auto validator = ordering_from_memcmp_t(env);
    bench_result_t base = bench_unary(env, "sz_order_serial", validator, ordering_from_sz<sz_order_serial>(env)).log();
    bench_result_t base_stl = bench_unary(env, "order<std::memcmp>", validator).log(base);

#if SZ_USE_HASWELL
    bench_unary(env, "sz_order_haswell", validator, ordering_from_sz<sz_order_haswell>(env)).log(base, base_stl);
#endif
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_order_skylake", validator, ordering_from_sz<sz_order_skylake>(env)).log(base, base_stl);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_order_neon", validator, ordering_from_sz<sz_order_neon>(env)).log(base, base_stl);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::lines_k);

    std::printf("Starting individual token-level benchmarks...\n");

    // Unary operations
    bench_checksums(env);
    bench_hashing(env);
    bench_stream_hashing(env);

    // Binary operations
    bench_comparing_equality(env);
    bench_comparing_order(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}