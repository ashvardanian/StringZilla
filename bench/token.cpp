/**
 *  @file bench/token.cpp
 *  @author Ash Vardanian
 *  @date January 4, 2024
 *  @brief Benchmarks token-level operations like hashing, equality, ordering, and copies.
 *
 *  Times the token-level dispatch points over the English lines, against the STL and LibC. Every
 *  capability's kernels are timed against the serial ones by the `cross_<arch>.cpp` files, through
 *  the adapters in `cross.hpp`.
 *
 *  Memory-bound: token hashing, equality, ordering, and copies are bandwidth-limited, so it reads
 *  the whole file by default.
 *
 *  Benchmarks include:
 *  - Checksum calculation and hashing for each token - @b bytesum and @b hash.
 *  - Stream hashing of a token (file, lines, or words) - @b sz_hash_state_init_best,
 *    @b sz_hash_state_update_best, @b sz_hash_state_digest_best.
 *  - Equality check between two tokens and their relative order - @b equal and @b ordering.
 *  - Multi-seed hashing, and SHA-256 digest computation - @b bench_hashing_multiseed,
 *    @b bench_sha256, @b bench_sha256_multistate.
 *
 *  For token operations, the number of operations per second are reported as the number of bytes
 *  processed or comparisons performed, depending on the specific operation being benchmarked.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines STRINGWARS_FILTER='hash|sha256' \
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGZILLA_STRESS=1 STRINGZILLA_STRESS_TIME_LIMIT=120s STRINGZILLA_STRESS_DIR=logs
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `find.cpp`, `sequence.cpp`, and `memory.cpp`.
 */
#include <numeric> // `std::accumulate`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

#pragma region Unary Functions

/** Wraps @c std::accumulate into a function object compatible with our benchmarking suite. */
struct bytesum_from_std_t {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        std::size_t bytesum = std::accumulate(
            buffer.begin(), buffer.end(), (std::size_t)0,
            [](std::size_t sum, char c) { return sum + static_cast<unsigned char>(c); });
        do_not_optimize(bytesum);
        return {buffer.size(), static_cast<check_value_t>(bytesum), 1};
    }
};

/** Wraps @c std::hash into a function object compatible with our benchmarking suite. */
struct hash_from_std_t {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        std::size_t hash = std::hash<std::string_view> {}(buffer);
        do_not_optimize(hash); //! The used function is not documented and can't be tested against anything
        return {buffer.size(), 0, 1};
    }
};

/** Baseline: hashes one token under every seed via independent @c sz_hash_best calls. */
template <sz_kernel_hash_t func_>
struct hash_multiseed_loop_from_sz {
    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }
    inline call_result_t operator()(std::string_view buffer) const noexcept {
        auto const seeds = multiway_seeds();
        sz_u64_t mixed = 0;
        for (sz_u64_t seed : seeds) {
            sz_u64_t hash = 0;
            func_(buffer.data(), buffer.size(), seed, &hash, nullptr);
            mixed ^= hash;
        }
        do_not_optimize(mixed);
        return {buffer.size() * seeds.size(), static_cast<check_value_t>(mixed), seeds.size()};
    }
};

void bench_checksums(environment_t const &env, corpus_t const &corpus) {
    auto base_call = bytesum_from_sz<cpu_best<sz_bytesum_best>> {corpus};
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_bytesum_best", base_call);
    print(base);
    print(bench_unary(env, corpus, "bytesum<std::accumulate>", base_call, bytesum_from_std_t {corpus}),
          baseline_of(base));
}

void bench_hashing(environment_t const &env, corpus_t const &corpus) {
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_hash_best",
                                                  hash_from_sz<cpu_best<sz_hash_best>> {corpus});
    print(base);
    print(bench_unary(env, corpus, "std::hash", hash_from_std_t {corpus}), baseline_of(base));
}

void bench_hashing_multiseed(environment_t const &env, corpus_t const &corpus) {

    // Baseline is the status quo: K dispatched single-shot `sz_hash_best` calls per token, so the
    // speedup isolates the multi-seed structural win rather than a backend difference.
    auto validator = hash_multiseed_loop_from_sz<cpu_best<sz_hash_best>> {corpus};
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_hash_best_loop", validator);
    print(base);
    print(bench_unary(env, corpus, "sz_hash_multiseed_best", validator,
                      hash_multiseed_from_sz<cpu_best<sz_hash_multiseed_best>> {corpus}),
          baseline_of(base));
}

void bench_stream_hashing(environment_t const &env, corpus_t const &corpus) {
    using best_t = hash_stream_from_sz<cpu_best<sz_hash_state_init_best>, cpu_best<sz_hash_state_update_best>,
                                       cpu_best<sz_hash_state_digest_best>>;
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_hash_stream_best", best_t {corpus});
    print(base);
    print(bench_unary(env, corpus, "std::hash", hash_from_std_t {corpus}), baseline_of(base));
}

/** Baseline: digests a batch of tokens through independent single-state SHA256 calls. */
template <typename lanes_>
struct sha256_multistate_loop_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        sz_sha256_state_t states[multistate_lanes_k];
        sz_u8_t digests[multistate_lanes_k * STRINGZILLA_SHA256_DIGEST_LENGTH];
        std::size_t bytes_passed = 0;
        sz_capability_t const capabilities = sz::default_capabilities();
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            std::string_view const token = corpus.tokens[(token_index + lane_index) % corpus.tokens.size()];
            std::size_t const lane_length = lanes_::length(lane_index, token.size());
            sz_sha256_state_init_best(&states[lane_index], capabilities, nullptr);
            sz_sha256_state_update_best(&states[lane_index], token.data(), lane_length, capabilities, nullptr);
            sz_sha256_state_digest_best(&states[lane_index], &digests[lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH],
                                        capabilities, nullptr);
            bytes_passed += lane_length;
        }
        // Multiplied rather than XOR-ed, so two lanes swapping digests cannot cancel out - lane
        // ordering is exactly what a batched kernel gets wrong.
        sz_u64_t mixed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            sz_u64_t lane_word;
            std::memcpy(&lane_word, &digests[lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH], sizeof(sz_u64_t));
            mixed = mixed * 31u + lane_word;
        }
        do_not_optimize(mixed);
        call_result_t result;
        result.bytes_passed = bytes_passed;
        result.check_value = static_cast<check_value_t>(mixed);
        result.operations_count = multistate_lanes_k;
        return result;
    }
};

/** Runs the multi-state dispatch point against one lane-length shape. */
template <typename lanes_>
void bench_sha256_multistate_shape(environment_t const &env, corpus_t const &corpus, std::string const &suffix) {

    // Baseline is the realistic status quo: one dispatched single-state hash per message, so the
    // speedup isolates the structural batch win rather than a backend difference. The baseline is
    // rebuilt per shape, so each row compares like with like.
    auto validator = sha256_multistate_loop_from_sz<lanes_> {corpus};
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_sha256_multistate_loop" + suffix, validator);
    print(base);
    using best_t = sha256_multistate_from_sz<cpu_best<sz_sha256_multistate_update_best>,
                                             cpu_best<sz_sha256_multistate_digest_best>, lanes_>;
    print(bench_unary(env, corpus, "sz_sha256_multistate_best" + suffix, validator, best_t {corpus}),
          baseline_of(base));
}

void bench_sha256_multistate(environment_t const &env, corpus_t const &corpus) {
    bench_sha256_multistate_shape<sha256_lanes_uniform_t>(env, corpus, sha256_lanes_uniform_t::name_k);
    bench_sha256_multistate_shape<sha256_lanes_one_short_t>(env, corpus, sha256_lanes_one_short_t::name_k);
    bench_sha256_multistate_shape<sha256_lanes_one_long_t>(env, corpus, sha256_lanes_one_long_t::name_k);
}

void bench_sha256(environment_t const &env, corpus_t const &corpus) {
    using best_t = sha256_stream_from_sz<cpu_best<sz_sha256_state_init_best>, cpu_best<sz_sha256_state_update_best>,
                                         cpu_best<sz_sha256_state_digest_best>>;
    print(bench_unary(env, corpus, "sz_sha256_best", best_t {corpus}));
}

#pragma endregion

#pragma region Binary Functions

/** Wraps LibC's string equality check for potentially different length inputs. */
struct equality_from_memcmp_t {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index], corpus.tokens[corpus.tokens.size() - 1 - token_index]);
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

/** Wraps LibC's string order-checking for potentially different length inputs. */
struct ordering_from_memcmp_t {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index], corpus.tokens[corpus.tokens.size() - 1 - token_index]);
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

    /** Wraps LibC's string comparison for potentially different length inputs. */
    static int memcmp_for_ordering(std::string_view a, std::string_view b) noexcept {
        auto order = memcmp(a.data(), b.data(), a.size() < b.size() ? a.size() : b.size());
        if (order == 0) return a.size() == b.size() ? 0 : (a.size() < b.size() ? -1 : 1);
        return order < 0 ? -1 : 1; // Normalize to sz_ordering_t's {-1,0,1}, not the raw memcmp delta.
    }
};

void bench_comparing_equality(environment_t const &env, corpus_t const &corpus) {
    auto base_call = equality_from_sz<cpu_best<sz_equal_best>> {corpus};
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_equal_best", base_call);
    print(base);
    print(bench_unary(env, corpus, "equal<std::memcmp>", base_call, equality_from_memcmp_t {corpus}),
          baseline_of(base));
}

void bench_comparing_order(environment_t const &env, corpus_t const &corpus) {
    auto base_call = ordering_from_sz<cpu_best<sz_order_best>> {corpus};
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_order_best", base_call);
    print(base);
    print(bench_unary(env, corpus, "order<std::memcmp>", base_call, ordering_from_memcmp_t {corpus}),
          baseline_of(base));
}

#pragma endregion

void bench_token(environment_t &env) {
    corpus_t const &corpus = env.corpora.lines();
    fmt::println("Starting individual token-level benchmarks...");

    // Unary operations
    bench_checksums(env, corpus);
    bench_hashing(env, corpus);
    bench_hashing_multiseed(env, corpus);
    bench_stream_hashing(env, corpus);
    bench_sha256(env, corpus);
    bench_sha256_multistate(env, corpus);

    // Binary operations
    bench_comparing_equality(env, corpus);
    bench_comparing_order(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
