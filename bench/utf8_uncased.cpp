/**
 *  @file scripts/bench_utf8_uncased.cpp
 *  @brief Benchmarks the @b `sz_utf8_uncased_*` family — case folding and uncased search.
 *         The program accepts a file path to a dataset and benchmarks the case folding operations,
 *         validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Case folding for Unicode text - @b utf8_uncased_fold.
 *  - Uncased substring search for Unicode text - @b utf8_uncased_search.
 *
 *  Its sibling @b `bench_utf8_iterate.cpp` covers the @b `sz_utf8_*` iteration/segmentation family
 *  (codepoint counting, Nth-codepoint, newline/whitespace scanning, UAX-29 word/grapheme/sentence boundaries,
 *  UAX-14 line breaking, transcoding).
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=line` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams.
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
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_uncased_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words STRINGWARS_UNIQUE=1 \
 *      build_release/stringzilla_bench_utf8_uncased_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_iterate.cpp`, `bench_token.cpp`, `bench_find.cpp`,
 *  `bench_sequence.cpp`, and `bench_memory.cpp`.
 */
#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Case Folding Functions

/** @brief Wraps a hardware-specific UTF-8 case folding backend. */
template <sz_utf8_uncased_fold_t func_>
struct utf8_uncased_fold_from_sz {

    environment_t const &env;
    mutable std::vector<char> output_buffer; // Reusable buffer to avoid repeated allocation

    utf8_uncased_fold_from_sz(environment_t const &env_) : env(env_) {
        // Pre-allocate worst-case buffer: 3x input size for worst-case expansion
        std::size_t max_token_size = 0;
        for (auto const &token : env.tokens) max_token_size = std::max(max_token_size, token.size());
        output_buffer.resize(max_token_size * 3 + 64); // Extra padding for safety
    }

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        // Ensure buffer is large enough
        if (output_buffer.size() < buffer.size() * 3) output_buffer.resize(buffer.size() * 3 + 64);

        sz_size_t result_length = func_(buffer.data(), buffer.size(), output_buffer.data());
        do_not_optimize(output_buffer.data());
        do_not_optimize(result_length);

        // Use sz_hash for validation checksum
        check_value_t checksum = sz_bytesum(output_buffer.data(), result_length);
        return {buffer.size(), checksum};
    }
};

void bench_utf8_uncased_fold(environment_t const &env) {

    auto validator = utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_uncased_fold_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_uncased_fold_icelake", validator,
                utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_icelake> {env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_uncased_fold_haswell", validator,
                utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_haswell> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_uncased_fold_neon", validator, utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_neon> {env})
        .log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_uncased_fold_sve2", validator, utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_sve2> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_uncased_fold_v128", validator, utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_v128> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_uncased_fold_rvv", validator, utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_rvv> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_uncased_fold_lasx", validator, utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_lasx> {env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_uncased_fold_powervsx", validator,
                utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_powervsx> {env})
        .log(base);
#endif
}

#pragma endregion

#pragma region Uncased Find Functions

/** @brief Wraps a hardware-specific UTF-8 uncased find backend. */
template <sz_utf8_uncased_search_t func_>
struct utf8_uncased_search_from_sz {

    environment_t const &env;

    utf8_uncased_search_from_sz(environment_t const &env_) : env(env_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        std::string_view haystack = env.dataset;
        std::string_view needle = env.tokens[token_index];
        return operator()(haystack, needle);
    }

    inline call_result_t operator()(std::string_view haystack, std::string_view needle) const noexcept {
        sz_size_t matched_length = 0;
        std::size_t count_matches = 0;
        sz_cptr_t h = haystack.data();
        sz_size_t h_len = haystack.size();
        sz_utf8_uncased_needle_metadata_t metadata;
        std::memset(&metadata, 0, sizeof(metadata));

        // Count all uncased matches
        while (h_len >= needle.size()) {
            sz_cptr_t match = func_(h, h_len, needle.data(), needle.size(), &metadata, &matched_length);
            if (!match) break;
            ++count_matches;
            // Move past the match
            std::size_t offset = (match - h) + (matched_length ? matched_length : 1);
            h += offset;
            h_len -= offset;
        }

        do_not_optimize(count_matches);
        // Operations = haystack_bytes * needle_bytes (worst case comparisons)
        std::size_t count_operations = haystack.size() * needle.size();
        return {haystack.size(), static_cast<check_value_t>(count_matches), count_operations};
    }
};

void bench_utf8_uncased_search(environment_t const &env) {

    auto validator = utf8_uncased_search_from_sz<sz_utf8_uncased_search_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_uncased_search_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_uncased_search_icelake", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_icelake> {env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_uncased_search_haswell", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_haswell> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_uncased_search_neon", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_neon> {env})
        .log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_uncased_search_sve2", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_sve2> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_uncased_search_v128", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_v128> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_uncased_search_rvv", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_rvv> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_uncased_search_lasx", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_lasx> {env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_uncased_search_powervsx", validator,
                utf8_uncased_search_from_sz<sz_utf8_uncased_search_powervsx> {env})
        .log(base);
#endif
}

#pragma endregion

#pragma region Uncased Order Functions

/** @brief Wraps a hardware-specific UTF-8 uncased ordering backend. */
template <sz_utf8_uncased_order_t func_>
struct utf8_uncased_order_from_sz {

    environment_t const &env;

    utf8_uncased_order_from_sz(environment_t const &env_) : env(env_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        std::string_view a = env.tokens[token_index];
        std::string_view b = env.tokens[(token_index + 1) % env.tokens.size()];
        sz_ordering_t ordering = func_(a.data(), a.size(), b.data(), b.size());
        do_not_optimize(ordering);
        // Worst case is a full case-insensitive scan of both operands.
        std::size_t count_operations = a.size() + b.size();
        // Fold the {-1,0,+1} verdict into a non-negative check value so SIMD must agree with serial.
        return {a.size() + b.size(), static_cast<check_value_t>(ordering + 1), count_operations};
    }
};

void bench_utf8_uncased_order(environment_t const &env) {

    auto validator = utf8_uncased_order_from_sz<sz_utf8_uncased_order_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_uncased_order_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_uncased_order_icelake", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_icelake> {env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_uncased_order_haswell", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_haswell> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_uncased_order_neon", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_neon> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_uncased_order_v128", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_v128> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_uncased_order_rvv", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_rvv> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_uncased_order_lasx", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_lasx> {env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_uncased_order_powervsx", validator,
                utf8_uncased_order_from_sz<sz_utf8_uncased_order_powervsx> {env})
        .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Case Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual testing
        environment_t::tokenization_t::lines_k);

    std::printf("Starting Unicode benchmarks...\n");

    // Unicode operations
    bench_utf8_uncased_fold(env);
    bench_utf8_uncased_search(env);
    bench_utf8_uncased_order(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
