/**
 *  @file scripts/bench_utf8_delimiters.cpp
 *  @brief Benchmarks the @b `sz_find_delimiters_utf8_*` family — first-UTF-8-delimiter scanning.
 *         The program accepts a file path to a dataset, tokenizes it, and benchmarks scanning each token
 *         for the first delimiter codepoint (punctuation/symbol/separator/whitespace), validating the
 *         SIMD-accelerated backends against the serial baseline.
 *
 *  Benchmarks include:
 *  - First-delimiter scan over Unicode text - @b find_delimiters_utf8.
 *
 *  As this is a scan-and-find operation, the number of operations per second is reported as the number of
 *  bytes scanned, matching the convention used by `bench_find.cpp`'s byte search.
 *
 *  Its sibling @b `bench_utf8_iterate.cpp` covers the @b `sz_utf8_*` iteration/segmentation family
 *  (codepoint counting, Nth-codepoint, newline/whitespace scanning, UAX-29 word/grapheme/sentence boundaries,
 *  UAX-14 line breaking, transcoding), and @b `bench_utf8_uncased.cpp` covers case folding and uncased search.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams.
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
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_delimiters_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      build_release/stringzilla_bench_utf8_delimiters_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_iterate.cpp`, `bench_utf8_uncased.cpp`, `bench_find.cpp`,
 *  `bench_token.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */
#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Find Delimiter Functions

/** @brief Function-pointer type of a per-ISA `sz_find_delimiters_utf8_<isa>` backend. */
typedef sz_cptr_t (*sz_find_delimiter_utf8_t)(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/** @brief Wraps a hardware-specific UTF-8 first-delimiter scanning backend. */
template <sz_find_delimiter_utf8_t func_>
struct find_delimiters_utf8_from_sz {

    environment_t const &env;

    find_delimiters_utf8_from_sz(environment_t const &env_) : env(env_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view token) const noexcept {
        // Scan the whole token, hopping past each delimiter, so the entire buffer is traversed
        // regardless of where the first delimiter sits - that keeps the throughput proportional to bytes.
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        std::size_t count_matches = 0;
        while (remaining) {
            sz_size_t matched_length = 0;
            sz_cptr_t const match = func_(cursor, remaining, &matched_length);
            if (!match) break;
            ++count_matches;
            std::size_t const advance = (std::size_t)(match - cursor) + (matched_length ? matched_length : 1);
            cursor += advance;
            remaining -= advance;
        }
        do_not_optimize(count_matches);
        return {token.size(), static_cast<check_value_t>(count_matches)};
    }
};

void bench_find_delimiters_utf8(environment_t const &env) {

    auto validator = find_delimiters_utf8_from_sz<sz_find_delimiters_utf8_serial> {env};
    bench_result_t base = bench_unary(env, "sz_find_delimiters_utf8_serial", validator).log();

#if SZ_USE_ICELAKE
    bench_unary(env, "sz_find_delimiters_utf8_icelake", validator,
                find_delimiters_utf8_from_sz<sz_find_delimiters_utf8_icelake> {env})
        .log(base);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_find_delimiters_utf8_haswell", validator,
                find_delimiters_utf8_from_sz<sz_find_delimiters_utf8_haswell> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_find_delimiters_utf8_neon", validator,
                find_delimiters_utf8_from_sz<sz_find_delimiters_utf8_neon> {env})
        .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Delimiter Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual testing
        environment_t::tokenization_t::lines_k);

    std::printf("Starting Unicode benchmarks...\n");

    bench_find_delimiters_utf8(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
