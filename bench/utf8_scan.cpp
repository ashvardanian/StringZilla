/**
 *  @file bench/utf8_scan.cpp
 *  @author Ash Vardanian
 *  @date June 8, 2026
 *  @brief Benchmarks the UTF-8 class-scan family (the @c utf8_tokens unit) against the serial
 *      baselines: the codepoint-class enumerators that emit every match of a character class.
 *
 *  Every kernel is benchmarked across all available SIMD backends side-by-side, and each backend's
 *  result is validated (via a per-call checksum) against the serial reference — so this file
 *  doubles as a differential harness.
 *
 *  Compute-bound: per-codepoint class scanning branches heavily, so a 64 MiB slice hits each path.
 *
 *  Benchmarks include:
 *  - Newline enumeration - @b utf8_newlines.
 *  - Whitespace enumeration - @b utf8_whitespaces (Unicode White_Space property).
 *  - Delimiter enumeration - @b utf8_delimiters (punctuation/symbol/separator/whitespace).
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or [1:200] for
 *    N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_scan_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      build_release/stringzilla_bench_utf8_scan_cpp20
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_segment.cpp`, and `utf8_uncased.cpp`.
 */
#include <vector>

#include <fmt/format.h>

#include "harness.hpp"

#include "stringzilla/utf8_tokens.h" // `sz_utf8_newlines`, `sz_utf8_whitespaces`, `sz_utf8_delimiters`

using namespace ashvardanian::stringzilla::bench;

#pragma region Wrappers

/** Enumerates every match of a codepoint class (newline, whitespace, delimiter) across each token
 *  via the multistep "find boundaries" API, resuming through the whole token in batches of
 *  @c sz_iterators_default_steps_k; the checksum is the total number of matches. */
template <sz_utf8_segmenter_t find_func_>
struct utf8_enumerate_delimiters {
    environment_t const &env;
    utf8_enumerate_delimiters(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t text = token.data();
        sz_size_t len = token.size();
        sz_size_t offsets[sz_iterators_default_steps_k], lengths[sz_iterators_default_steps_k];
        sz_size_t pos = 0, total = 0;
        while (pos < len) {
            sz_size_t consumed = 0;
            total += find_func_(text + pos, len - pos, offsets, lengths, sz_iterators_default_steps_k, &consumed);
            if (consumed == 0) break;
            pos += consumed;
        }
        do_not_optimize(total);
        return {token.size(), static_cast<check_value_t>(total)};
    }
};

#pragma endregion

#pragma region Benchmarks

void bench_utf8_newlines(environment_t const &env) {
    auto base_v = utf8_enumerate_delimiters<sz_utf8_newlines_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_newlines_serial", base_v).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_utf8_newlines_haswell", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_haswell> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_utf8_newlines_icelake", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_icelake> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_utf8_newlines_neon", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_neon> {env}).log(base);
#endif
#if STRINGZILLA_TARGET_SVE2
    bench_unary(env, "sz_utf8_newlines_sve2", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_sve2> {env}).log(base);
#endif
#if STRINGZILLA_TARGET_V128
    bench_unary(env, "sz_utf8_newlines_v128", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_v128> {env}).log(base);
#endif
#if STRINGZILLA_TARGET_RVV
    bench_unary(env, "sz_utf8_newlines_rvv", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_rvv> {env}).log(base);
#endif
#if STRINGZILLA_TARGET_POWERVSX
    bench_unary(env, "sz_utf8_newlines_powervsx", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_powervsx> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_LASX
    bench_unary(env, "sz_utf8_newlines_lasx", base_v, utf8_enumerate_delimiters<sz_utf8_newlines_lasx> {env}).log(base);
#endif
}

void bench_utf8_whitespaces(environment_t const &env) {
    auto base_v = utf8_enumerate_delimiters<sz_utf8_whitespaces_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_whitespaces_serial", base_v).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_utf8_whitespaces_haswell", base_v,
                utf8_enumerate_delimiters<sz_utf8_whitespaces_haswell> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_utf8_whitespaces_icelake", base_v,
                utf8_enumerate_delimiters<sz_utf8_whitespaces_icelake> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_utf8_whitespaces_neon", base_v, utf8_enumerate_delimiters<sz_utf8_whitespaces_neon> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_SVE2
    bench_unary(env, "sz_utf8_whitespaces_sve2", base_v, utf8_enumerate_delimiters<sz_utf8_whitespaces_sve2> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_V128
    bench_unary(env, "sz_utf8_whitespaces_v128", base_v, utf8_enumerate_delimiters<sz_utf8_whitespaces_v128> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_RVV
    bench_unary(env, "sz_utf8_whitespaces_rvv", base_v, utf8_enumerate_delimiters<sz_utf8_whitespaces_rvv> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_POWERVSX
    bench_unary(env, "sz_utf8_whitespaces_powervsx", base_v,
                utf8_enumerate_delimiters<sz_utf8_whitespaces_powervsx> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_LASX
    bench_unary(env, "sz_utf8_whitespaces_lasx", base_v, utf8_enumerate_delimiters<sz_utf8_whitespaces_lasx> {env})
        .log(base);
#endif
}

void bench_utf8_delimiters(environment_t const &env) {
    auto base_v = utf8_enumerate_delimiters<sz_utf8_delimiters_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_delimiters_serial", base_v).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_utf8_delimiters_haswell", base_v, utf8_enumerate_delimiters<sz_utf8_delimiters_haswell> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_utf8_delimiters_icelake", base_v, utf8_enumerate_delimiters<sz_utf8_delimiters_icelake> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_utf8_delimiters_neon", base_v, utf8_enumerate_delimiters<sz_utf8_delimiters_neon> {env})
        .log(base);
#endif
#if STRINGZILLA_TARGET_SVE2
    bench_unary(env, "sz_utf8_delimiters_sve2", base_v, utf8_enumerate_delimiters<sz_utf8_delimiters_sve2> {env})
        .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    log_environment();
    print_bench_environment();

    fmt::println("Building up the environment...");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual coverage
        environment_t::tokenization_t::lines_k, compute_bound_slice_bytes_k);

    fmt::println("Starting UTF-8 class-scan benchmarks...");

    bench_utf8_newlines(env);
    bench_utf8_whitespaces(env);
    bench_utf8_delimiters(env);

    fmt::println("All benchmarks passed.");
    return 0;
}
