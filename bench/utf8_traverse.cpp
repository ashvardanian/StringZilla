/**
 *  @file bench/utf8_traverse.cpp
 *  @brief Benchmarks the @b `sz_utf8_*` traversal/transcode family (the `utf8_runes` unit) against the serial
 *         baselines. Every kernel is benchmarked across all available SIMD backends side-by-side, and each
 *         backend's result is validated (via a per-call checksum) against the serial reference — so this file
 *         doubles as a differential correctness harness.
 *
 *  Benchmarks include:
 *  - Codepoint counting - @b utf8_count.
 *  - Nth-codepoint location - @b utf8_seek (the BMI/PDEP "Nth set bit" kernel on x86).
 *  - UTF-8 -> UTF-32 transcoding - @b utf8_decode.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_traverse_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      build_release/stringzilla_bench_utf8_traverse_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_scan.cpp`, `bench_utf8_segment.cpp`, and `bench_utf8_uncased.cpp`.
 */
#include <vector>

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

#include "stringzilla/utf8_runes.h" // `sz_utf8_count`, `sz_utf8_seek`, `sz_utf8_decode`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Wrappers

/** @brief  Counts the codepoints of each token; checksum = codepoint count. */
template <auto func_>
struct utf8_count_from_sz {
    environment_t const &env;
    utf8_count_from_sz(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_size_t count = func_(token.data(), token.size());
        do_not_optimize(count);
        return {token.size(), static_cast<check_value_t>(count)};
    }
};

/** @brief  Locates the middle codepoint of each token; checksum = byte offset of the located codepoint. */
template <auto func_>
struct utf8_seek_from_sz {
    environment_t const &env;
    std::vector<sz_size_t> targets; // Nth codepoint to locate per token, precomputed outside the timed call.
    utf8_seek_from_sz(environment_t const &env_) : env(env_) {
        targets.reserve(env.tokens.size());
        for (auto const &token : env.tokens) targets.push_back(sz_utf8_count_serial(token.data(), token.size()) / 2);
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t located = func_(token.data(), token.size(), targets[i]);
        do_not_optimize(located);
        check_value_t offset = located ? static_cast<check_value_t>(located - token.data()) : (check_value_t)-1;
        // Throughput should reflect the bytes actually scanned to reach the Nth codepoint, not the whole token.
        std::size_t scanned = located ? static_cast<std::size_t>(located - token.data()) : token.size();
        return {scanned, offset};
    }
};

/** @brief  Transcodes each token UTF-8 -> UTF-32 chunk by chunk; checksum = number of runes produced. */
template <auto func_>
struct utf8_unpack_from_sz {
    environment_t const &env;
    mutable std::vector<sz_rune_t> runes; // Reusable output buffer, sized to the largest token.
    utf8_unpack_from_sz(environment_t const &env_) : env(env_) {
        std::size_t max_token = 1;
        for (auto const &token : env.tokens) max_token = std::max(max_token, token.size());
        runes.resize(max_token + 1);
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        std::size_t produced = 0;
        while (remaining) {
            sz_size_t unpacked = 0;
            sz_cptr_t consumed = func_(cursor, remaining, runes.data(), runes.size(), &unpacked);
            std::size_t step = static_cast<std::size_t>(consumed - cursor);
            if (step == 0) break; // Incomplete trailing sequence; nothing more to do.
            produced += unpacked;
            cursor += step;
            remaining -= step;
        }
        do_not_optimize(runes.data());
        return {token.size(), static_cast<check_value_t>(produced)};
    }
};

#pragma endregion

#pragma region Benchmarks

void bench_utf8_count(environment_t const &env) {
    auto base_v = utf8_count_from_sz<sz_utf8_count_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_count_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_count_haswell", base_v, utf8_count_from_sz<sz_utf8_count_haswell> {env}).log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_count_icelake", base_v, utf8_count_from_sz<sz_utf8_count_icelake> {env}).log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_count_neon", base_v, utf8_count_from_sz<sz_utf8_count_neon> {env}).log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_count_sve2", base_v, utf8_count_from_sz<sz_utf8_count_sve2> {env}).log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_count_v128", base_v, utf8_count_from_sz<sz_utf8_count_v128> {env}).log(base);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_utf8_count_v128relaxed", base_v, utf8_count_from_sz<sz_utf8_count_v128relaxed> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_count_rvv", base_v, utf8_count_from_sz<sz_utf8_count_rvv> {env}).log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_count_powervsx", base_v, utf8_count_from_sz<sz_utf8_count_powervsx> {env}).log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_count_lasx", base_v, utf8_count_from_sz<sz_utf8_count_lasx> {env}).log(base);
#endif
}

void bench_utf8_seek(environment_t const &env) {
    auto base_v = utf8_seek_from_sz<sz_utf8_seek_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_seek_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_seek_haswell", base_v, utf8_seek_from_sz<sz_utf8_seek_haswell> {env}).log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_seek_icelake", base_v, utf8_seek_from_sz<sz_utf8_seek_icelake> {env}).log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_seek_neon", base_v, utf8_seek_from_sz<sz_utf8_seek_neon> {env}).log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_seek_sve2", base_v, utf8_seek_from_sz<sz_utf8_seek_sve2> {env}).log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_seek_v128", base_v, utf8_seek_from_sz<sz_utf8_seek_v128> {env}).log(base);
#endif
#if SZ_USE_V128RELAXED
    bench_unary(env, "sz_utf8_seek_v128relaxed", base_v, utf8_seek_from_sz<sz_utf8_seek_v128relaxed> {env}).log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_seek_rvv", base_v, utf8_seek_from_sz<sz_utf8_seek_rvv> {env}).log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_seek_powervsx", base_v, utf8_seek_from_sz<sz_utf8_seek_powervsx> {env}).log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_seek_lasx", base_v, utf8_seek_from_sz<sz_utf8_seek_lasx> {env}).log(base);
#endif
}

void bench_utf8_decode(environment_t const &env) {
    auto base_v = utf8_unpack_from_sz<sz_utf8_decode_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_decode_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_decode_haswell", base_v, utf8_unpack_from_sz<sz_utf8_decode_haswell> {env}).log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_decode_icelake", base_v, utf8_unpack_from_sz<sz_utf8_decode_icelake> {env}).log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_decode_neon", base_v, utf8_unpack_from_sz<sz_utf8_decode_neon> {env}).log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_decode_sve2", base_v, utf8_unpack_from_sz<sz_utf8_decode_sve2> {env}).log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_decode_v128", base_v, utf8_unpack_from_sz<sz_utf8_decode_v128> {env}).log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_decode_rvv", base_v, utf8_unpack_from_sz<sz_utf8_decode_rvv> {env}).log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_decode_powervsx", base_v, utf8_unpack_from_sz<sz_utf8_decode_powervsx> {env}).log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_decode_lasx", base_v, utf8_unpack_from_sz<sz_utf8_decode_lasx> {env}).log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Traversal Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual coverage
        environment_t::tokenization_t::lines_k);

    std::printf("Starting UTF-8 traversal benchmarks...\n");

    bench_utf8_count(env);
    bench_utf8_seek(env);
    bench_utf8_decode(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
