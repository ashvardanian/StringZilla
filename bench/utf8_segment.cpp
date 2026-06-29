/**
 *  @file bench/utf8_segment.cpp
 *  @brief Benchmarks the UTF-8 boundary-segmentation family against the serial baselines: the UAX-29 / UAX-14
 *         boundary engines. Every kernel is benchmarked across all available SIMD backends side-by-side, and
 *         each backend's result is validated (via a per-call checksum) against the serial reference — so this
 *         file doubles as a differential correctness harness.
 *
 *  Benchmarks include:
 *  - UAX-29 word-boundary segmentation - @b utf8_words.
 *  - UAX-29 grapheme-cluster segmentation - @b utf8_graphemes.
 *  - UAX-29 sentence-boundary segmentation - @b utf8_sentences.
 *  - UAX-14 line-break segmentation - @b utf8_linebreaks.
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
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_segment_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      build_release/stringzilla_bench_utf8_segment_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_traverse.cpp`, `bench_utf8_scan.cpp`, and `bench_utf8_uncased.cpp`.
 */
#include <vector>

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

#include "stringzilla/utf8_wordbreaks.h" // `sz_utf8_wordbreaks`
#include "stringzilla/utf8_graphemes.h"  // `sz_utf8_graphemes`
#include "stringzilla/utf8_sentences.h"  // `sz_utf8_sentences`
#include "stringzilla/utf8_linebreaks.h" // `sz_utf8_linebreaks`

using namespace ashvardanian::stringzilla::scripts;

#pragma region Wrappers

/** @brief  Segments each token into UAX-29 words/graphemes/sentences or UAX-14 line breaks forward;
 *          checksum = number of segments. */
template <auto func_>
struct utf8_word_forward_from_sz {
    environment_t const &env;
    utf8_word_forward_from_sz(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        sz_size_t starts[16], lengths[16];
        std::size_t words = 0;
        while (remaining) {
            sz_size_t consumed = 0;
            sz_size_t produced = func_(cursor, remaining, starts, lengths, 16, &consumed);
            words += static_cast<std::size_t>(produced);
            if (produced == 0 || consumed >= remaining) break; // Whole suffix segmented.
            cursor += consumed;                                // Resume from the first word that did not fit.
            remaining -= consumed;
        }
        do_not_optimize(words);
        return {token.size(), static_cast<check_value_t>(words)};
    }
};

#pragma endregion

#pragma region Benchmarks

void bench_utf8_words(environment_t const &env) {
    auto base_v = utf8_word_forward_from_sz<sz_utf8_wordbreaks_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_wordbreaks_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_wordbreaks_haswell", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_haswell> {env})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_wordbreaks_icelake", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_icelake> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_wordbreaks_neon", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_neon> {env})
        .log(base);
#endif
#if SZ_USE_SVE2
    bench_unary(env, "sz_utf8_wordbreaks_sve2", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_sve2> {env})
        .log(base);
#endif
#if SZ_USE_V128
    bench_unary(env, "sz_utf8_wordbreaks_v128", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_v128> {env})
        .log(base);
#endif
#if SZ_USE_RVV
    bench_unary(env, "sz_utf8_wordbreaks_rvv", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_rvv> {env})
        .log(base);
#endif
#if SZ_USE_POWERVSX
    bench_unary(env, "sz_utf8_wordbreaks_powervsx", base_v,
                utf8_word_forward_from_sz<sz_utf8_wordbreaks_powervsx> {env})
        .log(base);
#endif
#if SZ_USE_LASX
    bench_unary(env, "sz_utf8_wordbreaks_lasx", base_v, utf8_word_forward_from_sz<sz_utf8_wordbreaks_lasx> {env})
        .log(base);
#endif
}

void bench_utf8_graphemes(environment_t const &env) {
    auto base_v = utf8_word_forward_from_sz<sz_utf8_graphemes_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_graphemes_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_graphemes_haswell", base_v, utf8_word_forward_from_sz<sz_utf8_graphemes_haswell> {env})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_graphemes_icelake", base_v, utf8_word_forward_from_sz<sz_utf8_graphemes_icelake> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_graphemes_neon", base_v, utf8_word_forward_from_sz<sz_utf8_graphemes_neon> {env})
        .log(base);
#endif
}

void bench_utf8_sentences(environment_t const &env) {
    auto base_v = utf8_word_forward_from_sz<sz_utf8_sentences_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_sentences_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_sentences_haswell", base_v, utf8_word_forward_from_sz<sz_utf8_sentences_haswell> {env})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_sentences_icelake", base_v, utf8_word_forward_from_sz<sz_utf8_sentences_icelake> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_sentences_neon", base_v, utf8_word_forward_from_sz<sz_utf8_sentences_neon> {env})
        .log(base);
#endif
}

void bench_utf8_linebreaks(environment_t const &env) {
    auto base_v = utf8_word_forward_from_sz<sz_utf8_linebreaks_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_linebreaks_serial", base_v).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_utf8_linebreaks_haswell", base_v, utf8_word_forward_from_sz<sz_utf8_linebreaks_haswell> {env})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_utf8_linebreaks_icelake", base_v, utf8_word_forward_from_sz<sz_utf8_linebreaks_icelake> {env})
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_utf8_linebreaks_neon", base_v, utf8_word_forward_from_sz<sz_utf8_linebreaks_neon> {env})
        .log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Segmentation Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual coverage
        environment_t::tokenization_t::lines_k);

    std::printf("Starting UTF-8 segmentation benchmarks...\n");

    bench_utf8_words(env);
    bench_utf8_graphemes(env);
    bench_utf8_sentences(env);
    bench_utf8_linebreaks(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
