/**
 *  @file bench/utf8_segment.cpp
 *  @author Ash Vardanian
 *  @date June 8, 2026
 *  @brief Benchmarks the UTF-8 boundary-segmentation family: the UAX-29 / UAX-14 boundary engines.
 *
 *  Times the segmentation dispatch points over the multilingual lines. Every capability's kernels
 *  are timed and validated, through a per-call checksum, against the serial ones by the
 *  `cross_<arch>.cpp` files.
 *
 *  Compute-bound: UTF-8 segmentation branches per codepoint, so a 64 MiB slice covers all paths.
 *
 *  Benchmarks include:
 *  - UAX-29 word-boundary segmentation - @b utf8_wordbreaks.
 *  - UAX-29 grapheme-cluster segmentation - @b utf8_graphemes.
 *  - UAX-29 sentence-boundary segmentation - @b utf8_sentences.
 *  - UAX-14 line-break segmentation - @b utf8_linebreaks.
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
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      STRINGWARS_FILTER='utf8_(wordbreaks|graphemes|sentences|linebreaks)' build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_scan.cpp`, and `utf8_uncased.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

void bench_utf8_wordbreaks(environment_t const &env) {
    bench_unary(env, "sz_utf8_wordbreaks_best", utf8_word_forward_from_sz<cpu_best<sz_utf8_wordbreaks_best>> {env})
        .log();
}

void bench_utf8_graphemes(environment_t const &env) {
    bench_unary(env, "sz_utf8_graphemes_best", utf8_word_forward_from_sz<cpu_best<sz_utf8_graphemes_best>> {env}).log();
}

void bench_utf8_sentences(environment_t const &env) {
    bench_unary(env, "sz_utf8_sentences_best", utf8_word_forward_from_sz<cpu_best<sz_utf8_sentences_best>> {env}).log();
}

void bench_utf8_linebreaks(environment_t const &env) {
    bench_unary(env, "sz_utf8_linebreaks_best", utf8_word_forward_from_sz<cpu_best<sz_utf8_linebreaks_best>> {env})
        .log();
}

} // namespace

void bench_utf8_segment(corpora_t &corpora) {
    environment_t const &env = corpora.multilingual_slice();
    fmt::println("Starting UTF-8 segmentation benchmarks...");
    bench_utf8_wordbreaks(env);
    bench_utf8_graphemes(env);
    bench_utf8_sentences(env);
    bench_utf8_linebreaks(env);
}
