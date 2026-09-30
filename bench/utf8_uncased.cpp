/**
 *  @file bench/utf8_uncased.cpp
 *  @author Ash Vardanian
 *  @date November 28, 2025
 *  @brief Benchmarks the @b sz_utf8_uncased_* family — case folding and uncased search.
 *
 *  Times the case-folding dispatch points over the multilingual lines. Every capability's kernels
 *  are timed and validated against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Compute-bound: case-folded search is table- and branch-heavy per codepoint, so a 64 MiB slice
 *  exercises every path.
 *
 *  Benchmarks include:
 *  - Case folding for Unicode text - @b utf8_uncased_fold.
 *  - Uncased substring search for Unicode text - @b utf8_uncased_search.
 *  - Uncased ordering of Unicode text - @b utf8_uncased_order.
 *
 *  Its siblings @b utf8_traverse.cpp, @b utf8_scan.cpp, and @b utf8_segment.cpp cover the
 *  @b sz_utf8_* iteration and segmentation family: codepoint counting, Nth-codepoint, newline and
 *  whitespace scanning, UAX-29 word, grapheme and sentence boundaries, UAX-14 line breaks, and
 *  transcoding between UTFs.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *  - `STRINGWARS_UNIQUE=1` : Deduplicates tokens, sorting the set and dropping duplicates first.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words STRINGWARS_UNIQUE=1 STRINGWARS_FILTER=uncased \
 *      build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_scan.cpp`, `utf8_segment.cpp`,
 *  `token.cpp`, `find.cpp`, `sequence.cpp`, and `memory.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

void bench_utf8_uncased_fold(environment_t const &env) {
    bench_unary(env, "sz_utf8_uncased_fold_best", utf8_uncased_fold_from_sz<cpu_best<sz_utf8_uncased_fold_best>> {env})
        .log();
}

void bench_utf8_uncased_search(environment_t const &env) {
    bench_unary(env, "sz_utf8_uncased_search_best",
                utf8_uncased_search_from_sz<cpu_best<sz_utf8_uncased_search_best>> {env})
        .log();
}

void bench_utf8_uncased_order(environment_t const &env) {
    bench_unary(env, "sz_utf8_uncased_order_best",
                utf8_uncased_order_from_sz<cpu_best<sz_utf8_uncased_order_best>> {env})
        .log();
}

} // namespace

void bench_utf8_uncased(corpora_t &corpora) {
    environment_t const &env = corpora.multilingual_slice();
    fmt::println("Starting UTF-8 case-folding benchmarks...");
    bench_utf8_uncased_fold(env);
    bench_utf8_uncased_search(env);
    bench_utf8_uncased_order(env);
}
