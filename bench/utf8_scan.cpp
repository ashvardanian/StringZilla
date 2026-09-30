/**
 *  @file bench/utf8_scan.cpp
 *  @author Ash Vardanian
 *  @date June 8, 2026
 *  @brief Benchmarks the UTF-8 class-scan family (the @c utf8_tokens unit): the codepoint-class
 *      enumerators that emit every match of a character class.
 *
 *  Times the class-scan dispatch points over the multilingual lines. Every capability's kernels are
 *  timed and validated (via a per-call checksum) against the serial ones by the `cross_<arch>.cpp`
 *  files, through the adapters in `cross.hpp`.
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
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER='utf8_(newlines|whitespaces|delimiters)' \
 *      build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_segment.cpp`, and `utf8_uncased.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

void bench_utf8_newlines(environment_t const &env) {
    bench_unary(env, "sz_utf8_newlines_best", utf8_enumerate_delimiters<cpu_best<sz_utf8_newlines_best>> {env}).log();
}

void bench_utf8_whitespaces(environment_t const &env) {
    bench_unary(env, "sz_utf8_whitespaces_best", utf8_enumerate_delimiters<cpu_best<sz_utf8_whitespaces_best>> {env})
        .log();
}

void bench_utf8_delimiters(environment_t const &env) {
    bench_unary(env, "sz_utf8_delimiters_best", utf8_enumerate_delimiters<cpu_best<sz_utf8_delimiters_best>> {env})
        .log();
}

} // namespace

void bench_utf8_scan(corpora_t &corpora) {
    environment_t const &env = corpora.multilingual_slice();
    fmt::println("Starting UTF-8 class-scan benchmarks...");
    bench_utf8_newlines(env);
    bench_utf8_whitespaces(env);
    bench_utf8_delimiters(env);
}
