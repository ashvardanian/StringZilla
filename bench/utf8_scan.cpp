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
 *  Compute-bound: per-codepoint class scanning branches heavily, so a 64 MB slice hits each path.
 *
 *  Benchmarks include:
 *  - Newline enumeration - @b utf8_newlines.
 *  - Whitespace enumeration - @b utf8_whitespaces (Unicode White_Space property).
 *  - Delimiter enumeration - @b utf8_delimiters (punctuation/symbol/separator/whitespace).
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

namespace ashvardanian::stringzilla::bench {

void bench_utf8_newlines(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_newlines_best",
                      utf8_enumerate_delimiters<cpu_best<sz_utf8_newlines_best>> {corpus}));
}

void bench_utf8_whitespaces(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_whitespaces_best",
                      utf8_enumerate_delimiters<cpu_best<sz_utf8_whitespaces_best>> {corpus}));
}

void bench_utf8_delimiters(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_delimiters_best",
                      utf8_enumerate_delimiters<cpu_best<sz_utf8_delimiters_best>> {corpus}));
}

void bench_utf8_scan(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    fmt::println("Starting UTF-8 class-scan benchmarks...");
    bench_utf8_newlines(env, corpus);
    bench_utf8_whitespaces(env, corpus);
    bench_utf8_delimiters(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
