/**
 *  @file bench/utf8_uncased.cpp
 *  @author Ash Vardanian
 *  @date November 28, 2025
 *  @brief Benchmarks the @b sz_utf8_uncased_* family — case folding and uncased search.
 *
 *  Times the case-folding dispatch points over the multilingual lines. Every capability's kernels
 *  are timed and validated against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Compute-bound: case-folded search is table- and branch-heavy per codepoint, so a 64 MB slice
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
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words STRINGWARS_UNIQUE=1 STRINGWARS_FILTER=uncased \
 *      build_release/stringzilla_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_scan.cpp`, `utf8_segment.cpp`,
 *  `token.cpp`, `find.cpp`, `sequence.cpp`, and `memory.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_utf8_uncased_fold(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_uncased_fold_best",
                      utf8_uncased_fold_from_sz<cpu_best<sz_utf8_uncased_fold_best>> {corpus}));
}

void bench_utf8_uncased_search(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_uncased_search_best",
                      utf8_uncased_search_from_sz<cpu_best<sz_utf8_uncased_search_best>> {corpus}));
}

void bench_utf8_uncased_order(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_uncased_order_best",
                      utf8_uncased_order_from_sz<cpu_best<sz_utf8_uncased_order_best>> {corpus}));
}

void bench_utf8_uncased(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    fmt::println("Starting UTF-8 case-folding benchmarks...");
    bench_utf8_uncased_fold(env, corpus);
    bench_utf8_uncased_search(env, corpus);
    bench_utf8_uncased_order(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
