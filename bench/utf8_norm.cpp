/**
 *  @file bench/utf8_norm.cpp
 *  @author Ash Vardanian
 *  @date June 26, 2026
 *  @brief Benchmarks the @b sz_utf8_norm_* family: Unicode normalization and quick-check scans.
 *
 *  Times the normalization dispatch points over the multilingual lines. Every capability's kernels
 *  are timed and validated against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Compute-bound: Unicode normalization is table- and branch-heavy per codepoint, so a 64 MB slice
 *  exercises every path on the multilingual corpus.
 *
 *  Benchmarks include:
 *  - Unicode normalization for UTF-8 text - @b utf8_norm.
 *  - Normalization-form violation scanning (quick-check) - @b utf8_find_denormalized.
 *
 *  Both sections normalize to @b NFC, the most common interchange form.
 *
 *  Its sibling @b utf8_uncased.cpp covers the @b sz_utf8_uncased_* family (case folding and uncased
 *  substring search), and @b utf8_traverse.cpp, @b utf8_scan.cpp, and @b utf8_segment.cpp cover the
 *  @b sz_utf8_* iteration and segmentation family: codepoint counting, Nth-codepoint, newline and
 *  whitespace scanning, UAX-29 word, grapheme and sentence boundaries, UAX-14 line breaks, and
 *  transcoding between UTFs.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_UNIQUE=1 \
 *      STRINGWARS_FILTER='utf8_(norm|find_denormalized)' build_release/stringzilla_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_uncased.cpp`, `utf8_traverse.cpp`, `utf8_scan.cpp`,
 *  `utf8_segment.cpp`, `token.cpp`, `find.cpp`, `sequence.cpp`, and `memory.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_utf8_normalize(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_norm_best", utf8_norm_from_sz<cpu_best<sz_utf8_norm_best>> {corpus}));
}

void bench_utf8_find_denormalized(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_find_denormalized_best",
                      utf8_find_denormalized_from_sz<cpu_best<sz_utf8_find_denormalized_best>> {corpus}));
}

void bench_utf8_norm(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    fmt::println("Starting UTF-8 normalization benchmarks...");
    bench_utf8_normalize(env, corpus);
    bench_utf8_find_denormalized(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
