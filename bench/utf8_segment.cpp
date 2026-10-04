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
 *  Compute-bound: UTF-8 segmentation branches per codepoint, so a 64 MB slice covers all paths.
 *
 *  Benchmarks include:
 *  - UAX-29 word-boundary segmentation - @b utf8_wordbreaks.
 *  - UAX-29 grapheme-cluster segmentation - @b utf8_graphemes.
 *  - UAX-29 sentence-boundary segmentation - @b utf8_sentences.
 *  - UAX-14 line-break segmentation - @b utf8_linebreaks.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      STRINGWARS_FILTER='utf8_(wordbreaks|graphemes|sentences|linebreaks)' build_release/stringzilla_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_traverse.cpp`, `utf8_scan.cpp`, and `utf8_uncased.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_utf8_wordbreaks(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_wordbreaks_best",
                      utf8_word_forward_from_sz<cpu_best<sz_utf8_wordbreaks_best>> {corpus}));
}

void bench_utf8_graphemes(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_graphemes_best",
                      utf8_word_forward_from_sz<cpu_best<sz_utf8_graphemes_best>> {corpus}));
}

void bench_utf8_sentences(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_sentences_best",
                      utf8_word_forward_from_sz<cpu_best<sz_utf8_sentences_best>> {corpus}));
}

void bench_utf8_linebreaks(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_linebreaks_best",
                      utf8_word_forward_from_sz<cpu_best<sz_utf8_linebreaks_best>> {corpus}));
}

void bench_utf8_segment(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    fmt::println("Starting UTF-8 segmentation benchmarks...");
    bench_utf8_wordbreaks(env, corpus);
    bench_utf8_graphemes(env, corpus);
    bench_utf8_sentences(env, corpus);
    bench_utf8_linebreaks(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
