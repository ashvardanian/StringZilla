/**
 *  @file bench/utf8_traverse.cpp
 *  @author Ash Vardanian
 *  @date November 19, 2025
 *  @brief Benchmarks the @b sz_utf8_* traversal/transcode family (the @c utf8_runes unit).
 *
 *  Times the traversal dispatch points over the multilingual lines. Every capability's kernels are
 *  timed and validated (via a per-call checksum) against the serial ones by the `cross_<arch>.cpp`
 *  files, through the adapters in `cross.hpp`.
 *
 *  Compute-bound: codepoint iteration is branch-heavy, so a 64 MB slice exercises every path.
 *
 *  Benchmarks include:
 *  - Codepoint counting - @b utf8_count.
 *  - Validating and counting every line of a batch at once - @b utf8_count_strings.
 *  - Nth-codepoint location - @b utf8_seek (the BMI/PDEP "Nth set bit" kernel on x86).
 *  - UTF-8 → UTF-32 transcoding - @b utf8_decode.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER='utf8_(count|seek|decode)' \
 *      build_release/stringzilla_bench
 *  @endcode
 *
 *  This file is the sibling of `utf8_scan.cpp`, `utf8_segment.cpp`, and `utf8_uncased.cpp`.
 */
#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

void bench_utf8_count(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_count_best", utf8_count_from_sz<cpu_best<sz_utf8_count_best>> {corpus}));
}

/** The same counts one line at a time through @c sz_utf8_count_best, which does not validate. */
struct utf8_count_strings_by_loop_t {
    corpus_t const &lines;
    std::size_t batch;
    std::vector<sz_size_t> runes {};

    call_result_t operator()(std::size_t token_index) {
        std::size_t const count = std::min(batch, lines.tokens.size() - token_index);
        std::size_t bytes = 0;
        runes.resize(count);
        for (std::size_t position = 0; position != count; ++position) {
            token_view_t const line = lines.tokens[token_index + position];
            cpu_best<sz_utf8_count_best>(line.data(), line.size(), &runes[position], nullptr);
            bytes += line.size();
        }
        return {bytes, column_check_value(runes)};
    }
};

/** Times the column count over batches of lines against the loop over its lines. */
void bench_utf8_count_strings(environment_t &env, corpus_t const &corpus) {
    std::size_t const batch = candidates_per_call(env, corpus);
    strings_input_t lines(corpus);
    using count_strings_best_t = utf8_count_strings_from_sz<cpu_best<sz_utf8_count_strings_best>>;
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_utf8_count_strings_best",
                                                  count_strings_best_t {lines, batch});
    print(base);
    print(bench_unary(env, corpus, "utf8_count_strings<loop>", count_strings_best_t {lines, batch},
                      utf8_count_strings_by_loop_t {corpus, batch}),
          baseline_of(base));
}

void bench_utf8_seek(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_seek_best", utf8_seek_from_sz<cpu_best<sz_utf8_seek_best>> {corpus}));
}

void bench_utf8_decode(environment_t const &env, corpus_t const &corpus) {
    print(bench_unary(env, corpus, "sz_utf8_decode_best", utf8_unpack_from_sz<cpu_best<sz_utf8_decode_best>> {corpus}));
}

void bench_utf8_traverse(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    fmt::println("Starting UTF-8 traversal benchmarks...");
    bench_utf8_count(env, corpus);
    bench_utf8_count_strings(env, corpus);
    bench_utf8_seek(env, corpus);
    bench_utf8_decode(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
