/**
 *  @file bench/levenshtein.cpp
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Benchmarks for Levenshtein edit distances under unit costs.
 *
 *  Times the engine's verbs over the multilingual lines, which pick their kernel from the CPU's
 *  capabilities. Every capability's kernels, and the word-steps only the tier headers define, are
 *  timed against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Compute-bound: Myers' algorithm costs one word-step per query word per candidate byte, so a 64
 *  MB slice exercises every path while each call samples only what it needs.
 *
 *  Three shapes are measured, byte-level and rune-level alike, every candidate at its own length:
 *  - @c sz_levenshtein_engine_init plus one round over a single pair, which is what a caller
 *    scoring one pair pays: a batch of one, prepared and released around the round;
 *  - @c sz_levenshtein_distances from a prepared batch of queries against the next
 *    @c STRINGWARS_BATCH_PER_CORE tokens - by default as many median tokens as fill a 32 KB L1 - at
 *    two query lengths, the slice's median and the 1024 bytes whose match masks fill that L1, on
 *    every compiled backend. The batch is prepared once per arm, so what the arm times is the sweep
 *    and not the preparation the engine exists to hoist;
 *  - the exported building blocks one at a time, so the query's preparation, the staging of
 *    candidate bytes into class ids, and the word-steps over those ids each get a number.
 *
 *  Every arm's name carries its query length, as `:q117`. Throughput is reported as Cell Updates
 *  Per Second @b (CUPS): the query's length times the candidates' lengths. The building blocks
 *  count what each of them does instead: query bytes prepared, class ids staged, word-steps taken -
 *  so the ops/s column of one arm is read beside the next rather than against a shared denominator.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=levenshtein \
 *  build_release/stringzilla_bench
 *  @endcode
 */
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

/** The engine's init over the CPU's capabilities, in the shape of its init kernels. */
sz_status_t levenshtein_engine_init_cpu_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                         sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                         sz_stream_t stream) {
    return sz_levenshtein_engine_init(engine, queries, symbol, sz::default_capabilities(), allocator, stream);
}

#pragma region One Pair

/** One pair per call through a batch of one: the preparation and round a caller pays together. */
struct levenshtein_pair_from_sz {

    /** The tokens the pair is drawn from. */
    corpus_t const &corpus;

    /** Bytes the query is clamped to. */
    std::size_t query_bytes;

    /** Whether the distance counts bytes or runes. */
    sz_levenshtein_symbol_t symbol;

    levenshtein_pair_from_sz(corpus_t const &corpus, std::size_t query_bytes, sz_levenshtein_symbol_t symbol)
        : corpus(corpus), query_bytes(query_bytes), symbol(symbol) {}

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(corpus.tokens[token_index]).substr(0, query_bytes);
        std::string_view const candidate = corpus.tokens[(token_index + 1) % corpus.tokens.size()];
        sz_string_view_t const query_view {query.data(), query.size()};
        sz_string_view_t const candidate_view {candidate.data(), candidate.size()};
        sz_sequence_t queries, candidates;
        sz_sequence_from_string_views(&query_view, 1, &queries);
        sz_sequence_from_string_views(&candidate_view, 1, &candidates);

        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init(&engine, &queries, symbol, sz::default_capabilities(), nullptr, nullptr) !=
            sz_success_k)
            throw std::runtime_error("The engine could not be prepared.");
        sz_size_t distance = 0;
        sz_status_t const status = sz_levenshtein_distances(&engine, &candidates, &distance, 1, nullptr);
        sz_levenshtein_engine_free(&engine, nullptr);
        if (status != sz_success_k) throw std::runtime_error("The one-pair round failed.");
        return call_result_t(query.size() + candidate.size(), distance, query.size() * candidate.size());
    }
};

/** One-pair shape on the dispatched entry alone, as one pair fills one candidate on any backend. */
void bench_levenshtein_one_pair(environment_t const &env, corpus_t const &corpus, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    print(bench_unary(env, corpus, "sz_levenshtein_distances:pair" + suffix,
                      levenshtein_pair_from_sz {corpus, query_bytes, sz_levenshtein_bytes_k}));
    print(bench_unary(env, corpus, "sz_levenshtein_distances:pair:utf8" + suffix,
                      levenshtein_pair_from_sz {corpus, query_bytes, sz_levenshtein_runes_k}));
}

#pragma endregion

#pragma region Cross Product

/** Cross-product verbs at one query length, over bytes and runes, on the engine's kernel. */
void bench_levenshtein_cross_product(environment_t const &env, corpus_t const &corpus, std::size_t query_bytes,
                                     std::size_t candidates) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    std::optional<row_t> const base = bench_unary(
        env, corpus, "sz_levenshtein_distances" + suffix,
        levenshtein_distances_from_sz {levenshtein_engine_init_cpu_, sz_levenshtein_distances, corpus, query_bytes,
                                       candidates, sz_levenshtein_bytes_k});
    print(base);
    // The rune rows decode every candidate byte, so their cost over the byte rows is the decoder's.
    print(bench_unary(env, corpus, "sz_levenshtein_distances:utf8" + suffix,
                      levenshtein_distances_from_sz {levenshtein_engine_init_cpu_, sz_levenshtein_distances, corpus,
                                                     query_bytes, candidates, sz_levenshtein_runes_k}),
          baseline_of(base));
}

#pragma endregion

void bench_levenshtein(environment_t &env) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env, corpus);
    fmt::println("Starting Levenshtein benchmarks...");
    bench_levenshtein_one_pair(env, corpus, median_token_bytes(corpus));
    for (std::size_t const query_bytes : levenshtein_query_lengths(corpus))
        bench_levenshtein_cross_product(env, corpus, query_bytes, candidates);
}

} // namespace ashvardanian::stringzilla::bench
