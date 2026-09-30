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
 *  MiB slice exercises every path while each call samples only what it needs.
 *
 *  Three shapes are measured, byte-level and rune-level alike, every candidate at its own length:
 *  - @c sz_levenshtein_engine_init plus one round over a single pair, which is what a caller
 *    scoring one pair pays: a batch of one, prepared and released around the round;
 *  - @c sz_levenshtein_distances from a prepared batch of queries against the next
 *    @c STRINGWARS_BATCH tokens - by default as many median tokens as fill a 32 KiB L1 - at two
 *    query lengths, the slice's median and the 1024 bytes whose match masks fill that L1, on every
 *    compiled backend. The batch is prepared once per arm, so what the arm times is the sweep and
 *    not the preparation the engine exists to hoist;
 *  - the exported building blocks one at a time, so the query's preparation, the staging of
 *    candidate bytes into class ids, and the word-steps over those ids each get a number.
 *
 *  Every arm's name carries its query length, as `:q117`. Throughput is reported as Cell Updates
 *  Per Second @b (CUPS): the query's length times the candidates' lengths. The building blocks
 *  count what each of them does instead: query bytes prepared, class ids staged, word-steps taken -
 *  so the ops/s column of one arm is read beside the next rather than against a shared denominator.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or an integer
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
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines STRINGWARS_FILTER=levenshtein \
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 */
#include <stdexcept>   // `std::runtime_error`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

namespace {

/** The engine's init over the CPU's capabilities, in the shape of its init kernels. */
sz_status_t levenshtein_engine_init_cpu_(sz_levenshtein_engine_t *engine, sz_sequence_t const *queries,
                                         sz_levenshtein_symbol_t symbol, sz_size_t ordinal,
                                         sz_memory_allocator_t *allocator, void *stream) {
    return sz_levenshtein_engine_init(engine, queries, symbol, sz::default_capabilities(), ordinal, allocator, stream);
}

#pragma region One Pair

/** One pair per call through a batch of one: the preparation and round a caller pays together. */
struct levenshtein_pair_from_sz {

    /** The tokens the pair is drawn from. */
    environment_t const &env;

    /** Bytes the query is clamped to. */
    std::size_t query_bytes;

    /** Whether the distance counts bytes or runes. */
    sz_levenshtein_symbol_t symbol;

    levenshtein_pair_from_sz(environment_t const &env, std::size_t query_bytes, sz_levenshtein_symbol_t symbol)
        : env(env), query_bytes(query_bytes), symbol(symbol) {}

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        std::string_view const candidate = env.tokens[(token_index + 1) % env.tokens.size()];
        sz_string_view_t const query_view {query.data(), query.size()};
        sz_string_view_t const candidate_view {candidate.data(), candidate.size()};
        sz_sequence_t queries, candidates;
        sz_sequence_from_string_views(&query_view, 1, &queries);
        sz_sequence_from_string_views(&candidate_view, 1, &candidates);

        sz_levenshtein_engine_t engine {};
        if (sz_levenshtein_engine_init(&engine, &queries, symbol, sz::default_capabilities(), 0, nullptr, nullptr) !=
            sz_success_k)
            throw std::runtime_error("The engine could not be prepared.");
        sz_size_t distance = 0;
        sz_status_t const status = sz_levenshtein_distances(&engine, &candidates, &distance, 1, nullptr);
        sz_levenshtein_engine_free(&engine);
        if (status != sz_success_k) throw std::runtime_error("The one-pair round failed.");
        return call_result_t(query.size() + candidate.size(), distance, query.size() * candidate.size());
    }
};

/** One-pair shape on the dispatched entry alone, as one pair fills one candidate on any backend. */
void bench_levenshtein_one_pair(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    bench_unary(env, "sz_levenshtein_distances:pair" + suffix,
                levenshtein_pair_from_sz {env, query_bytes, sz_levenshtein_bytes_k})
        .log();
    bench_unary(env, "sz_levenshtein_distances:pair:utf8" + suffix,
                levenshtein_pair_from_sz {env, query_bytes, sz_levenshtein_runes_k})
        .log();
}

#pragma endregion

#pragma region Cross Product

/** Cross-product verbs at one query length, over bytes and runes, on the engine's kernel. */
void bench_levenshtein_cross_product(environment_t const &env, std::size_t query_bytes, std::size_t candidates) {
    using verbs_t = levenshtein_distances_from_sz<levenshtein_engine_init_cpu_, sz_levenshtein_distances>;
    std::string const suffix = ":q" + std::to_string(query_bytes);
    bench_result_t base = bench_unary(env, "sz_levenshtein_distances" + suffix,
                                      verbs_t {env, query_bytes, candidates, sz_levenshtein_bytes_k})
                              .log();
    // The rune rows decode every candidate byte, so their cost over the byte rows is the decoder's.
    bench_unary(env, "sz_levenshtein_distances:utf8" + suffix,
                verbs_t {env, query_bytes, candidates, sz_levenshtein_runes_k})
        .log(base);
}

#pragma endregion

} // namespace

void bench_levenshtein(corpora_t &corpora) {
    environment_t const &env = corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env);
    fmt::println("Starting Levenshtein benchmarks...");
    bench_levenshtein_one_pair(env, median_token_bytes(env));
    for (std::size_t const query_bytes : levenshtein_query_lengths(env))
        bench_levenshtein_cross_product(env, query_bytes, candidates);
}
