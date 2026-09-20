/**
 *  @file bench/levenshtein.cpp
 *  @brief Benchmarks for Levenshtein edit distances under unit costs.
 *         The program accepts a file path to a dataset, tokenizes it, and benchmarks the one-to-one and
 *         one-to-many entries of every backend, validating the SIMD-accelerated backends against the serial one.
 *
 *  Compute-bound: Myers' algorithm costs one word-step per query word per candidate byte, so a 64 MiB slice
 *  exercises every path while each call samples only what it needs.
 *
 *  Two shapes are measured, byte-level and rune-level alike, every candidate at its own length:
 *  - `sz_levenshtein_distance_*` between a median-length query and its successor, on the serial entries alone,
 *    since the one-to-one entries have no other backend;
 *  - `sz_levenshtein_distances_*` from one query against the next `STRINGWARS_BATCH` tokens - by default as many
 *    median tokens as fill a 32 KiB L1 - at two query lengths, the slice's median and the 1024 bytes whose match
 *    masks fill that L1, on every compiled backend.
 *
 *  Every arm's name carries its query length, as `:q117`.
 *  Throughput is reported as Cell Updates Per Second @b (CUPS): the query's length times the candidates' lengths.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or an integer [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_levenshtein_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_levenshtein_cpp20
 *  @endcode
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::bench;


#pragma region One to One

/** @brief One pair per call: a query clamped to @c query_bytes against its successor at its own length. */
template <sz_levenshtein_distance_t function_>
struct levenshtein_distance_from_sz {
    environment_t const &env;
    std::size_t query_bytes;
    sz_memory_allocator_t alloc;

    levenshtein_distance_from_sz(environment_t const &env, std::size_t query_bytes)
        : env(env), query_bytes(query_bytes) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        std::string_view const candidate = env.tokens[(token_index + 1) % env.tokens.size()];
        sz_size_t distance = 0;
        if (function_(query.data(), query.size(), candidate.data(), candidate.size(), &alloc, &distance) !=
            sz_success_k)
            throw std::runtime_error("The one-to-one entry failed.");
        return call_result_t(query.size() + candidate.size(), distance, query.size() * candidate.size());
    }
};

/** @brief The one-to-one entries on the serial walk alone, since one pair fills one candidate on every backend. */
void bench_levenshtein_one_to_one(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    bench_unary(env, "sz_levenshtein_distance_serial" + suffix,
                levenshtein_distance_from_sz<sz_levenshtein_distance_serial> {env, query_bytes})
        .log();
    bench_unary(env, "sz_levenshtein_distance_utf8_serial" + suffix,
                levenshtein_distance_from_sz<sz_levenshtein_distance_utf8_serial> {env, query_bytes})
        .log();
}

#pragma endregion

#pragma region One to Many

/** @brief One query per call, clamped to @c query_bytes, against the next @c candidates tokens at their own length. */
template <sz_levenshtein_distances_t function_>
struct levenshtein_distances_from_sz {
    environment_t const &env;
    std::size_t query_bytes;
    std::size_t candidates;
    sz_memory_allocator_t alloc;
    std::vector<sz_string_view_t> views;
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(environment_t const &env, std::size_t query_bytes, std::size_t candidates)
        : env(env), query_bytes(query_bytes), candidates(candidates), views(candidates), distances(candidates) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::string_view const query = std::string_view(env.tokens[token_index]).substr(0, query_bytes);
        std::size_t bytes = 0, cells = 0;
        for (std::size_t candidate = 0; candidate != candidates; ++candidate) {
            std::string_view const token = env.tokens[(token_index + 1 + candidate) % env.tokens.size()];
            views[candidate] = {token.data(), token.size()};
            bytes += token.size(), cells += query.size() * token.size();
        }
        sz_sequence_t sequence;
        sz_sequence_from_string_views(views.data(), candidates, &sequence);
        if (function_(query.data(), query.size(), &sequence, &alloc, distances.data()) != sz_success_k)
            throw std::runtime_error("The one-to-many entry failed.");
        // Multiplied rather than summed, so two candidates swapping distances cannot cancel out.
        check_value_t mixed = 0;
        for (sz_size_t const distance : distances) mixed = mixed * 31u + distance;
        call_result_t result(bytes, mixed, cells);
        result.inputs_processed = candidates;
        return result;
    }
};

/** @brief The one-to-many entries at one query length, byte and rune level, every backend against its serial base. */
void bench_levenshtein_one_to_many(environment_t const &env, std::size_t query_bytes, std::size_t candidates) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {env, query_bytes, candidates};
    bench_result_t base = bench_unary(env, "sz_levenshtein_distances_serial" + suffix, validator).log();
#if SZ_USE_HASWELL
    bench_unary(env, "sz_levenshtein_distances_haswell" + suffix, validator,
                levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {env, query_bytes, candidates})
        .log(base);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_levenshtein_distances_icelake" + suffix, validator,
                levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {env, query_bytes, candidates})
        .log(base);
#endif
    // The rune-level entries decode every candidate byte, so their cost over the byte entries is the decoder's.
    auto validator_utf8 = levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_serial> {env, query_bytes,
                                                                                               candidates};
    bench_result_t base_utf8 =
        bench_unary(env, "sz_levenshtein_distances_utf8_serial" + suffix, validator_utf8).log(base);
#if SZ_USE_HASWELL
    bench_unary(env, "sz_levenshtein_distances_utf8_haswell" + suffix, validator_utf8,
                levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_haswell> {env, query_bytes, candidates})
        .log(base_utf8);
#endif
#if SZ_USE_ICELAKE
    bench_unary(env, "sz_levenshtein_distances_utf8_icelake" + suffix, validator_utf8,
                levenshtein_distances_from_sz<sz_levenshtein_distances_utf8_icelake> {env, query_bytes, candidates})
        .log(base_utf8);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    // The arms throw on a failed status, so one bad call ends the run with its message rather than a crash.
    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        std::size_t const candidates = candidates_per_call(env);
        std::printf("Starting Levenshtein benchmarks...\n");
        bench_levenshtein_one_to_one(env, median_token_bytes(env));
        bench_levenshtein_one_to_many(env, median_token_bytes(env), candidates);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}
