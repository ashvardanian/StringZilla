/**
 *  @file bench/levenshtein.cpp
 *  @brief Benchmarks for Levenshtein edit distances under unit costs.
 *         The program accepts a file path to a dataset, tokenizes it, and benchmarks the one-to-one and
 *         one-to-many entries of every backend, validating the SIMD-accelerated backends against the serial one.
 *
 *  Compute-bound: Myers' algorithm costs one word-step per query word per candidate byte, so a 64 MiB slice
 *  exercises every path while each call samples only what it needs.
 *
 *  Two shapes are measured, byte-level and rune-level alike:
 *  - `sz_levenshtein_distance_*` between a token and its successor, both clamped to 512 bytes, on the serial
 *    entries alone, since the one-to-one entries have no other backend;
 *  - `sz_levenshtein_distances_*` from one token against the next 64 tokens, at two query widths - clamped to 64 bytes,
 *    one Myers word, and to 512 bytes, up to eight words - on every compiled backend.
 *
 *  Throughput is reported as Cell Updates Per Second @b (CUPS): the query's length times the candidates' lengths.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
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
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines build_release/stringzilla_bench_levenshtein_cpp20
 *  @endcode
 */
#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

static constexpr std::size_t candidates_per_query_k = 64;
static constexpr std::size_t candidate_bytes_k = 512;
static constexpr std::size_t query_widths_k[2] = {64, 512};

using levenshtein_one_to_one_t = sz_status_t (*)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_memory_allocator_t *,
                                                 sz_size_t *);
using levenshtein_one_to_many_t = sz_status_t (*)(sz_cptr_t, sz_size_t, sz_sequence_t const *, sz_memory_allocator_t *,
                                                  sz_size_t *);

#pragma region One to One

/** @brief One pair per call: a token against its successor, both clamped to @c candidate_bytes_k. */
struct levenshtein_distance_from_sz {
    environment_t const &env;
    levenshtein_one_to_one_t function;
    sz_memory_allocator_t alloc;

    levenshtein_distance_from_sz(environment_t const &env, levenshtein_one_to_one_t function)
        : env(env), function(function) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::size_t const lookup_mask = bit_floor(env.tokens.size()) - 1;
        std::string_view const first =
            std::string_view(env.tokens[token_index & lookup_mask]).substr(0, candidate_bytes_k);
        std::string_view const second =
            std::string_view(env.tokens[(token_index + 1) & lookup_mask]).substr(0, candidate_bytes_k);
        sz_size_t distance = 0;
        if (function(first.data(), first.size(), second.data(), second.size(), &alloc, &distance) != sz_success_k)
            throw std::runtime_error("The one-to-one entry failed.");
        return call_result_t(first.size() + second.size(), distance, first.size() * second.size());
    }
};

/** @brief The one-to-one entries have no backends: one pair fills one candidate, so every one runs the serial walk. */
void bench_levenshtein_one_to_one(environment_t const &env) {
    levenshtein_distance_from_sz serial(env, sz_levenshtein_distance_serial);
    bench_unary(env, "sz_levenshtein_distance_serial", serial).log();
    levenshtein_distance_from_sz serial_utf8(env, sz_levenshtein_distance_utf8_serial);
    bench_unary(env, "sz_levenshtein_distance_utf8_serial", serial_utf8).log();
}

#pragma endregion One to One

#pragma region One to Many

/** @brief One query per call against the next @c candidates_per_query_k tokens, all clamped. */
struct levenshtein_distances_from_sz {
    environment_t const &env;
    levenshtein_one_to_many_t function;
    std::size_t query_bytes;
    sz_memory_allocator_t alloc;
    std::vector<sz_string_view_t> views = std::vector<sz_string_view_t>(candidates_per_query_k);
    std::vector<sz_size_t> distances = std::vector<sz_size_t>(candidates_per_query_k);

    levenshtein_distances_from_sz(environment_t const &env, levenshtein_one_to_many_t function, std::size_t query_bytes)
        : env(env), function(function), query_bytes(query_bytes) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        std::size_t const lookup_mask = bit_floor(env.tokens.size()) - 1;
        std::string_view const query = std::string_view(env.tokens[token_index & lookup_mask]).substr(0, query_bytes);
        std::size_t bytes = 0, cells = 0;
        for (std::size_t candidate = 0; candidate != candidates_per_query_k; ++candidate) {
            std::string_view const token =
                std::string_view(env.tokens[(token_index + 1 + candidate) & lookup_mask]).substr(0, candidate_bytes_k);
            views[candidate] = {token.data(), token.size()};
            bytes += token.size(), cells += query.size() * token.size();
        }
        sz_sequence_t candidates;
        sz_sequence_from_string_views(views.data(), candidates_per_query_k, &candidates);
        if (function(query.data(), query.size(), &candidates, &alloc, distances.data()) != sz_success_k)
            throw std::runtime_error("The one-to-many entry failed.");
        std::size_t check = 0;
        for (sz_size_t const distance : distances) check += distance;
        call_result_t call_result(bytes, check, cells);
        call_result.inputs_processed = candidates_per_query_k;
        return call_result;
    }
};

void bench_levenshtein_one_to_many(environment_t const &env, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes);
    levenshtein_distances_from_sz serial(env, sz_levenshtein_distances_serial, query_bytes);
    bench_result_t base = bench_unary(env, "sz_levenshtein_distances_serial" + suffix, serial).log();
#if SZ_USE_HASWELL
    levenshtein_distances_from_sz haswell(env, sz_levenshtein_distances_haswell, query_bytes);
    bench_unary(env, "sz_levenshtein_distances_haswell" + suffix, serial, haswell).log(base);
#endif
#if SZ_USE_ICELAKE
    levenshtein_distances_from_sz icelake(env, sz_levenshtein_distances_icelake, query_bytes);
    bench_unary(env, "sz_levenshtein_distances_icelake" + suffix, serial, icelake).log(base);
#endif
    // The rune-level entries decode every candidate byte, so their cost over the byte entries is the decoder's.
    levenshtein_distances_from_sz serial_utf8(env, sz_levenshtein_distances_utf8_serial, query_bytes);
    bench_result_t base_utf8 = bench_unary(env, "sz_levenshtein_distances_utf8_serial" + suffix, serial_utf8).log(base);
#if SZ_USE_HASWELL
    levenshtein_distances_from_sz haswell_utf8(env, sz_levenshtein_distances_utf8_haswell, query_bytes);
    bench_unary(env, "sz_levenshtein_distances_utf8_haswell" + suffix, serial_utf8, haswell_utf8).log(base_utf8);
#endif
#if SZ_USE_ICELAKE
    levenshtein_distances_from_sz icelake_utf8(env, sz_levenshtein_distances_utf8_icelake, query_bytes);
    bench_unary(env, "sz_levenshtein_distances_utf8_icelake" + suffix, serial_utf8, icelake_utf8).log(base_utf8);
#endif
}

#pragma endregion One to Many

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "leipzig1M.txt", environment_t::tokenization_t::lines_k,
                                              compute_bound_slice_bytes_k);
        std::printf("Starting Levenshtein benchmarks...\n");
        bench_levenshtein_one_to_one(env);
        for (std::size_t const query_bytes : query_widths_k) bench_levenshtein_one_to_many(env, query_bytes);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}
