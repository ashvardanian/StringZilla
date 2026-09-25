/**
 *  @file bench/substrings.cpp
 *  @author Ash Vardanian
 *  @date August 8, 2026
 *  @brief Benchmarks multi-pattern search on the CPU: one compiled vocabulary against the corpus.
 *
 *  Memory-bound rather than compute-bound: the walk is one data-dependent load per byte, so what a
 *  row measures is how often that load hits a cache line the automaton already brought in.
 *  Vocabulary shape is what moves that, so every verb is measured twice - against the most frequent
 *  slice of the corpus vocabulary, where accepting states are common, and against the least
 *  frequent slice, where the walk sits near the root and almost never reports.
 *
 *  Compilation is timed separately, because a pipeline that recompiles per query is bound by that
 *  rather than by the walk, and because its cost scales with the vocabulary while every other row
 *  scales with the corpus. A compilation row reports needle bytes as its bytes and operations, and
 *  needles as its inputs; every other row reports the corpus bytes one call walks, and its
 *  haystacks as inputs.
 *
 *  There is no Standard row: the platform ships no multi-pattern search, so the serial backend is
 *  its own reference and the accelerated rows are logged against it.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Cross-check the backends against each other.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_substrings_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_substrings_cpp20
 *  @endcode
 *
 *  This file is the sibling of `substrings.cu`.
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include <fmt/format.h>

#include "harness.hpp"
#include "substrings.cuh"  // `substrings_dictionary_t`, `substrings_counts_from_sz`

using namespace ashvardanian::stringzilla::bench;

#pragma region Compilation

/** Compiles the vocabulary from scratch: the cost a pipeline pays once rather than per haystack. */
struct substrings_build_from_sz {

    /** The needles to compile, and the sensitivity to compile them at. */
    substrings_dictionary_t const &dictionary;

    call_result_t operator()(std::size_t) const {
        sz_memory_allocator_t allocator;
        sz_substrings_engine_t engine;
        sz_memory_allocator_init_default(&allocator);
        if (sz_substrings_engine_init_cpu(&dictionary.needle_sequence, dictionary.sensitivity,
                                          sz_substrings_overlapping_k, STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0,
                                          &allocator, &engine) != sz_success_k)
            throw std::runtime_error("The vocabulary would not compile.");
        check_value_t const mixed = (check_value_t)engine.state_count * 31u + engine.max_outputs_per_state;
        sz_substrings_engine_free(&engine);
        call_result_t result(dictionary.needle_bytes, mixed, dictionary.needle_bytes);
        result.inputs_processed = dictionary.needles.size();
        return result;
    }
};

#pragma endregion Compilation

#pragma region Verbs

/** Per-haystack counts on every CPU backend, the accelerated arms logged against the serial one. */
static void bench_substrings_counts(environment_t const &env, substrings_engine_t &engine,
                                    substrings_corpus_t const &corpus, std::string const &suffix) {
    auto validator = substrings_counts_from_sz<sz_substrings_counts_serial> {engine, corpus, corpus.haystacks};
    [[maybe_unused]] bench_result_t base = bench_unary(env, "sz_substrings_counts_serial" + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_substrings_counts_haswell" + suffix, validator,
                substrings_counts_from_sz<sz_substrings_counts_haswell> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_substrings_counts_icelake" + suffix, validator,
                substrings_counts_from_sz<sz_substrings_counts_icelake> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_substrings_counts_neon" + suffix, validator,
                substrings_counts_from_sz<sz_substrings_counts_neon> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
}

/** Every match on every CPU backend, the accelerated arms logged against the serial one. */
static void bench_substrings_find(environment_t const &env, substrings_engine_t &engine,
                                  substrings_corpus_t const &corpus, std::string const &suffix) {
    auto validator = substrings_find_from_sz<sz_substrings_find_serial> {engine, corpus, corpus.haystacks};
    [[maybe_unused]] bench_result_t base = bench_unary(env, "sz_substrings_find_serial" + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_substrings_find_haswell" + suffix, validator,
                substrings_find_from_sz<sz_substrings_find_haswell> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_substrings_find_icelake" + suffix, validator,
                substrings_find_from_sz<sz_substrings_find_icelake> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_substrings_find_neon" + suffix, validator,
                substrings_find_from_sz<sz_substrings_find_neon> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
}

/** The rewrite on every CPU backend, the accelerated arms logged against the serial one. */
static void bench_substrings_replace(environment_t const &env, substrings_engine_t &engine,
                                     substrings_dictionary_t const &dictionary, substrings_corpus_t const &corpus,
                                     std::string const &suffix) {
    auto validator = substrings_replace_from_sz<sz_substrings_replace_serial> {engine, corpus, corpus.haystacks,
                                                                               dictionary.replacements};
    [[maybe_unused]] bench_result_t base = bench_unary(env, "sz_substrings_replace_serial" + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_substrings_replace_haswell" + suffix, validator,
                substrings_replace_from_sz<sz_substrings_replace_haswell> {engine, corpus, corpus.haystacks,
                                                                           dictionary.replacements})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_substrings_replace_icelake" + suffix, validator,
                substrings_replace_from_sz<sz_substrings_replace_icelake> {engine, corpus, corpus.haystacks,
                                                                           dictionary.replacements})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_substrings_replace_neon" + suffix, validator,
                substrings_replace_from_sz<sz_substrings_replace_neon> {engine, corpus, corpus.haystacks,
                                                                        dictionary.replacements})
        .log(base);
#endif
}

/** BM25 scores on every CPU backend, the accelerated arms logged against the serial one. */
static void bench_substrings_bm25(environment_t const &env, substrings_engine_t &engine,
                                  substrings_corpus_t const &corpus, std::string const &suffix) {
    auto validator = substrings_bm25_from_sz<sz_substrings_bm25_scores_serial> {engine, corpus, corpus.haystacks};
    [[maybe_unused]] bench_result_t base =
        bench_unary(env, "sz_substrings_bm25_scores_serial" + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    bench_unary(env, "sz_substrings_bm25_scores_haswell" + suffix, validator,
                substrings_bm25_from_sz<sz_substrings_bm25_scores_haswell> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    bench_unary(env, "sz_substrings_bm25_scores_icelake" + suffix, validator,
                substrings_bm25_from_sz<sz_substrings_bm25_scores_icelake> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
#if STRINGZILLA_TARGET_NEON
    bench_unary(env, "sz_substrings_bm25_scores_neon" + suffix, validator,
                substrings_bm25_from_sz<sz_substrings_bm25_scores_neon> {engine, corpus, corpus.haystacks})
        .log(base);
#endif
}

/** One vocabulary slice, compiled, then walked by every verb under every policy it accepts. */
static void bench_substrings_slice(environment_t const &env, substrings_corpus_t const &corpus,
                                   substrings_slice_t slice, sz_substrings_case_sensitivity_t sensitivity) {
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_default(&allocator);
    substrings_dictionary_t const dictionary(env, slice, sensitivity, allocator);
    std::string const suffix = substrings_label(slice, sensitivity);
    if (dictionary.needles.empty()) {
        fmt::println("Vocabulary {} is empty on this corpus, skipping it.", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
        fmt::println("Vocabulary {} holds {} needles over {} states, {} of them hot.", suffix.c_str(),
                     dictionary.needles.size(), probe.engine.state_count, probe.engine.hot_count);
    }

    bench_unary(env, "sz_substrings_engine_init_cpu" + suffix, substrings_build_from_sz {dictionary}).log();
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
        substrings_engine_t engine(dictionary, policy, substrings_residency_t::host_k);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(env, engine, corpus, cover);
        bench_substrings_find(env, engine, corpus, cover);
    }
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
        substrings_engine_t engine(dictionary, policy, substrings_residency_t::host_k);
        bench_substrings_replace(env, engine, dictionary, corpus, suffix + substrings_policy_name(policy));
    }
    {
        substrings_engine_t engine(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
        bench_substrings_bm25(env, engine, corpus, suffix);
    }
}

#pragma endregion Verbs

int main(int argc, char const **argv) {
    install_bench_signal_handlers();
    log_environment();
    print_bench_environment();

    // The arms throw on a failed status, so one bad call ends the run with its message rather than a crash.
    try {
        fmt::println("Building up the environment...");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        substrings_corpus_t const corpus(env);
        fmt::println("Starting multi-pattern search benchmarks...");
        bench_substrings_slice(env, corpus, substrings_slice_t::frequent_k, sz_substrings_cased_k);
        bench_substrings_slice(env, corpus, substrings_slice_t::rare_k, sz_substrings_cased_k);
        bench_substrings_slice(env, corpus, substrings_slice_t::frequent_k, sz_substrings_uncased_k);
        bench_substrings_slice(env, corpus, substrings_slice_t::sampled_k, sz_substrings_cased_k);
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
