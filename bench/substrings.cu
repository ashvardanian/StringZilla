/**
 *  @file bench/substrings.cu
 *  @brief Benchmarks multi-pattern search on CUDA GPUs, against the CPU backend this build carries.
 *
 *  Memory-bound: every haystack byte is one data-dependent load into the automaton, and the device is filled
 *  by haystack chunks rather than haystacks, so a corpus fills it by its byte count rather than its item count.
 *
 *  The environment loads the dataset into unified memory, so the haystacks are already device-reachable and
 *  every call searches them in place. A corpus cutting into fewer chunks than one residency wave, or one whose
 *  match arrays would not fit the device, is refused rather than recorded. There is no Standard
 *  row: the platform ships no GPU multi-pattern search, so the baseline is the CPU backend, which is the
 *  comparison a dispatch decision actually turns on. Every row reports the corpus bytes one call walks as its
 *  bytes and operations, and its haystacks as inputs.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are
 *  used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=64mb` : Reads at most this many dataset bytes; `0` reads the whole file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test the GPU backend against the serial baseline.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D STRINGZILLA_BUILD_CUDA=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_substrings_cu20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_substrings_cu20
 *  @endcode
 *
 *  This file is the sibling of `substrings.cpp`.
 */
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`
#include "substrings.cuh"  // `substrings_dictionary_t`, `substrings_counts_from_sz`

using namespace ashvardanian::stringzilla::bench;

#pragma region Residency

/** Device accessors over @p views, which must themselves be device-reachable. */
static sz_sequence_t substrings_device_sequence(unified_vector<sz_string_view_t> const &views) {
    sz_sequence_t sequence {};
    if (sz_sequence_from_string_views_cuda(views.data(), views.size(), &sequence) != sz_success_k)
        throw std::runtime_error("The device accessors could not be bound.");
    return sequence;
}

/** Moves the corpus's managed pages to the device, so the first round does not time their migration. */
static void substrings_prefetch(substrings_corpus_t const &corpus) {
    cudaMemLocation where {};
    where.type = cudaMemLocationTypeDevice;
    cudaGetDevice(&where.id);
    cudaMemPrefetchAsync(corpus.views.data(), corpus.views.size() * sizeof(sz_string_view_t), where, 0, 0);
    for (sz_string_view_t const &view : corpus.views) cudaMemPrefetchAsync(view.start, view.length, where, 0, 0);
    cudaStreamSynchronize(0);
}

/**
 *  @brief Whether the corpus cuts into at least one residency wave of chunks, by the engine's own budget.
 *
 *  The width is reproduced here rather than read back, because the device derives it from the corpus total
 *  the same way: at least the corpus over the budget, and never under the warm-up a chunk has to pay.
 */
static bool substrings_fills_a_wave(sz_substrings_engine_t const &engine, substrings_corpus_t const &corpus) {
    std::size_t const budget = engine.chunk_budget ? engine.chunk_budget : 1;
    std::size_t const floor_bytes = std::max<std::size_t>(4 * engine.max_source_match_bytes, 1);
    std::size_t const chunk = std::max((corpus.bytes + budget - 1) / budget, floor_bytes);
    std::size_t chunks = 0;
    for (sz_string_view_t const &view : corpus.views)
        chunks += view.length == 0 ? 1 : (view.length + chunk - 1) / chunk;

    double const waves = (double)chunks / (double)budget;
    std::printf("> Corpus: %.1f MB in %zu haystacks, cut into %zu chunks of %zu B " //
                "against %zu resident threads - %.2f waves\n",                      //
                (double)corpus.bytes / 1e6, corpus.views.size(), chunks, chunk, budget, waves);
    if (waves >= 1.0) return true;
    std::printf("> Refusing the round: below one wave it times the launch, not the walk. " //
                "Raise STRINGWARS_DATASET_LIMIT.\n");
    return false;
}

/** Whether the corpus's overlapping matches fit the engine's own match budget, and that budget the device. */
static bool substrings_fits_the_budget(sz_substrings_engine_t &engine, substrings_corpus_t const &corpus,
                                       sz_sequence_t const &device_haystacks) {
    std::size_t free_bytes = 0, total_bytes = 0;
    unified_vector<sz_size_t> offsets(corpus.views.size() + 1, 0);
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        throw std::runtime_error("The device would not report its memory.");
    if (sz_substrings_find_cuda(&engine, &device_haystacks, nullptr, 0, offsets.data()) ==
        sz_device_memory_mismatch_k)
        throw std::runtime_error("The corpus is not device-resident.");
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) throw std::runtime_error("The sizing round failed.");

    std::size_t const emitted = engine.report->matches_emitted;
    std::printf("> Matches: %zu against a budget of %zu, in a %.1f GB arena and %.1f GB free\n", emitted,
                (std::size_t)engine.matches_budget, (double)engine.scratch_bytes / 1e9, (double)free_bytes / 1e9);
    if (emitted <= engine.matches_budget && engine.scratch_bytes <= free_bytes / 2) return true;
    std::printf("> Refusing the round: the round outruns its match budget. Lower STRINGWARS_DATASET_LIMIT.\n");
    return false;
}

#pragma endregion Residency

#pragma region Verbs

/** Per-haystack counts, the device arm logged against the CPU one. */
static void bench_substrings_counts(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                    substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                    std::string const &suffix) {
    auto validator = substrings_counts_from_sz<sz_substrings_counts_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_counts_serial" + suffix, validator).log();
    bench_unary(env, "sz_substrings_counts_cuda" + suffix, validator,
                substrings_counts_from_sz<sz_substrings_counts_cuda> {device, corpus, device_haystacks})
        .log(base);
}

/** Every match, the device arm logged against the CPU one. */
static void bench_substrings_find(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto validator = substrings_find_from_sz<sz_substrings_find_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_find_serial" + suffix, validator).log();
    bench_unary(env, "sz_substrings_find_cuda" + suffix, validator,
                substrings_find_from_sz<sz_substrings_find_cuda> {device, corpus, device_haystacks})
        .log(base);
}

/** The rewrite, the device arm logged against the CPU one. */
static void bench_substrings_replace(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                     substrings_dictionary_t const &dictionary, substrings_corpus_t const &corpus,
                                     sz_sequence_t const &device_haystacks, sz_sequence_t const &device_replacements,
                                     std::string const &suffix) {
    auto validator = substrings_replace_from_sz<sz_substrings_replace_serial> {host, corpus, corpus.haystacks,
                                                                               dictionary.replacements};
    bench_result_t base = bench_unary(env, "sz_substrings_replace_serial" + suffix, validator).log();
    bench_unary(env, "sz_substrings_replace_cuda" + suffix, validator,
                substrings_replace_from_sz<sz_substrings_replace_cuda> {device, corpus, device_haystacks,
                                                                        device_replacements})
        .log(base);
}

/** BM25 scores, the device arm logged against the CPU one. */
static void bench_substrings_bm25(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto validator = substrings_bm25_from_sz<sz_substrings_bm25_scores_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_bm25_scores_serial" + suffix, validator).log();
    bench_unary(env, "sz_substrings_bm25_scores_cuda" + suffix, validator,
                substrings_bm25_from_sz<sz_substrings_bm25_scores_cuda> {device, corpus, device_haystacks})
        .log(base);
}

/** One vocabulary slice, walked by every verb under every policy it accepts, once the round is sound. */
static void bench_substrings_slice(environment_t const &env, substrings_corpus_t const &corpus,
                                   sz_sequence_t const &device_haystacks, substrings_slice_t slice,
                                   sz_substrings_case_sensitivity_t sensitivity) {
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_unified(&allocator, SZ_NULL);
    substrings_dictionary_t const dictionary(env, slice, sensitivity, allocator);
    std::string const suffix = substrings_label(slice, sensitivity);
    if (dictionary.needles.empty()) {
        std::printf("Vocabulary %s is empty on this corpus, skipping it.\n", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        std::printf("Vocabulary %s holds %zu needles over %u states, %u of them hot.\n", suffix.c_str(),
                    dictionary.needles.size(), probe.engine.state_count, probe.engine.hot_count);
        if (!substrings_fills_a_wave(probe.engine, corpus) ||
            !substrings_fits_the_budget(probe.engine, corpus, device_haystacks))
            return;
    }

    sz_sequence_t const device_replacements = substrings_device_sequence(dictionary.replacement_views);
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, policy, substrings_residency_t::device_k);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(env, host, device, corpus, device_haystacks, cover);
        bench_substrings_find(env, host, device, corpus, device_haystacks, cover);
    }
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, policy, substrings_residency_t::device_k);
        bench_substrings_replace(env, host, device, dictionary, corpus, device_haystacks, device_replacements,
                                 suffix + substrings_policy_name(policy));
    }
    {
        substrings_engine_t host(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        bench_substrings_bm25(env, host, device, corpus, device_haystacks, suffix);
    }
}

#pragma endregion Verbs

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    // The arms throw on a failed status, so one bad call ends the run with its message rather than a crash.
    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        substrings_corpus_t const corpus(env);
        sz_sequence_t const device_haystacks = substrings_device_sequence(corpus.views);
        substrings_prefetch(corpus);
        std::printf("Starting multi-pattern search benchmarks...\n");
        bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::frequent_k, sz_substrings_cased_k);
        bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::rare_k, sz_substrings_cased_k);
        bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::frequent_k, sz_substrings_uncased_k);
        bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::sampled_k, sz_substrings_cased_k);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}
