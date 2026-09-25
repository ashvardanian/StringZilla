/**
 *  @file bench/overlap.cu
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Benchmarks window overlap on CUDA GPUs against the widest CPU backend this build carries.
 *
 *  Compute-bound: every candidate byte costs a modular multiply-add per width plus a B-tree
 *  descent, so a device-resident corpus of a few tens of megabytes keeps every multiprocessor busy
 *  for the whole round.
 *
 *  The environment loads the dataset into unified memory, so the candidates are already
 *  device-reachable and every call scores a wave-sized slice of them in place - which is the regime
 *  the GPU backend exists for. Scoring a handful of candidates per call would time the launch
 *  instead, and answer a question nobody is asking of a GPU.
 *
 *  The engine is built before the timing on both sides, because that is how it is meant to be used:
 *  The engine is built before the timing on both sides, because that is how it is meant to be used:
 *  one forest per batch of queries, many rounds of candidates against it. Only the round is timed.
 *
 *  There is no per-stage breakdown as in `overlap.cpp`: the device runs the prepared query, the
 *  chain and the probes inside one launch, so there is no boundary between them to time.
 *
 *  There is no Standard row here: the platform ships no stock GPU window-overlap scan to compare
 *  against, so the baseline is the widest CPU backend, which is the comparison a dispatch decision
 *  actually turns on.
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
 *  - `STRINGWARS_STRESS=1` : Test the GPU backend against the serial baseline.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_CUDA=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_overlap_cu20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_overlap_cu20
 *  @endcode
 *
 *  This file is the sibling of `overlap.cpp`.
 */
#include <algorithm> // `std::min`
#include <cmath>     // `std::ceil`, `std::log2`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <stringzilla/overlap.h> // `sz_overlap_*`

#include "harness.hpp"

using namespace ashvardanian::stringzilla::bench;

using overlap_engine_init_t = sz_status_t (*)(sz_sequence_t const *, sz_size_t const *, sz_size_t,
                                              sz_memory_allocator_t *, sz_overlap_engine_t *);

/** The width the corpus's collision entropy picks for a query of @p query_bytes against a mean
 *  candidate of the corpus. */
static std::size_t overlap_width_(environment_t const &env, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : env.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(env.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (token_view_t const token : env.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(env.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/**
 *  @brief The corpus as the device sees it: views over the tokens, and room for one round's scores.
 *
 *  Under CUDA the environment already loads the dataset into unified memory, so the candidates need
 *  no upload and the two sequences differ only in whose accessors they carry.
 */
struct overlap_cuda_corpus_t {

    /** One view per candidate; its size is the candidate count. */
    unified_vector<sz_string_view_t> views;

    /** @b [candidates], read back for the check value. */
    unified_vector<sz_f32_t> scores;

    /** Accessors a kernel calls, as the resident path requires. */
    sz_sequence_t device_candidates {};

    /** Accessors the CPU baseline calls, over the same views. */
    sz_sequence_t host_candidates {};

    /** Candidate bytes one round touches, which throughput divides by. */
    std::size_t bytes = 0;

    /** Windows this corpus offers at @p width, which the reported rate divides by. */
    std::size_t windows_at(std::size_t width) const noexcept {
        std::size_t total = 0;
        for (sz_string_view_t const &view : views) total += width <= view.length ? view.length - width + 1 : 0;
        return total;
    }

    overlap_cuda_corpus_t(environment_t const &env) {
        std::size_t const count = std::min<std::size_t>(env.tokens.size(), resident_candidates_per_call(env));
        views.resize(count), scores.resize(count);
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = env.tokens[index];
            views[index].start = token.data(), views[index].length = token.size();
            bytes += token.size();
        }
        if (sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) != sz_success_k)
            throw std::runtime_error("The device accessors could not be bound.");
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }
};

/** The leading token cut to @p query_bytes: the one query each arm's engine is built over. */
static std::string overlap_query_text_(environment_t const &env, std::size_t query_bytes) {
    token_view_t const whole = env.tokens[0];
    return std::string(whole.data(), std::min(whole.size(), query_bytes));
}

/** Scores the resident corpus against the fixed query, entirely on the device. */
struct overlap_scores_from_cuda {
    overlap_cuda_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t alloc;
    sz_overlap_engine_t engine {};

    overlap_scores_from_cuda(environment_t const &env, overlap_cuda_corpus_t &corpus, std::size_t query_bytes,
                             std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)) {
        sz_memory_allocator_init_unified(&alloc, STRINGZILLA_NULL);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (sz_overlap_engine_init_gpu(&queries, &scored_width, 1, &alloc, STRINGZILLA_NULL, &engine) != sz_success_k)
            throw std::runtime_error("The device forest could not be prepared.");
    }
    ~overlap_scores_from_cuda() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_cuda(overlap_scores_from_cuda const &) = delete;
    overlap_scores_from_cuda &operator=(overlap_scores_from_cuda const &) = delete;

    call_result_t operator()(std::size_t) {
        if (sz_overlap_scores_cuda(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.scores.size(), 1) !=
            sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        if (cudaStreamSynchronize(STRINGZILLA_NULL) != cudaSuccess)
            throw std::runtime_error("The GPU round did not finish.");
        check_value_t mixed = 0;
        for (sz_f32_t const score : corpus.scores) mixed = mixed * 31u + (check_value_t)(score * 1048576.0f);
        call_result_t result(corpus.bytes, mixed, windows);
        result.inputs_processed = corpus.views.size();
        return result;
    }
};

/** The same round on the CPU, so the two check values line up under @c STRINGWARS_STRESS. */
template <overlap_engine_init_t init_, sz_overlap_scores_t scores_>
struct overlap_scores_from_sz {
    overlap_cuda_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t alloc;
    std::vector<sz_f32_t> scores;
    sz_overlap_engine_t engine {};

    overlap_scores_from_sz(environment_t const &env, overlap_cuda_corpus_t &corpus, std::size_t query_bytes,
                           std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)),
          scores(corpus.scores.size()) {
        sz_memory_allocator_init_default(&alloc);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (init_(&queries, &scored_width, 1, &alloc, &engine) != sz_success_k)
            throw std::runtime_error("The host forest could not be prepared.");
    }
    ~overlap_scores_from_sz() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_sz(overlap_scores_from_sz const &) = delete;
    overlap_scores_from_sz &operator=(overlap_scores_from_sz const &) = delete;

    call_result_t operator()(std::size_t) {
        if (scores_(&engine, &corpus.host_candidates, scores.data(), scores.size(), 1) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        check_value_t mixed = 0;
        for (sz_f32_t const score : scores) mixed = mixed * 31u + (check_value_t)(score * 1048576.0f);
        call_result_t result(corpus.bytes, mixed, windows);
        result.inputs_processed = corpus.views.size();
        return result;
    }
};

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_overlap_scores(environment_t const &env, overlap_cuda_corpus_t &corpus, std::size_t query_bytes) {
    std::size_t const width = overlap_width_(env, query_bytes);
    std::string const suffix = ":w" + std::to_string(width);
    auto validator = overlap_scores_from_sz<sz_overlap_engine_init_serial, sz_overlap_scores_serial> {
        env, corpus, query_bytes, width};
    bench_result_t base = bench_unary(env, std::string("sz_overlap_scores_serial") + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    base = bench_unary(env, std::string("sz_overlap_scores_haswell") + suffix, validator,
                       overlap_scores_from_sz<sz_overlap_engine_init_haswell, sz_overlap_scores_haswell> {
                           env, corpus, query_bytes, width})
               .log(base);
#endif
#if STRINGZILLA_TARGET_SKYLAKE
    base = bench_unary(env, std::string("sz_overlap_scores_skylake") + suffix, validator,
                       overlap_scores_from_sz<sz_overlap_engine_init_skylake, sz_overlap_scores_skylake> {
                           env, corpus, query_bytes, width})
               .log(base);
#endif
    bench_unary(env, std::string("sz_overlap_scores_cuda") + suffix, validator,
                overlap_scores_from_cuda {env, corpus, query_bytes, width})
        .log(base);
}

int main(int argc, char const **argv) {
    install_bench_signal_handlers();
    log_environment();
    print_bench_environment();
    if (!log_cuda_device()) return 0;

    try {
        fmt::println("Building up the environment...");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        overlap_cuda_corpus_t corpus(env);
        fmt::println("Starting window overlap benchmarks over {} resident candidates...", corpus.views.size());
        bench_overlap_scores(env, corpus, median_token_bytes(env));
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
