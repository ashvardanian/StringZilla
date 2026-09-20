/**
 *  @file bench/levenshtein.cu
 *  @brief Benchmarks Levenshtein edit distances on CUDA GPUs, against the widest CPU backend this build carries.
 *
 *  Compute-bound: Myers costs one word-step per query word per candidate byte, so a device-resident corpus of a
 *  few tens of megabytes keeps every multiprocessor busy for the whole round.
 *
 *  The environment loads the dataset into unified memory, so the candidates are already device-reachable and
 *  every call scores a wave-sized slice of them in place - which is the regime the GPU backend exists for.
 *  Scoring a handful of candidates per call would time the launch instead, and answer a question nobody is
 *  asking of a GPU.
 *
 *  There is no per-stage breakdown as in `levenshtein.cpp`: the device settles every candidate inside one launch,
 *  so there is no boundary between the query's preparation and its sweep to time.
 *
 *  There is no `Standard` row here: the platform ships no stock GPU edit-distance kernel to compare against, so
 *  the baseline is the widest CPU backend, which is the comparison a dispatch decision actually turns on.
 *
 *  Throughput is reported as Cell Updates Per Second @b (CUPS): the query's length times the candidates' lengths,
 *  as the CPU benchmark reports it, though Myers settles sixty-four of those cells per word-step.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
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
 *  cmake --build build_release --config Release --target stringzilla_bench_levenshtein_cu20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_levenshtein_cu20
 *  @endcode
 *
 *  This file is the sibling of `levenshtein.cpp`.
 */
#include <algorithm> // `std::min`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <stringzilla/levenshtein.h> // `sz_levenshtein_*`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::bench;


/**
 *  @brief Candidates one call scores: the `STRINGWARS_BATCH` entry if set, else one residency wave.
 *
 *  Both kernels map one candidate per thread, so a wave is every multiprocessor filled to its thread ceiling -
 *  a count that moves with the part rather than a literal tuned on one of them.
 */
static std::size_t candidates_per_call(environment_t const &env, int device) {
    if (!env.batch_sizes_override.empty()) return env.batch_sizes_override.front();
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("The device would not report its geometry.");
    return (std::size_t)properties.multiProcessorCount * (std::size_t)properties.maxThreadsPerMultiProcessor;
}

/**
 *  @brief The corpus as the device sees it: views over the tokens, and the room for one round's distances.
 *
 *  Under CUDA the environment already loads the dataset into unified memory, so the candidates need no upload
 *  and the two sequences differ only in whose accessors they carry.
 */
struct levenshtein_cuda_corpus_t {
    unified_vector<sz_string_view_t> views; /**< One view per candidate; its size is the candidate count. */
    unified_vector<sz_size_t> distances;    /**< @b [candidates], read back for the check value. */
    sz_sequence_t device_candidates {};     /**< Accessors a kernel calls, as the resident path requires. */
    sz_sequence_t host_candidates {};       /**< Accessors the CPU baseline calls, over the same views. */
    std::size_t bytes = 0;                  /**< Candidate bytes one round touches, which throughput divides by. */

    levenshtein_cuda_corpus_t(environment_t const &env) {
        std::size_t const count = std::min<std::size_t>(env.tokens.size(), candidates_per_call(env, 0));
        views.resize(count), distances.resize(count);
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = env.tokens[index];
            views[index].start = token.data(), views[index].length = token.size();
            bytes += token.size();
        }
        if (sz_sequence_from_string_views_cuda(views.data(), views.size(), &device_candidates) !=
            sz_success_k)
            throw std::runtime_error("The device accessors could not be bound.");
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
    }
};

/** Scores the resident corpus against one token as the query, entirely on the device. */
struct levenshtein_distances_from_cuda {
    environment_t const &env;
    levenshtein_cuda_corpus_t &corpus;
    std::size_t query_bytes;
    sz_memory_allocator_t alloc;

    levenshtein_distances_from_cuda(environment_t const &env, levenshtein_cuda_corpus_t &corpus,
                                    std::size_t query_bytes)
        : env(env), corpus(corpus), query_bytes(query_bytes) {
        sz_memory_allocator_init_unified(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        token_view_t const whole = env.tokens[token_index % env.tokens.size()];
        token_view_t const query {whole.data(), std::min(whole.size(), query_bytes)};
        if (sz_levenshtein_distances_cuda(query.data(), query.size(), &corpus.device_candidates, &alloc,
                                          corpus.distances.data()) != sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        check_value_t mixed = 0;
        for (sz_size_t const distance : corpus.distances) mixed = mixed * 31u + (check_value_t)distance;
        call_result_t result(corpus.bytes, mixed, query.size() * corpus.bytes);
        result.inputs_processed = corpus.views.size();
        return result;
    }
};

/** The same round on the CPU, so the two check values line up under `STRINGWARS_STRESS`. */
template <sz_status_t (*distances_)(sz_cptr_t, sz_size_t, sz_sequence_t const *, sz_memory_allocator_t *, sz_size_t *)>
struct levenshtein_distances_from_sz {
    environment_t const &env;
    levenshtein_cuda_corpus_t &corpus;
    std::size_t query_bytes;
    sz_memory_allocator_t alloc;
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(environment_t const &env, levenshtein_cuda_corpus_t &corpus,
                                  std::size_t query_bytes)
        : env(env), corpus(corpus), query_bytes(query_bytes), distances(corpus.distances.size()) {
        sz_memory_allocator_init_default(&alloc);
    }

    call_result_t operator()(std::size_t token_index) {
        token_view_t const whole = env.tokens[token_index % env.tokens.size()];
        token_view_t const query {whole.data(), std::min(whole.size(), query_bytes)};
        if (distances_(query.data(), query.size(), &corpus.host_candidates, &alloc, distances.data()) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        check_value_t mixed = 0;
        for (sz_size_t const distance : distances) mixed = mixed * 31u + (check_value_t)distance;
        call_result_t result(corpus.bytes, mixed, query.size() * corpus.bytes);
        result.inputs_processed = corpus.views.size();
        return result;
    }
};

/** @brief Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_levenshtein_one_to_many(environment_t const &env, levenshtein_cuda_corpus_t &corpus,
                                          std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes) + ":c" + std::to_string(corpus.views.size());
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {env, corpus, query_bytes};
    bench_result_t base = bench_unary(env, std::string("sz_levenshtein_distances_serial") + suffix, validator).log();
#if SZ_USE_HASWELL
    base = bench_unary(env, std::string("sz_levenshtein_distances_haswell") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {env, corpus, query_bytes})
               .log(base);
#endif
#if SZ_USE_ICELAKE
    base = bench_unary(env, std::string("sz_levenshtein_distances_icelake") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {env, corpus, query_bytes})
               .log(base);
#endif
    bench_unary(env, std::string("sz_levenshtein_distances_cuda") + suffix, validator,
                levenshtein_distances_from_cuda {env, corpus, query_bytes})
        .log(base);
}

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    std::printf("Welcome to StringZilla!\n");
    if (auto code = log_environment(); code != 0) return code;

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        levenshtein_cuda_corpus_t corpus(env);
        std::printf("Starting Levenshtein benchmarks over %zu resident candidates...\n", corpus.views.size());
        bench_levenshtein_one_to_many(env, corpus, median_token_bytes(env));
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks passed.\n");
    return 0;
}
