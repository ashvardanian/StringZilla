/**
 *  @file bench/levenshtein.cu
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Benchmarks Levenshtein distances on CUDA GPUs against the widest CPU backend built.
 *
 *  Compute-bound: Myers costs one word-step per query word per candidate byte, so a device-resident
 *  corpus of a few tens of megabytes keeps every multiprocessor busy for the whole round.
 *
 *  The candidates are uploaded once into a device tape, so every call scores a wave-sized slice of
 *  memory the kernel already owns - which is the regime the GPU backend exists for. Scoring a
 *  handful of candidates per call would time the launch instead, and answer a question nobody is
 *  asking of a GPU. Leaving the candidates or the answers in managed memory answers a different
 *  question again: those pages follow whoever touched them last, so a short-candidate round times
 *  migration rather than the kernel. The query batch is prepared once per arm, so what an arm times
 *  is the sweep across `grid.y` and not the preparation the engine exists to hoist.
 *
 *  One candidate per thread means a warp costs the longest of its thirty-two, while throughput
 *  divides by the sum of their lengths - so length variance is a tax inside every number here. Each
 *  device arm therefore runs twice over the very same views, once in corpus order and once sorted
 *  by length descending, and the gap between `:shuffled` and `:sorted` measures that tax rather
 *  than assuming it.
 *
 *  There is no per-stage breakdown as in `levenshtein.cpp`: the device settles every candidate
 *  inside one launch, leaving no boundary between the query's preparation and its sweep to time.
 *
 *  There is no Standard row here: the platform ships no stock GPU edit-distance kernel to compare
 *  against, so the baseline is the widest CPU backend, which is the comparison a dispatch decision
 *  actually turns on.
 *
 *  Throughput is reported as Cell Updates Per Second @b (CUPS): the query's length times the
 *  candidates' lengths, as the CPU benchmark reports it, though Myers settles sixty-four of those
 *  cells per word-step.
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
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test the GPU backend against the serial baseline.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D STRINGZILLA_BUILD_CUDA=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_levenshtein_cu20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines build_release/stringzilla_bench_levenshtein_cu20
 *  @endcode
 *
 *  This file is the sibling of `levenshtein.cpp`.
 */
#include <algorithm> // `std::min`, `std::stable_sort`
#include <numeric>   // `std::iota`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <stringzilla/levenshtein.h> // `sz_levenshtein_*`

#include "shared.hpp"
#include "stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::bench;

/** Which ordering of the same candidates an arm scores. */
enum class levenshtein_cuda_order_t {

    /** Corpus order, where a warp's thirty-two candidates have whatever lengths they had. */
    shuffled_k,

    /** Length descending, so a warp's candidates sit as close in length as the corpus allows. */
    sorted_k,
};

/** The label an arm's name carries for the ordering it scored. */
static char const *levenshtein_cuda_order_name(levenshtein_cuda_order_t order) {
    return order == levenshtein_cuda_order_t::sorted_k ? ":sorted" : ":shuffled";
}

/** Queries one prepared batch carries, so `grid.y` spans an axis wider than one in every rung. */
enum { levenshtein_queries_per_batch_k = 4 };

/** Independent mixing chains the check value folds, so the hash is not what a round measures. */
enum { levenshtein_check_lanes_k = 8 };

/**
 *  @brief Folds one round's distances into the value the stress gate compares between backends.
 *
 *  One chain of `mixed * 31 + distance` is a loop-carried multiply with nothing else in flight, and
 *  over a wave of candidates it costs more than the kernel; eight chains over strided lanes retire
 *  in parallel instead.
 */
template <typename answers_type_>
static check_value_t levenshtein_check_value(answers_type_ const &answers) {
    check_value_t accumulators[levenshtein_check_lanes_k] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t const count = answers.size();
    std::size_t index = 0;
    for (; index + levenshtein_check_lanes_k <= count; index += levenshtein_check_lanes_k)
        for (std::size_t lane = 0; lane != levenshtein_check_lanes_k; ++lane)
            accumulators[lane] = accumulators[lane] * 31u + (check_value_t)answers[index + lane];
    for (; index != count; ++index) {
        std::size_t const lane = index % levenshtein_check_lanes_k;
        accumulators[lane] = accumulators[lane] * 31u + (check_value_t)answers[index];
    }
    check_value_t mixed = 0;
    for (std::size_t lane = 0; lane != levenshtein_check_lanes_k; ++lane) mixed = mixed * 31u + accumulators[lane];
    return mixed;
}

/**
 *  @brief The corpus as the device sees it: a tape of every candidate's bytes, two orderings of
 *      views into it, and the room for one round's distances.
 *
 *  The dataset the environment loads is managed, so a view into it is a page that follows whoever
 *  touched it last. The candidates therefore cross once into plain device memory the host cannot
 *  address, and the CPU arms keep their own views into the dataset to score the very same texts.
 */
struct levenshtein_cuda_corpus_t {

    /** Every candidate's bytes, back to back. */
    device_vector<char> tape;

    /** Tape addresses, corpus order. */
    device_vector<sz_string_view_t> device_views;

    /** The same tape addresses, length descending. */
    device_vector<sz_string_view_t> device_sorted_views;

    /** @b [queries, candidates], written by every round. */
    device_vector<sz_size_t> distances;

    /** The same matrix, one copy per round feeding the check. */
    pinned_vector<sz_size_t> answers;

    /** Dataset views the CPU arms read, corpus order. */
    std::vector<sz_string_view_t> host_views;

    /** The same dataset views, length descending. */
    std::vector<sz_string_view_t> host_sorted_views;

    /** Device accessors over @ref device_views. */
    sz_sequence_t device_shuffled {};

    /** Device accessors over @ref device_sorted_views. */
    sz_sequence_t device_sorted {};

    /** Host accessors over @ref host_views. */
    sz_sequence_t host_shuffled {};

    /** Host accessors over @ref host_sorted_views. */
    sz_sequence_t host_sorted {};

    /** Candidate bytes one round touches, whichever order. */
    std::size_t bytes = 0;

    levenshtein_cuda_corpus_t(environment_t const &env) {
        std::size_t const count = std::min<std::size_t>(env.tokens.size(), resident_candidates_per_call(env));
        for (std::size_t index = 0; index != count; ++index) bytes += env.tokens[index].size();
        if (tape.try_resize_uninitialized(bytes) != sz::status_t::success_k ||
            device_views.try_resize_uninitialized(count) != sz::status_t::success_k ||
            device_sorted_views.try_resize_uninitialized(count) != sz::status_t::success_k ||
            distances.try_resize_uninitialized(count * levenshtein_queries_per_batch_k) != sz::status_t::success_k)
            throw std::runtime_error("The device would not hold the corpus.");
        answers.resize(count * levenshtein_queries_per_batch_k);
        host_views.resize(count), host_sorted_views.resize(count);

        // Staged on the host once, so the candidates cross the bus as one block of addresses the host never reads.
        std::vector<char> staged;
        staged.reserve(bytes);
        std::vector<sz_string_view_t> tape_views(count);
        for (std::size_t index = 0, written = 0; index != count; ++index) {
            token_view_t const token = env.tokens[index];
            host_views[index] = {token.data(), token.size()};
            staged.insert(staged.end(), token.data(), token.data() + token.size());
            tape_views[index] = {tape.data() + written, token.size()};
            written += token.size();
        }

        // One permutation drives both sides, so the sorted arm's answers line up with its host baseline's even
        // where several candidates share a length.
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), (std::size_t)0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            return host_views[left].length > host_views[right].length;
        });
        std::vector<sz_string_view_t> tape_sorted_views(count);
        for (std::size_t index = 0; index != count; ++index)
            host_sorted_views[index] = host_views[order[index]], tape_sorted_views[index] = tape_views[order[index]];

        std::size_t const views_bytes = count * sizeof(sz_string_view_t);
        if (cuMemcpyHtoD((CUdeviceptr)tape.data(), staged.data(), staged.size()) != CUDA_SUCCESS ||
            cuMemcpyHtoD((CUdeviceptr)device_views.data(), tape_views.data(), views_bytes) != CUDA_SUCCESS ||
            cuMemcpyHtoD((CUdeviceptr)device_sorted_views.data(), tape_sorted_views.data(), views_bytes) !=
                CUDA_SUCCESS)
            throw std::runtime_error("The corpus would not upload.");

        if (sz_sequence_from_string_views_cuda(device_views.data(), count, &device_shuffled) != sz_success_k ||
            sz_sequence_from_string_views_cuda(device_sorted_views.data(), count, &device_sorted) != sz_success_k)
            throw std::runtime_error("The device accessors could not be bound.");
        sz_sequence_from_string_views(host_views.data(), count, &host_shuffled);
        sz_sequence_from_string_views(host_sorted_views.data(), count, &host_sorted);
    }

    /** Candidates one round scores, whichever order it walks them in. */
    std::size_t count() const { return host_views.size(); }

    /** The device accessors over one ordering of the same texts. */
    sz_sequence_t const &device_candidates(levenshtein_cuda_order_t order) const {
        return order == levenshtein_cuda_order_t::sorted_k ? device_sorted : device_shuffled;
    }

    /** The host accessors over one ordering of the same texts. */
    sz_sequence_t const &host_candidates(levenshtein_cuda_order_t order) const {
        return order == levenshtein_cuda_order_t::sorted_k ? host_sorted : host_shuffled;
    }
};

/**
 *  @brief The queries an arm of @p query_bytes prepares: that many dataset windows, each starting
 *      at one token.
 *
 *  Only three percent of XLSum lines reach sixteen kilobytes, so clamping a token to the width
 *  would leave the widest arms running at whatever length the token happened to have - a different
 *  rung per call, under a name that claims one width. The window is corpus text either way.
 *
 *  The windows are managed dataset memory, which the host reads to count symbols and classes and
 *  the device builder then reads from the copies the init stages for it.
 */
static std::vector<sz_string_view_t> levenshtein_cuda_queries(environment_t const &env, std::size_t query_bytes) {
    std::vector<sz_string_view_t> views(levenshtein_queries_per_batch_k);
    for (std::size_t query = 0; query != views.size(); ++query) {
        token_view_t const whole = env.tokens[(query * 7 + 1) % env.tokens.size()];
        std::size_t const reachable = (std::size_t)(env.dataset.data() + env.dataset.size() - whole.data());
        views[query] = {whole.data(), std::min(query_bytes, reachable)};
    }
    return views;
}

/** One prepared batch and the residency it was built for, released with the scope that named it. */
struct levenshtein_cuda_batch_t {

    /** The windows the batch was prepared from. */
    std::vector<sz_string_view_t> views;

    /** Host accessors over them, as both inits require. */
    sz_sequence_t queries {};

    /** The batch, prepared once and reused by every round. */
    sz_levenshtein_engine_t engine {};

    levenshtein_cuda_batch_t(environment_t const &env, std::size_t query_bytes, sz_levenshtein_symbol_t symbol,
                             sz_bool_t on_device)
        : views(levenshtein_cuda_queries(env, query_bytes)) {
        sz_sequence_from_string_views(views.data(), views.size(), &queries);
        sz_status_t const prepared = on_device == sz_true_k
                                         ? sz_levenshtein_engine_init_gpu(&queries, symbol, SZ_NULL, SZ_NULL, &engine)
                                         : sz_levenshtein_engine_init_cpu(&queries, symbol, SZ_NULL, &engine);
        if (prepared != sz_success_k) throw std::runtime_error("The engine could not be prepared.");
    }
    ~levenshtein_cuda_batch_t() { sz_levenshtein_engine_free(&engine); }
    levenshtein_cuda_batch_t(levenshtein_cuda_batch_t const &) = delete;
    levenshtein_cuda_batch_t &operator=(levenshtein_cuda_batch_t const &) = delete;

    /** Symbols the batch spans together: the row count of the cell budget an arm reports. */
    std::size_t symbols() const {
        std::size_t total = 0;
        for (sz_string_view_t const &view : views) total += view.length;
        return total;
    }
};

/** Scores the resident corpus against a prepared batch, entirely on the device. */
template <sz_levenshtein_distances_t distances_>
struct levenshtein_distances_from_cuda {

    /** The candidates, and the room for their distances. */
    levenshtein_cuda_corpus_t &corpus;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_cuda_order_t order;

    /** The queries, prepared on the device once. */
    levenshtein_cuda_batch_t batch;

    levenshtein_distances_from_cuda(environment_t const &env, levenshtein_cuda_corpus_t &corpus,
                                    std::size_t query_bytes, levenshtein_cuda_order_t order,
                                    sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : corpus(corpus), order(order), batch(env, query_bytes, symbol, sz_true_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &corpus.device_candidates(order), corpus.distances.data(), corpus.count()) !=
            sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        // Device memory cannot migrate, so the answers cross as one block instead of a page fault per four kilobytes.
        if (copy_device_to_host(corpus.distances, sz::span<sz_size_t> {corpus.answers.data(), corpus.answers.size()}) !=
            CUDA_SUCCESS)
            throw std::runtime_error("The answers would not come back.");
        call_result_t result(corpus.bytes, levenshtein_check_value(corpus.answers), batch.symbols() * corpus.bytes);
        result.inputs_processed = corpus.count() * levenshtein_queries_per_batch_k;
        return result;
    }
};

/** The same round on the CPU, so the two check values line up under @c STRINGWARS_STRESS. */
template <sz_levenshtein_distances_t distances_>
struct levenshtein_distances_from_sz {

    /** The candidates, whose texts both sides score. */
    levenshtein_cuda_corpus_t &corpus;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_cuda_order_t order;

    /** The same queries, prepared on the host. */
    levenshtein_cuda_batch_t batch;

    /** @b [queries, candidates], the CPU's own answers. */
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(environment_t const &env, levenshtein_cuda_corpus_t &corpus, std::size_t query_bytes,
                                  levenshtein_cuda_order_t order,
                                  sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : corpus(corpus), order(order), batch(env, query_bytes, symbol, sz_false_k),
          distances(corpus.count() * levenshtein_queries_per_batch_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &corpus.host_candidates(order), distances.data(), corpus.count()) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        call_result_t result(corpus.bytes, levenshtein_check_value(distances), batch.symbols() * corpus.bytes);
        result.inputs_processed = corpus.count() * levenshtein_queries_per_batch_k;
        return result;
    }
};

/** The device arm alone across query widths and both orders, for the word-count curve both ways. */
static void bench_levenshtein_word_counts(environment_t const &env, levenshtein_cuda_corpus_t &corpus) {
    std::size_t const widths[] = {8, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384};
    levenshtein_cuda_order_t const orders[] = {levenshtein_cuda_order_t::shuffled_k,
                                               levenshtein_cuda_order_t::sorted_k};
    for (std::size_t index = 0; index != sizeof(widths) / sizeof(widths[0]); ++index)
        for (levenshtein_cuda_order_t const order : orders) {
            std::size_t const query_bytes = widths[index];
            std::string const suffix = ":q" + std::to_string(query_bytes) + ":w" +
                                       std::to_string(sz_levenshtein_query_words(query_bytes)) +
                                       levenshtein_cuda_order_name(order);
            bench_unary(
                env, std::string("sz_levenshtein_distances_cuda") + suffix,
                levenshtein_distances_from_cuda<sz_levenshtein_distances_cuda> {env, corpus, query_bytes, order})
                .log();
        }
}

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_levenshtein_cross_product(environment_t const &env, levenshtein_cuda_corpus_t &corpus,
                                            std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes) + ":c" + std::to_string(corpus.count());
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_cuda_order_t::shuffled_k};
    bench_result_t base = bench_unary(env, std::string("sz_levenshtein_distances_serial") + suffix, validator).log();
#if SZ_USE_HASWELL
    base = bench_unary(env, std::string("sz_levenshtein_distances_haswell") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {
                           env, corpus, query_bytes, levenshtein_cuda_order_t::shuffled_k})
               .log(base);
#endif
#if SZ_USE_ICELAKE
    base = bench_unary(env, std::string("sz_levenshtein_distances_icelake") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {
                           env, corpus, query_bytes, levenshtein_cuda_order_t::shuffled_k})
               .log(base);
#endif
    // The warped rung spreads a query across a warp's thirty-two lanes, and a wider one is reported as out of
    // reach rather than thrown.
    if (sz_levenshtein_query_words(query_bytes) > sz_levenshtein_cuda_words_max_k) {
        fmt::println("Skipping `sz_levenshtein_distances_cuda{}`: {} words past the device's {}.", suffix.c_str(),
                     sz_levenshtein_query_words(query_bytes), (int)sz_levenshtein_cuda_words_max_k);
        return;
    }
    // Both orderings hold the same texts, so the gap between the two arms is the warp's `max(L)` tax alone.
    bench_unary(env, std::string("sz_levenshtein_distances_cuda") + suffix + ":shuffled", validator,
                levenshtein_distances_from_cuda<sz_levenshtein_distances_cuda> {env, corpus, query_bytes,
                                                                                levenshtein_cuda_order_t::shuffled_k})
        .log(base);
    auto validator_sorted = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_cuda_order_t::sorted_k};
    bench_unary(env, std::string("sz_levenshtein_distances_cuda") + suffix + ":sorted", validator_sorted,
                levenshtein_distances_from_cuda<sz_levenshtein_distances_cuda> {env, corpus, query_bytes,
                                                                                levenshtein_cuda_order_t::sorted_k})
        .log(base);

    // A window of this many bytes holds at most as many runes, so the byte guard above already covers the rune arm.
    // Cells are counted in bytes on both sides, as the CPU arms count them, so the rune rows compare across backends.
    auto validator_utf8 = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_cuda_order_t::shuffled_k, sz_levenshtein_runes_k};
    bench_result_t base_utf8 =
        bench_unary(env, std::string("sz_levenshtein_distances_serial:utf8") + suffix, validator_utf8).log(base);
    bench_unary(env, std::string("sz_levenshtein_distances_cuda:utf8") + suffix + ":shuffled", validator_utf8,
                levenshtein_distances_from_cuda<sz_levenshtein_distances_cuda> {
                    env, corpus, query_bytes, levenshtein_cuda_order_t::shuffled_k, sz_levenshtein_runes_k})
        .log(base_utf8);
}

int main(int argc, char const **argv) {
    install_test_signal_handlers();
    fmt::println("Welcome to StringZilla!");
    if (auto code = log_environment(); code != 0) return code;

    try {
        fmt::println("Building up the environment...");
        environment_t env = build_environment(argc, argv, "xlsum.csv", environment_t::tokenization_t::lines_k);
        levenshtein_cuda_corpus_t corpus(env);
        fmt::println("Starting Levenshtein benchmarks over {} resident candidates...", corpus.count());
        bench_levenshtein_cross_product(env, corpus, median_token_bytes(env));
        bench_levenshtein_word_counts(env, corpus);
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}
