/**
 *  @file bench/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the rows CUDA and ROCm share, each device arm against the CPU
 *      tiers this build carries.
 *
 *  Included by both `cross_cuda.cu` and `cross_rocm.hip`; each supplies its runtime and kernels.
 *
 *  Every row times a device-resident round, as that is the regime the GPU backends exist for:
 *  scoring a handful of candidates per call would time the launch instead. Each engine is built
 *  before the timing on both sides, as it is meant to be used - one batch of queries or needles,
 *  many rounds against it - so only the round is timed. There is no Standard row: the platforms
 *  ship no GPU edit distance, window overlap or multi-pattern search, so the baseline is the
 *  dispatched CPU entry, the widest tier this machine runs, which a dispatch decision turns on.
 *
 *  Levenshtein is compute-bound, Myers costing one word-step per query word per candidate byte.
 *  The candidates sit in unified tapes the host never touches again, so their pages migrate once,
 *  during the warm-up, and results come back through pinned memory. A warp's lanes that start
 *  together finish apart, so each device arm runs over the same views in corpus order and sorted
 *  by length, and over as many spans of the same tape with lengths log-uniform from one byte up,
 *  and the gaps between `:shuffled`, `:sorted` and `:skewed` measure how well a round balances
 *  them. Throughput is Cell Updates Per Second, as the CPU benchmark reports it.
 *
 *  Overlap is compute-bound too: every candidate byte costs a modular multiply-add per width and
 *  a B-tree descent, over candidates the unified dataset already makes device-reachable.
 *
 *  Substrings is memory-bound: every haystack byte is one data-dependent load into the automaton,
 *  and the device fills by haystack chunks rather than haystacks. A corpus that cuts into fewer
 *  chunks than one residency wave, or whose matches would not fit the device, is refused. Two rows
 *  stand apart: a small batch, where a round's chain of launches rather than its walk sets the
 *  rate, and BM25 over a few long documents, where one block per document would idle the device.
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_CUDA=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_FILTER=cuda build_release/stringzilla_bench
 *  @endcode
 */
#pragma once
#ifndef STRINGZILLA_BENCH_CROSS_SIMT_CUH
#define STRINGZILLA_BENCH_CROSS_SIMT_CUH

#include <cmath> // `std::ceil`, `std::log2`

#include <algorithm>        // `std::min`, `std::stable_sort`
#include <initializer_list> // `std::initializer_list`
#include <numeric>          // `std::iota`
#include <optional>         // `std::optional`
#include <span>             // `std::span`
#include <stdexcept>        // `std::runtime_error`
#include <string>           // `std::string`, `std::to_string`
#include <vector>           // `std::vector`

#include <fmt/format.h>

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp"
#include "cross.hpp" // Shared operation adapters

namespace ashvardanian::stringzilla::bench {

struct simt_backend_t {
    char const *name;
    device_backend_t const &runtime;
    sz_kernel_levenshtein_engine_init_t levenshtein_engine_init;
    sz_kernel_levenshtein_distances_t levenshtein_distances;
    sz_kernel_overlap_engine_init_t overlap_engine_init;
    sz_kernel_overlap_scores_t overlap_scores;
    sz_kernel_substrings_engine_init_t substrings_engine_init;
    sz_kernel_substrings_counts_t substrings_counts;
    sz_kernel_substrings_find_t substrings_find;
    sz_kernel_substrings_replace_t substrings_replace;
    sz_kernel_substrings_bm25_scores_t substrings_bm25_scores;
    sz_kernel_utf8_uncased_fold_t utf8_uncased_fold;
    sz_kernel_utf8_norm_t utf8_norm;
};

inline std::string simt_arm(simt_backend_t const &backend, char const *verb) {
    return std::string(verb) + "_" + backend.name + ":" + std::to_string(backend.runtime.selected.ordinal());
}

/**
 *  @brief @p count spans of @p text at seeded offsets, their lengths log-uniform from one byte to a
 *      ceiling whose mean is @p mean_bytes.
 *
 *  A warp of corpus lines differs in length by a few times; these differ by orders of magnitude
 *  while a round's bytes stay about what the corpus order walks, which is the skew a balanced round
 *  has to absorb.
 */
inline std::vector<sz_string_view_t> simt_skewed_views(char const *text, std::size_t bytes, std::size_t count,
                                                       double mean_bytes, std::uint64_t seed) {
    // A draw log-uniform over [1, L] has the mean (L - 1) / ln(L), which grows with L.
    double low = 2.0, high = 4.0;
    while ((high - 1.0) / std::log(high) < mean_bytes) high *= 2.0;
    for (int step = 0; step != 64; ++step) {
        double const middle = (low + high) / 2.0;
        ((middle - 1.0) / std::log(middle) < mean_bytes ? low : high) = middle;
    }
    double const ceiling = std::min(high, (double)bytes);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> exponent(0.0, std::log(ceiling));
    std::vector<sz_string_view_t> views(count);
    for (sz_string_view_t &view : views) {
        std::size_t const length = std::min((std::size_t)std::exp(exponent(generator)), bytes);
        view = {text + generator() % (bytes - length + 1), length};
    }
    return views;
}

#pragma region Levenshtein

/** Which ordering of the same candidates an arm scores. */
enum class levenshtein_simt_order_t {

    /** Corpus order, where a warp's thirty-two candidates have whatever lengths they had. */
    shuffled_k,

    /** Length descending, so a warp's candidates sit as close in length as the corpus allows. */
    sorted_k,

    /** Spans of the same tape, as many, with lengths log-uniform from one byte up. */
    skewed_k,
};

/** The label an arm's name carries for the ordering it scored. */
inline char const *levenshtein_simt_order_name(levenshtein_simt_order_t order) {
    return order == levenshtein_simt_order_t::sorted_k   ? ":sorted"
           : order == levenshtein_simt_order_t::skewed_k ? ":skewed"
                                                         : ":shuffled";
}

/** Host candidates and independent device tapes for each tested ordering. */
struct levenshtein_simt_corpus_t {

    /** Every candidate's bytes on the host, back to back, which the skewed spans point into. */
    std::vector<char> host_tape;

    /** The candidates in corpus order, in length descending order, and as skewed spans. */
    tape_t shuffled_tape, sorted_tape, skewed_tape;

    /** Dataset views the CPU arms read, corpus order. */
    std::vector<sz_string_view_t> host_views;

    /** The same dataset views, length descending. */
    std::vector<sz_string_view_t> host_sorted_views;

    /** The skewed spans of the host's tape. */
    std::vector<sz_string_view_t> host_skewed_views;

    levenshtein_simt_corpus_t(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus)
        : shuffled_tape(unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)),
          sorted_tape(unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)),
          skewed_tape(unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)) {
        std::size_t const count = std::min<std::size_t>(corpus.tokens.size(),
                                                        resident_candidates_per_call(env, backend.runtime));
        std::size_t bytes = 0;
        for (std::size_t index = 0; index != count; ++index) bytes += corpus.tokens[index].size();
        host_views.resize(count), host_sorted_views.resize(count);

        host_tape.reserve(bytes);
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = corpus.tokens[index];
            host_views[index] = {token.data(), token.size()};
            host_tape.insert(host_tape.end(), token.data(), token.data() + token.size());
        }

        // One permutation drives both sides, so the sorted arm's answers line up with its host
        // baseline's even where several candidates share a length.
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), (std::size_t)0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            return host_views[left].length > host_views[right].length;
        });
        for (std::size_t index = 0; index != count; ++index) host_sorted_views[index] = host_views[order[index]];

        double const mean_bytes = (double)bytes / (double)std::max<std::size_t>(count, 1);
        host_skewed_views = simt_skewed_views(host_tape.data(), bytes, count, mean_bytes, env.settings.seed.value);

        if (sz::failed(shuffled_tape.assign(host_views)))
            throw std::runtime_error("Unified memory could not hold the tape.");
        if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
        if (sz::failed(sorted_tape.assign(host_sorted_views)))
            throw std::runtime_error("Unified memory could not hold the tape.");
        if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
        if (sz::failed(skewed_tape.assign(host_skewed_views)))
            throw std::runtime_error("Unified memory could not hold the tape.");
        if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
    }

    /** Candidates one round scores, whichever order it walks them in. */
    std::size_t count() const { return host_views.size(); }

    std::size_t bytes(levenshtein_simt_order_t order) const {
        std::size_t total = 0;
        sz_sequence_t const candidates = host_candidates(order);
        for (sz_size_t index = 0; index != candidates.count; ++index)
            total += candidates.get_length(candidates.handle, index);
        return total;
    }

    /** The device accessors over one ordering of the texts. */
    sz_sequence_t device_candidates(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k   ? sorted_tape.sequence()
               : order == levenshtein_simt_order_t::skewed_k ? skewed_tape.sequence()
                                                             : shuffled_tape.sequence();
    }

    /** The host accessors over one ordering of the texts. */
    std::span<sz_string_view_t const> host_candidate_views(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k   ? std::span<sz_string_view_t const>(host_sorted_views)
               : order == levenshtein_simt_order_t::skewed_k ? std::span<sz_string_view_t const>(host_skewed_views)
                                                             : std::span<sz_string_view_t const>(host_views);
    }
    sz_sequence_t host_candidates(levenshtein_simt_order_t order) const {
        auto const views = host_candidate_views(order);
        sz_sequence_t result {};
        sz_sequence_from_string_views(views.data(), views.size(), &result);
        return result;
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
inline std::vector<sz_string_view_t> levenshtein_simt_queries(corpus_t const &corpus, std::size_t query_bytes) {
    std::vector<sz_string_view_t> views(4);
    for (std::size_t query = 0; query != views.size(); ++query) {
        token_view_t const whole = corpus.tokens[(query * 7 + 1) % corpus.tokens.size()];
        std::size_t const reachable = (std::size_t)(corpus.dataset.data() + corpus.dataset.size() - whole.data());
        views[query] = {whole.data(), std::min(query_bytes, reachable)};
    }
    return views;
}

/** The device arm alone across query widths and every order, for the word-count curve each way. */
inline void bench_levenshtein_word_counts(simt_backend_t const &backend, environment_t const &env,
                                          corpus_t const &corpus, levenshtein_simt_corpus_t &resident) {
    std::size_t const widths[] = {8, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384};
    levenshtein_simt_order_t const orders[] = {levenshtein_simt_order_t::shuffled_k, levenshtein_simt_order_t::sorted_k,
                                               levenshtein_simt_order_t::skewed_k};
    for (std::size_t index = 0; index != sizeof(widths) / sizeof(widths[0]); ++index)
        for (levenshtein_simt_order_t const order : orders) {
            std::size_t const query_bytes = widths[index];
            auto const queries = levenshtein_simt_queries(corpus, query_bytes);
            std::string const suffix = ":q" + std::to_string(query_bytes) + ":w" +
                                       std::to_string(sz_levenshtein_query_words(query_bytes)) +
                                       levenshtein_simt_order_name(order);
            print(bench_unary(
                env, corpus, simt_arm(backend, "sz_levenshtein_distances") + suffix,
                levenshtein_distances_from_sz {backend.levenshtein_engine_init, backend.levenshtein_distances, queries,
                                               resident.device_candidates(order), resident.host_candidate_views(order),
                                               sz_levenshtein_bytes_k, std::cref(backend.runtime)}));
        }
}

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
inline void bench_levenshtein_cross_product(simt_backend_t const &backend, environment_t const &env,
                                            corpus_t const &corpus, levenshtein_simt_corpus_t &resident,
                                            std::size_t query_bytes) {
    auto const queries = levenshtein_simt_queries(corpus, query_bytes);
    std::string const suffix = ":q" + std::to_string(query_bytes) + ":c" + std::to_string(resident.count());
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::string const shuffled_name = simt_arm(backend, "sz_levenshtein_distances") + suffix + ":shuffled",
                      sorted_name = simt_arm(backend, "sz_levenshtein_distances") + suffix + ":sorted",
                      skewed_name = simt_arm(backend, "sz_levenshtein_distances") + suffix + ":skewed",
                      utf8_name = simt_arm(backend, "sz_levenshtein_distances") + ":utf8" + suffix + ":shuffled";
    auto validator = levenshtein_distances_from_sz {sz_levenshtein_engine_init_serial,
                                                    sz_levenshtein_distances_serial,
                                                    queries,
                                                    resident.host_candidates(levenshtein_simt_order_t::shuffled_k),
                                                    resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
                                                    sz_levenshtein_bytes_k};
    auto dispatched = levenshtein_distances_from_sz {
        levenshtein_engine_init_cpu_,
        sz_levenshtein_distances,
        queries,
        resident.host_candidates(levenshtein_simt_order_t::shuffled_k),
        resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
        sz_levenshtein_bytes_k};
    std::optional<double> const base = bench_baseline(env, corpus, "sz_levenshtein_distances" + device_suffix + suffix,
                                                      {shuffled_name, sorted_name, skewed_name, utf8_name}, dispatched);
    // The warped rung spreads a query across a warp's thirty-two lanes, and a wider one is
    // reported as out of reach rather than thrown.
    if (sz_levenshtein_query_words(query_bytes) > sz_levenshtein_gpu_words_max_k) {
        fmt::println("Skipping `{}{}`: {} words past the device's {}.", simt_arm(backend, "sz_levenshtein_distances"),
                     suffix, sz_levenshtein_query_words(query_bytes), (int)sz_levenshtein_gpu_words_max_k);
        return;
    }
    // Both orderings hold the same texts, so the gap between the arms is the warp's `max(L)` tax,
    // and the skewed spans raise that tax by orders of magnitude.
    print(bench_unary(
              env, corpus, shuffled_name, validator,
              levenshtein_distances_from_sz {backend.levenshtein_engine_init, backend.levenshtein_distances, queries,
                                             resident.device_candidates(levenshtein_simt_order_t::shuffled_k),
                                             resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
                                             sz_levenshtein_bytes_k, std::cref(backend.runtime)}),
          base);
    auto validator_sorted = levenshtein_distances_from_sz {
        sz_levenshtein_engine_init_serial,
        sz_levenshtein_distances_serial,
        queries,
        resident.host_candidates(levenshtein_simt_order_t::sorted_k),
        resident.host_candidate_views(levenshtein_simt_order_t::sorted_k),
        sz_levenshtein_bytes_k};
    print(bench_unary(
              env, corpus, sorted_name, validator_sorted,
              levenshtein_distances_from_sz {backend.levenshtein_engine_init, backend.levenshtein_distances, queries,
                                             resident.device_candidates(levenshtein_simt_order_t::sorted_k),
                                             resident.host_candidate_views(levenshtein_simt_order_t::sorted_k),
                                             sz_levenshtein_bytes_k, std::cref(backend.runtime)}),
          base);
    auto validator_skewed = levenshtein_distances_from_sz {
        sz_levenshtein_engine_init_serial,
        sz_levenshtein_distances_serial,
        queries,
        resident.host_candidates(levenshtein_simt_order_t::skewed_k),
        resident.host_candidate_views(levenshtein_simt_order_t::skewed_k),
        sz_levenshtein_bytes_k};
    print(bench_unary(
              env, corpus, skewed_name, validator_skewed,
              levenshtein_distances_from_sz {backend.levenshtein_engine_init, backend.levenshtein_distances, queries,
                                             resident.device_candidates(levenshtein_simt_order_t::skewed_k),
                                             resident.host_candidate_views(levenshtein_simt_order_t::skewed_k),
                                             sz_levenshtein_bytes_k, std::cref(backend.runtime)}),
          base);

    // A window of this many bytes holds at most as many runes, so the byte guard above covers the
    // rune arm. Cells count bytes on both sides, as the CPU arms do, so the rune rows compare.
    auto validator_utf8 = levenshtein_distances_from_sz {
        sz_levenshtein_engine_init_serial,
        sz_levenshtein_distances_serial,
        queries,
        resident.host_candidates(levenshtein_simt_order_t::shuffled_k),
        resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
        sz_levenshtein_runes_k};
    auto dispatched_utf8 = levenshtein_distances_from_sz {
        levenshtein_engine_init_cpu_,
        sz_levenshtein_distances,
        queries,
        resident.host_candidates(levenshtein_simt_order_t::shuffled_k),
        resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
        sz_levenshtein_runes_k};
    std::optional<double> const base_utf8 = bench_baseline(
        env, corpus, "sz_levenshtein_distances" + device_suffix + ":utf8" + suffix, {utf8_name}, dispatched_utf8, base);
    print(bench_unary(
              env, corpus, utf8_name, validator_utf8,
              levenshtein_distances_from_sz {backend.levenshtein_engine_init, backend.levenshtein_distances, queries,
                                             resident.device_candidates(levenshtein_simt_order_t::shuffled_k),
                                             resident.host_candidate_views(levenshtein_simt_order_t::shuffled_k),
                                             sz_levenshtein_runes_k, std::cref(backend.runtime)}),
          base_utf8);
}

/** Every Levenshtein row over @p corpus, kept resident. */
inline void bench_levenshtein_simt(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus) {
    levenshtein_simt_corpus_t resident(backend, env, corpus);
    fmt::println("Starting Levenshtein benchmarks over {} resident candidates...", resident.count());
    bench_levenshtein_cross_product(backend, env, corpus, resident, median_token_bytes(corpus));
    bench_levenshtein_word_counts(backend, env, corpus, resident);
}

#pragma endregion Levenshtein

#pragma region Overlap

/** The leading tokens of @p corpus, one per resident thread of the bound device. */
inline std::vector<sz_string_view_t> overlap_token_views(simt_backend_t const &backend, environment_t const &env,
                                                         corpus_t const &corpus) {
    std::size_t const count = std::min<std::size_t>(corpus.tokens.size(),
                                                    resident_candidates_per_call(env, backend.runtime));
    std::vector<sz_string_view_t> views(count);
    for (std::size_t index = 0; index != count; ++index)
        views[index] = {corpus.tokens[index].data(), corpus.tokens[index].size()};
    return views;
}

/** Every arm at one query width, the width and the corpus @p label carried in each arm's name. */
inline void bench_overlap_scores(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                 overlap_corpus_t &resident, std::size_t query_bytes, char const *label) {
    std::size_t const width = overlap_width_(corpus, query_bytes);
    auto const query = std::string_view(corpus.tokens[0]).substr(0, query_bytes);
    std::string const suffix = ":w" + std::to_string(width) + label;
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::string const device_name = simt_arm(backend, "sz_overlap_scores") + suffix;
    auto validator = scores_from_sz {sz_overlap_engine_init_serial,
                                     sz_overlap_scores_serial,
                                     query,
                                     width,
                                     resident.host_candidates(),
                                     resident.views};
    auto dispatched = scores_from_sz {overlap_engine_init_cpu_,   sz_overlap_scores, query, width,
                                      resident.host_candidates(), resident.views};
    std::optional<double> const base = bench_baseline(env, corpus, "sz_overlap_scores" + device_suffix + suffix,
                                                      {device_name}, dispatched);
    print(bench_unary(env, corpus, device_name, validator,
                      scores_from_sz {backend.overlap_engine_init, backend.overlap_scores, query, width,
                                      resident.candidates.sequence(), resident.views, std::cref(backend.runtime)}),
          base);
}

/** Every overlap row over the leading tokens of @p corpus, then over as many skewed spans of
 *  the same bytes. */
inline void bench_overlap_simt(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus) {
    overlap_corpus_t resident(overlap_token_views(backend, env, corpus), backend.runtime);
    fmt::println("Starting window overlap benchmarks over {} resident candidates...", resident.candidates.size());
    bench_overlap_scores(backend, env, corpus, resident, median_token_bytes(corpus), "");
    std::size_t const bytes = resident.bytes();
    overlap_corpus_t skewed(
        simt_skewed_views(corpus.dataset.data(), corpus.dataset.size(), resident.candidates.size(),
                          (double)bytes / (double)resident.candidates.size(), env.settings.seed.value),
        backend.runtime);
    bench_overlap_scores(backend, env, corpus, skewed, median_token_bytes(corpus), ":skewed");
}

#pragma endregion Overlap

#pragma region Substrings Residency

/**
 *  @brief Whether the corpus cuts into at least one chunk per thread of @p resident_threads.
 *
 *  The width is reproduced here rather than read back, because the device derives it from the
 *  corpus total the same way: at least the corpus over the budget, and never under the warm-up a
 *  chunk has to pay. The budget may hold several chunks per resident thread, so the wave is
 *  measured against the device's own residency rather than against the budget.
 */
inline bool substrings_fills_a_wave(sz_substrings_engine_t const &engine, substrings_corpus_t const &resident,
                                    std::size_t resident_threads) {
    std::size_t const budget = engine.chunk_budget ? engine.chunk_budget : 1;
    std::size_t const floor_bytes = std::max<std::size_t>(4 * engine.max_source_match_bytes, 1);
    std::size_t const bytes = resident.bytes();
    std::size_t const chunk = std::max(sz::divide_round_up(bytes, budget), floor_bytes);
    std::size_t chunks = 0;
    for (sz_string_view_t const &view : resident.views)
        chunks += view.length == 0 ? 1 : sz::divide_round_up(view.length, chunk);

    double const waves = (double)chunks / (double)resident_threads;
    fmt::println("> Corpus: {:.1f} MB in {} haystacks, cut into {} chunks of {} B " //
                 "against {} resident threads - {:.2f} waves",                      //
                 (double)bytes / (1 << 20), resident.views.size(), chunks, chunk, resident_threads, waves);
    if (waves >= 1.0) return true;
    fmt::println("> Refusing the round: below one wave it times the launch, not the walk. " //
                 "Raise STRINGWARS_BYTES.");
    return false;
}

/** Overlapping matches one round of @p engine emits over @p resident, counted by a sizing round. */
inline std::size_t substrings_matches_emitted(simt_backend_t const &backend, sz_substrings_engine_t &engine,
                                              substrings_corpus_t const &resident,
                                              sz_sequence_t const &device_haystacks) {
    unified_vector<sz_size_t> offsets(resident.views.size() + 1, 0,
                                      unified_alloc<sz_size_t>(backend.runtime.unified, backend.runtime.stream));
    if (backend.substrings_find(&engine, &device_haystacks, nullptr, 0, offsets.data(), backend.runtime.stream) !=
        sz_success_k)
        throw std::runtime_error("The sizing round was refused.");
    if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
        throw std::runtime_error("The sizing round failed.");
    return engine.report->matches_emitted;
}

/** Whether every match of a round fits @p engine's budget, and its arena fits the device. */
inline bool substrings_fits_the_device(simt_backend_t const &backend, sz_substrings_engine_t const &engine,
                                       std::size_t emitted) {
    std::size_t const free_bytes = backend.runtime.free_bytes(backend.runtime.stream);
    if (free_bytes == 0) throw std::runtime_error("The device would not report its memory.");
    fmt::println("> Matches: {} against a budget of {}, in a {:.1f} GB arena and {:.1f} GB free", emitted,
                 (std::size_t)engine.matches_budget, (double)engine.scratch_bytes / (1 << 30),
                 (double)free_bytes / (1 << 30));
    if (emitted <= engine.matches_budget && engine.scratch_bytes <= free_bytes / 2) return true;
    fmt::println("> Refusing the round: its matches outgrow half the device's free memory. Lower STRINGWARS_BYTES.");
    return false;
}

#pragma endregion Substrings Residency

#pragma region Substrings Verbs

/** Per-haystack counts, the device arm logged against the CPU one. */
inline void bench_substrings_counts(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                    substrings_engine_t &host, substrings_engine_t &device,
                                    substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                    std::string const &suffix) {
    auto dispatched = substrings_counts_from_sz {sz_substrings_counts, host, resident, resident.haystacks()};
    std::string const device_name = simt_arm(backend, "sz_substrings_counts") + suffix;
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_counts" + device_suffix + suffix,
                                                      {device_name}, dispatched);
    print(bench_unary(env, corpus, device_name, dispatched,
                      substrings_counts_from_sz {backend.substrings_counts, device, resident, device_haystacks,
                                                 std::cref(backend.runtime)}),
          base);
}

/** Every match, the device arm logged against the CPU one. */
inline void bench_substrings_find(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                  substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto dispatched = substrings_find_from_sz {sz_substrings_find, host, resident, resident.haystacks()};
    std::string const device_name = simt_arm(backend, "sz_substrings_find") + suffix;
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_find" + device_suffix + suffix,
                                                      {device_name}, dispatched);
    print(bench_unary(env, corpus, device_name, dispatched,
                      substrings_find_from_sz {backend.substrings_find, device, resident, device_haystacks}),
          base);
}

/** The rewrite, the device arm logged against the CPU one. */
inline void bench_substrings_replace(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                     substrings_engine_t &host, substrings_engine_t &device,
                                     substrings_dictionary_t const &dictionary, substrings_corpus_t const &resident,
                                     sz_sequence_t const &device_haystacks, sz_sequence_t const &device_replacements,
                                     std::string const &suffix) {
    auto dispatched = substrings_replace_from_sz {sz_substrings_replace, host, resident, resident.haystacks(),
                                                  dictionary.replacements()};
    std::string const device_name = simt_arm(backend, "sz_substrings_replace") + suffix;
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_replace" + device_suffix + suffix,
                                                      {device_name}, dispatched);
    print(bench_unary(env, corpus, device_name, dispatched,
                      substrings_replace_from_sz {backend.substrings_replace, device, resident, device_haystacks,
                                                  device_replacements}),
          base);
}

/** BM25 scores, the device arm logged against the CPU one. */
inline void bench_substrings_bm25(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                  substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto dispatched = substrings_bm25_from_sz {sz_substrings_bm25_scores, host, resident, resident.haystacks()};
    std::string const device_name = simt_arm(backend, "sz_substrings_bm25_scores") + suffix;
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_bm25_scores" + device_suffix + suffix,
                                                      {device_name}, dispatched);
    print(bench_unary(env, corpus, device_name, dispatched,
                      substrings_bm25_from_sz {backend.substrings_bm25_scores, device, resident, device_haystacks,
                                               std::cref(backend.runtime)}),
          base);
}

/** One vocabulary slice walked by each verb under each accepted policy, once the round is sound. */
inline void bench_substrings_slice(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus,
                                   substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                   std::optional<substrings_ranking_t> &ranking, substrings_slice_t slice,
                                   sz_substrings_case_sensitivity_t sensitivity) {
    std::string const suffix = substrings_label(slice, sensitivity);
    if (!substrings_selects(env, simt_arm(backend, ""), suffix)) return;
    sz_allocator_t const allocator = backend.runtime.unified;
    substrings_dictionary_t dictionary(substrings_needles(env, corpus, ranking, slice), sensitivity, allocator,
                                       backend.runtime.stream);
    std::size_t matches_budget = 0;
    sz_allocator_t heap;
    if (sz_allocator_init_heap(&heap) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    if (dictionary.needles.empty()) {
        fmt::println("Vocabulary {} is empty on this corpus, skipping it.", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, backend.substrings_engine_init, allocator,
                                  matches_budget, resident.views.size(), backend.runtime.stream);
        fmt::println("Vocabulary {} holds {} needles over {} states, {} of them hot.", suffix.c_str(),
                     dictionary.needles.size(), probe.engine.state_count, probe.engine.hot_count);
        if (!substrings_fills_a_wave(probe.engine, resident)) return;
        // A benchmark round must keep every match it finds, whatever the default budget.
        std::size_t const emitted = substrings_matches_emitted(backend, probe.engine, resident, device_haystacks);
        if (emitted > probe.engine.matches_budget) matches_budget = emitted;
        substrings_engine_t sized(dictionary, sz_substrings_overlapping_k, backend.substrings_engine_init, allocator,
                                  matches_budget, resident.views.size(), backend.runtime.stream);
        if (!substrings_fits_the_device(backend, sized.engine, emitted)) return;
    }

    tape_t replacements_tape {unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)};
    if (sz::failed(replacements_tape.assign(dictionary.replacements())))
        throw std::runtime_error("Unified memory could not hold the tape.");
    sz_sequence_t const device_replacements = replacements_tape.sequence();
    if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
        throw std::runtime_error("The tape would not reach the device.");
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_engine_init_cpu_, heap);
        substrings_engine_t device(dictionary, policy, backend.substrings_engine_init, allocator, matches_budget,
                                   resident.views.size(), backend.runtime.stream);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(backend, env, corpus, host, device, resident, device_haystacks, cover);
        bench_substrings_find(backend, env, corpus, host, device, resident, device_haystacks, cover);
    }
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_engine_init_cpu_, heap);
        substrings_engine_t device(dictionary, policy, backend.substrings_engine_init, allocator, matches_budget,
                                   resident.views.size(), backend.runtime.stream);
        bench_substrings_replace(backend, env, corpus, host, device, dictionary, resident, device_haystacks,
                                 device_replacements, suffix + substrings_policy_name(policy));
    }
    {
        substrings_engine_t host(dictionary, sz_substrings_overlapping_k, substrings_engine_init_cpu_, heap);
        substrings_engine_t device(dictionary, sz_substrings_overlapping_k, backend.substrings_engine_init, allocator,
                                   matches_budget, resident.views.size(), backend.runtime.stream);
        bench_substrings_bm25(backend, env, corpus, host, device, resident, device_haystacks, suffix);
    }
}

/** Haystacks a small-batch round walks: too few to fill the device, so its chain of launches is
 *  most of what the round costs. */
enum { substrings_small_batch_haystacks_k = 64 };

/** Counting and finding over the leading few haystacks, where the gaps between a round's launches
 *  rather than its walk set the rate. */
inline void bench_substrings_small_batch(simt_backend_t const &backend, environment_t const &env,
                                         corpus_t const &corpus) {
    sz_allocator_t heap;
    if (sz_allocator_init_heap(&heap) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    std::size_t matches_budget = 0;
    sz_allocator_t const allocator = backend.runtime.unified;
    substrings_corpus_t const resident(corpus, substrings_small_batch_haystacks_k);
    std::string const suffix = substrings_label(substrings_slice_t::sampled_k, sz_substrings_cased_k) + ":h" +
                               std::to_string(resident.views.size());
    if (!substrings_selects(env, simt_arm(backend, ""), suffix)) return;
    substrings_dictionary_t const dictionary(substrings_sampled(env, corpus), sz_substrings_cased_k, allocator,
                                             backend.runtime.stream);
    if (dictionary.needles.empty()) return;
    tape_t haystacks_tape {unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)};
    if (sz::failed(haystacks_tape.assign(resident.haystacks())))
        throw std::runtime_error("Unified memory could not hold the tape.");
    sz_sequence_t const device_haystacks = haystacks_tape.sequence();
    if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
        throw std::runtime_error("The tape would not reach the device.");
    for (sz_substrings_overlap_policy_t const policy :
         {sz_substrings_overlapping_k, sz_substrings_leftmost_longest_k}) {
        substrings_engine_t host(dictionary, policy, substrings_engine_init_cpu_, heap);
        substrings_engine_t device(dictionary, policy, backend.substrings_engine_init, allocator, matches_budget,
                                   resident.views.size(), backend.runtime.stream);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(backend, env, corpus, host, device, resident, device_haystacks, cover);
        bench_substrings_find(backend, env, corpus, host, device, resident, device_haystacks, cover);
    }
}

/** Documents a long-document round cuts the corpus into: fewer than the device has multiprocessors,
 *  so one block per document would leave most of it idle. */
enum { substrings_documents_k = 8 };

/** BM25 over the corpus cut into a few long documents, the device arm against the CPU one. */
inline void bench_substrings_documents(simt_backend_t const &backend, environment_t const &env,
                                       corpus_t const &corpus) {
    sz_allocator_t heap;
    if (sz_allocator_init_heap(&heap) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    std::size_t matches_budget = 0;
    sz_allocator_t const allocator = backend.runtime.unified;
    std::string const suffix = substrings_label(substrings_slice_t::sampled_k, sz_substrings_cased_k) + ":d" +
                               std::to_string((std::size_t)substrings_documents_k);
    if (!substrings_selects(env, simt_arm(backend, ""), suffix)) return;
    substrings_dictionary_t const dictionary(substrings_sampled(env, corpus), sz_substrings_cased_k, allocator,
                                             backend.runtime.stream);
    if (dictionary.needles.empty()) return;
    char const *const first = corpus.tokens.front().data();
    std::size_t const bytes = (std::size_t)(corpus.tokens.back().data() + corpus.tokens.back().size() - first);
    std::vector<sz_string_view_t> documents(substrings_documents_k);
    for (std::size_t index = 0; index != documents.size(); ++index) {
        std::size_t const begin = bytes * index / documents.size(), end = bytes * (index + 1) / documents.size();
        documents[index] = {first + begin, end - begin};
    }
    substrings_corpus_t const resident(documents);
    tape_t haystacks_tape {unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)};
    if (sz::failed(haystacks_tape.assign(resident.haystacks())))
        throw std::runtime_error("Unified memory could not hold the tape.");
    sz_sequence_t const device_haystacks = haystacks_tape.sequence();
    if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
        throw std::runtime_error("The tape would not reach the device.");
    substrings_engine_t host(dictionary, sz_substrings_overlapping_k, substrings_engine_init_cpu_, heap);
    substrings_engine_t device(dictionary, sz_substrings_overlapping_k, backend.substrings_engine_init, allocator,
                               matches_budget, resident.views.size(), backend.runtime.stream);
    bench_substrings_bm25(backend, env, corpus, host, device, resident, device_haystacks, suffix);
}

/** Every substrings row over @p corpus: each vocabulary slice over the whole corpus kept resident,
 *  then the small batch and the few long documents. */
inline void bench_substrings_simt(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus) {
    std::pair<substrings_slice_t, sz_substrings_case_sensitivity_t> const slices[] = {
        {substrings_slice_t::frequent_k, sz_substrings_cased_k},
        {substrings_slice_t::rare_k, sz_substrings_cased_k},
        {substrings_slice_t::frequent_k, sz_substrings_uncased_k},
        {substrings_slice_t::sampled_k, sz_substrings_cased_k},
    };
    // The whole corpus crosses into a resident tape only for a slice the filter keeps.
    if (std::any_of(std::begin(slices), std::end(slices), [&](auto const &slice) {
            return substrings_selects(env, simt_arm(backend, ""), substrings_label(slice.first, slice.second));
        })) {
        substrings_corpus_t const resident(corpus);
        tape_t haystacks_tape {unified_alloc<char>(backend.runtime.unified, backend.runtime.stream)};
        if (sz::failed(haystacks_tape.assign(resident.haystacks())))
            throw std::runtime_error("Unified memory could not hold the tape.");
        sz_sequence_t const device_haystacks = haystacks_tape.sequence();
        if (sz_stream_synchronize_best(backend.runtime.capabilities, backend.runtime.stream) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
        fmt::println("Starting multi-pattern search benchmarks...");
        std::optional<substrings_ranking_t> ranking;
        for (auto const &[slice, sensitivity] : slices)
            bench_substrings_slice(backend, env, corpus, resident, device_haystacks, ranking, slice, sensitivity);
    }
    bench_substrings_small_batch(backend, env, corpus);
    bench_substrings_documents(backend, env, corpus);
}

#pragma endregion Substrings Verbs

#pragma region UTF8 Norm and Fold

/** Case folding and NFC normalization of every token, each device arm against the dispatch. */
inline void bench_utf8_norm_and_fold(simt_backend_t const &backend, environment_t const &env, corpus_t const &corpus) {
    std::string const device_suffix = std::string(":") + backend.name + ":" +
                                      std::to_string(backend.runtime.selected.ordinal());
    {
        auto dispatched = utf8_uncased_fold_from_sz {cpu_best<sz_utf8_uncased_fold_best>, corpus};
        std::string const device_name = simt_arm(backend, "sz_utf8_uncased_fold");
        std::optional<double> const base = bench_baseline(env, corpus, "sz_utf8_uncased_fold_best" + device_suffix,
                                                          {device_name}, dispatched);
        print(bench_unary(env, corpus, device_name, dispatched,
                          utf8_uncased_fold_from_sz {backend.utf8_uncased_fold, corpus, std::cref(backend.runtime)}),
              base);
    }
    {
        auto dispatched = utf8_norm_from_sz {cpu_best<sz_utf8_norm_best>, corpus};
        std::string const device_name = simt_arm(backend, "sz_utf8_norm");
        std::optional<double> const base = bench_baseline(env, corpus, "sz_utf8_norm_best" + device_suffix,
                                                          {device_name}, dispatched);
        print(bench_unary(env, corpus, device_name, dispatched,
                          utf8_norm_from_sz {backend.utf8_norm, corpus, std::cref(backend.runtime)}),
              base);
    }
}

#pragma endregion UTF8 Norm and Fold

#pragma region Drivers

/** Every family's rows over the multilingual lines, each on a corpus it keeps resident; a build
 *  host without a device logs as much and passes. */
inline int bench_cross_simt(environment_t &env, simt_backend_t const &backend) {
    try {
        corpus_t const &corpus = env.corpora.multilingual_lines();
        bench_levenshtein_simt(backend, env, corpus);
        bench_overlap_simt(backend, env, corpus);
        bench_substrings_simt(backend, env, corpus);
        fmt::println("Starting UTF-8 normalization and case folding benchmarks...");
        bench_utf8_norm_and_fold(backend, env, env.corpora.multilingual_slice());
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    return 0;
}

/** The substrings rows of a later CUDA tier, which carries no other family's kernels. */
inline int bench_cross_simt_substrings(environment_t &env, simt_backend_t const &backend) {
    try {
        bench_substrings_simt(backend, env, env.corpora.multilingual_lines());
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    return 0;
}

/** The Levenshtein and overlap rows of a later CUDA tier, which carries no other family's
 *  kernels. */
inline int bench_cross_simt_levenshtein_overlap(environment_t &env, simt_backend_t const &backend) {
    try {
        corpus_t const &corpus = env.corpora.multilingual_lines();
        bench_levenshtein_simt(backend, env, corpus);
        bench_overlap_simt(backend, env, corpus);
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    return 0;
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::bench

#endif // STRINGZILLA_BENCH_CROSS_SIMT_CUH
