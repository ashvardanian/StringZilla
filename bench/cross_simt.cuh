/**
 *  @file bench/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the rows CUDA and ROCm share, each device arm against the CPU
 *      tiers this build carries.
 *
 *  Included by one translation unit per binary, `cross_cuda.cu` or `cross_rocm.hip`, which picks
 *  its vendor's kernels by the runtime it is compiled for. HIP answers the CUDA runtime calls the
 *  rows and the harness make, under the names mapped below.
 *
 *  Every row times a device-resident round, as that is the regime the GPU backends exist for:
 *  scoring a handful of candidates per call would time the launch instead. Each engine is built
 *  before the timing on both sides, as it is meant to be used - one batch of queries or needles,
 *  many rounds against it - so only the round is timed. There is no Standard row: the platforms
 *  ship no GPU edit distance, window overlap or multi-pattern search, so the baseline is the widest
 *  CPU tier, which is the comparison a dispatch decision actually turns on.
 *
 *  Levenshtein is compute-bound, Myers costing one word-step per query word per candidate byte.
 *  The candidates cross once into plain device memory, as managed pages follow whoever touched
 *  them last and a short-candidate round would time migration. One candidate per thread means a
 *  warp costs the longest of its thirty-two, so each device arm runs twice over the same views,
 *  in corpus order and sorted by length, and the gap between `:shuffled` and `:sorted` measures
 *  that tax. Throughput is Cell Updates Per Second, as the CPU benchmark reports it.
 *
 *  Overlap is compute-bound too: every candidate byte costs a modular multiply-add per width and
 *  a B-tree descent, over candidates the unified dataset already makes device-reachable.
 *
 *  Substrings is memory-bound: every haystack byte is one data-dependent load into the automaton,
 *  and the device fills by haystack chunks rather than haystacks. A corpus that cuts into fewer
 *  chunks than one residency wave, or whose matches would not fit the device, is refused.
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
 *  cmake --build build_release --config Release --target stringzilla_cuda_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_FILTER=cuda build_release/stringzilla_cuda_bench
 *  @endcode
 */
#pragma once
#ifndef STRINGZILLA_BENCH_CROSS_SIMT_CUH
#define STRINGZILLA_BENCH_CROSS_SIMT_CUH

#include <algorithm> // `std::min`, `std::stable_sort`
#include <cmath>     // `std::ceil`, `std::log2`
#include <numeric>   // `std::iota`
#include <stdexcept> // `std::runtime_error`
#include <string>    // `std::string`, `std::to_string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include <stringzilla/stringzilla.h> // Primary C API

#include "harness.hpp"
#include "substrings.cuh" // `substrings_dictionary_t`, `substrings_counts_from_sz`

namespace ashvardanian::stringzilla::bench {

/*  The kernels and device exports of the vendor this translation unit is compiled for. */
#if STRINGZILLA_ARCH_ROCM_
inline constexpr char const *simt_vendor_k = "rocm";
inline constexpr sz_kernel_levenshtein_distances_t simt_levenshtein_distances_k = &sz_levenshtein_distances_rocm;
inline constexpr sz_kernel_overlap_scores_t simt_overlap_scores_k = &sz_overlap_scores_rocm;
inline constexpr sz_kernel_substrings_counts_t simt_substrings_counts_k = &sz_substrings_counts_rocm;
inline constexpr sz_kernel_substrings_find_t simt_substrings_find_k = &sz_substrings_find_rocm;
inline constexpr sz_kernel_substrings_replace_t simt_substrings_replace_k = &sz_substrings_replace_rocm;
inline constexpr sz_kernel_substrings_bm25_scores_t simt_substrings_bm25_scores_k = &sz_substrings_bm25_scores_rocm;
inline constexpr auto simt_sequence_from_string_views = &sz_rocm_sequence_from_string_views;
inline constexpr auto simt_memory_allocator_init_unified = &sz_rocm_memory_allocator_init_unified;
inline constexpr sz::device_kind_t simt_device_kind_k = sz::device_kind_t::rocm_k;
#else
inline constexpr char const *simt_vendor_k = "cuda";
inline constexpr sz_kernel_levenshtein_distances_t simt_levenshtein_distances_k = &sz_levenshtein_distances_cuda;
inline constexpr sz_kernel_overlap_scores_t simt_overlap_scores_k = &sz_overlap_scores_cuda;
inline constexpr sz_kernel_substrings_counts_t simt_substrings_counts_k = &sz_substrings_counts_cuda;
inline constexpr sz_kernel_substrings_find_t simt_substrings_find_k = &sz_substrings_find_cuda;
inline constexpr sz_kernel_substrings_replace_t simt_substrings_replace_k = &sz_substrings_replace_cuda;
inline constexpr sz_kernel_substrings_bm25_scores_t simt_substrings_bm25_scores_k = &sz_substrings_bm25_scores_cuda;
inline constexpr auto simt_sequence_from_string_views = &sz_cuda_sequence_from_string_views;
inline constexpr auto simt_memory_allocator_init_unified = &sz_cuda_memory_allocator_init_unified;
inline constexpr sz::device_kind_t simt_device_kind_k = sz::device_kind_t::cuda_k;
#endif

/** What device 0 of this vendor enables, or zero without a device, which the engines refuse. */
inline sz_capability_t simt_capabilities() noexcept {
    auto const [device, make_status] = sz::device_t::make(simt_device_kind_k, 0);
    return sz::succeeded(make_status) ? device.capabilities_enabled().value : 0;
}

/** The name a row carries for this vendor's kernel of @p verb, like @c sz_overlap_scores_cuda. */
inline std::string simt_arm(char const *verb) { return std::string(verb) + "_" + simt_vendor_k; }

#pragma region Levenshtein

/** Which ordering of the same candidates an arm scores. */
enum class levenshtein_simt_order_t {

    /** Corpus order, where a warp's thirty-two candidates have whatever lengths they had. */
    shuffled_k,

    /** Length descending, so a warp's candidates sit as close in length as the corpus allows. */
    sorted_k,
};

/** The label an arm's name carries for the ordering it scored. */
static char const *levenshtein_simt_order_name(levenshtein_simt_order_t order) {
    return order == levenshtein_simt_order_t::sorted_k ? ":sorted" : ":shuffled";
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
inline check_value_t levenshtein_check_value(answers_type_ const &answers) {
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
struct levenshtein_simt_corpus_t {

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

    levenshtein_simt_corpus_t(environment_t const &env) {
        std::size_t const count = std::min<std::size_t>(env.tokens.size(), resident_candidates_per_call(env));
        for (std::size_t index = 0; index != count; ++index) bytes += env.tokens[index].size();
        if (tape.resize_uninitialized(bytes) != sz::status_t::success_k ||
            device_views.resize_uninitialized(count) != sz::status_t::success_k ||
            device_sorted_views.resize_uninitialized(count) != sz::status_t::success_k ||
            distances.resize_uninitialized(count * levenshtein_queries_per_batch_k) != sz::status_t::success_k)
            throw std::runtime_error("The device would not hold the corpus.");
        answers.resize(count * levenshtein_queries_per_batch_k);
        host_views.resize(count), host_sorted_views.resize(count);

        // Staged once on the host, so the candidates cross the bus as a single block.
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

        // One permutation drives both sides, so the sorted arm's answers line up with its host
        // baseline's even where several candidates share a length.
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), (std::size_t)0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
            return host_views[left].length > host_views[right].length;
        });
        std::vector<sz_string_view_t> tape_sorted_views(count);
        for (std::size_t index = 0; index != count; ++index)
            host_sorted_views[index] = host_views[order[index]], tape_sorted_views[index] = tape_views[order[index]];

        if (copy_host_to_device(sz::span<char const> {staged.data(), staged.size()}, tape) != cudaSuccess ||
            copy_host_to_device(sz::span<sz_string_view_t const> {tape_views.data(), count}, device_views) !=
                cudaSuccess ||
            copy_host_to_device(sz::span<sz_string_view_t const> {tape_sorted_views.data(), count},
                                device_sorted_views) != cudaSuccess)
            throw std::runtime_error("The corpus would not upload.");

        if (simt_sequence_from_string_views(device_views.data(), count, &device_shuffled) != sz_success_k ||
            simt_sequence_from_string_views(device_sorted_views.data(), count, &device_sorted) != sz_success_k)
            throw std::runtime_error("The device accessors could not be bound.");
        sz_sequence_from_string_views(host_views.data(), count, &host_shuffled);
        sz_sequence_from_string_views(host_sorted_views.data(), count, &host_sorted);
    }

    /** Candidates one round scores, whichever order it walks them in. */
    std::size_t count() const { return host_views.size(); }

    /** The device accessors over one ordering of the same texts. */
    sz_sequence_t const &device_candidates(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k ? device_sorted : device_shuffled;
    }

    /** The host accessors over one ordering of the same texts. */
    sz_sequence_t const &host_candidates(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k ? host_sorted : host_shuffled;
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
static std::vector<sz_string_view_t> levenshtein_simt_queries(environment_t const &env, std::size_t query_bytes) {
    std::vector<sz_string_view_t> views(levenshtein_queries_per_batch_k);
    for (std::size_t query = 0; query != views.size(); ++query) {
        token_view_t const whole = env.tokens[(query * 7 + 1) % env.tokens.size()];
        std::size_t const reachable = (std::size_t)(env.dataset.data() + env.dataset.size() - whole.data());
        views[query] = {whole.data(), std::min(query_bytes, reachable)};
    }
    return views;
}

/** One prepared batch and the residency it was built for, released with the scope that named it. */
struct levenshtein_simt_batch_t {

    /** The windows the batch was prepared from. */
    std::vector<sz_string_view_t> views;

    /** Host accessors over them, as both inits require. */
    sz_sequence_t queries {};

    /** The batch, prepared once and reused by every round. */
    sz_levenshtein_engine_t engine {};

    levenshtein_simt_batch_t(environment_t const &env, std::size_t query_bytes, sz_levenshtein_symbol_t symbol,
                             sz_bool_t on_device)
        : views(levenshtein_simt_queries(env, query_bytes)) {
        sz_sequence_from_string_views(views.data(), views.size(), &queries);
        sz_capability_t const capabilities = on_device == sz_true_k ? simt_capabilities() : sz::default_capabilities();
        if (sz_levenshtein_engine_init(&engine, &queries, symbol, capabilities, 0, STRINGZILLA_NULL,
                                       STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The engine could not be prepared.");
    }
    ~levenshtein_simt_batch_t() { sz_levenshtein_engine_free(&engine); }
    levenshtein_simt_batch_t(levenshtein_simt_batch_t const &) = delete;
    levenshtein_simt_batch_t &operator=(levenshtein_simt_batch_t const &) = delete;

    /** Symbols the batch spans together: the row count of the cell budget an arm reports. */
    std::size_t symbols() const {
        std::size_t total = 0;
        for (sz_string_view_t const &view : views) total += view.length;
        return total;
    }
};

/** Scores the resident corpus against a prepared batch, entirely on the device. */
template <sz_kernel_levenshtein_distances_t distances_>
struct levenshtein_distances_from_simt {

    /** The candidates, and the room for their distances. */
    levenshtein_simt_corpus_t &corpus;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_simt_order_t order;

    /** The queries, prepared on the device once. */
    levenshtein_simt_batch_t batch;

    levenshtein_distances_from_simt(environment_t const &env, levenshtein_simt_corpus_t &corpus,
                                    std::size_t query_bytes, levenshtein_simt_order_t order,
                                    sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : corpus(corpus), order(order), batch(env, query_bytes, symbol, sz_true_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &corpus.device_candidates(order), corpus.distances.data(), corpus.count(),
                       STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        // Device memory cannot migrate, so the answers cross as one block, not a fault per page.
        if (copy_device_to_host(corpus.distances, sz::span<sz_size_t> {corpus.answers.data(), corpus.answers.size()}) !=
            cudaSuccess)
            throw std::runtime_error("The answers would not come back.");
        call_result_t result(corpus.bytes, levenshtein_check_value(corpus.answers), batch.symbols() * corpus.bytes);
        result.inputs_processed = corpus.count() * levenshtein_queries_per_batch_k;
        return result;
    }
};

/** The same round on the CPU, so the two check values line up under @c STRINGWARS_STRESS. */
template <sz_kernel_levenshtein_distances_t distances_>
struct levenshtein_distances_from_sz {

    /** The candidates, whose texts both sides score. */
    levenshtein_simt_corpus_t &corpus;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_simt_order_t order;

    /** The same queries, prepared on the host. */
    levenshtein_simt_batch_t batch;

    /** @b [queries, candidates], the CPU's own answers. */
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(environment_t const &env, levenshtein_simt_corpus_t &corpus, std::size_t query_bytes,
                                  levenshtein_simt_order_t order,
                                  sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : corpus(corpus), order(order), batch(env, query_bytes, symbol, sz_false_k),
          distances(corpus.count() * levenshtein_queries_per_batch_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &corpus.host_candidates(order), distances.data(), corpus.count(), nullptr) !=
            sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        call_result_t result(corpus.bytes, levenshtein_check_value(distances), batch.symbols() * corpus.bytes);
        result.inputs_processed = corpus.count() * levenshtein_queries_per_batch_k;
        return result;
    }
};

/** The device arm alone across query widths and both orders, for the word-count curve both ways. */
static void bench_levenshtein_word_counts(environment_t const &env, levenshtein_simt_corpus_t &corpus) {
    std::size_t const widths[] = {8, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384};
    levenshtein_simt_order_t const orders[] = {levenshtein_simt_order_t::shuffled_k,
                                               levenshtein_simt_order_t::sorted_k};
    for (std::size_t index = 0; index != sizeof(widths) / sizeof(widths[0]); ++index)
        for (levenshtein_simt_order_t const order : orders) {
            std::size_t const query_bytes = widths[index];
            std::string const suffix = ":q" + std::to_string(query_bytes) + ":w" +
                                       std::to_string(sz_levenshtein_query_words(query_bytes)) +
                                       levenshtein_simt_order_name(order);
            bench_unary(env, simt_arm("sz_levenshtein_distances") + suffix,
                        levenshtein_distances_from_simt<simt_levenshtein_distances_k> {env, corpus, query_bytes, order})
                .log();
        }
}

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_levenshtein_cross_product(environment_t const &env, levenshtein_simt_corpus_t &corpus,
                                            std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes) + ":c" + std::to_string(corpus.count());
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_simt_order_t::shuffled_k};
    bench_result_t base = bench_unary(env, std::string("sz_levenshtein_distances_serial") + suffix, validator).log();
#if STRINGZILLA_TARGET_HASWELL
    base = bench_unary(env, std::string("sz_levenshtein_distances_haswell") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {
                           env, corpus, query_bytes, levenshtein_simt_order_t::shuffled_k})
               .log(base);
#endif
#if STRINGZILLA_TARGET_ICELAKE
    base = bench_unary(env, std::string("sz_levenshtein_distances_icelake") + suffix, validator,
                       levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {
                           env, corpus, query_bytes, levenshtein_simt_order_t::shuffled_k})
               .log(base);
#endif
    // The warped rung spreads a query across a warp's thirty-two lanes, and a wider one is
    // reported as out of reach rather than thrown.
    if (sz_levenshtein_query_words(query_bytes) > sz_levenshtein_simt_words_max_k) {
        fmt::println("Skipping `{}{}`: {} words past the device's {}.", simt_arm("sz_levenshtein_distances"), suffix,
                     sz_levenshtein_query_words(query_bytes), (int)sz_levenshtein_simt_words_max_k);
        return;
    }
    // Both orderings hold the same texts, so the gap between the arms is the warp's `max(L)` tax.
    bench_unary(env, simt_arm("sz_levenshtein_distances") + suffix + ":shuffled", validator,
                levenshtein_distances_from_simt<simt_levenshtein_distances_k> {env, corpus, query_bytes,
                                                                               levenshtein_simt_order_t::shuffled_k})
        .log(base);
    auto validator_sorted = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_simt_order_t::sorted_k};
    bench_unary(env, simt_arm("sz_levenshtein_distances") + suffix + ":sorted", validator_sorted,
                levenshtein_distances_from_simt<simt_levenshtein_distances_k> {env, corpus, query_bytes,
                                                                               levenshtein_simt_order_t::sorted_k})
        .log(base);

    // A window of this many bytes holds at most as many runes, so the byte guard above covers the
    // rune arm. Cells count bytes on both sides, as the CPU arms do, so the rune rows compare.
    auto validator_utf8 = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        env, corpus, query_bytes, levenshtein_simt_order_t::shuffled_k, sz_levenshtein_runes_k};
    bench_result_t base_utf8 =
        bench_unary(env, std::string("sz_levenshtein_distances_serial:utf8") + suffix, validator_utf8).log(base);
    bench_unary(env, simt_arm("sz_levenshtein_distances") + ":utf8" + suffix + ":shuffled", validator_utf8,
                levenshtein_distances_from_simt<simt_levenshtein_distances_k> {
                    env, corpus, query_bytes, levenshtein_simt_order_t::shuffled_k, sz_levenshtein_runes_k})
        .log(base_utf8);
}

#pragma endregion Levenshtein

#pragma region Overlap

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
struct overlap_simt_corpus_t {

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

    overlap_simt_corpus_t(environment_t const &env) {
        std::size_t const count = std::min<std::size_t>(env.tokens.size(), resident_candidates_per_call(env));
        views.resize(count), scores.resize(count);
        for (std::size_t index = 0; index != count; ++index) {
            token_view_t const token = env.tokens[index];
            views[index].start = token.data(), views[index].length = token.size();
            bytes += token.size();
        }
        if (simt_sequence_from_string_views(views.data(), views.size(), &device_candidates) != sz_success_k)
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
struct overlap_scores_from_simt {
    overlap_simt_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    sz_overlap_engine_t engine {};

    overlap_scores_from_simt(environment_t const &env, overlap_simt_corpus_t &corpus, std::size_t query_bytes,
                             std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)) {
        simt_memory_allocator_init_unified(&allocator, 0);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (sz_overlap_engine_init(&engine, &queries, &scored_width, 1, 0, simt_capabilities(), 0, &allocator,
                                   STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The device forest could not be prepared.");
    }
    ~overlap_scores_from_simt() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_simt(overlap_scores_from_simt const &) = delete;
    overlap_scores_from_simt &operator=(overlap_scores_from_simt const &) = delete;

    call_result_t operator()(std::size_t) {
        if (simt_overlap_scores_k(&engine, &corpus.device_candidates, corpus.scores.data(), corpus.scores.size(), 1,
                                  STRINGZILLA_NULL) != sz_success_k)
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
template <sz_kernel_overlap_engine_init_t init_, sz_kernel_overlap_scores_t scores_>
struct overlap_scores_from_sz {
    overlap_simt_corpus_t &corpus;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    std::vector<sz_f32_t> scores;
    sz_overlap_engine_t engine {};

    overlap_scores_from_sz(environment_t const &env, overlap_simt_corpus_t &corpus, std::size_t query_bytes,
                           std::size_t width)
        : corpus(corpus), windows(corpus.windows_at(width)), query(overlap_query_text_(env, query_bytes)),
          scores(corpus.scores.size()) {
        sz_memory_allocator_init_default(&allocator);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (init_(&engine, &queries, &scored_width, 1, 0, 0, &allocator, nullptr) != sz_success_k)
            throw std::runtime_error("The host forest could not be prepared.");
    }
    ~overlap_scores_from_sz() { sz_overlap_engine_free(&engine); }
    overlap_scores_from_sz(overlap_scores_from_sz const &) = delete;
    overlap_scores_from_sz &operator=(overlap_scores_from_sz const &) = delete;

    call_result_t operator()(std::size_t) {
        if (scores_(&engine, &corpus.host_candidates, scores.data(), scores.size(), 1, nullptr) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        check_value_t mixed = 0;
        for (sz_f32_t const score : scores) mixed = mixed * 31u + (check_value_t)(score * 1048576.0f);
        call_result_t result(corpus.bytes, mixed, windows);
        result.inputs_processed = corpus.views.size();
        return result;
    }
};

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_overlap_scores(environment_t const &env, overlap_simt_corpus_t &corpus, std::size_t query_bytes) {
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
    bench_unary(env, simt_arm("sz_overlap_scores") + suffix, validator,
                overlap_scores_from_simt {env, corpus, query_bytes, width})
        .log(base);
}

#pragma endregion Overlap

#pragma region Substrings Residency

/** Device accessors over @p views, which must themselves be device-reachable. */
static sz_sequence_t substrings_device_sequence(unified_vector<sz_string_view_t> const &views) {
    sz_sequence_t sequence {};
    if (simt_sequence_from_string_views(views.data(), views.size(), &sequence) != sz_success_k)
        throw std::runtime_error("The device accessors could not be bound.");
    return sequence;
}

/** Moves the corpus's managed pages to the device, so the first round does not time migration. */
static void substrings_prefetch(substrings_corpus_t const &corpus) {
    sz_cuda_prefetch_(corpus.views.data(), corpus.views.size() * sizeof(sz_string_view_t), STRINGZILLA_NULL);
    for (sz_string_view_t const &view : corpus.views) sz_cuda_prefetch_(view.start, view.length, STRINGZILLA_NULL);
    sz_unused_(cudaStreamSynchronize(0));
}

/**
 *  @brief Whether the corpus cuts into a residency wave of chunks or more, by the engine's budget.
 *
 *  The width is reproduced here rather than read back, because the device derives it from the
 *  corpus total the same way: at least the corpus over the budget, and never under the warm-up a
 *  chunk has to pay.
 */
static bool substrings_fills_a_wave(sz_substrings_engine_t const &engine, substrings_corpus_t const &corpus) {
    std::size_t const budget = engine.chunk_budget ? engine.chunk_budget : 1;
    std::size_t const floor_bytes = std::max<std::size_t>(4 * engine.max_source_match_bytes, 1);
    std::size_t const chunk = std::max(sz::divide_round_up(corpus.bytes, budget), floor_bytes);
    std::size_t chunks = 0;
    for (sz_string_view_t const &view : corpus.views)
        chunks += view.length == 0 ? 1 : sz::divide_round_up(view.length, chunk);

    double const waves = (double)chunks / (double)budget;
    fmt::println("> Corpus: {:.1f} MB in {} haystacks, cut into {} chunks of {} B " //
                 "against {} resident threads - {:.2f} waves",                      //
                 (double)corpus.bytes / 1e6, corpus.views.size(), chunks, chunk, budget, waves);
    if (waves >= 1.0) return true;
    fmt::println("> Refusing the round: below one wave it times the launch, not the walk. " //
                 "Raise STRINGWARS_DATASET_LIMIT.");
    return false;
}

/** Whether overlapping matches fit the engine's match budget, and that budget fits the device. */
static bool substrings_fits_the_budget(sz_substrings_engine_t &engine, substrings_corpus_t const &corpus,
                                       sz_sequence_t const &device_haystacks) {
    std::size_t free_bytes = 0, total_bytes = 0;
    unified_vector<sz_size_t> offsets(corpus.views.size() + 1, 0);
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        throw std::runtime_error("The device would not report its memory.");
    if (simt_substrings_find_k(&engine, &device_haystacks, nullptr, 0, offsets.data(), STRINGZILLA_NULL) !=
        sz_success_k)
        throw std::runtime_error("The sizing round was refused.");
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) throw std::runtime_error("The sizing round failed.");

    std::size_t const emitted = engine.report->matches_emitted;
    fmt::println("> Matches: {} against a budget of {}, in a {:.1f} GB arena and {:.1f} GB free", emitted,
                 (std::size_t)engine.matches_budget, (double)engine.scratch_bytes / 1e9, (double)free_bytes / 1e9);
    if (emitted <= engine.matches_budget && engine.scratch_bytes <= free_bytes / 2) return true;
    fmt::println("> Refusing the round: the round outruns its match budget. Lower STRINGWARS_DATASET_LIMIT.");
    return false;
}

#pragma endregion Substrings Residency

#pragma region Substrings Verbs

/** Per-haystack counts, the device arm logged against the CPU one. */
static void bench_substrings_counts(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                    substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                    std::string const &suffix) {
    auto validator = substrings_counts_from_sz<sz_substrings_counts_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_counts_serial" + suffix, validator).log();
    bench_unary(env, simt_arm("sz_substrings_counts") + suffix, validator,
                substrings_counts_from_sz<simt_substrings_counts_k> {device, corpus, device_haystacks})
        .log(base);
}

/** Every match, the device arm logged against the CPU one. */
static void bench_substrings_find(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto validator = substrings_find_from_sz<sz_substrings_find_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_find_serial" + suffix, validator).log();
    bench_unary(env, simt_arm("sz_substrings_find") + suffix, validator,
                substrings_find_from_sz<simt_substrings_find_k> {device, corpus, device_haystacks})
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
    bench_unary(
        env, simt_arm("sz_substrings_replace") + suffix, validator,
        substrings_replace_from_sz<simt_substrings_replace_k> {device, corpus, device_haystacks, device_replacements})
        .log(base);
}

/** BM25 scores, the device arm logged against the CPU one. */
static void bench_substrings_bm25(environment_t const &env, substrings_engine_t &host, substrings_engine_t &device,
                                  substrings_corpus_t const &corpus, sz_sequence_t const &device_haystacks,
                                  std::string const &suffix) {
    auto validator = substrings_bm25_from_sz<sz_substrings_bm25_scores_serial> {host, corpus, corpus.haystacks};
    bench_result_t base = bench_unary(env, "sz_substrings_bm25_scores_serial" + suffix, validator).log();
    bench_unary(env, simt_arm("sz_substrings_bm25_scores") + suffix, validator,
                substrings_bm25_from_sz<simt_substrings_bm25_scores_k> {device, corpus, device_haystacks})
        .log(base);
}

/** One vocabulary slice walked by each verb under each accepted policy, once the round is sound. */
static void bench_substrings_slice(environment_t const &env, substrings_corpus_t const &corpus,
                                   sz_sequence_t const &device_haystacks, substrings_slice_t slice,
                                   sz_substrings_case_sensitivity_t sensitivity) {
    sz_memory_allocator_t allocator;
    simt_memory_allocator_init_unified(&allocator, 0);
    substrings_dictionary_t const dictionary(env, slice, sensitivity, allocator);
    std::string const suffix = substrings_label(slice, sensitivity);
    if (dictionary.needles.empty()) {
        fmt::println("Vocabulary {} is empty on this corpus, skipping it.", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        fmt::println("Vocabulary {} holds {} needles over {} states, {} of them hot.", suffix.c_str(),
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

#pragma endregion Substrings Verbs

#pragma region Drivers

/** Every family's rows over the multilingual lines, each on a corpus it keeps resident; a build
 *  host without a device logs as much and passes. */
inline int bench_simt_main(int argc, char const **argv) {
    install_bench_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    log_environment();
    print_bench_environment();
    if (!log_cuda_device()) return 0;

    // The arms throw on a failed status, so a bad call ends the run with its message, not a crash.
    try {
        corpora_t corpora(argc, argv);
        environment_t const &env = corpora.multilingual_lines();
        {
            levenshtein_simt_corpus_t corpus(env);
            fmt::println("Starting Levenshtein benchmarks over {} resident candidates...", corpus.count());
            bench_levenshtein_cross_product(env, corpus, median_token_bytes(env));
            bench_levenshtein_word_counts(env, corpus);
        }
        {
            overlap_simt_corpus_t corpus(env);
            fmt::println("Starting window overlap benchmarks over {} resident candidates...", corpus.views.size());
            bench_overlap_scores(env, corpus, median_token_bytes(env));
        }
        {
            substrings_corpus_t const corpus(env);
            sz_sequence_t const device_haystacks = substrings_device_sequence(corpus.views);
            substrings_prefetch(corpus);
            fmt::println("Starting multi-pattern search benchmarks...");
            bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::frequent_k,
                                   sz_substrings_cased_k);
            bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::rare_k, sz_substrings_cased_k);
            bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::frequent_k,
                                   sz_substrings_uncased_k);
            bench_substrings_slice(env, corpus, device_haystacks, substrings_slice_t::sampled_k, sz_substrings_cased_k);
        }
    }
    catch (std::exception const &e) {
        fmt::println(stderr, "Failed with: {}", e.what());
        return 1;
    }

    fmt::println("All benchmarks passed.");
    return 0;
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::bench

#endif // STRINGZILLA_BENCH_CROSS_SIMT_CUH
