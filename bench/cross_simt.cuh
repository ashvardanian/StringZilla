/**
 *  @file bench/cross_simt.cuh
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the rows CUDA and ROCm share, each device arm against the CPU
 *      tiers this build carries.
 *
 *  Included by one translation unit per binary, `cross_cuda.cu` or `cross_rocm.hip`, which picks
 *  its vendor's kernels by the runtime it is compiled for. The rows reach that runtime only through
 *  the library and the vendor helpers `harness.hpp` picks, so both vendors run them unchanged.
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
 *  them last and a short-candidate round would time migration. A warp's lanes that start together
 *  finish apart, so each device arm runs over the same views in corpus order and sorted by length,
 *  and over as many spans of the same tape with lengths log-uniform from one byte up, and the gaps
 *  between `:shuffled`, `:sorted` and `:skewed` measure how well a round balances them. Throughput
 *  is Cell Updates Per Second, as the CPU benchmark reports it.
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
 *  cmake --build build_release --config Release --target stringzilla_cuda_bench
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_FILTER=cuda build_release/stringzilla_cuda_bench
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
#include <stdexcept>        // `std::runtime_error`
#include <string>           // `std::string`, `std::to_string`
#include <vector>           // `std::vector`

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
inline constexpr sz::device_kind_t simt_device_kind_k = sz::device_kind_t::rocm_k;
#else
inline constexpr char const *simt_vendor_k = "cuda";
inline constexpr sz_kernel_levenshtein_distances_t simt_levenshtein_distances_k = &sz_levenshtein_distances_cuda;
inline constexpr sz_kernel_overlap_scores_t simt_overlap_scores_k = &sz_overlap_scores_cuda;
inline constexpr sz_kernel_substrings_counts_t simt_substrings_counts_k = &sz_substrings_counts_cuda;
inline constexpr sz_kernel_substrings_find_t simt_substrings_find_k = &sz_substrings_find_cuda;
inline constexpr sz_kernel_substrings_replace_t simt_substrings_replace_k = &sz_substrings_replace_cuda;
inline constexpr sz_kernel_substrings_bm25_scores_t simt_substrings_bm25_scores_k = &sz_substrings_bm25_scores_cuda;
inline constexpr sz::device_kind_t simt_device_kind_k = sz::device_kind_t::cuda_k;
#endif

/** What device 0 of this vendor enables, or zero without a device, which the engines refuse. */
inline sz_capability_t simt_capabilities() noexcept {
    auto const [device, make_status] = sz::device_t::make(simt_device_kind_k, 0);
    return sz::succeeded(make_status) ? device.capabilities_enabled().value : 0;
}

/** A sequence copied into one unified tape a kernel reads, returned to its allocator with it. */
struct simt_tape_t {
    sz_memory_allocator_t unified {};
    sz_sequence_t sequence {};
    sz_size_t bytes = 0;

    simt_tape_t() = default;
    simt_tape_t(simt_tape_t const &) = delete;
    simt_tape_t &operator=(simt_tape_t const &) = delete;
    ~simt_tape_t() noexcept {
        if (bytes) unified.free((void *)sequence.handle, bytes, unified.handle, nullptr);
    }

    /** Copies @p source in, once, migrating it to the device so no round times the first touch. */
    sz_sequence_t const &copy(sz_sequence_t const &source) {
        if (sz_memory_allocator_init_unified_best(&unified, simt_capabilities()) != sz_success_k ||
            sz_sequence_copy_best(&sequence, &source, &unified, &bytes, simt_capabilities(), nullptr) != sz_success_k ||
            sz_stream_synchronize_best(simt_capabilities(), nullptr) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
        return sequence;
    }
};

/** The name a row carries for this vendor's kernel of @p verb, like @c sz_overlap_scores_cuda. */
inline std::string simt_arm(char const *verb) { return std::string(verb) + "_" + simt_vendor_k; }

/**
 *  @brief @p count spans of @p text at seeded offsets, their lengths log-uniform from one byte to a
 *      ceiling whose mean is @p mean_bytes.
 *
 *  A warp of corpus lines differs in length by a few times; these differ by orders of magnitude
 *  while a round's bytes stay about what the corpus order walks, which is the skew a balanced round
 *  has to absorb.
 */
static std::vector<sz_string_view_t> simt_skewed_views(char const *text, std::size_t bytes, std::size_t count,
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
static char const *levenshtein_simt_order_name(levenshtein_simt_order_t order) {
    return order == levenshtein_simt_order_t::sorted_k   ? ":sorted"
           : order == levenshtein_simt_order_t::skewed_k ? ":skewed"
                                                         : ":shuffled";
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
 *  @brief The corpus as the device sees it: one tape per candidate order, and the room for
 *      one round's distances.
 *
 *  The dataset the corpus loads is managed, so a view into it is a page that follows whoever
 *  touched it last. The candidates therefore cross once into tapes migrated to the device, and the
 *  CPU arms keep their own views into the dataset, or into the host's copy of its bytes, to score
 *  the very same texts.
 */
struct levenshtein_simt_corpus_t {

    /** Every candidate's bytes on the host, back to back, which the skewed spans point into. */
    std::vector<char> host_tape;

    /** The candidates in corpus order, in length descending order, and as skewed spans. */
    simt_tape_t shuffled_tape, sorted_tape, skewed_tape;

    /** @ [queries, candidates], written by every round. */
    device_vector<sz_size_t> distances;

    /** The same matrix, one copy per round feeding the check. */
    pinned_vector<sz_size_t> answers;

    /** Dataset views the CPU arms read, corpus order. */
    std::vector<sz_string_view_t> host_views;

    /** The same dataset views, length descending. */
    std::vector<sz_string_view_t> host_sorted_views;

    /** The skewed spans of the host's tape. */
    std::vector<sz_string_view_t> host_skewed_views;

    /** Device accessors over @ref shuffled_tape. */
    sz_sequence_t device_shuffled {};

    /** Device accessors over @ref sorted_tape. */
    sz_sequence_t device_sorted {};

    /** Device accessors over @ref skewed_tape. */
    sz_sequence_t device_skewed {};

    /** Host accessors over @ref host_views. */
    sz_sequence_t host_shuffled {};

    /** Host accessors over @ref host_sorted_views. */
    sz_sequence_t host_sorted {};

    /** Host accessors over @ref host_skewed_views. */
    sz_sequence_t host_skewed {};

    /** Candidate bytes one round of the corpus order touches, which the sorted one shares. */
    std::size_t bytes = 0;

    /** Candidate bytes one skewed round touches. */
    std::size_t skewed_bytes = 0;

    levenshtein_simt_corpus_t(environment_t const &env, corpus_t const &corpus) {
        std::size_t const count = std::min<std::size_t>(corpus.tokens.size(), resident_candidates_per_call(env));
        for (std::size_t index = 0; index != count; ++index) bytes += corpus.tokens[index].size();
        if (distances.resize_uninitialized(count * levenshtein_queries_per_batch_k) != sz::status_t::success_k)
            throw std::runtime_error("The device would not hold the corpus.");
        answers.resize(count * levenshtein_queries_per_batch_k);
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
        for (sz_string_view_t const &view : host_skewed_views) skewed_bytes += view.length;

        sz_sequence_from_string_views(host_views.data(), count, &host_shuffled);
        sz_sequence_from_string_views(host_sorted_views.data(), count, &host_sorted);
        sz_sequence_from_string_views(host_skewed_views.data(), count, &host_skewed);
        device_shuffled = shuffled_tape.copy(host_shuffled);
        device_sorted = sorted_tape.copy(host_sorted);
        device_skewed = skewed_tape.copy(host_skewed);
    }

    /** Candidates one round scores, whichever order it walks them in. */
    std::size_t count() const { return host_views.size(); }

    /** Candidate bytes one round over @p order touches. */
    std::size_t bytes_of(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::skewed_k ? skewed_bytes : bytes;
    }

    /** The device accessors over one ordering of the texts. */
    sz_sequence_t const &device_candidates(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k   ? device_sorted
               : order == levenshtein_simt_order_t::skewed_k ? device_skewed
                                                             : device_shuffled;
    }

    /** The host accessors over one ordering of the texts. */
    sz_sequence_t const &host_candidates(levenshtein_simt_order_t order) const {
        return order == levenshtein_simt_order_t::sorted_k   ? host_sorted
               : order == levenshtein_simt_order_t::skewed_k ? host_skewed
                                                             : host_shuffled;
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
static std::vector<sz_string_view_t> levenshtein_simt_queries(corpus_t const &corpus, std::size_t query_bytes) {
    std::vector<sz_string_view_t> views(levenshtein_queries_per_batch_k);
    for (std::size_t query = 0; query != views.size(); ++query) {
        token_view_t const whole = corpus.tokens[(query * 7 + 1) % corpus.tokens.size()];
        std::size_t const reachable = (std::size_t)(corpus.dataset.data() + corpus.dataset.size() - whole.data());
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

    levenshtein_simt_batch_t(corpus_t const &corpus, std::size_t query_bytes, sz_levenshtein_symbol_t symbol,
                             sz_bool_t on_device)
        : views(levenshtein_simt_queries(corpus, query_bytes)) {
        sz_sequence_from_string_views(views.data(), views.size(), &queries);
        sz_capability_t const capabilities = on_device == sz_true_k ? simt_capabilities() : sz::default_capabilities();
        if (sz_levenshtein_engine_init(&engine, &queries, symbol, capabilities, STRINGZILLA_NULL, STRINGZILLA_NULL) !=
            sz_success_k)
            throw std::runtime_error("The engine could not be prepared.");
    }
    ~levenshtein_simt_batch_t() { sz_levenshtein_engine_free(&engine, STRINGZILLA_NULL); }
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
    levenshtein_simt_corpus_t &resident;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_simt_order_t order;

    /** The queries, prepared on the device once. */
    levenshtein_simt_batch_t batch;

    levenshtein_distances_from_simt(corpus_t const &corpus, levenshtein_simt_corpus_t &resident,
                                    std::size_t query_bytes, levenshtein_simt_order_t order,
                                    sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : resident(resident), order(order), batch(corpus, query_bytes, symbol, sz_true_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &resident.device_candidates(order), resident.distances.data(), resident.count(),
                       STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        // Device memory cannot migrate, so the answers cross as one block, not a fault per page.
        sz::span<sz_size_t> const answers {resident.answers.data(), resident.answers.size()};
        if (copy_device_to_host(resident.distances, answers) != sz_success_k ||
            sz_stream_synchronize_best(batch.engine.capability, STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The answers would not come back.");
        std::size_t const bytes = resident.bytes_of(order);
        return call_result_t(bytes, levenshtein_check_value(resident.answers), batch.symbols() * bytes);
    }
};

/** The same round on the CPU, so the two check values line up under @c STRINGZILLA_STRESS. */
template <sz_kernel_levenshtein_distances_t distances_>
struct levenshtein_distances_from_sz {

    /** The candidates, whose texts both sides score. */
    levenshtein_simt_corpus_t &resident;

    /** Which ordering of the candidates this arm walks. */
    levenshtein_simt_order_t order;

    /** The same queries, prepared on the host. */
    levenshtein_simt_batch_t batch;

    /** @b [queries, candidates], the CPU's own answers. */
    std::vector<sz_size_t> distances;

    levenshtein_distances_from_sz(corpus_t const &corpus, levenshtein_simt_corpus_t &resident, std::size_t query_bytes,
                                  levenshtein_simt_order_t order,
                                  sz_levenshtein_symbol_t symbol = sz_levenshtein_bytes_k)
        : resident(resident), order(order), batch(corpus, query_bytes, symbol, sz_false_k),
          distances(resident.count() * levenshtein_queries_per_batch_k) {}

    call_result_t operator()(std::size_t token_index) {
        sz_unused_(token_index);
        if (distances_(&batch.engine, &resident.host_candidates(order), distances.data(), resident.count(), nullptr) !=
            sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        std::size_t const bytes = resident.bytes_of(order);
        return call_result_t(bytes, levenshtein_check_value(distances), batch.symbols() * bytes);
    }
};

/** The device arm alone across query widths and every order, for the word-count curve each way. */
static void bench_levenshtein_word_counts(environment_t const &env, corpus_t const &corpus,
                                          levenshtein_simt_corpus_t &resident) {
    std::size_t const widths[] = {8, 64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384};
    levenshtein_simt_order_t const orders[] = {levenshtein_simt_order_t::shuffled_k, levenshtein_simt_order_t::sorted_k,
                                               levenshtein_simt_order_t::skewed_k};
    for (std::size_t index = 0; index != sizeof(widths) / sizeof(widths[0]); ++index)
        for (levenshtein_simt_order_t const order : orders) {
            std::size_t const query_bytes = widths[index];
            std::string const suffix = ":q" + std::to_string(query_bytes) + ":w" +
                                       std::to_string(sz_levenshtein_query_words(query_bytes)) +
                                       levenshtein_simt_order_name(order);
            print(bench_unary(
                env, corpus, simt_arm("sz_levenshtein_distances") + suffix,
                levenshtein_distances_from_simt<simt_levenshtein_distances_k> {corpus, resident, query_bytes, order}));
        }
}

/** Every arm at one query width, the width carried in each arm's name beside the resident count. */
static void bench_levenshtein_cross_product(environment_t const &env, corpus_t const &corpus,
                                            levenshtein_simt_corpus_t &resident, std::size_t query_bytes) {
    std::string const suffix = ":q" + std::to_string(query_bytes) + ":c" + std::to_string(resident.count());
    std::string const haswell_name = "sz_levenshtein_distances_haswell" + suffix,
                      icelake_name = "sz_levenshtein_distances_icelake" + suffix,
                      shuffled_name = simt_arm("sz_levenshtein_distances") + suffix + ":shuffled",
                      sorted_name = simt_arm("sz_levenshtein_distances") + suffix + ":sorted",
                      skewed_name = simt_arm("sz_levenshtein_distances") + suffix + ":skewed",
                      utf8_name = simt_arm("sz_levenshtein_distances") + ":utf8" + suffix + ":shuffled";
    auto validator = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        corpus, resident, query_bytes, levenshtein_simt_order_t::shuffled_k};
    std::optional<double> base = bench_baseline(
        env, corpus, "sz_levenshtein_distances_serial" + suffix,
        {haswell_name, icelake_name, shuffled_name, sorted_name, skewed_name, utf8_name}, validator);
#if STRINGZILLA_TARGET_HASWELL
    std::optional<row_t> const haswell = bench_unary(
        env, corpus, haswell_name, validator,
        levenshtein_distances_from_sz<sz_levenshtein_distances_haswell> {corpus, resident, query_bytes,
                                                                         levenshtein_simt_order_t::shuffled_k});
    print(haswell, base);
    if (std::optional<double> const rate = baseline_of(haswell)) base = rate;
#endif
#if STRINGZILLA_TARGET_ICELAKE
    std::optional<row_t> const icelake = bench_unary(
        env, corpus, icelake_name, validator,
        levenshtein_distances_from_sz<sz_levenshtein_distances_icelake> {corpus, resident, query_bytes,
                                                                         levenshtein_simt_order_t::shuffled_k});
    print(icelake, base);
    if (std::optional<double> const rate = baseline_of(icelake)) base = rate;
#endif
    // The warped rung spreads a query across a warp's thirty-two lanes, and a wider one is
    // reported as out of reach rather than thrown.
    if (sz_levenshtein_query_words(query_bytes) > sz_levenshtein_gpu_words_max_k) {
        fmt::println("Skipping `{}{}`: {} words past the device's {}.", simt_arm("sz_levenshtein_distances"), suffix,
                     sz_levenshtein_query_words(query_bytes), (int)sz_levenshtein_gpu_words_max_k);
        return;
    }
    // Both orderings hold the same texts, so the gap between the arms is the warp's `max(L)` tax,
    // and the skewed spans raise that tax by orders of magnitude.
    print(bench_unary(env, corpus, shuffled_name, validator,
                      levenshtein_distances_from_simt<simt_levenshtein_distances_k> {
                          corpus, resident, query_bytes, levenshtein_simt_order_t::shuffled_k}),
          base);
    auto validator_sorted = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        corpus, resident, query_bytes, levenshtein_simt_order_t::sorted_k};
    print(bench_unary(env, corpus, sorted_name, validator_sorted,
                      levenshtein_distances_from_simt<simt_levenshtein_distances_k> {
                          corpus, resident, query_bytes, levenshtein_simt_order_t::sorted_k}),
          base);
    auto validator_skewed = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        corpus, resident, query_bytes, levenshtein_simt_order_t::skewed_k};
    print(bench_unary(env, corpus, skewed_name, validator_skewed,
                      levenshtein_distances_from_simt<simt_levenshtein_distances_k> {
                          corpus, resident, query_bytes, levenshtein_simt_order_t::skewed_k}),
          base);

    // A window of this many bytes holds at most as many runes, so the byte guard above covers the
    // rune arm. Cells count bytes on both sides, as the CPU arms do, so the rune rows compare.
    auto validator_utf8 = levenshtein_distances_from_sz<sz_levenshtein_distances_serial> {
        corpus, resident, query_bytes, levenshtein_simt_order_t::shuffled_k, sz_levenshtein_runes_k};
    std::optional<double> const base_utf8 = bench_baseline(env, corpus, "sz_levenshtein_distances_serial:utf8" + suffix,
                                                           {utf8_name}, validator_utf8, base);
    print(bench_unary(env, corpus, utf8_name, validator_utf8,
                      levenshtein_distances_from_simt<simt_levenshtein_distances_k> {
                          corpus, resident, query_bytes, levenshtein_simt_order_t::shuffled_k, sz_levenshtein_runes_k}),
          base_utf8);
}

#pragma endregion Levenshtein

#pragma region Overlap

/** The width the corpus's collision entropy picks for a query of @p query_bytes against a mean
 *  candidate of the corpus. */
static std::size_t overlap_width_(corpus_t const &corpus, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : corpus.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(corpus.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (token_view_t const token : corpus.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(corpus.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/**
 *  @brief The corpus as the device sees it: views over the dataset, and room for a round's scores.
 *
 *  Under CUDA the corpus already loads the dataset into unified memory, so the candidates need
 *  no upload and the two sequences differ only in whose accessors they carry.
 */
struct overlap_simt_corpus_t {

    /** One view per candidate; its size is the candidate count. */
    unified_vector<sz_string_view_t> views;

    /** @b [candidates], read back for the check value. */
    unified_vector<sz_f32_t> scores;

    /** The candidates copied into one tape, which every device round reads. */
    simt_tape_t device_tape;

    /** The tape's accessors, which a kernel calls, as the resident path requires. */
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

    explicit overlap_simt_corpus_t(std::vector<sz_string_view_t> const &candidates)
        : views(candidates.size()), scores(candidates.size()) {
        for (std::size_t index = 0; index != candidates.size(); ++index)
            views[index] = candidates[index], bytes += candidates[index].length;
        sz_sequence_from_string_views(views.data(), views.size(), &host_candidates);
        device_candidates = device_tape.copy(host_candidates);
    }
};

/** The leading tokens of @p corpus, one per resident thread of the bound device. */
static std::vector<sz_string_view_t> overlap_token_views(environment_t const &env, corpus_t const &corpus) {
    std::size_t const count = std::min<std::size_t>(corpus.tokens.size(), resident_candidates_per_call(env));
    std::vector<sz_string_view_t> views(count);
    for (std::size_t index = 0; index != count; ++index)
        views[index] = {corpus.tokens[index].data(), corpus.tokens[index].size()};
    return views;
}

/** The leading token cut to @p query_bytes: the one query each arm's engine is built over. */
static std::string overlap_query_text_(corpus_t const &corpus, std::size_t query_bytes) {
    token_view_t const whole = corpus.tokens[0];
    return std::string(whole.data(), std::min(whole.size(), query_bytes));
}

/** Scores the resident corpus against the fixed query, entirely on the device. */
struct overlap_scores_from_simt {
    overlap_simt_corpus_t &resident;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    sz_overlap_engine_t engine {};

    overlap_scores_from_simt(corpus_t const &corpus, overlap_simt_corpus_t &resident, std::size_t query_bytes,
                             std::size_t width)
        : resident(resident), windows(resident.windows_at(width)), query(overlap_query_text_(corpus, query_bytes)) {
        sz_memory_allocator_init_unified_best(&allocator, simt_capabilities());
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (sz_overlap_engine_init(&engine, &queries, &scored_width, 1, 0, simt_capabilities(), &allocator,
                                   STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The device forest could not be prepared.");
    }
    ~overlap_scores_from_simt() { sz_overlap_engine_free(&engine, STRINGZILLA_NULL); }
    overlap_scores_from_simt(overlap_scores_from_simt const &) = delete;
    overlap_scores_from_simt &operator=(overlap_scores_from_simt const &) = delete;

    call_result_t operator()(std::size_t) {
        if (simt_overlap_scores_k(&engine, &resident.device_candidates, resident.scores.data(), resident.scores.size(),
                                  1, STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The GPU round failed.");
        if (sz_stream_synchronize_best(engine.capability, STRINGZILLA_NULL) != sz_success_k)
            throw std::runtime_error("The GPU round did not finish.");
        check_value_t mixed = 0;
        for (sz_f32_t const score : resident.scores) mixed = mixed * 31u + (check_value_t)(score * 1048576.0f);
        return call_result_t(resident.bytes, mixed, windows);
    }
};

/** The same round on the CPU, so the two check values line up under @c STRINGZILLA_STRESS. */
template <sz_kernel_overlap_engine_init_t init_, sz_kernel_overlap_scores_t scores_>
struct overlap_scores_from_sz {
    overlap_simt_corpus_t &resident;
    std::size_t windows;
    std::string query;
    sz_memory_allocator_t allocator;
    std::vector<sz_f32_t> scores;
    sz_overlap_engine_t engine {};

    overlap_scores_from_sz(corpus_t const &corpus, overlap_simt_corpus_t &resident, std::size_t query_bytes,
                           std::size_t width)
        : resident(resident), windows(resident.windows_at(width)), query(overlap_query_text_(corpus, query_bytes)),
          scores(resident.scores.size()) {
        sz_memory_allocator_init_default(&allocator);
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_size_t const scored_width = width;
        if (init_(&engine, &queries, &scored_width, 1, 0, &allocator, nullptr) != sz_success_k)
            throw std::runtime_error("The host forest could not be prepared.");
    }
    ~overlap_scores_from_sz() { sz_overlap_engine_free(&engine, nullptr); }
    overlap_scores_from_sz(overlap_scores_from_sz const &) = delete;
    overlap_scores_from_sz &operator=(overlap_scores_from_sz const &) = delete;

    call_result_t operator()(std::size_t) {
        if (scores_(&engine, &resident.host_candidates, scores.data(), scores.size(), 1, nullptr) != sz_success_k)
            throw std::runtime_error("The CPU round failed.");
        check_value_t mixed = 0;
        for (sz_f32_t const score : scores) mixed = mixed * 31u + (check_value_t)(score * 1048576.0f);
        return call_result_t(resident.bytes, mixed, windows);
    }
};

/** Every arm at one query width, the width and the corpus @p label carried in each arm's name. */
static void bench_overlap_scores(environment_t const &env, corpus_t const &corpus, overlap_simt_corpus_t &resident,
                                 std::size_t query_bytes, char const *label) {
    std::size_t const width = overlap_width_(corpus, query_bytes);
    std::string const suffix = ":w" + std::to_string(width) + label;
    std::string const haswell_name = "sz_overlap_scores_haswell" + suffix,
                      skylake_name = "sz_overlap_scores_skylake" + suffix,
                      device_name = simt_arm("sz_overlap_scores") + suffix;
    auto validator = overlap_scores_from_sz<sz_overlap_engine_init_serial, sz_overlap_scores_serial> {
        corpus, resident, query_bytes, width};
    std::optional<double> base = bench_baseline(env, corpus, "sz_overlap_scores_serial" + suffix,
                                                {haswell_name, skylake_name, device_name}, validator);
#if STRINGZILLA_TARGET_HASWELL
    std::optional<row_t> const haswell = bench_unary(
        env, corpus, haswell_name, validator,
        overlap_scores_from_sz<sz_overlap_engine_init_haswell, sz_overlap_scores_haswell> {corpus, resident,
                                                                                           query_bytes, width});
    print(haswell, base);
    if (std::optional<double> const rate = baseline_of(haswell)) base = rate;
#endif
#if STRINGZILLA_TARGET_SKYLAKE
    std::optional<row_t> const skylake = bench_unary(
        env, corpus, skylake_name, validator,
        overlap_scores_from_sz<sz_overlap_engine_init_skylake, sz_overlap_scores_skylake> {corpus, resident,
                                                                                           query_bytes, width});
    print(skylake, base);
    if (std::optional<double> const rate = baseline_of(skylake)) base = rate;
#endif
    print(bench_unary(env, corpus, device_name, validator,
                      overlap_scores_from_simt {corpus, resident, query_bytes, width}),
          base);
}

#pragma endregion Overlap

#pragma region Substrings Residency

/**
 *  @brief Whether the corpus cuts into a residency wave of chunks or more, by the engine's budget.
 *
 *  The width is reproduced here rather than read back, because the device derives it from the
 *  corpus total the same way: at least the corpus over the budget, and never under the warm-up a
 *  chunk has to pay.
 */
static bool substrings_fills_a_wave(sz_substrings_engine_t const &engine, substrings_corpus_t const &resident) {
    std::size_t const budget = engine.chunk_budget ? engine.chunk_budget : 1;
    std::size_t const floor_bytes = std::max<std::size_t>(4 * engine.max_source_match_bytes, 1);
    std::size_t const chunk = std::max(sz::divide_round_up(resident.bytes, budget), floor_bytes);
    std::size_t chunks = 0;
    for (sz_string_view_t const &view : resident.views)
        chunks += view.length == 0 ? 1 : sz::divide_round_up(view.length, chunk);

    double const waves = (double)chunks / (double)budget;
    fmt::println("> Corpus: {:.1f} MB in {} haystacks, cut into {} chunks of {} B " //
                 "against {} resident threads - {:.2f} waves",                      //
                 (double)resident.bytes / (1 << 20), resident.views.size(), chunks, chunk, budget, waves);
    if (waves >= 1.0) return true;
    fmt::println("> Refusing the round: below one wave it times the launch, not the walk. " //
                 "Raise STRINGWARS_BYTES.");
    return false;
}

/** Overlapping matches one round of @p engine emits over @p resident, counted by a sizing round. */
static std::size_t substrings_matches_emitted(sz_substrings_engine_t &engine, substrings_corpus_t const &resident,
                                              sz_sequence_t const &device_haystacks) {
    unified_vector<sz_size_t> offsets(resident.views.size() + 1, 0);
    if (simt_substrings_find_k(&engine, &device_haystacks, nullptr, 0, offsets.data(), STRINGZILLA_NULL) !=
        sz_success_k)
        throw std::runtime_error("The sizing round was refused.");
    if (sz_stream_synchronize_best(engine.capability, nullptr) != sz_success_k)
        throw std::runtime_error("The sizing round failed.");
    return engine.report->matches_emitted;
}

/** Whether every match of a round fits @p engine's budget, and its arena fits the device. */
static bool substrings_fits_the_device(sz_substrings_engine_t const &engine, std::size_t emitted) {
    std::size_t const free_bytes = gpu_free_bytes();
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
static void bench_substrings_counts(environment_t const &env, corpus_t const &corpus, substrings_engine_t &host,
                                    substrings_engine_t &device, substrings_corpus_t const &resident,
                                    sz_sequence_t const &device_haystacks, std::string const &suffix) {
    auto validator = substrings_counts_from_sz<sz_substrings_counts_serial> {host, resident, resident.haystacks};
    std::string const device_name = simt_arm("sz_substrings_counts") + suffix;
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_counts_serial" + suffix,
                                                      {device_name}, validator);
    print(bench_unary(env, corpus, device_name, validator,
                      substrings_counts_from_sz<simt_substrings_counts_k> {device, resident, device_haystacks}),
          base);
}

/** Every match, the device arm logged against the CPU one. */
static void bench_substrings_find(environment_t const &env, corpus_t const &corpus, substrings_engine_t &host,
                                  substrings_engine_t &device, substrings_corpus_t const &resident,
                                  sz_sequence_t const &device_haystacks, std::string const &suffix) {
    auto validator = substrings_find_from_sz<sz_substrings_find_serial> {host, resident, resident.haystacks};
    std::string const device_name = simt_arm("sz_substrings_find") + suffix;
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_find_serial" + suffix, {device_name},
                                                      validator);
    print(bench_unary(env, corpus, device_name, validator,
                      substrings_find_from_sz<simt_substrings_find_k> {device, resident, device_haystacks}),
          base);
}

/** The rewrite, the device arm logged against the CPU one. */
static void bench_substrings_replace(environment_t const &env, corpus_t const &corpus, substrings_engine_t &host,
                                     substrings_engine_t &device, substrings_dictionary_t const &dictionary,
                                     substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                     sz_sequence_t const &device_replacements, std::string const &suffix) {
    auto validator = substrings_replace_from_sz<sz_substrings_replace_serial> {host, resident, resident.haystacks,
                                                                               dictionary.replacements};
    std::string const device_name = simt_arm("sz_substrings_replace") + suffix;
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_replace_serial" + suffix,
                                                      {device_name}, validator);
    print(bench_unary(env, corpus, device_name, validator,
                      substrings_replace_from_sz<simt_substrings_replace_k> {device, resident, device_haystacks,
                                                                             device_replacements}),
          base);
}

/** BM25 scores, the device arm logged against the CPU one. */
static void bench_substrings_bm25(environment_t const &env, corpus_t const &corpus, substrings_engine_t &host,
                                  substrings_engine_t &device, substrings_corpus_t const &resident,
                                  sz_sequence_t const &device_haystacks, std::string const &suffix) {
    auto validator = substrings_bm25_from_sz<sz_substrings_bm25_scores_serial> {host, resident, resident.haystacks};
    std::string const device_name = simt_arm("sz_substrings_bm25_scores") + suffix;
    std::optional<double> const base = bench_baseline(env, corpus, "sz_substrings_bm25_scores_serial" + suffix,
                                                      {device_name}, validator);
    print(bench_unary(env, corpus, device_name, validator,
                      substrings_bm25_from_sz<simt_substrings_bm25_scores_k> {device, resident, device_haystacks}),
          base);
}

/** One vocabulary slice walked by each verb under each accepted policy, once the round is sound. */
static void bench_substrings_slice(environment_t const &env, corpus_t const &corpus,
                                   substrings_corpus_t const &resident, sz_sequence_t const &device_haystacks,
                                   substrings_slice_t slice, sz_substrings_case_sensitivity_t sensitivity) {
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_unified_best(&allocator, simt_capabilities());
    substrings_dictionary_t dictionary(env, corpus, slice, sensitivity, allocator);
    std::string const suffix = substrings_label(slice, sensitivity);
    if (dictionary.needles.empty()) {
        fmt::println("Vocabulary {} is empty on this corpus, skipping it.", suffix.c_str());
        return;
    }
    {
        substrings_engine_t probe(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        fmt::println("Vocabulary {} holds {} needles over {} states, {} of them hot.", suffix.c_str(),
                     dictionary.needles.size(), probe.engine.state_count, probe.engine.hot_count);
        if (!substrings_fills_a_wave(probe.engine, resident)) return;
        // A benchmark round must keep every match it finds, whatever the default budget.
        std::size_t const emitted = substrings_matches_emitted(probe.engine, resident, device_haystacks);
        if (emitted > probe.engine.matches_budget) dictionary.matches_budget = emitted;
        substrings_engine_t sized(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        if (!substrings_fits_the_device(sized.engine, emitted)) return;
    }

    simt_tape_t replacements_tape;
    sz_sequence_t const device_replacements = replacements_tape.copy(dictionary.replacements);
    for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, policy, substrings_residency_t::device_k);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(env, corpus, host, device, resident, device_haystacks, cover);
        bench_substrings_find(env, corpus, host, device, resident, device_haystacks, cover);
    }
    for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
        substrings_engine_t host(dictionary, policy, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, policy, substrings_residency_t::device_k);
        bench_substrings_replace(env, corpus, host, device, dictionary, resident, device_haystacks, device_replacements,
                                 suffix + substrings_policy_name(policy));
    }
    {
        substrings_engine_t host(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
        bench_substrings_bm25(env, corpus, host, device, resident, device_haystacks, suffix);
    }
}

/** Haystacks a small-batch round walks: too few to fill the device, so its chain of launches is
 *  most of what the round costs. */
enum { substrings_small_batch_haystacks_k = 64 };

/** Counting and finding over the leading few haystacks, where the gaps between a round's launches
 *  rather than its walk set the rate. */
static void bench_substrings_small_batch(environment_t const &env, corpus_t const &corpus) {
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_unified_best(&allocator, simt_capabilities());
    substrings_dictionary_t const dictionary(env, corpus, substrings_slice_t::sampled_k, sz_substrings_cased_k,
                                             allocator);
    if (dictionary.needles.empty()) return;
    substrings_corpus_t const resident(corpus, substrings_small_batch_haystacks_k);
    simt_tape_t haystacks_tape;
    sz_sequence_t const device_haystacks = haystacks_tape.copy(resident.haystacks);
    std::string const suffix = substrings_label(substrings_slice_t::sampled_k, sz_substrings_cased_k) + ":h" +
                               std::to_string(resident.views.size());
    for (sz_substrings_overlap_policy_t const policy :
         {sz_substrings_overlapping_k, sz_substrings_leftmost_longest_k}) {
        substrings_engine_t host(dictionary, policy, substrings_residency_t::host_k);
        substrings_engine_t device(dictionary, policy, substrings_residency_t::device_k);
        std::string const cover = suffix + substrings_policy_name(policy);
        bench_substrings_counts(env, corpus, host, device, resident, device_haystacks, cover);
        bench_substrings_find(env, corpus, host, device, resident, device_haystacks, cover);
    }
}

/** Documents a long-document round cuts the corpus into: fewer than the device has multiprocessors,
 *  so one block per document would leave most of it idle. */
enum { substrings_documents_k = 8 };

/** BM25 over the corpus cut into a few long documents, the device arm against the CPU one. */
static void bench_substrings_documents(environment_t const &env, corpus_t const &corpus) {
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_unified_best(&allocator, simt_capabilities());
    substrings_dictionary_t const dictionary(env, corpus, substrings_slice_t::sampled_k, sz_substrings_cased_k,
                                             allocator);
    if (dictionary.needles.empty()) return;
    char const *const first = corpus.tokens.front().data();
    std::size_t const bytes = (std::size_t)(corpus.tokens.back().data() + corpus.tokens.back().size() - first);
    std::vector<sz_string_view_t> documents(substrings_documents_k);
    for (std::size_t index = 0; index != documents.size(); ++index) {
        std::size_t const begin = bytes * index / documents.size(), end = bytes * (index + 1) / documents.size();
        documents[index] = {first + begin, end - begin};
    }
    substrings_corpus_t const resident(documents);
    simt_tape_t haystacks_tape;
    sz_sequence_t const device_haystacks = haystacks_tape.copy(resident.haystacks);
    substrings_engine_t host(dictionary, sz_substrings_overlapping_k, substrings_residency_t::host_k);
    substrings_engine_t device(dictionary, sz_substrings_overlapping_k, substrings_residency_t::device_k);
    bench_substrings_bm25(env, corpus, host, device, resident, device_haystacks,
                          substrings_label(substrings_slice_t::sampled_k, sz_substrings_cased_k) + ":d" +
                              std::to_string(documents.size()));
}

#pragma endregion Substrings Verbs

#pragma region Drivers

/** Every family's rows over the multilingual lines, each on a corpus it keeps resident; a build
 *  host without a device logs as much and passes. */
inline int bench_simt_main() {
    install_bench_signal_handlers();
    environment_t env {read_settings(), probe_machine()};
    print(env.machine);
    print(env.settings);
    if (env.machine.device_name.empty()) return 0;
    try {
        corpus_t const &corpus = env.corpora.multilingual_lines();
        {
            levenshtein_simt_corpus_t resident(env, corpus);
            fmt::println("Starting Levenshtein benchmarks over {} resident candidates...", resident.count());
            bench_levenshtein_cross_product(env, corpus, resident, median_token_bytes(corpus));
            bench_levenshtein_word_counts(env, corpus, resident);
        }
        {
            overlap_simt_corpus_t resident(overlap_token_views(env, corpus));
            fmt::println("Starting window overlap benchmarks over {} resident candidates...", resident.views.size());
            bench_overlap_scores(env, corpus, resident, median_token_bytes(corpus), "");
            overlap_simt_corpus_t skewed(
                simt_skewed_views(corpus.dataset.data(), corpus.dataset.size(), resident.views.size(),
                                  (double)resident.bytes / (double)resident.views.size(), env.settings.seed.value));
            bench_overlap_scores(env, corpus, skewed, median_token_bytes(corpus), ":skewed");
        }
        {
            substrings_corpus_t const resident(corpus);
            simt_tape_t haystacks_tape;
            sz_sequence_t const device_haystacks = haystacks_tape.copy(resident.haystacks);
            fmt::println("Starting multi-pattern search benchmarks...");
            bench_substrings_slice(env, corpus, resident, device_haystacks, substrings_slice_t::frequent_k,
                                   sz_substrings_cased_k);
            bench_substrings_slice(env, corpus, resident, device_haystacks, substrings_slice_t::rare_k,
                                   sz_substrings_cased_k);
            bench_substrings_slice(env, corpus, resident, device_haystacks, substrings_slice_t::frequent_k,
                                   sz_substrings_uncased_k);
            bench_substrings_slice(env, corpus, resident, device_haystacks, substrings_slice_t::sampled_k,
                                   sz_substrings_cased_k);
        }
        bench_substrings_small_batch(env, corpus);
        bench_substrings_documents(env, corpus);
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
