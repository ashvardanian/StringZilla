/**
 *  @file scripts/bench_substrings.cuh
 *  @brief Shared code for CPU and GPU multi-pattern search (Aho-Corasick) benchmarks.
 */
#include <cstring> // `std::memcpy`, `std::memcmp`

#include <algorithm>     // `std::sort`, `std::unique`
#include <numeric>       // `std::accumulate`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view` keys for the frequency map
#include <unordered_map> // `std::unordered_map` for the word-frequency count
#include <utility>       // `std::declval`
#include <vector>        // `std::vector`

#include "stringzillas/substrings/serial.hpp"

#if SZ_USE_CUDA
#include "stringzillas/substrings/cuda.cuh"
#endif

#include "shared.hpp"
#include "stringzilla.hpp" // `status_name`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.

// StringZillas library symbols available on every backend:
using ashvardanian::stringzillas::substrings_alphabet_size_k;
using ashvardanian::stringzillas::substrings_cased_k;
using ashvardanian::stringzillas::substrings_match_t;
using ashvardanian::stringzillas::substrings_case_sensitivity_t;
using ashvardanian::stringzillas::substrings_u32_dictionary_t;
using ashvardanian::stringzillas::substrings_u32_parallel_t;
using ashvardanian::stringzillas::substrings_u32_serial_t;
using ashvardanian::stringzillas::substrings_uncased_k;
using ashvardanian::stringzillas::forkunion_executor_t;

// StringZillas library symbols provided only by the CUDA backend:
#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::cuda_status_t;
using ashvardanian::stringzillas::substrings_u32_cuda_t;
using ashvardanian::stringzillas::gpu_specs_fetch;
#endif

#pragma region Needle Dictionaries

/**
 *  @brief One contiguous slice of the frequency-ordered vocabulary, plus its byte total.
 *
 *  Borrows twice over: the terms are a subrange of the vocabulary's own array, and each term points into
 *  `env.dataset`. `total_bytes` is the denominator construction throughput is reported against, since a
 *  needle count alone compares badly across corpora.
 */
struct needle_slice_t {
    span<span<char const> const> terms;
    size_t total_bytes = 0;

    size_t size() const noexcept { return terms.size(); }
};

/**
 *  @brief The post-cutoff vocabulary: distinct corpus words with both noisy ends removed, frequency-ordered.
 *
 *  Words are cut from the raw dataset by whitespace regardless of how `STRINGWARS_TOKENS` shapes the
 *  haystacks, so `lines` and `file` searches draw needles from the same vocabulary a `words` search does.
 *  Two cutoffs, both applied before any slice is taken. The most frequent one percent are stopwords that
 *  match on nearly every byte while building no trie depth, so they measure output cost rather than
 *  automaton behaviour. Terms occurring once are hapax - mostly OCR noise and URLs no haystack reaches twice.
 *  Every span borrows from `env.dataset`, so building this copies no token bytes.
 */
struct vocabulary_t {
    unified_vector<span<char const>> terms; // ? Frequency-descending, ties broken by first appearance.
    unified_vector<size_t> counts;          // ? Parallel to `terms`.
    size_t dropped_frequent = 0;
    size_t dropped_hapax = 0;
    size_t dropped_short = 0;
    size_t dropped_long = 0;
    size_t total_occurrences = 0;

    size_t size() const noexcept { return terms.size(); }
};

/** @brief Shortest word admitted into the vocabulary: anything under three bytes matches at nearly every
 *         position, so the benchmark would measure match materialization rather than the automaton walk. */
constexpr size_t vocabulary_min_word_bytes_k = 3;

/** @brief Longest word admitted into the vocabulary. Longer whitespace-cut tokens are unsegmented CJK or
 *         Thai runs and URLs rather than words - and the longest needle sets `max_match_bytes`, which every
 *         GPU chunk re-walks as its warm-up, so one runaway "word" would tax every chunk in the corpus. */
constexpr size_t vocabulary_max_word_bytes_k = 32;

/** @brief Orders spans by content, so a sort groups equal terms into runs a single pass can count. */
bool spans_less(span<char const> left, span<char const> right) noexcept {
    size_t const shared = left.size() < right.size() ? left.size() : right.size();
    int const ordering = std::memcmp(left.data(), right.data(), shared);
    return ordering != 0 ? ordering < 0 : left.size() < right.size();
}

/**
 *  @brief Counts word frequencies over the whole dataset and drops both noisy ends.
 *  @param[in] frequent_cutoff Fraction of the frequency-ordered vocabulary discarded from the top.
 *  @param[in] minimum_occurrences Terms appearing fewer times than this are discarded outright.
 *
 *  One hashed counting pass over the corpus, then a sort over only the distinct survivors - the corpus has
 *  hundreds of millions of words but only a few million distinct ones, so sorting every occurrence instead
 *  used to dominate the whole suite's start-up.
 */
vocabulary_t build_vocabulary(environment_t const &env, double frequent_cutoff = 0.01, size_t minimum_occurrences = 2) {
    // No UTF-8 validation: words split on ASCII whitespace, so a malformed word means a malformed corpus,
    // which should fail loudly from `try_build` rather than silently shrink the vocabulary here.
    // The library's own primitives do the walking: each `split` step is one `sz_find_byteset` SIMD scan and
    // the map hashes through `sz_hash`. No `reserve` - growth rehashes move only the distinct entries,
    // never the occurrences, so their total cost is a few multiples of the final table.
    std::unordered_map<std::string_view, size_t, sz::hash, sz::equal_to> frequencies;
    size_t dropped_short = 0, dropped_long = 0;
    for (auto word : sz::string_view {env.dataset.data(), env.dataset.size()}.split()) {
        if (word.size() == 0) continue; // ? Runs of separators yield empty segments, not words
        if (word.size() < vocabulary_min_word_bytes_k) {
            ++dropped_short;
            continue;
        }
        if (word.size() > vocabulary_max_word_bytes_k) {
            ++dropped_long;
            continue;
        }
        ++frequencies[std::string_view {word.data(), word.size()}];
    }

    // The hapax filter runs before any sort, so the ranking pass touches only the few percent that survive.
    // The top cutoff keeps its base: a fraction of ALL distinct terms, hapax included, so the slice matches
    // what the run-length formulation selected.
    vocabulary_t vocabulary;
    vocabulary.dropped_short = dropped_short;
    vocabulary.dropped_long = dropped_long;
    vocabulary.terms.reserve(frequencies.size());
    vocabulary.counts.reserve(frequencies.size());
    for (auto const &entry : frequencies) {
        vocabulary.total_occurrences += entry.second;
        if (entry.second < minimum_occurrences) {
            ++vocabulary.dropped_hapax;
            continue;
        }
        vocabulary.terms.push_back({entry.first.data(), entry.first.size()});
        vocabulary.counts.push_back(entry.second);
    }

    // Order by frequency first, then by content, so the ranking is deterministic across runs and platforms.
    unified_vector<size_t> order(vocabulary.terms.size());
    for (size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) noexcept {
        if (vocabulary.counts[left] != vocabulary.counts[right])
            return vocabulary.counts[left] > vocabulary.counts[right];
        return spans_less(vocabulary.terms[left], vocabulary.terms[right]);
    });

    size_t const drop_from_top = (size_t)((double)frequencies.size() * frequent_cutoff);
    vocabulary.dropped_frequent = drop_from_top < order.size() ? drop_from_top : order.size();
    unified_vector<span<char const>> kept_terms;
    unified_vector<size_t> kept_counts;
    kept_terms.reserve(order.size());
    kept_counts.reserve(order.size());
    for (size_t rank = vocabulary.dropped_frequent; rank < order.size(); ++rank) {
        size_t const index = order[rank];
        kept_terms.push_back(vocabulary.terms[index]);
        kept_counts.push_back(vocabulary.counts[index]);
    }
    vocabulary.terms = std::move(kept_terms);
    vocabulary.counts = std::move(kept_counts);
    return vocabulary;
}

/** @brief Which end of the frequency-ordered vocabulary a dictionary is drawn from, and how much of it. */
enum class vocabulary_slice_t {
    most_frequent_1_percent_k,
    most_frequent_10_percent_k,
    least_frequent_1_percent_k,
    least_frequent_10_percent_k,
    entire_k,
};

/** @brief Human-readable slice tag for every benchmark label.
 *  @note Reaches `std::printf` as an argument to a `%s`, never as a format, so the `%` stays literal. */
constexpr char const *vocabulary_slice_name(vocabulary_slice_t slice) noexcept {
    switch (slice) {
    case vocabulary_slice_t::most_frequent_1_percent_k: return "most_frequent_1%";
    case vocabulary_slice_t::most_frequent_10_percent_k: return "most_frequent_10%";
    case vocabulary_slice_t::least_frequent_1_percent_k: return "least_frequent_1%";
    case vocabulary_slice_t::least_frequent_10_percent_k: return "least_frequent_10%";
    case vocabulary_slice_t::entire_k: return "entire";
    }
    return "unknown";
}

/**
 *  @brief Takes one slice of @p vocabulary as a dictionary.
 *
 *  Slices are taken by term count, so the frequent and the rare slice of the same percentage hold the same
 *  number of needles and differing byte totals - Zipf makes frequent terms short.
 */
needle_slice_t needle_slice_of(vocabulary_t const &vocabulary, vocabulary_slice_t slice) {
    size_t const available = vocabulary.size();
    size_t wanted = available;
    size_t first = 0;
    switch (slice) {
    case vocabulary_slice_t::most_frequent_1_percent_k: wanted = available / 100; break;
    case vocabulary_slice_t::most_frequent_10_percent_k: wanted = available / 10; break;
    case vocabulary_slice_t::least_frequent_1_percent_k: wanted = available / 100, first = available - wanted; break;
    case vocabulary_slice_t::least_frequent_10_percent_k: wanted = available / 10, first = available - wanted; break;
    case vocabulary_slice_t::entire_k: break;
    }
    if (wanted == 0) return {};

    needle_slice_t result;
    result.terms = {vocabulary.terms.data() + first, wanted};
    for (span<char const> const &term : result.terms) result.total_bytes += term.size();
    return result;
}

#pragma endregion Needle Dictionaries

#pragma region Reporting

/** @brief Structural facts about one compiled automaton. Construction @b throughput is not printed here -
 *         it is measured by `bench_nullary` like every other operation, so it carries the same units and
 *         the same min-of-N latency. */
void print_dictionary_properties(substrings_u32_dictionary_t const &dictionary, needle_slice_t const &needles) {
    size_t const state_count = dictionary.count_states();
    size_t const hot_count = dictionary.hot_count();
    size_t const hot_tier_bytes = hot_count * substrings_alphabet_size_k * sizeof(u32_t);
    size_t const cold_tier_bytes = dictionary.transitions_bytes() - hot_tier_bytes;

    std::printf(" - Needles: %zu requested, %zu inserted, %zu bytes\n", needles.size(), dictionary.count_needles(),
                needles.total_bytes);
    std::printf(" - States: %zu total, %zu hot (%.1f%%), %zu cold\n", state_count, hot_count,
                state_count ? 100.0 * (double)hot_count / (double)state_count : 0.0, state_count - hot_count);
    std::printf(" - Hot tier: %zu bytes (%.2f MiB)\n", hot_tier_bytes, hot_tier_bytes / (1024.0 * 1024.0));
    std::printf(" - Cold tier: %zu bytes (%.2f MiB)\n", cold_tier_bytes, cold_tier_bytes / (1024.0 * 1024.0));
    std::printf(" - Max match length: %zu bytes, max outputs per state: %zu\n", (size_t)dictionary.max_match_bytes(),
                (size_t)dictionary.view().max_outputs_per_state);
}

#pragma endregion Reporting

#pragma region Timed Callables

/**
 *  @brief Wraps one `substrings` engine call into a `bench_nullary`-compatible nullary callable.
 *
 *  @p invocable names the method and its arguments at the call site, so this template never learns which
 *  operation it is timing. Reports the device-measured kernel GB/s once it goes out of scope, as
 *  `similarities_callable` does; CPU engines accumulate no device time and are skipped.
 */
template <typename invocable_type_, typename output_type_>
struct substrings_callable {
    invocable_type_ invocable; // ? Returns the engine's status; the call site supplies the call.
    size_t total_bytes = 0;
    output_type_ const *output = nullptr; // ? The container `arrays_equality` compares two runs on.

    double kernel_milliseconds_total = 0.0;
    double kernel_bytes_total = 0.0;

    ~substrings_callable() {
        if (kernel_milliseconds_total <= 0.0 || kernel_bytes_total <= 0.0) return;
        double const kernel_gigabytes_per_second = kernel_bytes_total / (kernel_milliseconds_total * 1e6);
        std::printf("> Kernel: %.3f GB/s @ %.3f ms device-measured (excludes host materialization)\n",
                    kernel_gigabytes_per_second, kernel_milliseconds_total);
    }

    call_result_t operator()() noexcept(false) {
        engine_timing_t const timing = invoke_engine_(invocable);
        if (timing.status != status_t::success_k)
            throw std::runtime_error(std::string("substrings operation failed: ") + status_name(timing.status));
        kernel_milliseconds_total += timing.kernel_milliseconds;
        kernel_bytes_total += (double)total_bytes;

        call_result_t result;
        result.bytes_passed = total_bytes;
        result.operations = total_bytes; // ? Lets `log()` print bytes/cycle from its shared efficiency line.
        result.check_value = reinterpret_cast<check_value_t>(output);
        return result;
    }
};

/** @brief Builds a `substrings_callable` with both template arguments deduced. */
template <typename invocable_type_, typename output_type_>
substrings_callable<invocable_type_, output_type_> make_substrings_callable( //
    size_t total_bytes, output_type_ const &output, invocable_type_ &&invocable) {
    return {std::forward<invocable_type_>(invocable), total_bytes, &output};
}

#pragma endregion Timed Callables

#pragma region Sweep

/** @brief A `try_find` output buffer bigger than this is skipped, so a dictionary of very common short
 *         needles can't run the box out of memory. */
static constexpr size_t substrings_matches_safety_cap_k = 200'000'000;

/**
 *  @brief One measured configuration of the sweep.
 *
 *  The frequent slices are short terms that keep the walk near the root; the rare slices are long terms that
 *  drive it deep. Every printed line carries this label, so no number is read positionally.
 */
struct substrings_sweep_cell_t {
    vocabulary_slice_t slice;
    substrings_case_sensitivity_t sensitivity;

    char const *sensitivity_name() const noexcept { return sensitivity == substrings_uncased_k ? "uncased" : "cased"; }
    std::string label() const { return std::string(sensitivity_name()) + ":" + vocabulary_slice_name(slice); }
};

/** @brief The sweep both the CPU and the CUDA entry points walk, declared once so they cannot drift apart. */
static substrings_sweep_cell_t const substrings_sweep_k[] = {
    {vocabulary_slice_t::most_frequent_1_percent_k, substrings_cased_k},
    {vocabulary_slice_t::most_frequent_10_percent_k, substrings_cased_k},
    {vocabulary_slice_t::least_frequent_1_percent_k, substrings_cased_k},
    {vocabulary_slice_t::least_frequent_10_percent_k, substrings_cased_k},
    {vocabulary_slice_t::entire_k, substrings_cased_k},
    {vocabulary_slice_t::most_frequent_1_percent_k, substrings_uncased_k},
    {vocabulary_slice_t::most_frequent_10_percent_k, substrings_uncased_k},
    {vocabulary_slice_t::least_frequent_1_percent_k, substrings_uncased_k},
    {vocabulary_slice_t::least_frequent_10_percent_k, substrings_uncased_k},
    {vocabulary_slice_t::entire_k, substrings_uncased_k},
};

/**
 *  @brief Benchmarks one sweep cell on every backend this build links, so a GPU build reports its own CPU
 *         baselines beside the device numbers. Device work is gated inline, as in `similarities.cuh`.
 */
void bench_substrings_dictionary(                                        //
    environment_t const &env, size_t haystack_bytes,                     //
    vocabulary_t const &vocabulary, substrings_sweep_cell_t const &cell, //
    cpu_specs_t const &cpu_specs, forkunion_executor_t &pool) {

    std::string const dictionary_label = cell.label();
    std::printf("\n=== Dictionary %s ===\n", dictionary_label.c_str());

    needle_slice_t const needles = needle_slice_of(vocabulary, cell.slice);
    if (needles.size() == 0) {
        std::printf("Skipping: the post-cutoff vocabulary is too small for this slice\n");
        return;
    }

    // Construction reports needle-bytes per second through the same callable as the searches below.
    substrings_u32_serial_t rebuilt_engine;
    auto build_call = make_substrings_callable(needles.total_bytes, rebuilt_engine, [&] {
        rebuilt_engine.reset(); // ? Rebuilding from scratch on every call IS the measured operation
        return rebuilt_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    });
    bench_nullary(env, "substrings_build_serial:" + dictionary_label, build_call).log();

    substrings_u32_serial_t serial_engine;
    status_t const serial_build_status = serial_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    if (serial_build_status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the serial dictionary: ") +
                                 status_name(serial_build_status));

    substrings_u32_parallel_t parallel_engine;
    status_t const parallel_build_status = parallel_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    if (parallel_build_status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the parallel dictionary: ") +
                                 status_name(parallel_build_status));

    print_dictionary_properties(serial_engine.dictionary(), needles);

    unified_vector<size_t> serial_counts(env.tokens.size());
    unified_vector<size_t> parallel_counts(env.tokens.size());

    size_t serial_total = 0, parallel_total = 0;
    auto serial_call = make_substrings_callable(haystack_bytes, serial_counts, [&] {
        return serial_engine.try_count(env.tokens, span<size_t>(serial_counts.data(), serial_counts.size()),
                                       serial_total);
    });
    std::string const serial_name = "substrings_count_serial:" + dictionary_label;
    bench_result_t const serial_result = bench_nullary(env, serial_name, serial_call).log();

    auto parallel_call = make_substrings_callable(haystack_bytes, parallel_counts, [&] {
        return parallel_engine.try_count(env.tokens, span<size_t>(parallel_counts.data(), parallel_counts.size()),
                                         parallel_total, pool, cpu_specs);
    });
    std::string const parallel_name = "substrings_count_parallel:" + dictionary_label;
    bench_result_t const parallel_result = bench_nullary(env, parallel_name, serial_call, parallel_call,
                                                         callable_no_op_t {}, arrays_equality<size_t> {})
                                               .log(serial_result);

    // Printed once per cell: the same number for every backend, and a bytes-per-second figure means little
    // without it, since a match-saturated dictionary and a near-miss one differ by an order of magnitude.
    size_t const total_occurrences = serial_total;
    std::printf(" - Occurrences: %zu across %zu haystacks\n", total_occurrences, serial_counts.size());

    // `try_find` materializes every match, so a dictionary of very common short needles can blow up the
    // output buffer; size it from the `try_count` pass above and skip outright past the safety cap.
    if (total_occurrences == 0 || total_occurrences > substrings_matches_safety_cap_k) {
        std::printf("Skipping try_find: %zu occurrences %s the safety cap of %zu\n", total_occurrences,
                    total_occurrences == 0 ? "is" : "exceeds", substrings_matches_safety_cap_k);
        return;
    }

    size_t serial_found = 0, parallel_found = 0;
    unified_vector<substrings_match_t> serial_matches(total_occurrences);
    auto serial_find_call = make_substrings_callable(haystack_bytes, serial_matches, [&] {
        return serial_engine.try_find(
            env.tokens, span<substrings_match_t>(serial_matches.data(), serial_matches.size()), serial_found);
    });
    std::string const serial_find_name = "substrings_find_serial:" + dictionary_label;
    bench_result_t const serial_find_result = bench_nullary(env, serial_find_name, serial_find_call).log();

    unified_vector<substrings_match_t> parallel_matches(total_occurrences);
    auto parallel_find_call = make_substrings_callable(haystack_bytes, parallel_matches, [&] {
        return parallel_engine.try_find(env.tokens,
                                        span<substrings_match_t>(parallel_matches.data(), parallel_matches.size()),
                                        parallel_found, pool, cpu_specs);
    });
    std::string const parallel_find_name = "substrings_find_parallel:" + dictionary_label;
    bench_result_t const parallel_find_result =
        bench_nullary(env, parallel_find_name, parallel_find_call).log(serial_find_result);

#if SZ_USE_CUDA
    gpu_specs_t gpu_specs;
    if (gpu_specs_fetch(gpu_specs) != status_t::success_k) throw std::runtime_error("Failed to fetch GPU specs.");

    substrings_u32_cuda_t device_engine;
    cuda_status_t const device_build_status = device_engine.try_build(needles.terms, cell.sensitivity);
    if (device_build_status.status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the device dictionary: ") +
                                 status_name(device_build_status.status));

    // The device engine reports the same per-haystack breakdown, so this compares element-wise against the
    // serial pass that already ran.
    unified_vector<size_t> device_counts(env.tokens.size(), 0);
    size_t device_total = 0;

    auto cuda_call = make_substrings_callable(haystack_bytes, device_counts, [&] {
        return device_engine.try_count(env.tokens, span<size_t>(device_counts.data(), device_counts.size()),
                                       device_total, cuda_executor_t {}, gpu_specs);
    });
    std::string const cuda_name = "substrings_count_cuda:" + dictionary_label;
    bench_result_t const cuda_result = bench_nullary(env, cuda_name, cuda_call).log(serial_result);
    if (!cuda_result.skipped) {
        bool const matches_reference = arrays_equality<size_t> {}(reinterpret_cast<check_value_t>(&device_counts),
                                                                  reinterpret_cast<check_value_t>(&serial_counts));
        std::printf("> Checksum: %zu occurrences (device) vs %zu (host reference)%s, %zu calls, %.3f raw seconds\n",
                    device_total, total_occurrences, matches_reference ? "" : " -- MISMATCH",
                    cuda_result.profiled_calls, cuda_result.profiled_seconds);
    }

    // Sized from the serial `try_count` above, like the CPU `try_find` calls, so the scatter never grows
    // device memory mid-benchmark.
    size_t device_found = 0;
    unified_vector<substrings_match_t> device_matches(total_occurrences);
    auto cuda_find_call = make_substrings_callable(haystack_bytes, device_matches, [&] {
        return device_engine.try_find(env.tokens,
                                      span<substrings_match_t>(device_matches.data(), device_matches.size()),
                                      device_found, cuda_executor_t {}, gpu_specs);
    });
    std::string const cuda_find_name = "substrings_find_cuda:" + dictionary_label;
    bench_result_t const cuda_find_result = bench_nullary(env, cuda_find_name, cuda_find_call).log(serial_find_result);
    if (!cuda_find_result.skipped && device_found != total_occurrences)
        std::printf("> Checksum: %zu matches (device) vs %zu (host reference) -- MISMATCH\n", device_found,
                    total_occurrences);
#endif
}

/** @brief The whole sweep: builds the vocabulary once, then walks every slice and sensitivity on every
 *         backend this build links. */
void bench_substrings(environment_t const &env) {
    // `STRINGWARS_UNIQUE` deduplicates `env.tokens` in place, leaving both cutoffs nothing to rank by.
    if (env.unique)
        std::printf(                                                                                   //
            "WARNING: STRINGWARS_UNIQUE is set, so every term counts once and both frequency cutoffs " //
            "are meaningless. Unset it for a real vocabulary.\n");
    size_t const haystack_bytes = std::accumulate(
        env.tokens.begin(), env.tokens.end(), (size_t)0,
        [](size_t total, token_view_t const &token) noexcept { return total + token.size(); });
    std::printf(" - Haystacks: %zu spans, %zu bytes total\n", env.tokens.size(), haystack_bytes);
    vocabulary_t const vocabulary = build_vocabulary(env);
    std::printf(                                                                                  //
        " - Vocabulary: %zu terms after cutoffs, %zu occurrences, dropped %zu most-frequent and " //
        "%zu under-2-occurrence\n",                                                               //
        vocabulary.size(), vocabulary.total_occurrences, vocabulary.dropped_frequent, vocabulary.dropped_hapax);
    std::printf(                                                                       //
        " - Length gates: dropped %zu words under %zu bytes and %zu over %zu bytes\n", //
        vocabulary.dropped_short, vocabulary_min_word_bytes_k, vocabulary.dropped_long, vocabulary_max_word_bytes_k);

    forkunion_executor_t pool;
    if (pool.try_spawn(std::thread::hardware_concurrency()) != status_t::success_k)
        throw std::runtime_error("Failed to spawn the thread pool.");
    cpu_specs_t const cpu_specs = pool.specs();

    std::printf("Starting substrings benchmarks...\n");
    for (substrings_sweep_cell_t const &cell : substrings_sweep_k)
        bench_substrings_dictionary(env, haystack_bytes, vocabulary, cell, cpu_specs, pool);
}

#pragma endregion Sweep

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
