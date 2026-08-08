/**
 *  @file scripts/bench_find_many.cuh
 *  @brief Shared code for CPU and GPU multi-pattern search (Aho-Corasick) benchmarks.
 */
#include <cstring> // `std::memcpy`, `std::memcmp`

#include <algorithm> // `std::sort`, `std::unique`
#include <numeric>   // `std::accumulate`
#include <string>    // `std::string`
#include <tuple>     // `std::tuple`, `std::apply`
#include <utility>   // `std::declval`
#include <vector>    // `std::vector`

#include "stringzillas/find_many/serial.hpp"

#if SZ_USE_CUDA
#include "stringzillas/find_many/cuda.cuh"
#endif

#include "shared.hpp"
#include "stringzilla.hpp" // `status_name`

namespace ashvardanian {
namespace stringzilla {
namespace scripts {

// Per-symbol: a using-directive re-exports our `memcpy` and nvcc then finds the call ambiguous.

// StringZillas library symbols available on every backend:
using ashvardanian::stringzillas::find_many_alphabet_size_k;
using ashvardanian::stringzillas::find_many_cased_k;
using ashvardanian::stringzillas::find_many_match_t;
using ashvardanian::stringzillas::find_many_case_sensitivity_t;
using ashvardanian::stringzillas::find_many_u32_dictionary_t;
using ashvardanian::stringzillas::find_many_u32_parallel_t;
using ashvardanian::stringzillas::find_many_u32_serial_t;
using ashvardanian::stringzillas::find_many_uncased_k;
using ashvardanian::stringzillas::forkunion_executor_t;

// StringZillas library symbols provided only by the CUDA backend:
#if SZ_USE_CUDA
using ashvardanian::stringzillas::cuda_executor_t;
using ashvardanian::stringzillas::cuda_status_t;
using ashvardanian::stringzillas::find_many_u32_cuda_t;
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
 *  @brief The post-cutoff vocabulary: distinct corpus terms with both noisy ends removed, frequency-ordered.
 *
 *  Two cutoffs, both applied before any slice is taken. The most frequent one percent are stopwords that
 *  match on nearly every byte while building no trie depth, so they measure output cost rather than
 *  automaton behaviour. Terms occurring once are hapax - mostly OCR noise and URLs no haystack reaches twice.
 *  Every span borrows from `env.dataset`, so building this copies no token bytes.
 *
 *  @note Needs `STRINGWARS_UNIQUE` unset or `0`. `build_environment` deduplicates `env.tokens` in place when
 *        it is set, which destroys the very frequency signal this selection rests on.
 */
struct vocabulary_t {
    unified_vector<span<char const>> terms; // ? Frequency-descending, ties broken by first appearance.
    unified_vector<size_t> counts;          // ? Parallel to `terms`.
    size_t dropped_frequent = 0;
    size_t dropped_hapax = 0;
    size_t total_occurrences = 0;

    size_t size() const noexcept { return terms.size(); }
};

/** @brief Orders spans by content, so a sort groups equal terms into runs a single pass can count. */
bool spans_less(span<char const> left, span<char const> right) noexcept {
    size_t const shared = left.size() < right.size() ? left.size() : right.size();
    int const ordering = std::memcmp(left.data(), right.data(), shared);
    return ordering != 0 ? ordering < 0 : left.size() < right.size();
}

/**
 *  @brief Counts term frequencies over `env.tokens` and drops both noisy ends.
 *  @param[in] frequent_cutoff Fraction of the frequency-ordered vocabulary discarded from the top.
 *  @param[in] minimum_occurrences Terms appearing fewer times than this are discarded outright.
 *
 *  Frequencies come from run lengths in a sorted array of spans, not a hash map.
 */
vocabulary_t build_vocabulary(environment_t const &env, double frequent_cutoff = 0.01, size_t minimum_occurrences = 2) {
    unified_vector<span<char const>> sorted;
    sorted.reserve(env.tokens.size());
    // No UTF-8 validation: tokens split on ASCII whitespace, so a malformed token means a malformed corpus,
    // which should fail loudly from `try_build` rather than silently shrink the vocabulary here.
    for (token_view_t const &token : env.tokens) {
        if (token.size() < 3 || token.size() > 32) continue;
        sorted.push_back({token.data(), token.size()});
    }
    std::sort(sorted.begin(), sorted.end(), spans_less);

    // Equal terms are adjacent after the sort, so one linear pass over the runs yields every frequency.
    auto const same_term = [](span<char const> left, span<char const> right) noexcept {
        return left.size() == right.size() && std::memcmp(left.data(), right.data(), left.size()) == 0;
    };
    vocabulary_t vocabulary;
    for (size_t run_start = 0; run_start < sorted.size();) {
        size_t run_end = run_start + 1;
        while (run_end < sorted.size() && same_term(sorted[run_start], sorted[run_end])) ++run_end;
        vocabulary.terms.push_back(sorted[run_start]);
        vocabulary.counts.push_back(run_end - run_start);
        vocabulary.total_occurrences += run_end - run_start;
        run_start = run_end;
    }

    // Order by frequency first, then by content, so the ranking is deterministic across runs and platforms.
    unified_vector<size_t> order(vocabulary.terms.size());
    for (size_t index = 0; index < order.size(); ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) noexcept {
        if (vocabulary.counts[left] != vocabulary.counts[right])
            return vocabulary.counts[left] > vocabulary.counts[right];
        return spans_less(vocabulary.terms[left], vocabulary.terms[right]);
    });

    size_t const drop_from_top = (size_t)((double)order.size() * frequent_cutoff);
    unified_vector<span<char const>> kept_terms;
    unified_vector<size_t> kept_counts;
    kept_terms.reserve(order.size());
    kept_counts.reserve(order.size());
    for (size_t rank = 0; rank < order.size(); ++rank) {
        size_t const index = order[rank];
        if (rank < drop_from_top) {
            ++vocabulary.dropped_frequent;
            continue;
        }
        if (vocabulary.counts[index] < minimum_occurrences) {
            ++vocabulary.dropped_hapax;
            continue;
        }
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
void print_dictionary_properties(find_many_u32_dictionary_t const &dictionary, needle_slice_t const &needles) {
    size_t const state_count = dictionary.count_states();
    size_t const hot_count = dictionary.hot_count();
    size_t const hot_tier_bytes = hot_count * find_many_alphabet_size_k * sizeof(u32_t);
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
 *  @brief Wraps one `find_many` engine call into a `bench_nullary`-compatible nullary callable.
 *
 *  @p invocable names the method and its arguments at the call site, so this template never learns which
 *  operation it is timing. Reports the device-measured kernel GB/s once it goes out of scope, as
 *  `similarities_callable` does; CPU engines accumulate no device time and are skipped.
 */
template <typename invocable_type_, typename output_type_>
struct find_many_callable {
    invocable_type_ invocable; // ? Returns the engine's status; the call site supplies the call.
    size_t total_bytes = 0;
    output_type_ const *output = nullptr; // ? The container `arrays_equality` compares two runs on.

    double kernel_milliseconds_total = 0.0;
    double kernel_bytes_total = 0.0;

    ~find_many_callable() {
        if (kernel_milliseconds_total <= 0.0 || kernel_bytes_total <= 0.0) return;
        double const kernel_gigabytes_per_second = kernel_bytes_total / (kernel_milliseconds_total * 1e6);
        std::printf("> Kernel: %.3f GB/s @ %.3f ms device-measured (excludes host materialization)\n",
                    kernel_gigabytes_per_second, kernel_milliseconds_total);
    }

    call_result_t operator()() noexcept(false) {
        engine_timing_t const timing = invoke_engine_(invocable);
        if (timing.status != status_t::success_k)
            throw std::runtime_error(std::string("find_many operation failed: ") + status_name(timing.status));
        kernel_milliseconds_total += timing.kernel_milliseconds;
        kernel_bytes_total += (double)total_bytes;

        call_result_t result;
        result.bytes_passed = total_bytes;
        result.operations = total_bytes; // ? Lets `log()` print bytes/cycle from its shared efficiency line.
        result.check_value = reinterpret_cast<check_value_t>(output);
        return result;
    }
};

/** @brief Builds a `find_many_callable` with both template arguments deduced. */
template <typename invocable_type_, typename output_type_>
find_many_callable<invocable_type_, output_type_> make_find_many_callable( //
    size_t total_bytes, output_type_ const &output, invocable_type_ &&invocable) {
    return {std::forward<invocable_type_>(invocable), total_bytes, &output};
}

#pragma endregion Timed Callables

#pragma region Sweep

/** @brief A `try_find` output buffer bigger than this is skipped, so a dictionary of very common short
 *         needles can't run the box out of memory. */
static constexpr size_t find_many_matches_safety_cap_k = 200'000'000;

/**
 *  @brief One measured configuration of the sweep.
 *
 *  The frequent slices are short terms that keep the walk near the root; the rare slices are long terms that
 *  drive it deep. Every printed line carries this label, so no number is read positionally.
 */
struct find_many_sweep_cell_t {
    vocabulary_slice_t slice;
    find_many_case_sensitivity_t sensitivity;

    char const *sensitivity_name() const noexcept { return sensitivity == find_many_uncased_k ? "uncased" : "cased"; }
    std::string label() const { return std::string(sensitivity_name()) + ":" + vocabulary_slice_name(slice); }
};

/** @brief The sweep both the CPU and the CUDA entry points walk, declared once so they cannot drift apart. */
static find_many_sweep_cell_t const find_many_sweep_k[] = {
    {vocabulary_slice_t::most_frequent_1_percent_k, find_many_cased_k},
    {vocabulary_slice_t::most_frequent_10_percent_k, find_many_cased_k},
    {vocabulary_slice_t::least_frequent_1_percent_k, find_many_cased_k},
    {vocabulary_slice_t::least_frequent_10_percent_k, find_many_cased_k},
    {vocabulary_slice_t::entire_k, find_many_cased_k},
    {vocabulary_slice_t::most_frequent_1_percent_k, find_many_uncased_k},
    {vocabulary_slice_t::most_frequent_10_percent_k, find_many_uncased_k},
    {vocabulary_slice_t::least_frequent_1_percent_k, find_many_uncased_k},
    {vocabulary_slice_t::least_frequent_10_percent_k, find_many_uncased_k},
    {vocabulary_slice_t::entire_k, find_many_uncased_k},
};

/**
 *  @brief Benchmarks one sweep cell on every backend this build links, so a GPU build reports its own CPU
 *         baselines beside the device numbers. Device work is gated inline, as in `similarities.cuh`.
 */
void bench_find_many_dictionary(                                        //
    environment_t const &env, size_t haystack_bytes,                    //
    vocabulary_t const &vocabulary, find_many_sweep_cell_t const &cell, //
    cpu_specs_t const &cpu_specs, forkunion_executor_t &pool) {

    std::string const dictionary_label = cell.label();
    std::printf("\n=== Dictionary %s ===\n", dictionary_label.c_str());

    needle_slice_t const needles = needle_slice_of(vocabulary, cell.slice);
    if (needles.size() == 0) {
        std::printf("Skipping: the post-cutoff vocabulary is too small for this slice\n");
        return;
    }

    // Construction reports needle-bytes per second through the same callable as the searches below.
    find_many_u32_serial_t rebuilt_engine;
    auto build_call = make_find_many_callable(needles.total_bytes, rebuilt_engine, [&] {
        rebuilt_engine.reset(); // ? Rebuilding from scratch on every call IS the measured operation
        return rebuilt_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    });
    bench_nullary(env, "find_many_build_serial:" + dictionary_label, build_call).log();

    find_many_u32_serial_t serial_engine;
    status_t const serial_build_status = serial_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    if (serial_build_status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the serial dictionary: ") +
                                 status_name(serial_build_status));

    find_many_u32_parallel_t parallel_engine;
    status_t const parallel_build_status = parallel_engine.try_build(needles.terms, cell.sensitivity, cpu_specs);
    if (parallel_build_status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the parallel dictionary: ") +
                                 status_name(parallel_build_status));

    print_dictionary_properties(serial_engine.dictionary(), needles);

    unified_vector<size_t> serial_counts(env.tokens.size());
    unified_vector<size_t> parallel_counts(env.tokens.size());

    size_t serial_total = 0, parallel_total = 0;
    auto serial_call = make_find_many_callable(haystack_bytes, serial_counts, [&] {
        return serial_engine.try_count(env.tokens, span<size_t>(serial_counts.data(), serial_counts.size()),
                                       serial_total);
    });
    std::string const serial_name = "find_many_count_serial:" + dictionary_label;
    bench_result_t const serial_result = bench_nullary(env, serial_name, serial_call).log();

    auto parallel_call = make_find_many_callable(haystack_bytes, parallel_counts, [&] {
        return parallel_engine.try_count(env.tokens, span<size_t>(parallel_counts.data(), parallel_counts.size()),
                                         parallel_total, pool, cpu_specs);
    });
    std::string const parallel_name = "find_many_count_parallel:" + dictionary_label;
    bench_result_t const parallel_result = bench_nullary(env, parallel_name, serial_call, parallel_call,
                                                         callable_no_op_t {}, arrays_equality<size_t> {})
                                               .log(serial_result);

    // Printed once per cell: the same number for every backend, and a bytes-per-second figure means little
    // without it, since a match-saturated dictionary and a near-miss one differ by an order of magnitude.
    size_t const total_occurrences = serial_total;
    std::printf(" - Occurrences: %zu across %zu haystacks\n", total_occurrences, serial_counts.size());

    // `try_find` materializes every match, so a dictionary of very common short needles can blow up the
    // output buffer; size it from the `try_count` pass above and skip outright past the safety cap.
    if (total_occurrences == 0 || total_occurrences > find_many_matches_safety_cap_k) {
        std::printf("Skipping try_find: %zu occurrences %s the safety cap of %zu\n", total_occurrences,
                    total_occurrences == 0 ? "is" : "exceeds", find_many_matches_safety_cap_k);
        return;
    }

    size_t serial_found = 0, parallel_found = 0;
    unified_vector<find_many_match_t> serial_matches(total_occurrences);
    auto serial_find_call = make_find_many_callable(haystack_bytes, serial_matches, [&] {
        return serial_engine.try_find(env.tokens, span<find_many_match_t>(serial_matches.data(), serial_matches.size()),
                                      serial_found);
    });
    std::string const serial_find_name = "find_many_find_serial:" + dictionary_label;
    bench_result_t const serial_find_result = bench_nullary(env, serial_find_name, serial_find_call).log();

    unified_vector<find_many_match_t> parallel_matches(total_occurrences);
    auto parallel_find_call = make_find_many_callable(haystack_bytes, parallel_matches, [&] {
        return parallel_engine.try_find(env.tokens,
                                        span<find_many_match_t>(parallel_matches.data(), parallel_matches.size()),
                                        parallel_found, pool, cpu_specs);
    });
    std::string const parallel_find_name = "find_many_find_parallel:" + dictionary_label;
    bench_result_t const parallel_find_result =
        bench_nullary(env, parallel_find_name, parallel_find_call).log(serial_find_result);

#if SZ_USE_CUDA
    gpu_specs_t gpu_specs;
    if (gpu_specs_fetch(gpu_specs) != status_t::success_k) throw std::runtime_error("Failed to fetch GPU specs.");

    find_many_u32_cuda_t device_engine;
    cuda_status_t const device_build_status = device_engine.try_build(needles.terms, cell.sensitivity);
    if (device_build_status.status != status_t::success_k)
        throw std::runtime_error(std::string("Failed to build the device dictionary: ") +
                                 status_name(device_build_status.status));

    // The device engine reports the same per-haystack breakdown, so this compares element-wise against the
    // serial pass that already ran.
    unified_vector<size_t> device_counts(env.tokens.size(), 0);
    size_t device_total = 0;

    auto cuda_call = make_find_many_callable(haystack_bytes, device_counts, [&] {
        return device_engine.try_count(env.tokens, span<size_t>(device_counts.data(), device_counts.size()),
                                       device_total, cuda_executor_t {}, gpu_specs);
    });
    std::string const cuda_name = "find_many_count_cuda:" + dictionary_label;
    bench_result_t const cuda_result = bench_nullary(env, cuda_name, cuda_call).log(serial_result);
    if (!cuda_result.skipped) {
        bool const matches_reference = arrays_equality<size_t> {}(reinterpret_cast<check_value_t>(&device_counts),
                                                                  reinterpret_cast<check_value_t>(&serial_counts));
        std::printf("> Checksum: %zu occurrences (device) vs %zu (host reference)%s, %zu calls, %.3f raw seconds\n",
                    device_total, total_occurrences, matches_reference ? "" : " -- MISMATCH",
                    cuda_result.profiled_calls, cuda_result.profiled_seconds);
    }
#endif
}

/** @brief The whole sweep: builds the vocabulary once, then walks every slice and sensitivity on every
 *         backend this build links. */
void bench_find_many(environment_t const &env) {
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

    cpu_specs_t const cpu_specs;
    forkunion_executor_t pool;
    if (pool.try_spawn(std::thread::hardware_concurrency()) != status_t::success_k)
        throw std::runtime_error("Failed to spawn the thread pool.");

    std::printf("Starting find-many benchmarks...\n");
    for (find_many_sweep_cell_t const &cell : find_many_sweep_k)
        bench_find_many_dictionary(env, haystack_bytes, vocabulary, cell, cpu_specs, pool);
}

#pragma endregion Sweep

} // namespace scripts
} // namespace stringzilla
} // namespace ashvardanian
