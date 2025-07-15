/**
 *  @file   bench_find_many.cuh
 *  @brief  Shared code for CPU and GPU batched parallel exact substring search.
 */
#include <tuple> // `std::tuple`
#include <span>  // `std::span`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include <stringzillas/find_many.hpp> // C++ templates for string processing

#if SZ_USE_CUDA
#include <stringzillas/find_many.cuh> // Parallel string processing in CUDA
#endif

#include "bench.hpp"

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

using namespace ashvardanian::stringzilla::scripts;

using counts_t = unified_vector<size_t>;
using matches_t = unified_vector<find_many_match_t>;

#pragma region Multi-Pattern Search

/** @brief Wraps a hardware-specific multi-pattern search backend into something @b `bench_nullary`-compatible . */
template <typename engine_type_, typename results_type_, typename... extra_args_>
struct find_many_callable {
    using engine_t = engine_type_;
    using results_t = results_type_;
    using result_t = typename results_t::value_type;

    environment_t const &env;
    counts_t &results_counts_per_haystack;
    matches_t &results_matches_per_haystack;
    find_many_u32_dictionary_t const &dictionary;
    engine_t engine = {};
    std::tuple<extra_args_...> extra_args = {};

    find_many_callable(environment_t const &env, counts_t &counts, matches_t &matches,
                       find_many_u32_dictionary_t const &dict, engine_t eng = {}, extra_args_... args)
        : env(env), results_counts_per_haystack(counts), results_matches_per_haystack(matches), dictionary(dict),
          engine(std::move(eng)), extra_args(args...) {}

    call_result_t operator()() noexcept(false) {

        using chars_view_t = span<char const>;
        chars_view_t const dataset_view = {env.dataset.data(), env.dataset.size()};
        span<chars_view_t const> haystacks = {&dataset_view, 1};

        // Without `volatile`, the serial logic keeps being optimized out!
        volatile status_t status = engine.try_build(dictionary);
        if (status != status_t::success_k) throw std::runtime_error("Failed to build dictionary.");
        span<size_t> counts_span = {results_counts_per_haystack.data(), results_counts_per_haystack.size()};
        span<find_many_match_t> matches_span = {results_matches_per_haystack.data(),
                                                results_matches_per_haystack.size()};

        // Unpack the extra arguments from `std::tuple` into the engine call using `std::apply`
        constexpr bool only_counts_k = std::is_same_v<results_t, counts_t>;
        if constexpr (only_counts_k)
            status = std::apply(
                [&](auto &&...rest) mutable {
                    auto result = engine.try_count(haystacks, counts_span, rest...);
                    for (auto &count : counts_span) do_not_optimize(count);
                    return result;
                },
                extra_args);
        else
            status = std::apply(
                [&](auto &&...rest) mutable {
                    auto result = engine.try_find(haystacks, counts_span, matches_span, rest...);
                    for (auto &match : matches_span) do_not_optimize(match);
                    return result;
                },
                extra_args);

        do_not_optimize(status);
        if (status != status_t::success_k) throw std::runtime_error("Failed multi-pattern search.");

        std::size_t needle_characters = engine.dictionary().total_needles_length();
        std::size_t bytes_passed = 0, character_comparisons = 0;
        for (std::size_t i = 0; i < haystacks.size(); ++i) {
            bytes_passed += haystacks[i].size();
            character_comparisons += haystacks[i].size() * needle_characters;
        }
        volatile call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = character_comparisons;
        call_result.inputs_processed = haystacks.size();
        call_result.check_value = only_counts_k ? reinterpret_cast<check_value_t>(&results_counts_per_haystack)
                                                : reinterpret_cast<check_value_t>(&results_matches_per_haystack);
        return (call_result_t const &)call_result;
    }
};

void bench_find_many(environment_t const &env) {

    using namespace std::string_literals; // for "s" suffix

#if SZ_USE_CUDA
    gpu_specs_t specs = *gpu_specs();
#endif
    std::vector<std::size_t> vocabulary_sizes = {
        1024,
        64,
        32 * 1024,
        1,
    };
#if SZ_DEBUG
    vocabulary_sizes = {1, 2, 64};
#endif
    counts_t counts_baseline, counts_accelerated;
    matches_t matches_baseline, matches_accelerated;

    using counts_equality_t = arrays_equality<size_t>;
    using matches_equality_t = arrays_equality<find_many_match_t>;

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    fork_union_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");
    static_assert(executor_like<fork_union_t>);

    auto scramble_accelerated_results = [&](auto &results_accelerated) {
        std::shuffle(results_accelerated.begin(), results_accelerated.end(), global_random_generator());
    };

    for (std::size_t vocabulary_size : vocabulary_sizes) {
        auto shape_suffix = vocabulary_size == 1 ? std::to_string(vocabulary_size) + "needle"s
                                                 : std::to_string(vocabulary_size) + "needles"s;
        if (vocabulary_size > env.tokens.size()) continue;

        // Construct the dictionary for the current vocabulary size
        find_many_u32_dictionary_t dict;
        if (dict.try_reserve(vocabulary_size) != status_t::success_k)
            throw std::runtime_error("Failed to reserve space for dictionary.");
        for (std::size_t token_index = 0; dict.count_needles() < vocabulary_size && token_index < env.tokens.size();
             ++token_index) {
            auto const &token = env.tokens[token_index];
            auto status = dict.try_insert({token.data(), token.size()});
            if (status == status_t::contains_duplicates_k) continue; // Skip duplicates
            if (status != status_t::success_k) throw std::runtime_error("Failed to insert token into dictionary.");
        }
        if (dict.try_build() != status_t::success_k) throw std::runtime_error("Failed to build dictionary.");

        // Estimate the amount of memory needed for the results
        std::size_t const results_count = dict.count({env.dataset.data(), env.dataset.size()});
        counts_baseline.resize(1), counts_accelerated.resize(1);
        matches_baseline.resize(results_count), matches_accelerated.resize(results_count);

        // Perform the benchmarks, passing the dictionary to the engines
        auto call_count_baseline =
            find_many_callable<find_many_u32_serial_t, counts_t>(env, counts_baseline, matches_baseline, dict);
        auto name_count_baseline = "count_many_serial:"s + shape_suffix;
        bench_result_t count_baseline = bench_nullary(env, name_count_baseline, call_count_baseline).log();

        auto call_find_baseline =
            find_many_callable<find_many_u32_serial_t, matches_t>(env, counts_baseline, matches_baseline, dict);
        auto name_find_baseline = "find_many_serial:"s + shape_suffix;
        bench_result_t find_baseline = bench_nullary(env, name_find_baseline, call_find_baseline).log();

        // Parallel search
        bench_nullary( //
            env, "count_many_parallel:"s + shape_suffix, call_count_baseline,
            find_many_callable<find_many_u32_parallel_t, counts_t, fork_union_t &>( //
                env, counts_accelerated, matches_accelerated, dict, {}, pool),
            callable_no_op_t {},  // preprocessing
            counts_equality_t {}) // equality check
            .log(count_baseline);

        bench_nullary( //
            env, "find_many_parallel:"s + shape_suffix, call_find_baseline,
            find_many_callable<find_many_u32_parallel_t, matches_t, fork_union_t &>( //
                env, counts_accelerated, matches_accelerated, dict, {}, pool),
            callable_no_op_t {},   // preprocessing
            matches_equality_t {}) // equality check
            .log(find_baseline);

        scramble_accelerated_results(counts_accelerated);
        scramble_accelerated_results(matches_accelerated);

        // CUDA-accelerated search
#if SZ_USE_CUDA
        bench_nullary( //
            env, "count_many_cuda:"s + shape_suffix, call_count_baseline,
            find_many_callable<find_many_u32_cuda_t, counts_t, cuda_executor_t, gpu_specs_t>( //
                env, counts_accelerated, matches_accelerated, dict, {}, {}, specs),
            callable_no_op_t {},  // preprocessing
            counts_equality_t {}) // equality check
            .log(count_baseline);

        bench_nullary( //
            env, "find_many_cuda:"s + shape_suffix, call_find_baseline,
            find_many_callable<find_many_u32_cuda_t, matches_t, cuda_executor_t, gpu_specs_t>( //
                env, counts_accelerated, matches_accelerated, dict, {}, {}, specs),
            callable_no_op_t {},   // preprocessing
            matches_equality_t {}) // equality check
            .log(find_baseline);

        scramble_accelerated_results(counts_accelerated);
        scramble_accelerated_results(matches_accelerated);
#endif
    }
}

#pragma endregion

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian