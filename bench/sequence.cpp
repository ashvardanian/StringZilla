
/**
 *  @file bench/sequence.cpp
 *  @author Ash Vardanian
 *  @date June 26, 2023
 *  @brief Benchmarks sorting, partitioning, and merging operations on string sequences.
 *
 *  Times the sorting and intersection dispatch points over the English words, against the STL and
 *  @c qsort. Every capability's kernels, and the pgram sorts only the tier headers define, are
 *  timed against the serial ones by the `cross_<arch>.cpp` files.
 *
 *  Memory-bound: sort cost is dominated by cache-missing permutation over the whole collection, so
 *  it reads the whole file by default.
 *
 *  Benchmarks include:
 *  - String sequence sorting algorithms - @b argsort and @b pgrams_sort.
 *  - String sequences intersections - @b intersect.
 *
 *  For sorting, the number of operations per second are reported as the worst-case time complexity
 *  of a comparison-based sorting algorithm, meaning O(N × log(N)) for N elements. For
 *  intersections, the number of operations is estimated as the total number of characters in the
 *  two input sequences combined.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment
 *  variables are used:
 *  - `STRINGWARS_DATASET=path` : Path to the dataset file.
 *  - `STRINGWARS_DATASET_LIMIT=0` : Reads at most this many dataset bytes; `0` reads the whole
 *    file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer
 *    [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_MAX_SECONDS=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER=pattern` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words STRINGWARS_FILTER='sort|intersect' \
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `find.cpp`, `token.cpp`, and `memory.cpp`.
 */
#include <memory>        // `std::memcpy`
#include <numeric>       // `std::iota`
#include <unordered_set> // `std::unordered_set`

#if __linux__ && defined(_GNU_SOURCE)
#include <stdlib.h> // `qsort_r`
#endif

#include <fmt/format.h>

#include "cross.hpp"

using namespace ashvardanian::stringzilla::bench;

#if __linux__ && defined(_GNU_SOURCE) && !defined(__BIONIC__)
#define STRINGZILLA_HAS_QSORT_R_ 1
#else
#define STRINGZILLA_HAS_QSORT_R_ 0
#endif
#if defined(_MSC_VER)
#define STRINGZILLA_HAS_QSORT_S_ 1
#else
#define STRINGZILLA_HAS_QSORT_S_ 0
#endif

namespace {

#pragma region C Callbacks

#if STRINGZILLA_HAS_QSORT_R_ || STRINGZILLA_HAS_QSORT_S_

/**
 *  @brief Callback function for the @b qsort_r re-entrant sorting function.
 *  @note The @c qsort_r function is not available on all platforms, and is not in the C standard.
 */
#if defined(_MSC_VER)
int _get_qsort_order(void *arg, void const *a, void const *b) {
#else
int _get_qsort_order(void const *a, void const *b, void *arg) {
#endif
    sz_sequence_t *sequence = (sz_sequence_t *)arg;
    sz_size_t idx_a = *(sz_size_t *)a;
    sz_size_t idx_b = *(sz_size_t *)b;

    sz_cptr_t str_a = sequence->get_start(sequence->handle, idx_a);
    sz_cptr_t str_b = sequence->get_start(sequence->handle, idx_b);
    sz_size_t len_a = sequence->get_length(sequence->handle, idx_a);
    sz_size_t len_b = sequence->get_length(sequence->handle, idx_b);

    int result = strncmp(str_a, str_b, len_a < len_b ? len_a : len_b);
    return result ? result : (int)(len_a - len_b);
}

#endif

#pragma endregion

#pragma region Sorting Benchmarks

struct argsort_strings_via_std_t {
    strings_t const &input;
    permute_t &output;

    argsort_strings_via_std_t(strings_t const &input, permute_t &output) : input(input), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);
        std::sort(output.begin(), output.end(),
                  [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return input[i] < input[j]; });

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

#if STRINGZILLA_HAS_QSORT_R_ || STRINGZILLA_HAS_QSORT_S_

struct argsort_strings_via_qsort_t {
    strings_t const &input;
    permute_t &output;

    argsort_strings_via_qsort_t(strings_t const &input, permute_t &output) : input(input), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);

        // Prepare the sequence structure for the callback.
        sz_sequence_t array;
        array.count = input.size();
        array.handle = &input;
        array.get_start = get_start;
        array.get_length = get_length;
#if STRINGZILLA_HAS_QSORT_R_
        qsort_r(output.data(), array.count, sizeof(sz_sorted_idx_t), _get_qsort_order, &array);
#elif STRINGZILLA_HAS_QSORT_S_
        qsort_s(output.data(), array.count, sizeof(sz_sorted_idx_t), _get_qsort_order, &array);
#endif

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

#endif

/**
 *  @brief Find the array permutation that sorts the input strings.
 *  @warning Some algorithms use more memory than others; this benchmark does not account for it.
 */
void bench_sequencing_strings(environment_t const &env) {
    permute_t permute_buffer(env.tokens.size());

    auto base_call = argsort_strings_via_sz<cpu_best<sz_sequence_argsort_best>> {env.tokens, permute_buffer};
    bench_result_t base = bench_nullary(env, "sz_sequence_argsort_best", base_call).log();

    // Include STL functionality
    auto std_call = argsort_strings_via_std_t {env.tokens, permute_buffer};
    bench_nullary(env, "sequence_argsort<std::sort>", base_call, std_call).log(base);

    // Include POSIX and WinAPI functionality
#if STRINGZILLA_HAS_QSORT_R_ || STRINGZILLA_HAS_QSORT_S_
    auto qsort_call = argsort_strings_via_qsort_t {env.tokens, permute_buffer};
    bench_nullary(env, "sequence_argsort<qsort>", base_call, qsort_call).log(base);
#endif
}

struct argsort_ci_strings_via_std_t {
    strings_t const &input;
    std::vector<std::string> const &folded;
    permute_t &output;

    argsort_ci_strings_via_std_t(strings_t const &input, std::vector<std::string> const &folded, permute_t &output)
        : input(input), folded(folded), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);
        std::stable_sort(output.begin(), output.end(),
                         [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return folded[i] < folded[j]; });

        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_uncased_sorting_permutation(folded, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

/**
 *  @brief Find the array permutation that sorts the input strings in UTF-8 case-folded order.
 *  @warning Some algorithms use more memory than others; this benchmark does not account for it.
 */
void bench_sequencing_strings_uncased(environment_t const &env) {
    permute_t permute_buffer(env.tokens.size());
    std::vector<std::string> const folded = fold_tokens(env.tokens);

    auto base_call = argsort_ci_strings_via_sz<cpu_best<sz_sequence_argsort_uncased_best>> {env.tokens, folded,
                                                                                            permute_buffer};
    bench_result_t base = bench_nullary(env, "sz_sequence_argsort_uncased_best", base_call).log();

    // Include STL functionality, as the case-folded reference
    auto std_call = argsort_ci_strings_via_std_t {env.tokens, folded, permute_buffer};
    bench_nullary(env, "sequence_argsort_uncased<std::stable_sort>", base_call, std_call).log(base);
}

#pragma endregion

#pragma region P-grams Sorting Benchmarks

struct sort_pgrams_via_std_t {
    pgrams_t const &input;
    permute_t &output;

    sort_pgrams_via_std_t(pgrams_t const &input, permute_t &output) : input(input), output(output) {}

    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);
        std::sort(output.begin(), output.end(),
                  [&](sz_sorted_idx_t i, sz_sorted_idx_t j) { return input[i] < input[j]; });

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output);
        std::size_t bytes_passed = input.size() * sizeof(sz_pgram_t);
        return {bytes_passed, checksum, ops_performed};
    }
};

/**
 *  @brief Sort the tokens' leading bytes, which are integers, before the strings themselves.
 *
 *  The pgram sorts have no dispatch point, so the STL stands alone here, and the sorts themselves
 *  are timed by the cross files.
 */
void bench_sequencing_pgrams(environment_t const &env) {
    permute_t permute_buffer(env.tokens.size());
    pgrams_t const pgrams_buffer = pgrams_from_tokens(env);
    bench_nullary(env, "pgrams_sort<std::sort>", sort_pgrams_via_std_t {pgrams_buffer, permute_buffer}).log();
}

#pragma endregion

#pragma region Intersections Benchmarks

/** Uses the STL's @c std::unordered_map to find the intersections between two string sequences. */
struct intersect_strings_via_std_t {
    strings_t const &input_a;
    strings_t const &input_b;
    permute_t &output_a;
    permute_t &output_b;

    explicit intersect_strings_via_std_t(intersect_inputs_t &inputs)
        : input_a(inputs.tokens_a), input_b(inputs.tokens_b), output_a(inputs.permute_a), output_b(inputs.permute_b) {}

    call_result_t operator()() const {
        auto const &input_small = input_a.size() < input_b.size() ? input_a : input_b;
        auto const &input_large = input_a.size() < input_b.size() ? input_b : input_a;
        auto &output_small = input_a.size() < input_b.size() ? output_a : output_b;
        auto &output_large = input_a.size() < input_b.size() ? output_b : output_a;

        // Construct an unordered map for the smaller input
        std::unordered_map<std::string_view, sz_sorted_idx_t> map_small;
        for (sz_sorted_idx_t idx_in_small = 0; idx_in_small < input_small.size(); ++idx_in_small)
            map_small[input_small[idx_in_small]] = idx_in_small;

        // Iterate through the larger input and find the intersections
        std::size_t intersections = 0;
        for (sz_sorted_idx_t idx_in_large = 0; idx_in_large < input_large.size(); ++idx_in_large) {
            auto it = map_small.find(input_large[idx_in_large]);
            if (it == map_small.end()) continue;
            output_large[intersections] = idx_in_large;
            output_small[intersections] = it->second;
            ++intersections;
        }

        // Prepare stats
        check_value_t checksum = static_cast<check_value_t>(intersections);
        std::size_t bytes_passed = accumulate_lengths(input_a) + accumulate_lengths(input_b);
        return {bytes_passed, checksum, input_a.size() + input_b.size()};
    }
};

/**
 *  @brief Intersect every distinct token with a sample of half as many.
 *  @warning Some algorithms use more memory than others; this benchmark does not account for it.
 */
void bench_intersections(environment_t const &env) {
    intersect_inputs_t inputs(env);

    auto base_call = intersect_strings_via_sz<cpu_best<sz_sequence_intersect_best>> {inputs};
    bench_result_t base = bench_nullary(env, "sz_sequence_intersect_best", base_call).log();

    // Include STL functionality
    auto std_call = intersect_strings_via_std_t {inputs};
    bench_nullary(env, "intersect<std::unordered_map>", base_call, std_call).log(base);
}

#pragma endregion

} // namespace

void bench_sequence(corpora_t &corpora) {
    environment_t const &env = corpora.words();
    fmt::println("Starting sequence benchmarks...");
    bench_sequencing_pgrams(env);
    bench_sequencing_strings(env);
    bench_sequencing_strings_uncased(env);
    bench_intersections(env);
}
