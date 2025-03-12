
/**
 *  @file   bench_sort.cpp
 *  @brief  Benchmarks sorting, partitioning, and merging operations on string sequences.
 *          The program accepts a file path to a dataset, tokenizes it, and benchmarks the search operations,
 *          validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - String sequence sorting algorithms - @b argsort and @b pgrams_sort.
 *  - String sequences intersections - @b intersect.
 *
 *  For sorting, the number of operations per second are reported as the worst-case time complexity of a
 *  comparison-based sorting algorithm, meaning O(N*log(N)) for N elements. For intersections, the number of
 *  operations is estimated as the total number of characters in the two input sequences.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_sort
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words build_release/stringzilla_bench_sort
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_sort
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_search.cpp`, `bench_token.cpp`, `bench_similarity.cpp`, and `bench_memory.cpp`.
 */
#include <memory>  // `std::memcpy`
#include <numeric> // `std::iota`

#if __linux__ && defined(_GNU_SOURCE)
#include <stdlib.h> // `qsort_r`
#endif

#define SZ_USE_MISALIGNED_LOADS (1)
#include "bench.hpp"

using namespace ashvardanian::stringzilla::scripts;

using pgrams_t = std::vector<sz_pgram_t>;
using strings_t = std::vector<std::string_view>;
using permute_t = std::vector<sz_sorted_idx_t>;

#if __linux__ && defined(_GNU_SOURCE) && !defined(__BIONIC__)
#define _SZ_HAS_QSORT_R 1
#elif defined(_MSC_VER)
#define _SZ_HAS_QSORT_S 1
#endif

/** @brief Helper function to distill a large @b `permute_t` object down to a single comparable hash integer. */
template <typename entries_type_>
bool is_sorting_permutation(entries_type_ const &entries, permute_t const &permute) {
    return std::is_sorted(permute.begin(), permute.end(),
                          [&](std::size_t i, std::size_t j) { return entries[i] < entries[j]; });
}

/** @brief Helper function to accumulate the total length of all strings in a sequence. */
std::size_t accumulate_lengths(strings_t const &strings) {
    return std::accumulate(strings.begin(), strings.end(), (std::size_t)0,
                           [](std::size_t sum, std::string_view const &str) { return sum + str.size(); });
}

#pragma region C Callbacks

/** @brief Trampoline function to access @b `sz_cptr_t[]` arrays via @b `sz_sequence_t::get_start`. */
static sz_cptr_t get_start(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].data();
}

/** @brief Trampoline function to access @b `sz_cptr_t[]` arrays via @b `sz_sequence_t::get_length`. */
static sz_size_t get_length(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].size();
}

/**
 *  @brief Callback function for the @b `qsort_r` re-entrant sorting function.
 *  @note The `qsort_r` function is not available on all platforms, and is not part of the C standard.
 */
#if defined(_MSC_VER)
static int _get_qsort_order(void *arg, void const *a, void const *b) {
#else
static int _get_qsort_order(void const *a, void const *b, void *arg) {
#endif
    sz_sequence_t *sequence = (sz_sequence_t *)arg;
    sz_size_t idx_a = *(sz_size_t *)a;
    sz_size_t idx_b = *(sz_size_t *)b;

    sz_cptr_t str_a = sequence->get_start(sequence->handle, idx_a);
    sz_cptr_t str_b = sequence->get_start(sequence->handle, idx_b);
    sz_size_t len_a = sequence->get_length(sequence->handle, idx_a);
    sz_size_t len_b = sequence->get_length(sequence->handle, idx_b);

    int res = strncmp(str_a, str_b, len_a < len_b ? len_a : len_b);
    return res ? res : (int)(len_a - len_b);
}

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

#if defined(_SZ_HAS_QSORT_R) || defined(_SZ_HAS_QSORT_S)

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
#if defined(_SZ_HAS_QSORT_R)
        qsort_r(output.data(), array.count, sizeof(sz_sorted_idx_t), _get_qsort_order, &array);
#elif defined(_SZ_HAS_QSORT_S)
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

template <sz_sequence_argsort_t func_>
struct argsort_strings_via_sz {
    strings_t const &input;
    permute_t &output;

    argsort_strings_via_sz(strings_t const &input, permute_t &output) : input(input), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);

        // Prepare the sequence structure for the callback.
        sz_sequence_t array;
        array.count = input.size();
        array.handle = &input;
        array.get_start = get_start;
        array.get_length = get_length;
        sz::_with_alloc<std::allocator<char>>(
            [&](sz_memory_allocator_t &alloc) { return func_(&array, &alloc, output.data()); });

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

/**
 *  @brief Find the array permutation that sorts the input strings.
 *  @warning Some algorithms use more memory than others and memory usage is not accounted for in this benchmark.
 */
void bench_sorting_strings(environment_t const &env) {
    permute_t permute_buffer(env.tokens.size());

    // First, benchmark the STL function
    auto base_call = argsort_strings_via_std_t {env.tokens, permute_buffer};
    benchmark_result_t base = benchmark_nullary(env, "std::sort(positions)", base_call).log();
    auto serial_call = argsort_strings_via_sz<sz_sequence_argsort_serial> {env.tokens, permute_buffer};
    benchmark_nullary(env, "sz_sequence_argsort_serial(positions)", base_call, serial_call).log(base);

// Conditionally include SIMD-accelerated backends
#if SZ_USE_SKYLAKE
    auto skylake_call = argsort_strings_via_sz<sz_sequence_argsort_skylake> {env.tokens, permute_buffer};
    benchmark_nullary(env, "sz_sequence_argsort_skylake(positions)", base_call, skylake_call).log(base);
#endif
#if SZ_USE_SVE
    auto sve_call = argsort_strings_via_sz<sz_sequence_argsort_sve> {env.tokens, permute_buffer};
    benchmark_nullary(env, "sz_sequence_argsort_sve(positions)", base_call, sve_call).log(base);
#endif

    // Include POSIX functionality
#if __linux__ && defined(_GNU_SOURCE) && !defined(__BIONIC__)
    auto qsort_call = argsort_strings_via_qsort_t {env.tokens, permute_buffer};
    benchmark_nullary(env, "qsort_r(positions)", base_call, qsort_call).log(base);
#endif
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

template <sz_pgrams_sort_t func_>
struct sort_pgrams_via_sz {
    pgrams_t const &input;
    pgrams_t &output_sorted;
    permute_t &output_permutation;

    sort_pgrams_via_sz(pgrams_t const &input, pgrams_t &output_sorted, permute_t &output_permutation)
        : input(input), output_sorted(output_sorted), output_permutation(output_permutation) {}
    call_result_t operator()() const {
        std::copy(input.begin(), input.end(), output_sorted.begin());
        std::iota(output_permutation.begin(), output_permutation.end(), 0);

        // Prepare the sequence structure for the callback.
        sz::_with_alloc<std::allocator<char>>([&](sz_memory_allocator_t &alloc) {
            return func_(output_sorted.data(), output_sorted.size(), &alloc, output_permutation.data());
        });

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output_permutation);
        std::size_t bytes_passed = input.size() * sizeof(sz_pgram_t);
        return {bytes_passed, checksum, ops_performed};
    }
};

/**
 *  @brief Find the array permutation that sorts the input strings.
 *  @warning Some algorithms use more memory than others and memory usage is not accounted for in this benchmark.
 */
void bench_sorting_pgrams(environment_t const &env) {
    permute_t permute_buffer(env.tokens.size());

    // Before sorting the strings themselves, which is a heavy operation,
    // let's sort some prefixes to understand how the sorting algorithm behaves.
    pgrams_t pgrams_buffer(env.tokens.size()), pgrams_sorted(env.tokens.size());
    std::transform(env.tokens.begin(), env.tokens.end(), pgrams_buffer.begin(), [](std::string_view const &str) {
        sz_pgram_t pgram = 0;
        std::memcpy(&pgram, str.data(), (std::min)(sizeof(pgram), str.size()));
        return pgram;
    });

    // First, benchmark the STL function
    auto base_call = sort_pgrams_via_std_t {pgrams_buffer, permute_buffer};
    benchmark_result_t base = benchmark_nullary(env, "std::sort(pgrams)", base_call).log();
    auto serial_call = sort_pgrams_via_sz<sz_pgrams_sort_serial> {pgrams_buffer, pgrams_sorted, permute_buffer};
    benchmark_nullary(env, "sz_pgrams_sort_serial(pgrams)", base_call, serial_call).log(base);

    // Conditionally include SIMD-accelerated backends
#if SZ_USE_SKYLAKE
    auto skylake_call = sort_pgrams_via_sz<sz_pgrams_sort_skylake> {pgrams_buffer, pgrams_sorted, permute_buffer};
    benchmark_nullary(env, "sz_pgrams_sort_skylake(pgrams)", base_call, skylake_call).log(base);
#endif
#if SZ_USE_SVE
    auto sve_call = sort_pgrams_via_sz<sz_pgrams_sort_sve> {pgrams_buffer, pgrams_sorted, permute_buffer};
    benchmark_nullary(env, "sz_pgrams_sort_sve(pgrams)", base_call, sve_call).log(base);
#endif
}

#pragma endregion

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::words_k);

    std::printf("Starting search benchmarks...\n");
    bench_sorting_pgrams(env);
    bench_sorting_strings(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}