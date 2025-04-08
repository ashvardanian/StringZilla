/**
 *  @file   bench_similarity.cpp
 *  @brief  Benchmarks string similarity computations.
 *          It accepts a file with a list of words, and benchmarks the levenshtein edit-distance computations,
 *          alignment scores, and fingerprinting techniques combined with the Hamming distance.
 *
 *  Benchmarks include:
 *  - Linear-complexity basic & bounded Hamming distance computations.
 *  - Quadratic-complexity basic & bounded Levenshtein edit-distance computations.
 *  - Quadratic-complexity Needleman-Wunsch alignment scores for bioinformatics.
 *
 *  For Dynamic Programming algorithms, the number of operations per second are reported as the worst-case time
 *  complexity of the Cells Updates Per Second @b (CUPS) metric, meaning O(N*M) for a pair of strings with N and M
 *  characters, respectively.
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
 *  cmake --build build_release --config Release --target stringzilla_bench_similarity
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=words build_release/stringzilla_bench_similarity
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=proteins.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_similarity
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_search.cpp`, `bench_token.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */

#include "bench.hpp"
#include "test_stringcuzilla.cuh" // `levenshtein_baseline`, `error_costs_256x256_unary`

#if SZ_USE_CUDA
#include <stringcuzilla/similarity.cuh> // Parallel string processing on CUDA or OpenMP
#endif

#if SZ_USE_OPENMP
#include <stringcuzilla/similarity.hpp> // OpenMP templates for string similarity measures
#endif

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using namespace std::literals; // for ""sv

using similarities_t = unified_vector<sz_ssize_t>;
using levenshtein_serial_t = sz::levenshtein_distances<sz_cap_parallel_k, char, std::allocator<char>>;
using levenshtein_cuda_t = sz::levenshtein_distances<sz_cap_cuda_k, char>;

#pragma region Levenshtein Distance and Alignment Scores

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <typename engine_type_>
struct batch_callable {
    using engine_t = engine_type_;

    environment_t const &env;
    similarities_t &results;
    sz_size_t bound = SZ_SIZE_MAX;
    engine_t engine = {};

    batch_callable(environment_t const &env, similarities_t &res, sz_size_t batch_size) : env(env), results(res) {
        if (env.tokens.size() <= batch_size) throw std::runtime_error("Batch size is too large.");
    }

    inline call_result_t operator()(std::size_t batch_index) noexcept(false) {
        std::size_t const batch_size = results.size();
        std::size_t const forward_token_index = (batch_index * batch_size) % (env.tokens.size() - batch_size);
        std::size_t const backward_token_index = env.tokens.size() - forward_token_index - batch_size;

        return operator()({env.tokens.data() + forward_token_index, batch_size},
                          {env.tokens.data() + backward_token_index, batch_size});
    }

    inline call_result_t operator()(std::span<token_view_t const> a, std::span<token_view_t const> b) noexcept(false) {
        sz::status_t status = engine(a, b, results.data());
        if (status != sz::status_t::success_k) throw std::runtime_error("Failed to compute Levenshtein distance.");
        do_not_optimize(results);
        std::size_t bytes_passed = 0, cells_passed = 0;
        for (std::size_t i = 0; i < results.size(); ++i) {
            bytes_passed += std::min(a[i].size(), b[i].size());
            cells_passed += a[i].size() * b[i].size();
        }
        call_result_t call_result;
        call_result.bytes_passed = bytes_passed;
        call_result.operations = cells_passed;
        call_result.inputs_processed = results.size();
        call_result.check_value = reinterpret_cast<check_value_t>(&results);
        return call_result;
    }
};

struct similarities_equality_t {
    bool operator()(check_value_t const &a, check_value_t const &b) const {
        similarities_t const &a_ = *reinterpret_cast<similarities_t const *>(a);
        similarities_t const &b_ = *reinterpret_cast<similarities_t const *>(b);
        if (a_.size() != b_.size()) return false;
        for (std::size_t i = 0; i < a_.size(); ++i)
            if (a_[i] != b_[i]) {
                std::printf("Mismatch at index %zu: %zd != %zd\n", i, a_[i], b_[i]);
                return false;
            }
        return true;
    }
};

void bench_levenshtein(environment_t const &env) {

    std::vector<std::size_t> batch_sizes = {1024 / 32, 1024, 1024 * 32};
    similarities_t results_baseline, results_accelerated;

    for (std::size_t batch_size : batch_sizes) {
        results_baseline.resize(batch_size);
        results_accelerated.resize(batch_size);

        auto call_baseline = batch_callable<levenshtein_serial_t>(env, results_baseline, batch_size);
        auto name_baseline = "levenshtein_serial:batch"s + std::to_string(batch_size);
        bench_result_t baseline = bench_unary(env, name_baseline, call_baseline).log();

        bench_result_t accelerated =
            bench_unary(env, "levenshtein_cuda:batch"s + std::to_string(batch_size), call_baseline,
                        batch_callable<levenshtein_cuda_t>(env, results_accelerated, batch_size),
                        callable_no_op_t {},        // preprocessing
                        similarities_equality_t {}) // equality check
                .log(baseline);
    }
}

#pragma endregion

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    try {
        std::printf("Building up the environment...\n");
        environment_t env = build_environment( //
            argc, argv,                        //
            "xlsum.csv",                       // Preferred for UTF-8 content
            environment_t::tokenization_t::lines_k);

        std::printf("Starting string similarity benchmarks...\n");
        bench_levenshtein(env);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}