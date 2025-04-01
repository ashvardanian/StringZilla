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
#include "test.hpp" // `levenshtein_baseline`, `unary_substitution_costs`

#if SZ_USE_CUDA
#include <stringzilla/similarity.cuh> // Parallel string processing on CUDA or OpenMP
#endif

#if SZ_USE_OPENMP
#include <stringzilla/similarity.hpp> // OpenMP templates for string similarity measures
#endif

using namespace ashvardanian::stringzilla::scripts;

#pragma region Hamming Distance

/** @brief Wraps a hardware-specific Hamming-distance backend into something @b `bench_unary`-compatible . */
template <sz_hamming_distance_t hamming_distance_>
struct hamming_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env[token_index], env[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_size_t result_distance;
        sz_status_t status = hamming_distance_( //
            a.data(), a.size(),                 //
            b.data(), b.size(),                 //
            bound, &result_distance);
        do_not_optimize(status);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        return {bytes_passed, static_cast<check_value_t>(result_distance)};
    }
};

void bench_hamming(environment_t const &env) {
    auto base_call = hamming_from_sz<sz_hamming_distance_serial>(env);
    bench_result_t base = bench_unary(env, "sz_hamming_distance_serial", base_call).log();
    auto base_utf8_call = hamming_from_sz<sz_hamming_distance_utf8_serial>(env);
    bench_result_t base_utf8 = bench_unary(env, "sz_hamming_distance_utf8_serial", base_utf8_call).log(base);
    sz_unused(base_utf8);
}

#pragma endregion

#pragma region Levenshtein Distance and Alignment Scores

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <sz_levenshtein_distance_t levenshtein_distance_>
struct levenshtein_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env[token_index], env[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_size_t result_distance;
        sz_status_t status = levenshtein_distance_( //
            a.data(), a.size(),                     //
            b.data(), b.size(),                     //
            bound, NULL, &result_distance);
        do_not_optimize(status);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
template <sz_needleman_wunsch_score_t needleman_wunsch_>
struct alignment_score_from_sz {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;
    error_costs_256x256_t costs = unary_substitution_costs();

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env[token_index], env[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_ssize_t result_score;
        sz_status_t status = needleman_wunsch_( //
            a.data(), a.size(),                 //
            b.data(), b.size(),                 //
            costs.data(), (sz_error_cost_t)-1,  //
            NULL, &result_score);
        do_not_optimize(status);
        sz_size_t result_distance = (sz_size_t)(-result_score);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

#if SZ_USE_OPENMP

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
struct levenshtein_from_sz_openmp {

    environment_t const &env;
    sz_size_t bound = SZ_SIZE_MAX;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(env[token_index], env[env.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept(false) {
        sz_size_t result_distance = sz::openmp::levenshtein_distance(a, b, std::allocator<char>());
        do_not_optimize(result_distance);
        std::size_t bytes_passed = std::min(a.size(), b.size());
        std::size_t cells_passed = a.size() * b.size();
        return {bytes_passed, static_cast<check_value_t>(result_distance), cells_passed};
    }
};

#endif

#if SZ_USE_CUDA

/** @brief Wraps a hardware-specific Levenshtein-distance backend into something @b `bench_unary`-compatible . */
struct levenshtein_from_sz_cuda {

    environment_t const &env;
    std::vector<sz_size_t, sz::cuda::unified_alloc<sz_size_t>> results;
    sz_size_t bound = SZ_SIZE_MAX;

    levenshtein_from_sz_cuda(environment_t const &env, sz_size_t batch_size) : env(env), results(batch_size) {
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
        sz::status_t status = sz::cuda::levenshtein_distances(a, b, results.data());
        if (status != sz::status_t::success_k) throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
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
        return call_result;
    }
};

#endif

void bench_edits(environment_t const &env) {
    auto base_call = levenshtein_from_sz<sz_levenshtein_distance_serial>(env);
    bench_result_t base = bench_unary(env, "sz_levenshtein_distance_serial", base_call).log();
    auto base_utf8_call = levenshtein_from_sz<sz_levenshtein_distance_utf8_serial>(env);
    bench_result_t base_utf8 = bench_unary(env, "sz_levenshtein_distance_utf8_serial", base_utf8_call).log(base);
    sz_unused(base_utf8);

#if SZ_USE_OPENMP
    bench_unary(env, "sz::openmp::levenshtein_distance", levenshtein_from_sz_openmp(env)).log(base);
#endif
#if SZ_USE_CUDA
    bench_unary(env, "sz::cuda::levenshtein_distances(x1024)", levenshtein_from_sz_cuda(env, 1024)).log(base);
#endif

#if SZ_USE_ICE
    auto ice_call = levenshtein_from_sz<sz_levenshtein_distance_ice>(env);
    bench_unary(env, "sz_levenshtein_distance_ice", ice_call).log(base);
#endif

    auto needleman_wunsch_call = alignment_score_from_sz<sz_needleman_wunsch_score_serial>(env);
    bench_unary(env, "sz_needleman_wunsch_score_serial", needleman_wunsch_call).log(base);
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
        bench_hamming(env);
        bench_edits(env);
    }
    catch (std::exception const &e) {
        std::fprintf(stderr, "Failed with: %s\n", e.what());
        return 1;
    }

    std::printf("All benchmarks finished.\n");
    return 0;
}